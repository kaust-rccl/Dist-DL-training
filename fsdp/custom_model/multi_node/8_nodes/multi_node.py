import os, time, re, string
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint
import pynvml; pynvml.nvmlInit()
import numpy as np
import wandb
from transformers import (
    TrainerCallback, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, BloomTokenizerFast
)
from datasets import load_dataset

# Config placeholders
MODEL_NAME      = os.getenv("MODEL_NAME", "bigscience/bloom-560m")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "default_experiment")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "./outputs")
MAX_LENGTH      = int(os.getenv("MAX_LENGTH", 512))
TRAIN_SIZE      = int(os.getenv("TRAIN_SIZE", 1000))
EVAL_SIZE       = int(os.getenv("EVAL_SIZE", 100))
BATCH_SIZE      = int(os.getenv("PER_DEVICE_BATCH_SIZE", 1))
GRAD_ACC_STEPS  = int(os.getenv("GRAD_ACC_STEPS", 4))
NUM_EPOCHS      = int(os.getenv("NUM_EPOCHS", 5))
LR              = float(os.getenv("LEARNING_RATE", 5e-5))
WEIGHT_DECAY    = float(os.getenv("WEIGHT_DECAY", 0.01))
FP16            = os.getenv("FP16", "True") == "True"
BF16            = os.getenv("BF16", "False") == "True"

VOCAB_SIZE   = int(os.getenv("VOCAB_SIZE",   50000))
HIDDEN_SIZE  = int(os.getenv("HIDDEN_SIZE",  3072))
NUM_LAYERS   = int(os.getenv("NUM_LAYERS",   24))
NUM_HEADS    = int(os.getenv("NUM_HEADS",    24))
FF_DIM       = int(os.getenv("FF_DIM",       12288))
SEQ_LENGTH   = int(os.getenv("SEQ_LENGTH",   MAX_LENGTH))

class SimpleGPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 2048,
        num_layers: int = 2,
        num_heads: int = 16,
        dim_feedforward: int = 5504,
        seq_length: int = 128,
    ):
        super().__init__()
        self.seq_length = seq_length

        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.lm_head.weight = self.token_embedding.weight

    # ⭐ forward now accepts the kwargs Trainer will pass
    def forward(
        self,
        input_ids,              # (batch, seq_len)
        attention_mask=None,    # ignored in this simple example
        labels=None,            # optional – return loss if provided
        **kwargs,
    ):
        #self.seq_length = input_ids.size(1)
        # (batch, seq_len, hidden)
        x = self.token_embedding(input_ids)
        self.seq_length = x.size(1)
        #x = x.transpose(0, 1)  # (seq_len, batch, hidden)

        # causal mask so tokens can’t see the future
        causal_mask = torch.triu(
            torch.full((self.seq_length, self.seq_length), float("-inf"), device=x.device), diagonal=1
        )
        # Transformer
        x = self.transformer(x, mask=causal_mask)
        #x = x.transpose(0, 1)  # (batch, seq_len, hidden)

        # logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab)

        if labels is not None:
            # shift so that token i predicts token i+1
            shift_logits  = logits[:, :-1].contiguous()
            shift_labels  = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, **kwargs):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"]          # (B, L, V)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

class UtilisationCallback(TrainerCallback):
    def __init__(self):
        self.sum_alloc = 0.0
        self.sum_reserved = 0.0
        self.sum_total = 0.0
        self.sum_util = 0.0
        self.sum_util2 = 0.0
        self.n_samples = 0
        self.final_local=None

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def on_step_end(self, args, state, control, **kwargs):
        gpu = torch.cuda.current_device()

        self.sum_alloc += torch.cuda.memory_allocated(gpu) / 1024**2
        self.sum_reserved += torch.cuda.memory_reserved(gpu) / 1024**2
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.sum_total += meminfo.used / 1024**2

        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        self.sum_util += util
        self.sum_util2 += util ** 2

        self.n_samples += 1
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.n_samples == 0:
            self.final_local = torch.tensor([0.,0.,0.,0.,0.,0.], device=args.device)
        else:
            self.final_local = torch.tensor(
                [
                    self.sum_alloc,
                    self.sum_reserved,
                    self.sum_total,
                    self.sum_util,
                    self.sum_util2,
                    self.n_samples,
                ],
                device=args.device,
            )
        return control


class TrainSpeedCallback(TrainerCallback):
    def __init__(self):
        self.t_prev = time.time()
        self.s_prev = 0
        self.total_steps = 0.0
        self.total_seconds = 0.0

    def on_log(self, args, state, control, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return control

        steps_now = state.global_step
        if steps_now == 0:
            return control

        dt = time.time() - self.t_prev
        ds = steps_now - self.s_prev
        if dt == 0:
            return control

        world = dist.get_world_size() if dist.is_initialized() else 1
        eff_batch = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * world
        )

        sps = ds / dt
        samp = sps * eff_batch

        wandb.log(
            {
                "train/steps_per_second": sps,
                "train/samples_per_second": samp,
            },
            step=steps_now,
        )

        self.total_steps += ds
        self.total_seconds += dt

        self.t_prev, self.s_prev = time.time(), steps_now
        return control

    def on_train_end(self, args, state, control, **kwargs):
        local = torch.tensor(
            [self.total_steps, self.total_seconds], device=args.device
        )
        if dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)

        world = dist.get_world_size() if dist.is_initialized() else 1
        eff_batch = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * world
        )

        if dist.get_rank() == 0 and local[1] > 0:
            mean_sps = local[0].item() / local[1].item()
            mean_samp = mean_sps * eff_batch
            wandb.run.summary.update(
                {
                    "train/steps_per_second_mean": round(mean_sps, 3),
                    "train/samples_per_second_mean": round(mean_samp, 3),
                }
            )

def load_squad():
    """Load and preprocess the SQuAD dataset for generative QA."""
    tokenizer = BloomTokenizerFast.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad")

    def preprocess_function(examples):
        inputs = [
            f"Question: {q} Context: {c} Answer:"
            for q, c in zip(examples["question"], examples["context"])
        ]
        answers = [
            a["text"][0] if a["text"] else "No Answer" for a in examples["answers"]
        ]

        model_inputs = tokenizer(
            inputs, truncation=True, padding="max_length", max_length=512
        )
        labels = tokenizer(
            answers, truncation=True, padding="max_length", max_length=512
        )
        labels["input_ids"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in lab]
            for lab in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized_dataset = tokenized_dataset.remove_columns(
        [c for c in tokenized_dataset["train"].column_names if c not in keep_cols]
    )
    return tokenized_dataset, tokenizer


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def compute_em_and_f1(predicted: str, ground_truth: str):
    """Return Exact-Match and F1 for a single prediction/reference pair."""
    pred = normalize_answer(predicted)
    gt = normalize_answer(ground_truth)

    em = int(pred == gt)
    pred_tokens = pred.split()
    gt_tokens = gt.split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return em, 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def evaluate_model(trainer, dataset, tokenizer):
    """Evaluate a model on the validation split using EM and F1."""
    em_scores, f1_scores = [], []

    for example in dataset:
        input_ids = torch.tensor([example["input_ids"]]).to(trainer.args.device)
        attention_mask = torch.tensor(
            [example["attention_mask"]]
        ).to(trainer.args.device)

        with trainer.model.summon_full_params(module=trainer.model):
            outputs = trainer.model.generate(
                input_ids, attention_mask=attention_mask, max_new_tokens=50, do_sample=False
            )

        generated_answer = tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).split("Answer:")[-1].strip()

        reference_answer = tokenizer.decode(
            [tok for tok in example["labels"] if tok != -100], skip_special_tokens=True
        ).strip()

        em, f1 = compute_em_and_f1(generated_answer, reference_answer)
        em_scores.append(em)
        f1_scores.append(f1)

    return {"exact_match": np.mean(em_scores), "f1": np.mean(f1_scores)}

def main():
    # Initialize distributed backend
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Prepare data and model
    ds, tokenizer = load_squad()
    train_ds = ds["train"].shuffle(42).select(range(TRAIN_SIZE))
    eval_ds = ds["validation"].shuffle(42).select(range(EVAL_SIZE))

    model = SimpleGPTModel(
            vocab_size      = tokenizer.vocab_size,
            hidden_size     = HIDDEN_SIZE,
            num_layers      = NUM_LAYERS,
            num_heads       = NUM_HEADS,
            dim_feedforward = FF_DIM,
            seq_length      = SEQ_LENGTH,
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

    # LM objective – drop labels
    train_ds = train_ds.remove_columns("labels")

    # FSDP configuration
    fsdp_cfg = {
        "mixed_precision": {
            "enabled": True,
            "param_dtype": torch.float16,
            "reduce_dtype": torch.float32,
            "buffer_dtype": torch.float16,
        },
        "transformer_layer_cls_to_wrap": ["torch.nn.modules.transformer.TransformerEncoderLayer"],
        "backward_prefetch": "backward_post",
        "forward_prefetch": True,
        "sync_module_states": True,
        "device_id":torch.cuda.current_device(),
        "state_dict_type": "sharded_state_dict",
        "offload_to_cpu": True,
        }

    # Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=EXPERIMENT_NAME,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=4,
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        fsdp="full_shard",
        fsdp_config=fsdp_cfg,
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[UtilisationCallback(), TrainSpeedCallback()]
    )

    # Start training
    trainer.train()

    util_cb = next(c for c in trainer.callback_handler.callbacks if isinstance(c, UtilisationCallback))

    # Move tensor to device for reduction
    local = util_cb.final_local.to(trainer.args.device)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.SUM)

    tot_alloc, tot_reserved, tot_total, tot_util, tot_util2, n = local.tolist()
    n = int(n)
    mean = lambda x: x / n if n else 0.0
    std_util = ((tot_util2 / n) - (mean(tot_util) ** 2)) ** 0.5 if n else 0.0

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        import wandb
        wandb.run.summary.update({
            "avg_mem_alloc_MB": round(mean(tot_alloc), 1),
            "avg_mem_reserved_MB": round(mean(tot_reserved), 1),
            "avg_mem_total_MB": round(mean(tot_total), 1),
            "avg_gpu_util_%": round(mean(tot_util), 1),
            "std_gpu_util_%": round(std_util, 1),
        })

    metrics = evaluate_model(trainer, eval_ds, tokenizer)
    print("Final Evaluation Metrics:", metrics)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
