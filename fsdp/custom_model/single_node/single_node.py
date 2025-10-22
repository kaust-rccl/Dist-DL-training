import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BloomTokenizerFast
)
from datasets import load_dataset
import numpy as np
import re
import string

# Config placeholders
MODEL_NAME = os.getenv("MODEL_NAME", "bigscience/bloomz-560m")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "default_experiment")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", 1000))
EVAL_SIZE = int(os.getenv("EVAL_SIZE", 100))
BATCH_SIZE = int(os.getenv("PER_DEVICE_BATCH_SIZE", 1))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1))
LR = float(os.getenv("LEARNING_RATE", 5e-5))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
FP16 = os.getenv("FP16", "False") == "True"
BF16 = os.getenv("BF16", "True") == "True"

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
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Final LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    # ⭐ forward now accepts the kwargs Trainer will pass
    def forward(
        self,
        input_ids,              # (batch, seq_len)
        attention_mask=None,    # ignored in this simple example
        labels=None,            # optional – return loss if provided
        **kwargs,
    ):
        self.seq_length = input_ids.size(1)
        # (batch, seq_len, hidden)
        x = self.token_embedding(input_ids)
        x = x.transpose(0, 1)  # (seq_len, batch, hidden)

        # causal mask so tokens can’t see the future
        causal_mask = torch.triu(
            torch.full(
                (self.seq_length, self.seq_length), float("-inf"), device=x.device
            ),
            diagonal=1,
        )

        # Transformer
        x = self.transformer(x, mask=causal_mask)
        x = x.transpose(0, 1)  # (batch, seq_len, hidden)

        # logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab)

        # Return loss + logits if labels are supplied
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),   # (B × L, V)
                labels.view(-1)                     # (B × L)
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

        with torch.no_grad():
            outputs = trainer.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
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
    # Load dataset and tokenizer
    ds, tokenizer = load_squad()

    train_ds = ds["train"].shuffle(42).select(range(TRAIN_SIZE))
    eval_ds  = ds["validation"].shuffle(42).select(range(EVAL_SIZE))

    # Initialize model
    model = SimpleGPTModel(
        vocab_size      = tokenizer.vocab_size,
        hidden_size     = HIDDEN_SIZE,
        num_layers      = NUM_LAYERS,
        num_heads       = NUM_HEADS,
        dim_feedforward = FF_DIM,
        seq_length      = SEQ_LENGTH,
    )
    # Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
        return_tensors="pt"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=EXPERIMENT_NAME,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        remove_unused_columns=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator
    )

    # Train and evaluate
    trainer.train()
    metrics = evaluate_model(trainer, eval_ds, tokenizer)
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    main()
