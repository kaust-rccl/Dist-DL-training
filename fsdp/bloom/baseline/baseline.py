import os
import torch
import torch.distributed as dist
from transformers import (
    BloomForCausalLM,
    BloomTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import numpy as np
import re
import string

# Environment/config placeholders
MODEL_NAME = os.getenv("MODEL_NAME", "bigscience/bloom-560m")
EXPERIMENT_NAME = os.getenv("EXPERIMENT", "default_experiment")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 512))
TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", 500))
EVAL_SIZE = int(os.getenv("EVAL_SIZE", 100))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))
LR = float(os.getenv("LEARNING_RATE", 5e-5))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC", 4))
FP16 = os.getenv("FP16", "False") == "True"
BF16 = os.getenv("BF16", "True") == "True"


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
            a["text"][0] if a["text"] else "No Answer"
            for a in examples["answers"]
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
    return tokenized_dataset, tokenizer


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def compute_em_and_f1(predicted: str, ground_truth: str):
    """Return Exact-Match and F1 for one prediction/reference pair."""
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
    """Evaluate a model on `dataset` using EM and F1."""
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
            [tok for tok in example["labels"] if tok != -100],
            skip_special_tokens=True,
        ).strip()

        em, f1 = compute_em_and_f1(generated_answer, reference_answer)
        em_scores.append(em)
        f1_scores.append(f1)

    return {"exact_match": np.mean(em_scores), "f1": np.mean(f1_scores)}



def main():
    tokenized_ds, tokenizer = load_squad()
    train_ds = tokenized_ds["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    eval_ds = tokenized_ds["validation"].shuffle(seed=42).select(range(EVAL_SIZE))

    model = BloomForCausalLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=EXPERIMENT_NAME,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        fp16=FP16,
        bf16=BF16,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = evaluate_model(trainer, eval_ds, tokenizer)
    print("Evaluation Metrics:", metrics)

    model.save_pretrained("./bloom-qa-finetuned")
    tokenizer.save_pretrained("./bloom-qa-finetuned")

if __name__ == "__main__":
    main()
