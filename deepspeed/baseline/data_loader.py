from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "bigscience/bloom-560m"

def load_squad(subset_size: int = None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def preprocess_function(examples):
        inputs = ["Question: " + q + " Context: " + c + " Answer:" for q, c in zip(examples["question"], examples["context"])]
        answers = [a["text"][0] if a["text"] else "No Answer" for a in examples["answers"]]
        texts = [inp + " " + ans for inp, ans in zip(inputs, answers)]
        tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    if subset_size:
        dataset["train"] = dataset["train"].select(range(subset_size))
        dataset["validation"] = dataset["validation"].select(range(min(50, subset_size)))

    return dataset.map(preprocess_function, batched=True)