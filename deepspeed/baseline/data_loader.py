from datasets import load_dataset
from transformers import AutoTokenizer

# Define the pre-trained model to use for tokenization
MODEL_NAME = "bigscience/bloom-560m"

def load_squad(subset_size: int = None):
    """
    Loads and preprocesses the SQuAD dataset for fine-tuning a causal language model.

    Args:
        subset_size (int, optional): If provided, limits the training and validation datasets
                                     to this many examples for faster experimentation.

    Returns:
        DatasetDict: Tokenized and preprocessed train/validation datasets ready for training.
    """
    # Load the tokenizer associated with the BLOOM-560m model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the SQuAD dataset from Hugging Face Hub
    dataset = load_dataset("squad")
    # Set the tokenizer's padding token to the end-of-sequence token to mark the end pf sequence, since BLOOM doesn't have a default pad token
    tokenizer.pad_token = tokenizer.eos_token
    # Pad on the right side (important for generation tasks)
    tokenizer.padding_side = "right"

    # Define a preprocessing function to convert raw examples into model-ready inputs
    def preprocess_function(examples):
        # Format each input as: "Question: ... Context: ... Answer:"
        inputs = ["Question: " + q + " Context: " + c + " Answer:" for q, c in zip(examples["question"], examples["context"])]
        # Extract the first available answer string for each example
        answers = [a["text"][0] if a["text"] else "No Answer" for a in examples["answers"]]
        # Concatenate the formatted prompt with its corresponding answer
        texts = [inp + " " + ans for inp, ans in zip(inputs, answers)]

        # Tokenize each example:
        # - truncate to max length
        # - pad to max length
        # - return input_ids and attention_mask
        tokenized = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        # For causal language modeling, labels = input_ids (predict next token)
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    # If a subset size is given, reduce the size of both train and validation sets
    if subset_size:
        dataset["train"] = dataset["train"].select(range(subset_size))
        dataset["validation"] = dataset["validation"].select(range(min(50, subset_size)))

    # Apply the preprocessing function to the dataset in batches
    return dataset.map(preprocess_function, batched=True)