# ===================================================
# SQuAD Data Loading & Preprocessing for BLOOM Models
# ===================================================

from datasets import load_dataset
from transformers import AutoTokenizer

# --------------------------------------------
# Step 1: Define the Pretrained Model Name
# --------------------------------------------
# This model checkpoint is used to load the appropriate tokenizer
MODEL_NAME = "bigscience/bloom-560m"


# --------------------------------------------
# Step 2: Function to Load and Preprocess SQuAD
# --------------------------------------------

def load_squad(subset_size: int = None):
    """
    Loads and preprocesses the SQuAD dataset for fine-tuning a causal language model.

    Args:
        subset_size (int, optional): If provided, limits the size of the training and
                                     validation sets for quick prototyping or debugging.

    Returns:
        DatasetDict: A dictionary containing the tokenized and formatted training
                     and validation splits, ready to be passed to a Trainer.
    """

    # Load the tokenizer corresponding to the chosen model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the full SQuAD dataset (train + validation)
    dataset = load_dataset("squad")

    # BLOOM does not have a default pad token, so we use the EOS token instead
    tokenizer.pad_token = tokenizer.eos_token

    # For generation-style models, it's standard to pad on the right
    tokenizer.padding_side = "right"

    # --------------------------------------------
    # Step 3: Define Preprocessing Logic
    # --------------------------------------------

    def preprocess_function(examples):
        """
        Converts raw SQuAD examples into tokenized model inputs.

        Each input format:
            "Question: <q> Context: <c> Answer: <a>"

        Returns:
            A dictionary with input_ids, attention_mask, and labels.
        """
        # Format input as question + context + answer prompt
        inputs = [
            f"Question: {q} Context: {c} Answer:"
            for q, c in zip(examples["question"], examples["context"])
        ]

        # Extract the first answer text (some entries contain multiple answers)
        answers = [
            a["text"][0] if a["text"] else "No Answer"
            for a in examples["answers"]
        ]

        # Combine input prompt with the correct answer
        texts = [inp + " " + ans for inp, ans in zip(inputs, answers)]

        # Tokenize each full example (prompt + answer)
        tokenized = tokenizer(
            texts,
            truncation=True,  # Truncate sequences to max length
            padding="max_length",  # Pad to the max sequence length
            max_length=512  # Hard limit on token length
        )

        # For causal language modeling, model tries to predict next token
        # So we use the input tokens themselves as the labels
        tokenized["labels"] = tokenized["input_ids"]

        return tokenized

    # --------------------------------------------
    # Step 4: Optional Dataset Subsetting
    # --------------------------------------------

    if subset_size:
        # Reduce training set to first `subset_size` examples
        dataset["train"] = dataset["train"].select(range(subset_size))

        # Validation set is optionally reduced (but at most 50 samples)
        dataset["validation"] = dataset["validation"].select(range(min(50, subset_size)))

    # --------------------------------------------
    # Step 5: Apply Preprocessing to Entire Dataset
    # --------------------------------------------

    # Use batched mapping for faster performance
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    return tokenized_dataset
