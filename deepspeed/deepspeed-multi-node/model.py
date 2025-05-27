from transformers import AutoModelForCausalLM, AutoTokenizer

# Model to use for fine-tuning
MODEL_NAME = "bigscience/bloom-560m"
# Directory to save the fine-tuned model and tokenizer
SAVE_DIR = "./bloom-finetuned"

def load_model():
    """
    Loads the pre-trained BLOOM model and its tokenizer for causal language modeling.

    Returns:
        model (AutoModelForCausalLM): A transformer model pre-trained for causal LM tasks.
        tokenizer (AutoTokenizer): Tokenizer that corresponds to the BLOOM model.

    Notes:
        - BLOOM is a causal (auto-regressive) language model.
        - The model can be fine-tuned using a causal language modeling objective (predict the next token).
    """
    # Load tokenizer from Hugging Face Hub (includes vocab and tokenization rules)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the pre-trained causal language model (BLOOM-560m)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def save_model(model, tokenizer):
    """
    Saves the fine-tuned model and tokenizer to disk.

    Args:
        model (PreTrainedModel): A Hugging Face model to save.
        tokenizer (PreTrainedTokenizer): The tokenizer used during training.

    Side Effects:
        - Creates or overwrites the directory defined in SAVE_DIR.
        - Writes model weights, config, and tokenizer files for future reuse.

    Example:
        After training, call: save_model(trainer.model, tokenizer)
    """
    # Save the model's weights and configuration
    model.save_pretrained(SAVE_DIR)
    # Save the tokenizer vocabulary and settings
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")