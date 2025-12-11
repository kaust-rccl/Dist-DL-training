# ============================================
# BLOOM Model Loader & Saver for Fine-Tuning
# ============================================

from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------
# Configuration
# --------------------------------------------

# Pre-trained model name from Hugging Face Hub
MODEL_NAME = "bigscience/bloom-1b7"

# Local directory to save fine-tuned model and tokenizer
SAVE_DIR = "./bloom-finetuned"


# --------------------------------------------
# Step 1: Load Model and Tokenizer
# --------------------------------------------

def load_model():
    """
    Loads the pre-trained BLOOM-560m model and its tokenizer for causal language modeling.

    Returns:
        model (AutoModelForCausalLM): A transformer model configured for auto-regressive (causal) generation.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.

    Details:
        - BLOOM is designed for causal language modeling (next-token prediction).
        - This function loads both weights and vocab from the Hugging Face Hub.
        - You can pass these objects directly to the HuggingFace Trainer or other training routines.
    """
    # Load the tokenizer (handles tokenization and decoding)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load the pre-trained causal language model (supports generation & fine-tuning)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    return model, tokenizer


# --------------------------------------------
# Step 2: Save Model and Tokenizer
# --------------------------------------------

def save_model(model, tokenizer):
    """
    Saves the fine-tuned model and tokenizer to disk.

    Args:
        model (PreTrainedModel): A HuggingFace model (e.g., after training) to be saved.
        tokenizer (PreTrainedTokenizer): The tokenizer used for training, to be saved as well.

    Side Effects:
        - Creates or overwrites the target directory at SAVE_DIR.
        - Saves all necessary files to re-load and re-use the model and tokenizer.

    Usage:
        >>> save_model(trainer.model, tokenizer)
        Saves to './bloom-finetuned'
    """
    # Save model weights, config, and training heads
    model.save_pretrained(SAVE_DIR)

    # Save tokenizer vocab, merges, and config
    tokenizer.save_pretrained(SAVE_DIR)

    print(f"✅ Model and tokenizer saved to {SAVE_DIR}")
