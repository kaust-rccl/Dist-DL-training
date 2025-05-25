from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "bigscience/bloom-560m"
SAVE_DIR = "./bloom-finetuned"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def save_model(model, tokenizer):
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")