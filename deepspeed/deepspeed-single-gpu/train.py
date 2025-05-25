from transformers import Trainer, DataCollatorForLanguageModeling
from model import load_model, save_model
from data_loader import load_squad
from config import TRAINING_ARGS

# ============================================
#  BLOOM Fine-Tuning Script (Baseline Setup)
# ============================================
#
# This script fine-tunes a causal language model (BLOOM-560m) using Hugging Face's
# `Trainer` API on a question-answer formatted subset of the SQuAD dataset.
#
# Key components:
# - Loads model/tokenizer
# - Prepares tokenized dataset
# - Uses DataCollator for causal language modeling
# - Trains using Hugging Face Trainer
# - Saves the fine-tuned model for later use
#

# -------------------------------
# 1. Load Pre-trained Model
# -------------------------------
# This function loads both the BLOOM model and its corresponding tokenizer.
# The tokenizer is reused throughout for data preprocessing and padding.
model, tokenizer = load_model()

# -------------------------------
# 2. Load and Tokenize Dataset
# -------------------------------
# Loads a subset of the SQuAD dataset and tokenizes each example as:
# "Question: ... Context: ... Answer: ..."
# Labels are aligned with input_ids for causal LM training.
tokenized_datasets = load_squad(subset_size=500)

# -------------------------------
# 3. Create Data Collator
# -------------------------------
# The DataCollator dynamically pads sequences in a batch and ensures that
# labels match input_ids. Setting `mlm=False` means we are NOT using
# masked language modeling (e.g., BERT) — this is for causal LM (e.g., BLOOM).
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# 4. Set Up Trainer
# -------------------------------
# Trainer handles the full training loop — forward, backward, optimizer step,
# checkpoint saving, evaluation, and logging. It requires:
# - model and tokenizer
# - training/evaluation datasets
# - training arguments (batch size, epochs, learning rate, etc.)
# - a data collator to manage padding and batching
trainer = Trainer(
    model=model,
    args=TRAINING_ARGS,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
# 5. Train the Model
# -------------------------------
# This runs the training loop according to the configuration provided.
trainer.train()

# -------------------------------
# 6. Save the Fine-tuned Model
# -------------------------------
# Saves both the model weights and tokenizer to disk.
# This allows reloading the model later for inference or further fine-tuning.
save_model(model, tokenizer)
