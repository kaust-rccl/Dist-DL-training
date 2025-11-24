from transformers import Trainer, DataCollatorForLanguageModeling
from model import load_model, save_model
from data_loader import load_squad
from config import TRAINING_ARGS
import torch
import torch.distributed as dist

import os
import time
from datetime import timedelta

# =================================================
#  Distributed Initialization Logging (TCP-based)
# =================================================
print(f"Initializing process group for rank {os.environ.get('RANK')}")
print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

# =============================
#  Device Assignment
# =============================
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# ======================================
#  Verify Process Group Initialization
# ======================================
if dist.is_initialized():
    print(f"torch.distributed initialized: rank {dist.get_rank()} / world size {dist.get_world_size()}")

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

# Base number of samples per GPU
base_size_per_gpu = 10000

# Total number of processes (GPUs) across all nodes
world_size = dist.get_world_size()

# GPUs per node (from SLURM or fallback to all local GPUs)
gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))

# Number of nodes = total GPUs // GPUs per node
num_nodes = world_size // gpus_per_node

print(f"Running on {num_nodes} nodes × {gpus_per_node} GPUs = {world_size} total GPUs")

# Compute total subset size for weak scaling
subset_size = base_size_per_gpu * world_size
print(f"Loading subset_size = {subset_size} examples")

# Load the dataset with the computed subset size
tokenized_datasets = load_squad(subset_size=subset_size)

# -------------------------------
# 3. Create Data Collator
# -------------------------------
# The DataCollator dynamically pads sequences in a batch and ensures that
# labels match input_ids. Setting `mlm=False` means we are NOT using
# masked language modeling (e.g., BERT) — this is for causal LM (e.g., BLOOM).
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define Trainer
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