# ============================================
# Configuration for Fine-Tuning with DeepSpeed
# ============================================

import argparse
import os

from transformers import TrainingArguments

# --------------------------------------------
# Step 1: Define and Parse Command Line Arguments
# --------------------------------------------
# This allows external configuration when launching the training script.
# For example: python train.py --deepspeed ds_configs/zero1_cpu_offload.json

parser = argparse.ArgumentParser()

# Add a command-line argument to specify the DeepSpeed config file
parser.add_argument(
    "--deepspeed",
    type=str,
    default=None,
    help="Path to DeepSpeed config JSON file"
)

# Parse the arguments into a namespace object called `args`
args = parser.parse_args()


# --------------------------------------------
# Step 2: Create HuggingFace TrainingArguments
# --------------------------------------------
# This object defines all hyperparameters and runtime settings for training.
# It will be used by HuggingFace's `Trainer` class.
EXPERIMENT_NAME = os.getenv("EXPERIMENT", "default_experiment")

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))  # set by torchrun/deepspeed
TARGET_GBS = 16  # from your 1-GPU baseline

# Choose a per-device batch you *want* to try
base_per_device_bs = 4

# Compute gradient_accumulation_steps to hit TARGET_GBS
grad_acc_steps = max(
    1,
    TARGET_GBS // (WORLD_SIZE * base_per_device_bs)
)

TRAINING_ARGS = TrainingArguments(
    run_name=EXPERIMENT_NAME,

    report_to="wandb",

    output_dir="./bloom-qa-finetuned",  # Path to save checkpoints, logs, and final model

    # Evaluation Strategy: Run evaluation at the end of each epoch
    eval_strategy="epoch",

    # Save Strategy: Save a checkpoint at the end of each epoch
    save_strategy="epoch",

    # Per-device batch sizes during training and evaluation
    per_device_train_batch_size=base_per_device_bs,
    per_device_eval_batch_size=base_per_device_bs,

    # Accumulate gradients over 4 steps to simulate a larger batch size
    gradient_accumulation_steps=grad_acc_steps,

    # Number of full passes over the training dataset
    num_train_epochs=3,

    # Optimizer hyperparameters
    learning_rate=5e-5,     # AdamW optimizer learning rate
    weight_decay=0.01,      # L2 regularization to reduce overfitting

    # Enable automatic mixed precision (float16) for faster training on supported GPUs
    fp16=True,

    # Disable gradient checkpointing for now (set to True to reduce memory usage at cost of speed)
    gradient_checkpointing=False,

    # Avoid uploading to Hugging Face Hub
    push_to_hub=False,

    # DeepSpeed integration: use the path passed via --deepspeed CLI argument
    deepspeed=args.deepspeed
)
