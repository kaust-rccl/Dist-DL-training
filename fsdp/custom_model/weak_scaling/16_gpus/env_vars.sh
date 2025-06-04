# Conda setup
export CONDA_SH_PATH="/path/to/miniforge/etc/profile.d/conda.sh"
export CONDA_ENV="bloom_env"

# Wandb/offline‐run settings
export EXPERIMENT_NAME="Custom_Model_Weak_Scaling_16_GPUs"
export LOG_DIR="/path/to/logs"
export WANDB_API_KEY="your_wandb_api_key"


export OUTPUT_DIR="/path/to/outputs/${EXPERIMENT_NAME}"

export MAX_LENGTH=512
export TRAIN_SIZE=500
export EVAL_SIZE=100
export NUM_EPOCHS=5
export BATCH_SIZE=1
export LEARNING_RATE=5e-5
export WEIGHT_DECAY=0.01
export GRAD_ACC=1
export FP16=True
export BF16=False

# ── Model architecture ───────────────────────────────
export VOCAB_SIZE=50000          # ≈ tokenizer.vocab_size
export HIDDEN_SIZE=2048          # d_model
export NUM_LAYERS=24             # transformer depth
export NUM_HEADS=16              # attention heads
export FF_DIM=8192               # feed-forward width
export SEQ_LENGTH=512            # max position (same as MAX_LENGTH)
