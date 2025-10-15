# Conda setup
export CONDA_ENV="bloom_env"

# Wandb/offline‐run settings
export EXPERIMENT_NAME="BLOOM_Multi_Nodes_2_NODES"
export LOG_DIR="$PWD/logs"
export WANDB_API_KEY="your_wandb_api_key"


export MODEL_NAME="bigscience/bloom-560m"
export OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"
export MAX_LENGTH=512
export TRAIN_SIZE=500
export EVAL_SIZE=100
export NUM_EPOCHS=5
export BATCH_SIZE=1
export LEARNING_RATE=5e-5
export WEIGHT_DECAY=0.01
export GRAD_ACC=4
export FP16=True
export BF16=False
