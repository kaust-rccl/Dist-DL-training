#!/bin/bash
#SBATCH --job-name=JOB_NAME                      
#SBATCH --output=JOB_NAME_%j.out                 # Stdout file
#SBATCH --error=JOB_NAME_%j.err                  # Stderr file
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks=1                               # Single process
#SBATCH --gpus=1                                 # Single GPU
#SBATCH --constraint=a100		         # e.g. a100
#SBATCH --cpus-per-task=4		         # CPU cores per task
#SBATCH --time=01:00:00	                         # HH:MM:SS
#SBATCH --mem=0		                         # Memory per node

# Activate environment
source ${CONDA_SH_PATH}
conda activate ${CONDA_ENV}

# Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT_NAME=${EXPERIMENT_NAME}
export LOG_DIR=${LOG_DIR}/${EXPERIMENT_NAME}
mkdir -p ${LOG_DIR}/wandb_runs
export WANDB_DIR=${LOG_DIR}/wandb_runs
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}
export WANDB_API_KEY=${WANDB_API_KEY}

# Execute fine-tuning
python3 single_node.py

# Post-training: sync with Wandb
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
