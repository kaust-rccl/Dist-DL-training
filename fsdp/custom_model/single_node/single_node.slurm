#!/bin/bash
#SBATCH --job-name=single_custom                      
#SBATCH --output=logs/single_custom_%j.out
#SBATCH --error=logs/single_custom_%j.err 
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4		         # CPU cores per task
#SBATCH --time=00:30:00	                         # HH:MM:SS
#SBATCH --mem=64G		                 # Memory per node

source env_vars.sh

# Activate environment
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Setup Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}
export WANDB_API_KEY=${WANDB_API_KEY}

# Execute fine-tuning
python single_node.py

# Post-training: sync with Wandb
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
