#!/bin/bash
#SBATCH --job-name=name	          	    # Job name
#SBATCH --output=name_%j.out    	    # Standard output
#SBATCH --error=name_%j.err     	    # Standard error
#SBATCH --ntasks=1                          # Total MPI tasks
#SBATCH --tasks-per-node=1                  # MPI tasks per node
#SBATCH --gpus=1                            # Total GPUs per node
#SBATCH --gpus-per-node=1                   # GPUs allocated per node
#SBATCH --constraint=a100		    # e.g. a100,8gpus
#SBATCH --cpus-per-task=4		    # CPU cores per task
#SBATCH --time=01:00:00                     # HH:MM:SS
#SBATCH --mem=0		                    # Total memory per node

# Load and activate conda environment
source ${CONDA_SH_PATH}
conda activate ${CONDA_ENV}

# Setup Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}

# Wandb API key for later sync (online mode)
export WANDB_API_KEY=${WANDB_API_KEY}

# Execute fine-tuning script
python3 baseline.py

# After training: sync offline logs to Wandb cloud
wandb online
cd $WANDB_DIR
wandb sync --include-offline --sync-all
