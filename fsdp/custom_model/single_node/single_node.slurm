#!/bin/bash
#SBATCH --job-name=F-C-1G1N
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

# Setup Wandb online logging
export WANDB_MODE=online
export EXPERIMENT="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_API_KEY=${WANDB_API_KEY}

# Export caching to user space
export HF_HOME=/ibex/user/$USER/.hf
export TRANSFORMERS_CACHE=/ibex/user/$USER/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/ibex/user/$USER/.cache/huggingface/datasets
export HF_MODULES_CACHE=/ibex/user/$USER/.cache/huggingface/modules
export XDG_CACHE_HOME=/ibex/user/$USER/.cache
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

# Execute fine-tuning
python single_node.py


