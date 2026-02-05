#!/bin/bash
#SBATCH --job-name=F-B-1G1N     	          	    # Job name
#SBATCH --output=logs/%x-%j.out             	    # Standard output
#SBATCH --ntasks=1   		                          # Total MPI tasks
#SBATCH --tasks-per-node=1      	                # MPI tasks per node
#SBATCH --gpus=1                            	    # Total GPUs per node
#SBATCH --gpus-per-node=1                   	    # GPUs allocated per node
#SBATCH --constraint=a100,4gpus                   # Request specific GPU Node (A100)
#SBATCH --reservation=distributedDL_001           # Workshop reservation
#SBATCH --cpus-per-task=4		                      # CPU cores per task
#SBATCH --time=00:30:00                        	  # HH:MM:SS
#SBATCH --mem=64G		                              # Total memory per node

source env_vars.sh

# Load and activate conda environment
module load dl-workshops fsdp

# Setup Wandb online logging
export WANDB_MODE=online
export EXPERIMENT="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

# Export caching to user space
export HF_HOME=/ibex/user/$USER/.hf
export TRANSFORMERS_CACHE=/ibex/user/$USER/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/ibex/user/$USER/.cache/huggingface/datasets
export HF_MODULES_CACHE=/ibex/user/$USER/.cache/huggingface/modules
export XDG_CACHE_HOME=/ibex/user/$USER/.cache
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

# Wandb API key for later sync (online mode)
export WANDB_API_KEY=${WANDB_API_KEY}

# Execute fine-tuning script
python baseline.py

