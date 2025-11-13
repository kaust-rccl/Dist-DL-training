#!/bin/bash
#SBATCH --job-name=F-B-1G1N     	          	    # Job name
#SBATCH --error=logs/single_node_%j.err     	    # Standard error
#SBATCH --output=logs/single_node_%j.out    	    # Standard output
#SBATCH --ntasks=1   		                          # Total MPI tasks
#SBATCH --tasks-per-node=1      	                # MPI tasks per node
#SBATCH --gpus=1                            	    # Total GPUs per node
#SBATCH --gpus-per-node=1                   	    # GPUs allocated per node
#SBATCH --constraint=v100			                    # e.g. v100
#SBATCH --cpus-per-task=4		                      # CPU cores per task
#SBATCH --time=00:30:00                        	  # HH:MM:SS
#SBATCH --mem=64G		                              # Total memory per node

source env_vars.sh

# Load and activate conda environment
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Setup Wandb online logging
export WANDB_MODE=online
export EXPERIMENT="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

# Wandb API key for later sync (online mode)
export WANDB_API_KEY=${WANDB_API_KEY}

# Execute fine-tuning script
python baseline.py

