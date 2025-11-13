#!/bin/bash
#SBATCH --job-name=F-B-8G1N
#SBATCH --output=logs/multi_gpu_8_%j.out
#SBATCH --error=logs/multi_gpu_8_%j.err            
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=8
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00                	   # HH:MM:SS
#SBATCH --mem=0 		           # Memory per node

source env_vars.sh

# Activate environment
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Wandb online logging
export WANDB_MODE=online
export EXPERIMENT="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

# Distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')   
export WORLD_SIZE=8

srun --nodes=1 --ntasks=1 --gpus=8 \
     python -m torch.distributed.launch --use_env \
       --nproc_per_node=8 \
       --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
       multi_gpu.py


# Post-training: sync with Wandb cloud
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
