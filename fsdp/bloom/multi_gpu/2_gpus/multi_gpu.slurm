#!/bin/bash
#SBATCH --job-name=multi_gpu_2
#SBATCH --output=logs/multi_gpu_2_%j.out
#SBATCH --error=logs/multi_gpu_2_%j.err            
#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=2
#SBATCH --constraint=v100,cpu_intel_platinum_8260,gpu_ai
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00                	   # HH:MM:SS
#SBATCH --mem=0		                   # Memory per node

source env_vars.sh

# Activate environment
source ${CONDA_SH_PATH}
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=0,1

# Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}

# Distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')                                    
export WORLD_SIZE=2


srun --nodes=1 --ntasks=1 --gpus=2 \
     python -m torch.distributed.launch --use_env \
       --nproc_per_node=2 \
       --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
       multi_gpu.py

# Post-training: sync with Wandb cloud
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
