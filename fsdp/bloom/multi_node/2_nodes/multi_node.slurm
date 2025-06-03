#!/bin/bash
#SBATCH --job-name=multi_node_2 
#SBATCH --output=logs/multi_node_2_%j.out
#SBATCH --error=logs/multi_node_2_%j.err
#SBATCH --gpus=2
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4		      # CPU cores per process
#SBATCH --time=01:00:00	                      # HH:MM:SS
#SBATCH --mem=0		                      # Memory per node

source env_vars.sh

# Activate your virtual environment
source ${CONDA_SH_PATH}
conda activate ${CONDA_ENV}

#export CUDA_VISIBLE_DEVICES=0

# Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}


# Distributed setup
echo "Initializing distributed backend"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')  
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

# Launch multi-node training using torchrun
srun torchrun \
  --nnodes=$WORLD_SIZE \
  --nproc_per_node=1 \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  multi_node.py

# Post-training: sync with Wandb cloud
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
