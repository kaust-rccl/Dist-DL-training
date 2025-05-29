#!/bin/bash
#SBATCH --job-name=JOB_NAME                # e.g. o_multi_gpu
#SBATCH --output=JOB_NAME_%j.out           # Stdout file
#SBATCH --error=JOB_NAME_%j.err            # Stderr file
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks=2	                   # Total processes (should equal GPUs)
#SBATCH --gpus=2		           # GPUs per node
#SBATCH --constraint=a100		   # e.g. a100
#SBATCH --cpus-per-task=4		   # CPU cores per process
#SBATCH --time=01:00:00                	   # HH:MM:SS
#SBATCH --mem=0		                   # Memory per node

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

# Distributed setup
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=29500  
export WORLD_SIZE=2


# Launch with torchrun for FSDP
torchrun \
  --nnodes=1 \
  --nproc_per_node=$WORLD_SIZE \
  --rdzv_backend=c10d \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  multi_gpu.py

# Post-training: sync with Wandb cloud
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
