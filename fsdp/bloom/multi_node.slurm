#!/bin/bash
#SBATCH --job-name=JOB_NAME               
#SBATCH --output=JOB_NAME_%j.out              # Stdout file
#SBATCH --error=JOB_NAME_%j.err               # Stderr file
#SBATCH --ntasks=2			      # Total processes
#SBATCH --gpus=1		              # GPUs per node
#SBATCH --tasks-per-node=1		      # Tasks per node
#SBATCH --constraint=a100		      # e.g. a100
#SBATCH --cpus-per-task=4		      # CPU cores per process
#SBATCH --time=01:00:00	                      # HH:MM:SS
#SBATCH --mem=0		                      # Memory per node

# Activate your virtual environment
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
echo "Initializing distributed backend"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

# Launch multi-node training using torchrun
srun torchrun \
  --nnodes=$WORLD_SIZE \
  --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  multi_node.py

# Post-training: sync with Wandb cloud
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
