#!/bin/bash
#SBATCH --job-name=custom_weak_scaling_2                        
#SBATCH --output=logs/custom_weak_scaling_2_%j.out
#SBATCH --error=logs/custom_weak_scaling_2_%j.err
#SBATCH --ntasks=2		                   # Total processes (should equal GPUS_PER_NODE * NNODES)
#SBATCH --ntasks-per-node=2		           # Processes per node
#SBATCH --gpus-per-node=8		           # GPUs per node
#SBATCH --constraint=v100,cpu_intel_platinum_8260,gpu_ai
#SBATCH --cpus-per-task=4		           # CPU cores per process
#SBATCH --time=00:30:00	                           # HH:MM:SS
#SBATCH --mem=0		                           # Memory per node

source env_vars.sh

# Activate environment
torch_fsdpenv=/ibex/user/$USER/miniforge/etc/profile.d/conda.sh
source $torch_fsdpenv
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=0,1

# Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}
export WANDB_API_KEY=${WANDB_API_KEY}

# Distributed setup
echo "Init dist backend"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=${master_addr}
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()') 
export SLURM_GPUS_PER_NODE=2

srun --nodes=1 --ntasks=1 --gpus=2 \
     python -m torch.distributed.launch --use_env \
       --nproc_per_node=2 \
       --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
       weak_scaling.py


# Post-training: sync with Wandb
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
