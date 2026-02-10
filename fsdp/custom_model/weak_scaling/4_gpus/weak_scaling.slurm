#!/bin/bash
#SBATCH --job-name=F-CW-4G1N
#SBATCH --output=logs/%x-%j.out
#SBATCH --ntasks=1		                   # Total processes (should equal GPUS_PER_NODE * NNODES)
#SBATCH --ntasks-per-node=1		           # Processes per node
#SBATCH --gpus-per-node=4		           # GPUs per node
#SBATCH --constraint=a100,4gpus
#SBATCH --reservation=distributedDL           # Workshop reservation
#SBATCH --cpus-per-task=16		           # CPU cores per process
#SBATCH --time=00:30:00	                           # HH:MM:SS
#SBATCH --mem=64G	                           # Memory per node

source env_vars.sh

# Activate environment
module load dl-workshops fsdp


export CUDA_VISIBLE_DEVICES=0,1,2,3

# Wandb online logging
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

# Distributed setup
echo "Init dist backend"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=${master_addr}
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()') 
export SLURM_GPUS_PER_NODE=4

srun --nodes=1 --ntasks=1 --gpus=${SLURM_GPUS_PER_NODE} \
     python -m torch.distributed.launch --use_env \
       --nproc_per_node=${SLURM_GPUS_PER_NODE} \
       --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
       weak_scaling.py


