#!/bin/bash
#SBATCH --job-name=multi_node_custom_8                 
#SBATCH --output=logs/multi_node_custom_8_%j.out                   # Stdout file
#SBATCH --error=logs/multi_node_custom_8_%j.err                    # Stderr file
#SBATCH --gpus=64
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=1
#SBATCH --constraint=v100,cpu_intel_platinum_8260,gpu_ai          
#SBATCH --cpus-per-task=4		           # CPU cores per process
#SBATCH --time=00:30:00                            # HH:MM:SS
#SBATCH --mem=0		                           # Memory per node

source env_vars.sh

# Activate environment
source ${CONDA_SH_PATH}
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=0

# Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}
export WANDB_API_KEY=${WANDB_API_KEY}

module purge
module load cuda/11.8 nccl/2.17.1-cuda11.8 openmpi/4.1.4/gnu11.2.1-cuda11.8

# ---------- optional tuning / diagnostics ----------
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1      # fail fast on hangs 
export NCCL_ALGO=Tree                         # stable perf across IB hops
export NCCL_NET_GDR_LEVEL=4                   # allow GPUDirect-RDMA if available 
export NCCL_IB_HCA=mlx5                       # pick Mellanox HCAs

# ---------- run the benchmark ----------
LOG="logs/nccl_${SLURM_JOB_ID}.log"
srun --cpu-bind=none --ntasks-per-node=1 --gpus-per-task=1 all_reduce_perf -b 4G -e 4G -f 2 -g 1 -c 0 -n 50 -w 20 > "$LOG" 2>&1

# ---------- parse result & gate ----------
BW=$(awk '/^# Avg bus bandwidth/ {print $(NF)}' "$LOG")

if [[ -z "$BW" ]]; then
    echo "NCCL parse failed – check $LOG"; exit 44
fi
echo "Measured AllReduce BW: $BW GB/s"
if (( $(echo "$BW < 5" | bc -l) )); then
    echo "Bandwidth below threshold – aborting"; exit 46
fi

module purge

export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Setup dist
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR=${master_addr}
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

# Launch training
srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
     	torchrun \
  	--nnodes=$SLURM_JOB_NUM_NODES \
  	--nproc_per_node=1 \
  	--node_rank=$SLURM_NODEID \
  	--rdzv_backend=c10d \
  	--rdzv_id=$SLURM_JOB_ID \
  	--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  	multi_node.py

# Sync logs
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
