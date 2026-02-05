#!/bin/bash
#SBATCH --job-name=F-C-8G4N
#SBATCH --output=logs/%x-%j.out
#SBATCH --gpus=8
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=1
#SBATCH --constraint=a100,4gpus
#SBATCH --reservation=distributedDL_001           # Workshop reservation
#SBATCH --cpus-per-task=8		           # CPU cores per process
#SBATCH --time=00:30:00                            # HH:MM:SS
#SBATCH --mem=64G	                           # Memory per node

source env_vars.sh

# Activate environment
module load dl-workshops fsdp

export CUDA_VISIBLE_DEVICES=0

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

# Distributed setup
echo "Initializing distributed backend"

nodes_array=($(scontrol show hostnames "$SLURM_NODELIST"))   # ordered list :contentReference[oaicite:0]{index=0}
export MASTER_ADDR=${nodes_array[0]}                         # first node is master
export MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()          # open socket
s.bind(('', 0))              # bind to ephemeral port
print(s.getsockname()[1])    # emit port number
s.close()
PY
)

echo "Master ↦ ${MASTER_ADDR}:${MASTER_PORT}"
export WORLD_SIZE=$(( ${#nodes_array[@]} * 1 ))

# Avoid CPU over-subscription inside every rank
export OMP_NUM_THREADS=1

for (( i=0; i<${SLURM_JOB_NUM_NODES}; i++ )); do
    srun --nodes=1 --ntasks=1 \
         --gpus=${SLURM_GPUS_PER_NODE} \
         --cpus-per-task=${SLURM_CPUS_PER_TASK} \
         -w ${nodes_array[$i]} \
         python -m torch.distributed.launch --use_env \
            --nproc_per_node=${SLURM_GPUS_PER_NODE} \
            --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${i} \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            multi_node.py &
done
wait

# Sync logs
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
