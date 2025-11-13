#!/bin/bash
#SBATCH --job-name=F-C-16G2N
#SBATCH --output=logs/custom_weak_scaling_16_%j.out
#SBATCH --error=logs/custom_weak_scaling_16_%j.err
#SBATCH --ntasks=16		                   # Total processes (should equal GPUS_PER_NODE * NNODES)
#SBATCH --ntasks-per-node=8		           # Processes per node
#SBATCH --gpus-per-node=8		           # GPUs per node
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4		           # CPU cores per process
#SBATCH --time=00:30:00	                           # HH:MM:SS
#SBATCH --mem=0		                           # Memory per node

source env_vars.sh

# Activate environment
torch_fsdpenv=/ibex/user/$USER/miniforge/etc/profile.d/conda.sh
source $torch_fsdpenv
conda activate ${CONDA_ENV}

export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Wandb online logging
export WANDB_MODE=online
export EXPERIMENT="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_API_KEY=${WANDB_API_KEY}

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
export WORLD_SIZE=$(( ${#nodes_array[@]} * ${SLURM_GPUS_PER_NODE} ))

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
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            weak_scaling.py &
done
wait


