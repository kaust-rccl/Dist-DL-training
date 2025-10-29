#!/bin/bash
#SBATCH --job-name=multi_node_8
#SBATCH --output=logs/multi_node_8_%j.out
#SBATCH --error=logs/multi_node_8_%j.err
#SBATCH --gpus=16
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=4		      # CPU cores per process
#SBATCH --time=00:30:00	                      # HH:MM:SS
#SBATCH --mem=64G		                      # Memory per node

source env_vars.sh

# Activate your virtual environment
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export CUDA_VISIBLE_DEVICES=0

# Wandb offline logging
export WANDB_MODE=offline
export EXPERIMENT="${EXPERIMENT_NAME}"
export WANDB_DIR=${LOG_DIR}/$EXPERIMENT/wandb_runs
mkdir -p $WANDB_DIR
export WANDB_RUN_ID=${EXPERIMENT_NAME}
export WANDB_NAME=${EXPERIMENT_NAME}


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
         --gpus=1 \
         --cpus-per-task=${SLURM_CPUS_PER_TASK} \
         -w ${nodes_array[$i]} \
         python -m torch.distributed.launch --use_env \
            --nproc_per_node=1 \
            --nnodes=${SLURM_JOB_NUM_NODES} --node_rank=${i} \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            multi_node.py &
done
wait

# Post-training: sync with Wandb cloud
wandb online
cd ${WANDB_DIR}
wandb sync --include-offline --sync-all
