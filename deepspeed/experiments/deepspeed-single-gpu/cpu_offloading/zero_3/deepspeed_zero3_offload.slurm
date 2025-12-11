#!/bin/bash
#SBATCH --job-name=3ZO-1G1N                    # Name of the job shown in SLURM queue
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --tasks-per-node=1                     # One task per node
#SBATCH --cpus-per-task=4                      # Number of CPUs allocated per task
#SBATCH --gpus=1                               # Request 1 GPU
#SBATCH --gpus-per-node=1                      # GPUs per node
#SBATCH --mem=32G                              # Request 32 GB of system memory
#SBATCH --constraint=v100                      # Request specific GPU type (V100)
#SBATCH --time=01:00:00                        # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                 # Redirect SLURM .out log to log/ directory

# ------------------------------
# Resolve directory paths
# ------------------------------
SCRIPT_DIR=$(pwd)

cd "$SCRIPT_DIR/../../../.."   # → Go to project root: Dist-DL-training/deepspeed

JOB_ID=${SLURM_JOB_ID:-manual}
EXPERIMENT_DIR="$SCRIPT_DIR"

# Log directories (all adjacent to SLURM script)
LOG_OUT_DIR="$EXPERIMENT_DIR/log"
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory/$JOB_ID"
CPU_LOG_DIR="$EXPERIMENT_DIR/cpu_memory/$JOB_ID"
mkdir -p "$LOG_OUT_DIR" "$GPU_LOG_DIR" "$CPU_LOG_DIR"

export EXPERIMENT_NAME="Deepspeed_1G_Zero3_Off"
export WANDB_API_KEY="543b7e544b7135b5a155d75b41180fbf7046d28d"

# Setup Wandb online logging
export WANDB_MODE=online
export EXPERIMENT="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_DIR="$LOG_OUT_DIR/$EXPERIMENT/wandb_runs"
mkdir -p $WANDB_DIR
export WANDB_RUN_ID="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
export WANDB_NAME="${EXPERIMENT_NAME}_${SLURM_JOB_ID}"

export HF_HOME=/ibex/user/$USER/.hf
export TRANSFORMERS_CACHE=/ibex/user/$USER/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/ibex/user/$USER/.cache/huggingface/datasets
export HF_MODULES_CACHE=/ibex/user/$USER/.cache/huggingface/modules
export XDG_CACHE_HOME=/ibex/user/$USER/.cache
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

# ------------------------------
# Environment Setup
# ------------------------------
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate deepspeed-finetune
module load cuda/12.4.1

# ------------------------------
# GPU Memory Logging
# ------------------------------
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total \
  --format=csv,nounits -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
GPU_LOG_PID=$!

# ----------------------------
# Discover participating nodes
# ----------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")         # Get list of hostnames in this job
nodes_array=($nodes)                                           # Split the list into a Bash array
echo "Node IDs of participating nodes: ${nodes_array[*]}"      # Print out the node hostnames

# -----------------------------------------
# Determine MASTER node IP and open port
# -----------------------------------------
head_node="${nodes_array[0]}"                                           # Choose the first node as the master
echo "Head node hostname: ${head_node}"
master_ip=$(srun -n1 -N1 -w ${head_node} hostname -I | cut -d " " -f2)  # Fetch the IP address of the head node
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')  # Find a free TCP port dynamically
echo "Master endpoint: ${master_ip}:${master_port}"

# ------------------------------
# Launch Training
# ------------------------------
python -m torch.distributed.run \
--rdzv_endpoint="$master_ip":"$master_port" \
scripts/train.py --deepspeed ds_configs/zero3_cpu_offload.json &

TRAIN_PID=$!

# ------------------------------
# CPU Memory Logging
# ------------------------------
psrecord $TRAIN_PID --include-children --interval 5 \
  --log "$CPU_LOG_DIR/cpu_memory_log.txt" &
CPU_LOG_PID=$!

# ------------------------------
# Wait & Cleanup
# ------------------------------
wait $TRAIN_PID
kill $GPU_LOG_PID
kill $CPU_LOG_PID

# Wandb API key for later sync (online mode)
export WANDB_API_KEY=${WANDB_API_KEY}

echo "Analyzing memory logs for job ${SLURM_JOB_ID:-manual}"
python scripts/analyze_memory.py "${SLURM_JOB_ID:-manual}" --path "$SCRIPT_DIR"