#!/bin/bash
#SBATCH --job-name=2-gpus-zero1-offload-deepspeed-bloom-finetune     # Name of the job shown in SLURM queue
#SBATCH --ntasks=1                                           # Number of tasks
#SBATCH --tasks-per-node=1                                   # One task per node
#SBATCH --cpus-per-task=4                                    # Number of CPUs allocated per task
#SBATCH --gpus=2                                             # Request 2 GPU
#SBATCH --gpus-per-node=2                                    # 2 GPUs per node
#SBATCH --mem=32G                                            # Request 32 GB of system memory
#SBATCH --constraint=v100                                    # Request specific GPU type (V100)
#SBATCH --time=12:00:00                                      # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                               # Redirect SLURM .out log to log/ directory

# ------------------------------
# Resolve directory paths
# ------------------------------
SCRIPT_DIR=$(pwd)

cd "$SCRIPT_DIR/../../../../.."   # → Go to project root: Dist-DL-training/deepspeed

JOB_ID=${SLURM_JOB_ID:-manual}
EXPERIMENT_DIR="$SCRIPT_DIR"

# Log directories (all adjacent to SLURM script)
LOG_OUT_DIR="$EXPERIMENT_DIR/log"
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory/$JOB_ID"
CPU_LOG_DIR="$EXPERIMENT_DIR/cpu_memory/$JOB_ID"
mkdir -p "$LOG_OUT_DIR" "$GPU_LOG_DIR" "$CPU_LOG_DIR"

# ------------------------------
# Environment Setup
# ------------------------------
source /ibex/user/x_mohameta/miniforge/etc/profile.d/conda.sh
conda activate deepspeed-finetune
module load cuda/12.4.1

# ------------------------------
# GPU Memory Logging
# ------------------------------
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total \
  --format=csv,nounits -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
GPU_LOG_PID=$!


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Hostname or IP of the master node for NCCL initialization
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')                                                      # Rendezvous port (if 9999 is in use, try another such as 6000 or 29500)
export WORLD_SIZE=$SLURM_GPUS_ON_NODE                                        # Total number of GPUs being used on this node
export RANK=0                                                                 # Global rank of this process (0 for single-node jobs)
export LOCAL_RANK=0                                                           # Local GPU index for this process (0–N-1)


# ------------------------------
# Launch Training
# ------------------------------
python -m torch.distributed.run --nproc_per_node=$SLURM_GPUS_ON_NODE  scripts/train.py --deepspeed ds_configs/zero1_cpu_offload.json &

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
