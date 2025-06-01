#!/bin/bash
#SBATCH --job-name=baseline-bloom-finetune     # Name of the job shown in SLURM queue
#SBATCH --gpus=1                                # Request 1 GPU
#SBATCH --mem=32G                               # Request 32 GB of system memory
#SBATCH --constraint=v100                       # V100 GPU (optional)
#SBATCH --time=12:00:00                         # Max runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                  # ✅ SLURM .out into local log/

# ------------------------------
# Resolve directory paths
# ------------------------------
SCRIPT_DIR=$(PWD)

cd "$SCRIPT_DIR/../.."   # → Go to project root: Dist-DL-training/deepspeed

JOB_ID=${SLURM_JOB_ID:-manual}
EXPERIMENT_DIR="$SCRIPT_DIR"

# Log directories (all adjacent to SLURM script)
LOG_OUT_DIR="$EXPERIMENT_DIR/log"
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory"
CPU_LOG_DIR="$EXPERIMENT_DIR/cpu_memory"

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
  --format=csv,nounits -l 5 > "$GPU_LOG_DIR/gpu_memory_log_${JOB_ID}.csv" &
GPU_LOG_PID=$!

# ------------------------------
# Launch Training
# ------------------------------
python scripts/train.py

TRAIN_PID=$!

# ------------------------------
# CPU Memory Logging
# ------------------------------
psrecord $TRAIN_PID --include-children --interval 5 \
  --log "$CPU_LOG_DIR/cpu_memory_log_${JOB_ID}.txt" &
CPU_LOG_PID=$!

# ------------------------------
# Wait & Cleanup
# ------------------------------
wait $TRAIN_PID
kill $GPU_LOG_PID
kill $CPU_LOG_PID
