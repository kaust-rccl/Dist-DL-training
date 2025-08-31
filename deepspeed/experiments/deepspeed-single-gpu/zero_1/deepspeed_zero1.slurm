#!/bin/bash
#SBATCH --job-name=zero1-deepspeed-bloom-finetune     # Name of the job shown in SLURM queue
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --tasks-per-node=1                     # One task per node
#SBATCH --cpus-per-task=4                      # Number of CPUs allocated per task
#SBATCH --gpus=1                               # Request 1 GPU
#SBATCH --gpus-per-node=1                      # GPUs per node
#SBATCH --mem=32G                              # Request 32 GB of system memory
#SBATCH --constraint=v100                      # Request specific GPU type (V100)
#SBATCH --time=12:00:00                        # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                 # Redirect SLURM .out log to log/ directory

# ------------------------------
# Resolve directory paths
# ------------------------------
SCRIPT_DIR=$(pwd)

cd "$SCRIPT_DIR/../../.."   # → Go to project root: Dist-DL-training/deepspeed

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

# ------------------------------
# Launch Training
# ------------------------------
python scripts/train.py --deepspeed ds_configs/zero1.json &

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

echo "Analyzing memory logs for job ${SLURM_JOB_ID:-manual}"
python scripts/analyze_memory.py ${SLURM_JOB_ID:-manual} --path "$SCRIPT_DIR"