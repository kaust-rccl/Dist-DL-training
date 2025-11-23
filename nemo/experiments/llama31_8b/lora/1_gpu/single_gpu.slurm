#!/bin/bash
#SBATCH --job-name=m-lo-1g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --constraint=a100
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out

# ------------------------------
# Resolve directory paths
# ------------------------------

SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPT_DIR="$( cd "$( dirname "$SCRIPT_PATH" )" && pwd )"
GPU_LOG_DIR="$SCRIPT_DIR/gp u_memory/$JOB_ID"
mkdir -p "$GPU_LOG_DIR"
NEMO_ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
MODEL_DIR="$NEMO_ROOT_DIR/models/llama31_8b/model"
CACHE_DIR=$NEMO_ROOT_DIR

export TORCH_HOME="$CACHE_DIR/torch"
export NEMO_DATASETS_CACHE="$CACHE_DIR/datasets"
export HF_HOME="$CACHE_DIR/hf_cache"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets_cache"
export TMPDIR="$CACHE_DIR/tmp"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export NEMO_HOME="$CACHE_DIR/nemo_cache"
mkdir -p "$TORCH_HOME" "$NEMO_DATASETS_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$NEMO_HOME"

module load cuda/12.4.1
module load singularity

dmesg --ctime | tail -n 50

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
GPU_MONITOR_PID=$!

echo "==============================="
echo " Job started"
echo " Start time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================="

# record start timestamp (seconds since epoch)
start_ts=$(date +%s)

srun singularity exec --nv /ibex/user/x_mohameta/nemo/image/nemo_25.07.sif \
nemo llm finetune \
--factory llama31_8b --yes \
trainer.devices="${SLURM_GPUS_PER_NODE}" \
trainer.max_steps=250 \
trainer.limit_val_batches=10 \
optim.lr_scheduler.warmup_steps=50 \
data.global_batch_size=8 \
resume.restore_config.path="$MODEL_DIR"

end_ts=$(date +%s)
runtime=$(( end_ts - start_ts ))

# convert seconds → HH:MM:SS
hours=$(( runtime/3600 ))
mins=$(( (runtime%3600)/60 ))
secs=$(( runtime%60 ))

echo "==============================="
echo " Job finished"
echo " End time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total time : $(printf "%02d:%02d:%02d" $hours $mins $secs)"
echo "==============================="

