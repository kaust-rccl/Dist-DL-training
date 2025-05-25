#!/bin/bash
#SBATCH --job-name=baseline--bloom-finetune
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --constraint=v100
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

source /ibex/user/x_mohameta/miniforge/etc/profile.d/conda.sh
conda activate deepspeed-finetune
module load cuda/12.4.1

# Start memory logging in background
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv,nounits -l 5 > baseline-single-gpu_memory_log.csv &
MEMORY_LOG_PID=$!

python train.py

# Stop logging
kill $MEMORY_LOG_PID
