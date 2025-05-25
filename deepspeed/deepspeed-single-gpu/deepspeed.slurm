#!/bin/bash
#SBATCH --job-name=deepspeed-bloom-finetune  # Name of the job shown in SLURM queue
#SBATCH --gpus=1                             # Request 1 GPU
#SBATCH --mem=32G                            # Request 32 GB of system memory
#SBATCH --constraint=v100                   # Specify V100 GPU (optional constraint)
#SBATCH --time=12:00:00                      # Maximum job runtime (HH:MM:SS)
#SBATCH --output=logs/%x-%j.out              # Output log file (%x=job name, %j=job ID)

# Load conda environment activation script
source /ibex/user/x_mohameta/miniforge/etc/profile.d/conda.sh
# Activate the environment containing DeepSpeed, Transformers, etc.
conda activate deepspeed-finetune
# Load the correct CUDA module (matching PyTorch + DeepSpeed install)
module load cuda/12.4.1

# Start memory logging in background
# Log GPU memory usage every 5 seconds to a CSV file
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total --format=csv,nounits -l 5 > deepspeed-single-gpu_memory_log.csv &
# Store background logging process ID so it can be killed later
MEMORY_LOG_PID=$!


# Run the training script using deepspeed
deepspeed train.py

# Stop logging
# Kill the background logging process after training completes
kill $MEMORY_LOG_PID
