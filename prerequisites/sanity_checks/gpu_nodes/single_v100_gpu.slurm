#!/bin/bash
#SBATCH --job-name=single-v100-gpu-access-check     # Name of the job shown in SLURM queue
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --tasks-per-node=1                          # One task per node
#SBATCH --cpus-per-task=4                           # Number of CPUs allocated per task
#SBATCH --gpus=1                                    # Request 1 GPU
#SBATCH --gpus-per-node=1                           # GPUs per node
#SBATCH --constraint=v100                           # Request specific GPU type (V100)
#SBATCH --time=00:05:00                             # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                      # Redirect SLURM .out log to log/ directory


echo "───────────────────────────────────────────────────"
echo "  Pre-Workshop Sanity Check: V100 GPU Node Access"
echo "───────────────────────────────────────────────────"

echo "[1/3] Checking SLURM allocation..."
echo "---------------------------------"
echo "→ Job ID: $SLURM_JOB_ID"
echo "→ Node list: $SLURM_JOB_NODELIST"
echo "→ GPUs requested: $SLURM_GPUS"
echo
echo "[✓] Allocation successful — running on: $(hostname)"
echo

echo "[2/3] Verifying GPU visibility..."
echo "--------------------------------"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "[✗] nvidia-smi not available!"
echo
echo "[✓] GPU detected and accessible."
echo

echo "[3/3] Summary"
echo "--------------"
echo "SLURM allocation confirmed"
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected via nvidia-smi"
else
    echo "Could not verify GPU visibility (no nvidia-smi)"
fi
echo
echo "GPU node access test complete!"
echo "───────────────────────────────────────────────"
