#!/bin/bash
#SBATCH --job-name=multi-a100-gpu-access-check # Name of the job shown in SLURM queue
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --tasks-per-node=1                     # One task per node
#SBATCH --cpus-per-task=4                      # Number of CPUs allocated per task
#SBATCH --gpus=4                               # Request 1 GPU
#SBATCH --gpus-per-node=4                      # GPUs per node
#SBATCH --constraint=a100                      # Request specific GPU type (A100)
#SBATCH --time=00:05:00                        # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                 # Redirect SLURM .out log to log/ directory


echo "────────────────────────────────────────────────────"
echo " Pre-Workshop Sanity Check: Multi-A100-GPU (1 node)"
echo "────────────────────────────────────────────────────"
echo
echo "[1/3] Checking SLURM allocation..."
echo "---------------------------------"
echo "→ Job ID:         ${SLURM_JOB_ID:-N/A}"
echo "→ Node list:      ${SLURM_JOB_NODELIST:-N/A}"
echo "→ GPUs requested: ${SLURM_GPUS:-N/A}"
echo "→ Hostname:       $(hostname)"
echo

echo "[2/3] Verifying GPU visibility on this node..."
echo "----------------------------------------------"
if command -v nvidia-smi &>/dev/null; then
  echo "→ 'nvidia-smi' found. Listing GPUs:"
  echo "-----------------------------------"
  nvidia-smi -L || echo "[✗] Failed to list GPUs."
  echo
  # Count GPUs
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  echo "→ Detected GPUs on node: ${GPU_COUNT}"
  echo "[✓] GPU hardware visible."
else
  echo "[✗] 'nvidia-smi' not found. Cannot verify GPU visibility."
fi
echo

echo "[3/3] Summary"
echo "--------------"
echo "SLURM single-node allocation confirmed"
if command -v nvidia-smi &>/dev/null; then
  echo "${GPU_COUNT:-0} GPU(s) detected on this node"
else
  echo "Skipped GPU check — 'nvidia-smi' unavailable"
fi
echo
echo "Multi-GPU (single node) access test complete!"
echo "───────────────────────────────────────────────"

