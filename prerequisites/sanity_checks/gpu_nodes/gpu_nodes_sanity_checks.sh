#!/bin/bash
#======================================================================#
#  GPU Nodes Sanity Check Launcher
#----------------------------------------------------------------------
#  This script submits all GPU access sanity check jobs for the workshop:
#    - V100 single-GPU / multi-GPU / multi-node
#    - A100 single-GPU / multi-GPU / multi-node
#
#  Purpose:
#    Quickly verify that each participant can:
#      1) Land on the correct GPU partitions (V100 & A100)
#      2) Access GPUs inside the SLURM allocation
#      3) Reach multiple nodes when allocated
#
#  Notes:
#    - Each sbatch command submits an independent job.
#    - Logs are automatically written to the /log directory
#      as defined inside each .slurm script.
#======================================================================#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ---------------------- V100 CHECKS ------------------------
# These scripts target the V100 GPU partition.
echo "[V100] Submitting single-GPU check..."
sbatch "${SCRIPT_DIR}/single_v100_gpu.slurm"
echo "[V100] Submitting multi-GPU (same node) check..."
sbatch "${SCRIPT_DIR}/multi_v100_gpus.slurm"
echo "[V100] Submitting multi-node check..."
sbatch "${SCRIPT_DIR}/multi_v100_nodes.slurm"

# ---------------------- A100 CHECKS ------------------------
# These scripts target the A100 GPU partition.
echo "[A100] Submitting single-GPU check..."
sbatch "${SCRIPT_DIR}/single_a100_gpu.slurm"
echo "[A100] Submitting multi-GPU (same node) check..."
sbatch "${SCRIPT_DIR}/multi_a100_gpus.slurm"
echo "[A100] Submitting multi-node check..."
sbatch "${SCRIPT_DIR}/multi_a100_nodes.slurm"

# ---------------------- SUMMARY ----------------------------
# Print a footer message with a reminder to check logs.
echo
echo "───────────────────────────────────────────────"
echo "✅ All sanity check jobs submitted!"
echo "   Use 'squeue -u $USER' to monitor progress."
echo "   Logs will appear under ./log/ after completion."
echo "───────────────────────────────────────────────"
