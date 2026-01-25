#!/bin/bash
#======================================================================#
#  Cluster Sanity Check Launcher
#======================================================================#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FS_CHECK="${SCRIPT_DIR}/file_system/file_system_access.sh"
GPU_CHECK="${SCRIPT_DIR}/gpu_nodes/gpu_nodes_sanity_checks.sh"

echo "───────────────────────────────────────────────"
echo " Verifying sub-scripts..."
echo "───────────────────────────────────────────────"

for script in "$FS_CHECK" "$GPU_CHECK"; do
  if [[ ! -f "$script" ]]; then
    echo "ERROR: Missing script: $script" >&2
    exit 1
  fi
  if [[ ! -x "$script" ]]; then
    echo "ERROR: Script is not executable: $script" >&2
    echo "Fix with: chmod +x $script" >&2
    exit 1
  fi
done

echo "All required sub-scripts are present and executable."
echo

FS_RC=0
GPU_RC=0

# ---------------------- File system checks -----------------------
echo "───────────────────────────────────────────────"
echo " Running file system sanity checks..."
echo " (user space + TinyImageNet dataset)"
echo "───────────────────────────────────────────────"
echo

# Running inside an `if` prevents `set -e` from aborting the script
if ! bash "$FS_CHECK"; then
  FS_RC=$?
  echo
  echo "[!] File system checks failed with exit code: ${FS_RC}"
else
  FS_RC=0
  echo
  echo "File system checks completed successfully."
fi

# ---------------------- GPU node checks --------------------------
echo
echo "───────────────────────────────────────────────"
echo " Submitting GPU node sanity check jobs..."
echo " (V100 + A100, single/multi-GPU, multi-node)"
echo "───────────────────────────────────────────────"
echo

if ! bash "$GPU_CHECK"; then
  GPU_RC=$?
  echo
  echo "[!] GPU node sanity launcher failed with exit code: ${GPU_RC}"
else
  GPU_RC=0
  echo
  echo "GPU node sanity jobs submitted successfully."
fi

# ---------------------- Summary ---------------------------------
echo
echo "=================================================================="
echo " Cluster sanity check launcher completed."
echo
echo "File system checks exit code : ${FS_RC}"
echo "GPU node checks launcher code: ${GPU_RC}"
echo
echo "• Use 'squeue -u $USER' to monitor GPU sanity jobs."
echo "• Check ./log/ under the GPU sanity scripts for job outputs."
echo "=================================================================="

# Final exit code: non-zero if either part failed
if [[ $FS_RC -ne 0 || $GPU_RC -ne 0 ]]; then
  exit 1
fi

exit 0
