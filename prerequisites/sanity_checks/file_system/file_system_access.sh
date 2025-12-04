#!/bin/bash
#============================================================#
#  Combined Access Checks
#  Runs:
#    1) user_space_access.sh   – verifies /ibex/user/$USER
#    2) dataset_access.sh      – verifies /ibex/reference/CV/tinyimagenet
#  Note:
#    Each sub-script handles its own detailed echoes and exit codes.
#============================================================#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_CHECK="${SCRIPT_DIR}/user_space_access.sh"
DATA_CHECK="${SCRIPT_DIR}/dataset_access.sh"

# -------- Verify scripts exist and are executable --------
for script in "$USER_CHECK" "$DATA_CHECK"; do
  if [[ ! -f "$script" ]]; then
    echo "ERROR: Missing script: $script" >&2
    echo "Please ensure all prerequisite scripts are present." >&2
    exit 1
  fi
  if [[ ! -x "$script" ]]; then
    echo "ERROR: Script is not executable: $script" >&2
    echo "Fix with: chmod +x $script" >&2
    exit 1
  fi
done

# -------- Run the checks (their own echoes will appear) --------
bash "$USER_CHECK"
echo
bash "$DATA_CHECK"
