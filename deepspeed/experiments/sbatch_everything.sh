#!/bin/bash
set -euo pipefail

echo "==================== Submitting ALL DeepSpeed experiments ===================="

# Helper function to submit a SLURM script from its own directory
submit() {
    local SCRIPT_PATH="$(realpath "$1")"
    local SCRIPT_DIR
    SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
    local SCRIPT_FILE
    SCRIPT_FILE="$(basename "$SCRIPT_PATH")"

    echo ""
    echo ">>> Submitting: $SCRIPT_PATH"
    echo "    (cd $SCRIPT_DIR && sbatch $SCRIPT_FILE)"

    (
        cd "$SCRIPT_DIR"
        sbatch "$SCRIPT_FILE"
    )
}

################################################################################
# BASELINE
################################################################################
submit baseline/baseline.slurm

#################################################################################
## SINGLE GPU (no offloading)
#################################################################################
#submit deepspeed-single-gpu/zero_0/deepspeed_zero0.slurm
#submit deepspeed-single-gpu/zero_1/deepspeed_zero1.slurm
#submit deepspeed-single-gpu/zero_2/deepspeed_zero2.slurm
#submit deepspeed-single-gpu/zero_3/deepspeed_zero3.slurm

################################################################################
# SINGLE GPU (CPU OFFLOADING)
################################################################################
submit deepspeed-single-gpu/cpu_offloading/zero_1/deepspeed_zero1_offload.slurm
submit deepspeed-single-gpu/cpu_offloading/zero_2/deepspeed_zero2_offload.slurm
submit deepspeed-single-gpu/cpu_offloading/zero_3/deepspeed_zero3_offload.slurm
submit deepspeed-single-gpu/cpu_offloading/pinned_memory/deepspeed_zero2_offload_pinned_memory.slurm

################################################################################
# MULTI-GPU
################################################################################
submit deepspeed-multi-gpu/2_gpus/deepspeed_2_gpus.slurm

# Stages comparison 2 GPUs (no offload)
submit deepspeed-multi-gpu/2_gpus_stages_comparison/zero_1/deepspeed_2_gpus_zero1.slurm
submit deepspeed-multi-gpu/2_gpus_stages_comparison/zero_2/deepspeed_2_gpus_zero2.slurm
submit deepspeed-multi-gpu/2_gpus_stages_comparison/zero_3/deepspeed_2_gpus_zero3.slurm

## Stages comparison 2 GPUs (offload)
#submit deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_1/deepspeed_2_gpus_zero1_offload.slurm
#submit deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_2/deepspeed_2_gpus_zero2_offload.slurm
#submit deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_3/deepspeed_2_gpus_zero3_offload.slurm

# 4 GPUs
submit deepspeed-multi-gpu/4_gpus/deepspeed_4_gpus.slurm

# 8 GPUs
submit deepspeed-multi-gpu/8_gpus/deepspeed_8_gpus.slurm

################################################################################
# MULTI-NODE
################################################################################
#submit deepspeed-multi-node/2_nodes/deepspeed_2_nodes.slurm
#submit deepspeed-multi-node/4_nodes/deepspeed_4_nodes.slurm
#submit deepspeed-multi-node/8_nodes/deepspeed_8_nodes.slurm

echo ""
echo "==================== All jobs have been submitted ===================="
