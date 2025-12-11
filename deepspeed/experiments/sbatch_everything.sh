#!/bin/bash
set -euo pipefail

echo "==================== Submitting ALL DeepSpeed experiments ===================="


################################################################################
# BASELINE
################################################################################
echo ""
echo ">>> Submitting BASELINE experiment"
sbatch baseline/baseline.slurm


################################################################################
# SINGLE GPU (no offloading)
################################################################################
#echo ""
#echo ">>> Submitting SINGLE-GPU experiments (no offloading)"
#
#echo " - Zero-0"
#sbatch deepspeed-single-gpu/zero_0/deepspeed_zero0.slurm
#
#echo " - Zero-1"
#sbatch deepspeed-single-gpu/zero_1/deepspeed_zero1.slurm
#
#echo " - Zero-2"
#sbatch deepspeed-single-gpu/zero_2/deepspeed_zero2.slurm
#
#echo " - Zero-3"
#sbatch deepspeed-single-gpu/zero_3/deepspeed_zero3.slurm


################################################################################
# SINGLE GPU (CPU OFFLOADING)
################################################################################
echo ""
echo ">>> Submitting SINGLE-GPU CPU-OFFLOADING experiments"

echo " - Zero-1 Offload"
sbatch deepspeed-single-gpu/cpu_offloading/zero_1/deepspeed_zero1_offload.slurm

echo " - Zero-2 Offload"
sbatch deepspeed-single-gpu/cpu_offloading/zero_2/deepspeed_zero2_offload.slurm

echo " - Zero-3 Offload"
sbatch deepspeed-single-gpu/cpu_offloading/zero_3/deepspeed_zero3_offload.slurm

echo " - Zero-2 Offload + Pinned Memory"
sbatch deepspeed-single-gpu/cpu_offloading/pinned_memory/deepspeed_zero2_offload_pinned_memory.slurm



################################################################################
# MULTI-GPU EXPERIMENTS
################################################################################
echo ""
echo ">>> Submitting MULTI-GPU experiments"


######## 2 GPUs ########
echo ""
echo ">>> 2-GPU baseline"
sbatch deepspeed-multi-gpu/2_gpus/deepspeed_2_gpus.slurm


### 2-GPU ZeRO Stage Comparison (NO offloading)
echo ""
echo ">>> 2-GPU ZeRO Stage Comparison (NO offloading)"

echo " - Zero-1"
sbatch deepspeed-multi-gpu/2_gpus_stages_comparison/zero_1/deepspeed_2_gpus_zero1.slurm

echo " - Zero-2"
sbatch deepspeed-multi-gpu/2_gpus_stages_comparison/zero_2/deepspeed_2_gpus_zero2.slurm

echo " - Zero-3"
sbatch deepspeed-multi-gpu/2_gpus_stages_comparison/zero_3/deepspeed_2_gpus_zero3.slurm


#### 2-GPU ZeRO Stage Comparison (OFFLOADING)
#echo ""
#echo ">>> 2-GPU ZeRO Stage Comparison (CPU OFFLOADING)"
#
#echo " - Zero-1 Offload"
#sbatch deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_1/deepspeed_2_gpus_zero1_offload.slurm
#
#echo " - Zero-2 Offload"
#sbatch deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_2/deepspeed_2_gpus_zero2_offload.slurm
#
#echo " - Zero-3 Offload"
#sbatch deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_3/deepspeed_2_gpus_zero3_offload.slurm


######## 4 GPUs ########
echo ""
echo ">>> Submitting 4-GPU experiment"
sbatch deepspeed-multi-gpu/4_gpus/deepspeed_4_gpus.slurm


######## 8 GPUs ########
echo ""
echo ">>> Submitting 8-GPU experiment"
sbatch deepspeed-multi-gpu/8_gpus/deepspeed_8_gpus.slurm



################################################################################
# MULTI-NODE EXPERIMENTS
################################################################################
#echo ""
#echo ">>> Submitting MULTI-NODE experiments"
#
#
######### 2 Nodes ########
#echo " - 2 Nodes"
#sbatch deepspeed-multi-node/2_nodes/deepspeed_2_nodes.slurm
#
######### 4 Nodes ########
#echo " - 4 Nodes"
#sbatch deepspeed-multi-node/4_nodes/deepspeed_4_nodes.slurm
#
######### 8 Nodes ########
#echo " - 8 Nodes"
#sbatch deepspeed-multi-node/8_nodes/deepspeed_8_nodes.slurm
#


echo ""
echo "==================== All jobs have been submitted ===================="
