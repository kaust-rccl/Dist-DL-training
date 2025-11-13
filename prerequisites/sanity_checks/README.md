# Sanity Checks Overview

This directory contains all **pre-workshop system sanity checks**.  
Each check verifies a different layer of the cluster environment to ensure participants can successfully run distributed GPU workloads during the workshop.

## Purpose

These checks ensure:

1. **SLURM allocation works** for both V100 and A100 partitions  
2. **GPU nodes are reachable** and properly configured  
3. **`nvidia-smi` detects the GPUs** from within allocated jobs  
4. (Optionally) Multi-node jobs can communicate via `srun`

If any of these fail, participants may face runtime issues later (e.g., missing GPUs, invalid partitions, or step creation errors).

---

## Current Checks

| Check               | Description                                                                                  | Location                          |
|---------------------|----------------------------------------------------------------------------------------------|-----------------------------------|
| **GPU Node Access** | Verifies SLURM allocation, node reachability, and GPU visibility on both V100 and A100 nodes | [gpu_nodes/](gpu_nodes/README.md) |

---

