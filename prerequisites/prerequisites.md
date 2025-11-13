# Workshop Prerequisites

This document describes all pre-workshop sanity checks and environment requirements.  
Each prerequisite is isolated under its own directory for clarity, and all shared setup logic is provided under `common_setup/`.

---

## Directory Structure

```
prerequisites/
├── sanity_checks/
│ ├── gpu_nodes
│ │ ├── single_v100_gpu.slurm
│ │ ├── multi_v100_gpus.slurm
│ │ ├── multi_v100_nodes.slurm
│ │ ├── single_a100_gpu.slurm
│ │ ├── multi_a100_gpus.slurm
│ │ ├── multi_a100_nodes.slurm
│ │ ├──gpu_sanity_checks.sh # Launcher script for all GPU checks
│ │ └── README.md
│ └──  README.md
├── wandb_access/
│ └──  README.md
└── prerequisites.md 
```

## Available Checks

| Category                    | Description                                                         | Location                                  |
|-----------------------------|---------------------------------------------------------------------|-------------------------------------------|
| **Sanity Checks**           | Core cluster access checks (GPU visibility, SLURM allocation, etc.) | [sanity_checks/](sanity_checks/README.md) |
| **Weights & Biases Access** | Verifies your W&B credentials and API connectivity                  | [wandb_access/](wandb_access/README.md)   |

---

