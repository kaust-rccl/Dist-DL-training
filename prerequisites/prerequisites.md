# Workshop Prerequisites

This document describes all pre-workshop environment requirements and how to run the corresponding validation checks.
Each prerequisite lives in its own directory for clarity, and common logic is provided under the relevant sub-folders.

All participants **must** complete these checks before the workshop to ensure smooth execution of all hands-on
exercises.

---

## Directory Structure

```
prerequisites/
├── sanity_checks/
│   ├── file_system
│   │   ├── user_space_access.sh
│   │   ├── dataset_access.sh
│   │   ├── file_system_access.sh
│   │   └── README.md
│   ├── gpu_nodes
│   │   ├── single_v100_gpu.slurm
│   │   ├── multi_v100_gpus.slurm
│   │   ├── multi_v100_nodes.slurm
│   │   ├── single_a100_gpu.slurm
│   │   ├── multi_a100_gpus.slurm
│   │   ├── multi_a100_nodes.slurm
│   │   ├── gpu_nodes_sanity_checks.sh
│   │   └── README.md
│   ├── cluster_sanity_check.sh   # Combined launcher (file system + GPU)
│   └── README.md
│
├── wandb_access/
│   └── README.md
│
└── prerequisites.md   # (this file)

```
## 1. Hardware Requirements

Participants must have access to:

- **A laptop or workstation** with a stable terminal or SSH client.

- **Linux/macOS**: native terminal is sufficient.

- **A modern web browser** (Chrome, Firefox, or Edge) for Zoom.

- **A headset and microphone** for communication during online sessions

## 2. Network Access

- Participants connecting **from outside KAUST** must ensure they have a working **KAUST VPN connection**.

  - Verify VPN access at least one day before the workshop.

- Test that you can connect to the IBEX login node via:

    ```commandline
    ssh username@glogin.ibex.kaust.edu.sa
    ```

## 3. IBEX Login and Environment Setup

Before the workshop:

- Ensure your **IBEX account is active and accessible**.

  - If you don’t yet have access, request temporary credentials via the registration form.

- Once logged in, clone the workshop materials:

    ```commandline
    git clone https://github.com/kaust-rccl/Dist-DL-training.git
    ```
## 4. Recommended Prior Sessions

To get the most out of this workshop, it is **strongly recommended** that participants have:

Completed the **KAUST Data Science Onboarding** session (or equivalent).

Basic familiarity with:

- Python programming

- HPC environments (modules, Slurm jobs)

- Version control with Git

## 5. Pre-Workshop Verification (Sample Run)

### Checklist (Complete Before Workshop)

| Check                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                     | What to do                                                                               | Status |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|--------|
| Cluster sanity checks   | Runs **all required cluster validations**, including:<br>• File system access (read/write/delete under `/ibex/user/$USER`)<br>• Optional dataset readability check<br>• GPU node access on both **V100** and **A100** partitions<br>• Single-GPU, multi-GPU, and multi-node SLURM job submission<br>• GPU visibility via `nvidia-smi` inside jobs<br><br>**GPU results appear in `sanity_checks/gpu_nodes/log/` and must be checked manually.** | Run [`sanity_checks/cluster_sanity_checks.sh`](./sanity_checks/cluster_sanity_checks.sh) |        |
| W&B API key setup       | Ensure you have a valid Weights & Biases account and API key to enable metric logging during training.                                                                                                                                                                                                                                                                                                                                          | See [`wandb_access/README.md`](./wandb_access/README.md)                                 |        |
| Internet access for W&B | Node must be able to reach `api.wandb.ai` during training to upload logs.                                                                                                                                                                                                                                                                                                                                                                       | See [`wandb_access/README.md`](./wandb_access/README.md)                                 |        |

## Expected Output

### File System Check Output

#### Access check for personal directory `/ibex/user/$USER` Summary

```commandline
[✓] Path exists: /ibex/user/<username>
[✓] Permissions OK (rwx)
[✓] Write test passed
[✓] Read test passed
[✓] Delete test passed
/ibex/user/<username> is accessible.
```

#### Access check for `tinyimagenet` directory

```commandline
[✓] Path exists and is a directory
[✓] Permissions allow read/list
[✓] Readability test passed (listed contents and read a file)

Access to /ibex/reference/CV/tinyimagenet verified.
```

### GPU Check Output (Submission Only)

Example:

```commandline
[V100] Submitting single-GPU check...
Submitted batch job xxxxxxxx
[A100] Submitting multi-node check...
Submitted batch job xxxxxxxx
```

**This is NOT the actual sanity result** — you must inspect the logs.

#### ⚠️ You MUST Check the GPU Logs

For all GPU jobs, the real outputs are written to:

```commandline
<submission_path>/log/
```

Each job creates a log file such as:

```commandline
log/single_v100_gpu-42123456.out
log/multi_a100_nodes-42123460.out
```

**Success Criteria**

A participant passes the sanity checks when:

- **All logs** show `[✓]` for allocation, reachability, and GPU visibility

- No `[✗]` markers or SLURM step errors appear

- `nvidia-smi` lists the expected GPU models (V100 or A100)

### Weights and Biases Authentication

Run:

```commandline
wandb login
```

You should see:

```commandline
Successfully logged in to Weights & Biases!
```

## If Any Check Fails

If any check fails, please contact <insert TA / workshop support contact>.

### Include:

- The failing check name
- Any relevant log file(s)
- Your username
- Error message or screenshot

This helps us diagnose issues quickly before the workshop begins.