# Sanity Checks Overview

This directory contains all pre-workshop system sanity checks used to validate the environment on Ibex before running any distributed GPU training.
These checks ensure that every participant can read required files, write to their user directory, and successfully submit GPU jobs on both V100 and A100 partitions.

### Running these checks before the workshop helps avoid runtime issues such as:
- Missing GPU access
- Wrong partition usage
- File permission errors
- Multi-node job failures
- Missing datasets
- SLURM step creation errors

---
## What These Checks Validate
### File System Access

- Ensures the participant can:

- Access their personal directory /ibex/user/$USER

- Read/write/delete files normally

- Access the TinyImageNet dataset under /ibex/reference/CV/tinyimagenet

### GPU Node Access

- Ensures the participant can:

- Allocate SLURM jobs on V100 and A100 partitions

- Launch single-GPU, multi-GPU (same node), and multi-node jobs

- Detect GPUs from inside SLURM allocations (nvidia-smi)

> **Important**:
GPU job results are not printed to the terminal.
They are written to log files under the gpu_nodes/log/ directory.
You must check these logs after the jobs finish.

---


---
## How to Run All Checks (Recommended)

Run everything with one command:
```commandline
bash cluster_sanity_checks.sh
```
This will:

1. Run file system checks immediately

2. Submit all GPU jobs to SLURM

3. Print job submission confirmations

4. Remind you to check logs after jobs finish

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

## Running Checks Separately
If you prefer to run only part of the sanity suite:

- File system checks only
    ```commandline
    cd file_system
    ./file_system_access.sh
    ```
  → Full explanation available in [file_system/README.md](file_system/README.md)

- GPU node checks only
    ```commandline
    cd gpu_nodes
    ./gpu_nodes_sanity_launcher.sh
    ```
  → Full explanation available in [gpu_nodes/README.md](gpu_nodes/README.md)