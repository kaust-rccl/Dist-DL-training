# Jobs Submission Manual

## Purpose of This Guide

This guide is designed to help workshop participants submit all **BLOOM and custom-model** experiment jobs early in the
session, allowing sufficient time for jobs to queue, start, and complete while the workshop progresses.

Participants are encouraged to submit **all jobs upfront**, then focus on understanding the concepts and analyzing
results as jobs finish.

---

## Working Directory Context

All commands in this guide assume that you start from the root [`fsdp`](.) directory, which contains `bloom/`,
`custom_model/`, and the root `README.md`.

### ⚠️ Important rule

- Every job must be submitted from inside its own directory

- Always return to the [`fsdp`](.) directory after each submission using cd -

--- 

## Track Your Submissions

Keep track of your submitted jobs throughout the workshop.

You can check job status at any time using:

```commandline
squeue --me
```

### What this shows

- Job ID

- Job name

- State (PENDING, RUNNING, COMPLETED)

- Elapsed time

- Number of nodes

- Reason / assigned node

Participants are encouraged to periodically check this output while the workshop progresses.

---

## Before Submitting Any Jobs — Configure Your W&B API Key

All experiments rely on Weights & Biases (W&B) for logging.
Before submitting any jobs, you must inject your personal `WANDB_API_KEY` into all `env_vars.sh` files.

A helper script is provided to update everything at once.

```commandline
./wandb_update.sh <your_wandb_api_key>``
```

---

## 1) BLOOM — Baseline (Single GPU)

This section corresponds to Exercise: [BLOOM Baseline Training](./README.md#run-the-baseline-fine-tuning-job)

```commandline
cd bloom/baseline
sbatch baseline.slurm
cd -
```

---

## 2) BLOOM — Multi-GPU Scaling (2 → 4 → 8 GPUs)

This section corresponds to
Exercise: [BLOOM Multi-GPU Scaling](./README.md#-exercise-run-multi-gpu-experiments-and-populate-results-table)

### 2 GPUs

```commandline
cd bloom/multi_gpu/2_gpus
sbatch multi_gpu.slurm
cd -
```

### 4 GPUs

```commandline
cd bloom/multi_gpu/4_gpus
sbatch multi_gpu.slurm
cd -
```

### 8 GPUs

```commandline
cd bloom/multi_gpu/8_gpus
sbatch multi_gpu.slurm
cd -
```

---

## 3) BLOOM — Multi-Node Scaling (2 → 4 → 8 Nodes)

This section corresponds to
Exercise: [BLOOM Multi-Node Scaling](./README.md#-exercise-run-multi-node-experiments-and-populate-results-table)

## 2 nodes

```commandline
cd bloom/multi_node/2_nodes
sbatch multi_node.slurm
cd -
```

## 4 nodes

```commandline
cd bloom/multi_node/4_nodes
sbatch multi_node.slurm
cd -
```

## 8 nodes

```commandline
cd bloom/multi_node/8_nodes
sbatch multi_node.slurm
cd -
```

---


## Conclusion

After submitting the jobs, participants are expected to **periodically check their job status** using the provided SLURM
commands.

Once the workshop reaches the results analysis phase, please return to the **generated log files and outputs**
corresponding to each experiment. These logs will be used to:

- Analyze performance and scaling behavior
- Discuss trade-offs observed across single-GPU, multi-GPU, and multi-node runs

Having jobs submitted early and tracking their progress ensures that **meaningful results are available for discussion**
by the end of the session.

