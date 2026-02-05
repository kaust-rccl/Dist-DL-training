# Jobs Submission Manual

## Purpose of This Guide

This guide is designed to help workshop participants **submit all DDP experiment jobs early in the session**,
allowing sufficient time for jobs to queue, start, and complete while the workshop progresses.



---

## Working Directory Context

All commands in this guide assume that you start from the [experiments/](.) directory.

```commandline
cd experiments
```

---

## Track Your Submissions

You can check the status of your submitted jobs at any time using:

```commandline
squeue --me
```

### What this shows

- Job ID

- Job name

- State (PENDING, RUNNING, COMPLETED)

- Elapsed time

- Time limit

- Number of nodes

- Reason / assigned node

---

## 1) Multi-GPU scaling (1 → 2 → 4 → 8 GPUs)

This section corresponds to Exercise: [Baseline Training](../README.md#multi-gpu-scaling-1-node)

### 1 GPU - Baseline

```commanline
cd baseline
sbatch baseline.slurm
cd -
```

### 2 GPUs

```commanline
cd multi_gpu/2_gpus
sbatch multi_gpu.slurm
cd -
```

### 4 GPUs

```commanline
cd multi_gpu/4_gpus
sbatch multi_gpu.slurm
cd -
```

### 8 GPUs

```commanline
cd multi_gpu/8_gpus
sbatch multi_gpu.slurm
cd -
```

---

## 3) Multi-node scaling (2 → 4 → 8 nodes)

This section corresponds to Exercise: [Scaling on Multi Nodes](../README.md#multi-node-scaling-fixed-gpus-per-node)

### 2 nodes

```commandline
cd multi_node/2_nodes
sbatch multi_node.slurm
cd -
```

### 4 nodes

```commandline
cd multi_node/4_nodes
sbatch multi_node.slurm
cd -
```

### 8 nodes

```commandline
cd multi_node/8_nodes
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
- Analyze performance and scaling behavior
- Discuss trade-offs observed across single-GPU, multi-GPU, and multi-node runs

Having jobs submitted early and tracking their progress ensures that **meaningful results are available for discussion**
by the end of the session.
