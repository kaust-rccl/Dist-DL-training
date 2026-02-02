# Submit Jobs Manual

## ⚠️ Important rule

Make sure you start from the [`experiments`](.) directory.

```commandline
pwd
```

All jobs must be submitted from inside their own directory, and you must return to the experiments directory after each
submission.

---

## Track your submissions (recommended)

Keep this running in a separate terminal tab while submitting jobs.

```commandline

watch squeue --me
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