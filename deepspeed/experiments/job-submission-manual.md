# Submit Jobs Manual

## ⚠️ Important rule:

Make sure you start from [experiments](.) dir.

---

## Track your submissions (recommended)

Keep this running in a separate terminal tab while submitting jobs.

```commandline
watch -n squeue --me
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

## 1) Baseline

This section corresponds to
Exercise: [Baseline Training](../README.md#exercise-run-the-baseline-training--fill-evaluation-summary-table)

```commandline
cd baseline
sbatch baseline.slurm
cd -
```

---

## 2) Single GPU — CPU offloading (ignore pinned)

This section corresponds to
Exercise: [ ZeRO Stages with CPU Offloading](../README.md#exercise-benchmarking-zero-stages-with-cpu-offloading)

## Zero-1 offload

```commandline
cd deepspeed-single-gpu/cpu_offloading/zero_1
sbatch deepspeed_zero1_offload.slurm
cd -
```

## Zero-2 offload

```commandline
cd deepspeed-single-gpu/cpu_offloading/zero_2
sbatch deepspeed_zero2_offload.slurm
cd -
```

## Zero-3 offload

```commandline
cd deepspeed-single-gpu/cpu_offloading/zero_3
sbatch deepspeed_zero3_offload.slurm
cd -
```
---

## 3) Multi-GPU scaling (2 → 4 → 8 GPUs)

This section corresponds to Exercise: [Scaling on Multi GPUs](../README.md#part-1-scaling-on-multi-gpu-setup)

### 2 GPUs

```commandline
cd deepspeed-multi-gpu/2_gpus
sbatch deepspeed_2_gpus.slurm
cd -
```

### 4 GPUs

```commandline
cd deepspeed-multi-gpu/4_gpus
sbatch deepspeed_4_gpus.slurm
cd -
```

### 8 GPUs

```commandline
cd deepspeed-multi-gpu/8_gpus
sbatch deepspeed_8_gpus.slurm
cd -
```

---

## 4) 2 GPUs — stages comparison

This section corresponds to Exercise: [2 GPUs Zero Stages Comparison](../README.md#part-2-2-gpu-zero-stage-comparison)

### Zero-1

``` commandline 
cd deepspeed-multi-gpu/2_gpus_stages_comparison/zero_1
sbatch deepspeed_2_gpus_zero1.slurm
cd -

```

### Zero-2

``` commandline 
cd deepspeed-multi-gpu/2_gpus_stages_comparison/zero_2
sbatch deepspeed_2_gpus_zero2.slurm
cd -

```

### Zero-3

``` commandline 
cd deepspeed-multi-gpu/2_gpus_stages_comparison/zero_3
sbatch deepspeed_2_gpus_zero3.slurm
cd -
```

---

## 5) Multi-node scaling (2 → 4 → 8 nodes)

This section corresponds to Exercise: [Scale on Multi Nodes](../README.md#exercise-multinode-scaling)

### 2 nodes

```commandline
cd deepspeed-multi-node/2_nodes
sbatch deepspeed_2_nodes.slurm
cd -
```

### 4 nodes

```commandline
cd deepspeed-multi-node/4_nodes
sbatch deepspeed_4_nodes.slurm
cd -
```

### 8 nodes

```commandline
cd deepspeed-multi-node/8_nodes
sbatch deepspeed_8_nodes.slurm
cd -
```