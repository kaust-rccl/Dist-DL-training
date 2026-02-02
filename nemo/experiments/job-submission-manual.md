# Jobs Submission Manual

## Purpose of This Guide

This guide is designed to help workshop participants **submit all NeMo experiment jobs early in the session**, allowing sufficient time for jobs to queue, start, and complete while the workshop progresses.



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

## 1) Data Parallel — LLaMA 3.1 8B (1 → 2 → 4 → 8 GPUs)

This section corresponds to
Exercise: [Data Parallel Training — LLaMA 3.1 8B](data_parallel/README.md#31-llama-31-8b-scaling-table-lora-data-parallel)

### 1 GPU

```commandline
cd data_parallel/llama31_8b/1_gpu
sbatch single_gpu.slurm
cd -
```

### 2 GPUs

```commandline
cd data_parallel/llama31_8b/2_gpus
sbatch 2_gpus.slurm
cd -
```

### 4 GPUs

```commandline
cd data_parallel/llama31_8b/4_gpus
sbatch 4_gpus.slurm
cd -
```

### 8 GPUs

```commandline
cd data_parallel/llama31_8b/8_gpus
sbatch 8_gpus.slurm
cd -
```

---

## 2) Data Parallel — Mixtral 8×7B (2 → 4 → 8 GPUs)

This section corresponds to
Exercise: [Data Parallel Training — Mixtral 8×7B](data_parallel/README.md#32-mixtral-87b-scaling-table-lora-data-parallel--expert-parallel)

### 2 GPUs

```commandline
cd data_parallel/mixtral_8x7b/2_gpus
sbatch 2_gpus.slurm
cd -
```
---

### 4 GPUs

```commandline
cd data_parallel/mixtral_8x7b/4_gpus
sbatch 4_gpus.slurm
cd -
```
---

### 8 GPUs

```commandline
cd data_parallel/mixtral_8x7b/8_gpus
sbatch 8_gpus.slurm
cd -
```
---

## 3) Model Parallel — LLaMA 3.1 8B (2 → 4 → 8 GPUs)

This section corresponds to Exercise: [Model Parallel Training — LLaMA 3.1 8B](model_parallel/README.md#31-llama-31-8b-scaling-table-)

### 2 GPUs
```commandline
cd model_parallel/llama31_8b/2_gpus
sbatch 2_gpus.slurm
cd -
```

### 4 GPUs
```commandline
cd model_parallel/llama31_8b/4_gpus
sbatch 4_gpus.slurm
cd -
```

### 8 GPUs
```commandline
cd model_parallel/llama31_8b/8_gpus
sbatch 8_gpus.slurm
cd -
```
---

## 4) Model Parallel — Mixtral 8×7B (4 → 8 GPUs)

This section corresponds to Exercise: [Model Parallel Training — Mixtral 8×7B](model_parallel/README.md#32-mixtral-87b-scaling-table)

### 4 GPUs
```commandline
cd model_parallel/mixtral_8x7b/4_gpus
sbatch 4_gpus.slurm
cd -
```

### 8 GPUs
```commandline
cd model_parallel/mixtral_8x7b/8_gpus
sbatch 8_gpus.slurm
cd -
```
---
## Conclusion

After submitting the jobs, participants are expected to **periodically check their job status** using the provided SLURM commands.

Once the workshop reaches the results analysis phase, please return to the **generated log files and outputs** corresponding to each experiment. These logs will be used to:
- Analyze performance and scaling behavior
- Discuss trade-offs observed across single-GPU, multi-GPU, model-parallel, and data-parallel runs.

Having jobs submitted early and tracking their progress ensures that **meaningful results are available for discussion** by the end of the session.
