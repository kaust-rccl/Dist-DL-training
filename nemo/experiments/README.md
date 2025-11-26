# Experiments Overview

This directory contains the two main training experiment tracks for the workshop:

- [**`data_parallel/`** ](./data_parallel) — Fine-tuning large models using *data parallelism* with **LoRA enabled**
- [**`model_parallel/`**](./model_parallel0) — Fine-tuning large models using *model parallelism* with **LoRA disabled**

Each track demonstrates a different strategy for training large language models (LLMs) on multi-GPU systems, and each
highlights different scaling behaviors, performance patterns, and memory characteristics.

Both tracks use the same NeMo Factory infrastructure, container environment, and caching system — but differ in how the
model is distributed across GPUs.

---

## 1. Data-Parallel Experiments (`data_parallel/`)

The **data-parallel track** focuses on the most common form of distributed training: replicating the model on multiple
GPUs and splitting the input batch across them.

To make this feasible on A100 nodes, we use **LoRA (Low-Rank Adaptation)**:

- LoRA dramatically reduces trainable parameters
- The full model remains frozen
- Memory usage stays low enough to fit large models on a single GPU
- This lets us scale from **1 → 2 → 4 → 8 GPUs**

The goal of this track is to explore:

- How throughput changes with more GPUs
- How memory usage changes
- How step time and total job time scale
- How LoRA interacts with DP in practice

Inside `data_parallel/`:

```text
data_parallel/
│
├── llama31_8b/ # LLaMA 3.1 8B LoRA data-parallel runs
└── mixtral_8x7b/ # Mixtral 8×7B LoRA data-parallel runs
```

The runs are short, designed for quick iteration and hands-on scaling analysis.

---

## 2. Model-Parallel Experiments (`model_parallel/`)

The **model-parallel track** explores how to train very large models *without* parameter-efficient fine-tuning.

Because LoRA is disabled in this track:

> **There are no single-GPU runs in model-parallel experiments.**

Instead, we use the relevant parallelism strategies:

### Dense models:

- **Tensor Model Parallelism (TP)**  
  Splits individual weight matrices across GPUs.

### Mixture-of-Experts (MoE) models:

- **Expert Parallelism (EP)**  
  Splits MoE experts across GPUs.

Depending on the model architecture and configuration, the total number of GPUs required is:

```text
total_gpus = TP × EP
```

Inside `model_parallel/`:

```text
model_parallel/
│
├── llama31_8b/ # LLaMA 3.1 8B TP runs
└── mixtral_8x7b/ # Mixtral 8×7B TP × EP runs
```

Each model contains subdirectories corresponding to the chosen parallelism configurations (e.g., 2 GPUs, 4 GPUs, 8
GPUs).

---

## 3. How to Navigate This Folder

Each subdirectory contains:

- The SLURM scripts (`*.slurm`)
- A `README.md` describing how that experiment works
- The expected NeMo overrides
- Instructions for extracting metrics
- A scaling table to fill in manually

The workflow is:

1. Choose an experiment (`data_parallel` or `model_parallel`)
2. Choose a model (`llama` or `mixtral`)
3. Select the GPU count or parallel configuration
4. Submit the job using `sbatch`
5. Inspect the logs and fill in the scaling tables


