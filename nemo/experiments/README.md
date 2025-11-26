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

