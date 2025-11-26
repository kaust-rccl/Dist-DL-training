# NeMo Workshop Directory Overview

This directory contains all components needed to run the hands-on NeMo Factory fine-tuning workshop. It is organized
into two main parts:

- [**`models/`**](./models) — where pretrained base models and import scripts live  
- [**`experiments/`**](./experiments) — where all data-parallel and model-parallel fine-tuning runs are executed

Together, these folders provide a complete workflow:  
**import → prepare → run → analyze**.

---

## 1. `models/` — Base Models & Import Scripts

This directory contains the pretrained model checkpoints used for all experiments in the workshop.  
Each model subdirectory includes:

- the **import script** used to download or prepare the model using NeMo Factory  
- a `model/` folder where the converted NeMo checkpoint will be stored  
- a small README describing how to import the model

Directory structure:

```text
models/
│
├── llama31_8b/
│ ├── import_llama.slurm
│ └── model/ # generated after import
│
├── mixtral_8x7b/
│ ├── import_mixtral.slurm
│ └── model/ # generated after import
│
└── template/ # reference layout for additional models
```

### What this directory is for

- Running a one-time import step for each model  
- Ensuring all participants start from identical NeMo-formatted checkpoints  
- Keeping models separate from experiment outputs  
- Enabling reusable cached model weights across Data-Parallel and Model-Parallel runs

The `model/` folders remain untouched by experiments — they are **read-only sources** for all fine-tuning jobs.

---

## 2. `experiments/` — Data-Parallel & Model-Parallel Training

All fine-tuning experiments live here.  
The directory is split into two tracks:
```text
experiments/
│
├── data_parallel/
└── model_parallel/
```

### `data_parallel/`

This track demonstrates scaling behavior using **LoRA fine-tuning** with:

- 1 GPU
- 2 GPUs
- 4 GPUs
- 8 GPUs

It focuses on:

- global batch size scaling  
- step-time behavior  
- GPU memory usage  
- how LoRA enables single-GPU fine-tuning of large models  

Models supported:

```text
data_parallel/llama31_8b/
data_parallel/mixtral_8x7b/
```

Each model directory includes subfolders for **1/2/4/8 GPUs**, SLURM scripts, and scaling tables.

### `model_parallel/`

This track demonstrates **true large-model training** with LoRA disabled.  
These runs require **multiple GPUs**, because the full model must be sharded using:

- Tensor Model Parallelism (TP)
- Expert Parallelism (EP)
- Sequence Parallelism

There is **no single-GPU experiment** in this track.

Models supported:

```text
model_parallel/llama31_8b/
model_parallel/mixtral_8x7b/
```

This track highlights:

- how large models are split across GPUs  
- differences between dense (LLaMA) and MoE (Mixtral) architectures  
- how memory footprint and communication patterns change with TP/EP  

---

## 3. How the Two Main Folders Work Together

The workflow across the `nemo/` top-level folder is:
```text
      ┌────────────────────────┐
      │     models/ (import)   │
      │  - import scripts      │
      │  - base checkpoints    │
      └─────────────┬──────────┘
                    │
                    ▼
      ┌────────────────────────┐
      │   experiments/         │
      │  - DP (LoRA)           │
      │  - MP (TP/EP)          │
      │  runs read from models/│
      └────────────────────────┘
```
- `models/` is used **once** to import each base model  
- `experiments/` is used **many times** to run DP and MP training jobs  
- Both share the same container, caching strategy, and overall workflow  

This separation keeps the repo clean and avoids mixing:

- heavy model checkpoints  
- experimental outputs  
- GPU logs  
- SLURM logs  
- per-configuration directories

---

## 4. Recommended Workflow for Participants

1. **Import models**  
   Go into `models/` and run the import scripts for LLaMA and Mixtral.

2. **Choose a training mode**  
   Navigate to `experiments/data_parallel/` or `experiments/model_parallel/`.

3. **Pick GPU count or parallelism configuration**  
   - DP: 1 / 2 / 4 / 8 GPUs  
   - MP: TP × EP configurations (2 GPUs minimum)

4. **Submit the SLURM script**  
   Each experiment folder contains its own `sbatch` files.

5. **Analyze logs**  
   - GPU memory  
   - train-step timing  
   - wall-clock time  
   - scaling behavior  

6. **Fill in the provided scaling tables**  
   Compare your results with other participants.