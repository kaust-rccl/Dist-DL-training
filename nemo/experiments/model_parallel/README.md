# Model Parallel Fine-Tuning (No LoRA / No PEFT)

While the [Data Parallel](../data_parallel/README.md) section focused on **data-parallel training with LoRA**, this part
of the workshop shifts to a
different goal:

**Understanding how very large models are trained *without* PEFT — by splitting the model across multiple GPUs.**

For this section, **LoRA is intentionally disabled**.  
We want to demonstrate how large models (dense or MoE) can be trained using:

- **Tensor Model Parallelism (TP)**
- **Expert Parallelism (EP)** for Mixture-of-Experts models
- **Sequence Parallelism**

These techniques divide the *model weights themselves* across multiple GPUs, allowing models that do *not* fit on a
single device to be trained.

This is in contrast to LoRA:

- LoRA reduces *trainable parameters* and *memory*, enabling single-GPU fine-tuning.
- Model parallelism reduces the **model's memory footprint by distributing its layers**, enabling multi-GPU training.

Here, PEFT would “hide” the real behavior of model parallelism, so we disable it:

```bash
peft=none 
```

This section is therefore about **true large-model training**, not parameter-efficient approximations.

---

## Models Used in This Section

For this workshop, we will practice data-parallel fine-tuning with **two example models**, each representing a different
architectural family. This allows us to observe LoRA behavior and scaling characteristics across distinct transformer
designs.

### 1. **LLaMA 3.1 8B — Dense Decoder-Only Transformer**

LLaMA is a **dense** model:

- 8 billion parameters
- standard decoder-only stack (attention + MLP)
- no mixture-of-experts layers
- full model must be loaded during training

Without LoRA, **LLaMA 8B does NOT fit on a single A100 80GB GPU** (gradients + optimizer states exceed memory), so we
use:

- **Tensor Model Parallelism (TP)**  
  Splits large weight matrices across GPUs.

This allows LLaMA 8B to be trained on 2× A100 GPUs.

LLaMA is our example of **dense model parallelism**.

---

### 2. **Mixtral 8×7B — Sparse Mixture-of-Experts Transformer**

Mixtral uses **Mixture-of-Experts (MoE)** layers:

- 8 experts per MoE layer
- only 2 experts active per token
- total parameters ~24B
- far too large for a single GPU

Even with LoRA disabled, model parallelism makes Mixtral trainable via:

- **Expert Parallelism (EP)**  
  Shards the experts across GPUs (required to fit into memory).

To run Mixtral:

- **EP ≥ 2 is required** — it cannot fit on 1 GPU.

Mixtral is our example of **sparse model parallelism**.

---

## Why Disable LoRA Here?

To showcase the **actual mechanisms** used to train massive models:

- Tensor Parallelism  
  (splitting linear layers across GPUs)

- Expert Parallelism  
  (splitting MoE experts across GPUs)

- Sequence Parallelism  
  (splitting activations)

With LoRA, models fit easily and parallelism becomes unnecessary.  
By disabling LoRA:

- LLaMA 8B forces us to use **TP**
- Mixtral 8×7B forces us to use **EP**
- We see real **memory footprint reduction**, GPU communication, and scaling behavior

This provides a clear, hands-on understanding of the *true* model-parallel strategies used for very large LLMs.

---

## Directory Overview

TODO

---

## Before We Launch NeMo: Environment Notes

Before we run any NeMo Factory commands, there are a few Ibex-specific setup steps to take care of. These are not NeMo
requirements, but cluster requirements that ensure everything runs smoothly—especially caching, storage, and GPU
allocation. Once these are in place, we can launch NeMo Factory without encountering quota or resource errors.

It’s also important to note that NeMo Factory requires **A100 GPUs** on Ibex. The provided container and training
configuration are built around CUDA 12.x and Triton kernels optimized for A100. Running these experiments on **V100 GPUs
is not supported** and will fail during initialization.

Once the environment is prepared and we’re on A100 nodes, we can safely launch NeMo Factory for LoRA fine-tuning.

## 1. Environment Setup

### 1.1 Disk Quota & Cache Paths

By default, NeMo, PyTorch, Triton, and Hugging Face store models, datasets, and compiled kernels under `$HOME`.  
On Ibex, `$HOME` has **very limited quota**, and large models (like LLaMA 3.1 8B) can easily cause **disk quota exceeded
** errors.

#### **Solution: Redirect all cache paths to project storage**

Use a project directory (e.g., `/ibex/project/<project>/`), or you user space (e.g., `/ibex/user/$USER/`) with enough
space. Example:

```bash
export TORCH_HOME=<NEW_PATH>/NeMo/experiments/llama3180b/torch
export NEMO_DATASETS_CACHE=<NEW_PATH>/NeMo/experiments/llama3180b/datasets
export HF_HOME=<NEW_PATH>/NeMo/experiments/llama3180b/hf_cache
export HF_DATASETS_CACHE=<NEW_PATH>/NeMo/experiments/llama3180b/datasets_cache
export TMPDIR=<NEW_PATH>/NeMo/experiments/llama3180b/tmp
export TRITON_CACHE_DIR=<NEW_PATH>/NeMo/experiments/llama3180b/triton
export NEMO_HOME=<NEW_PATH>/NeMo/experiments/llama3180b/nemo_cache
```

## 2. SLURM Directives

Lightning and NeMo Factory expect very specific SLURM configurations. Incorrect directives will cause training to fail
before it even starts.

### 3.1 `--ntasks` vs `--ntasks-per-node`

Do not use:

```bash
#SBATCH --ntasks=2      # Unsupported with Lightning
```

This produces errors like:

```commandline
Unexpected error: You set `--ntasks=2` in your SLURM bash script,
but this variable is not supported.
HINT: Use `--ntasks-per-node=2` instead.
```

#### ✔ Correct:

`--ntasks-per-node`, and it must equal the number of GPUs/devices Lightning will use.

For example, if `trainer.devices=2`, use:

```bash
#SBATCH --ntasks-per-node=2
```

### 3.2 Number of GPUs

Avoid mixing `--gpus-per-task` with Lightning. It leads to **mismatched GPU visibility**.

#### Common pitfall:

For example, if `trainer.devices=2`, don't:

```bash
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1   # ❌
```

This causes:

```commandline
Unexpected error: You requested gpu: [0, 1]
But your machine only has: [0]
```

#### ✔ Correct approach:

Use only `--gpus-per-node`:
For `trainer.devices=N`, use:

```bash
#SBATCH --ntasks-per-node=N
#SBATCH --gpus-per-node=N
```

You should se:

```commandline
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/N
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/N
.
.
.
Initializing distributed: GLOBAL_RANK: N-1, MEMBER: N/N
```

These checks ensure NeMo Factory runs smoothly on Ibex without quota errors, GPU mismatches, or Lightning initialization
failures.

### 4. GPU Requirements: A100 Only

NeMo Factory fine-tuning requires **A100 GPUs** on Ibex.  
The official NeMo containers used in this workshop are built against:

- CUDA 12.x
- Triton kernels optimized for A100 tensor cores
- PyTorch builds that do **not** support V100 compute capability

Attempting to run on **V100 GPUs will fail** during initialization or container startup.  
All experiments in this workshop must be submitted to A100 nodes.

---

## Understanding the SLURM Script for This Workshop

Before diving into model–parallel experiments, it is important to understand how the SLURM script is structured and why
we need **multiple GPUs** for every run in this section.

Because **LoRA is disabled**, the full model weights, gradients, optimizer states, and activations must fit into GPU
memory. None of the models we use in this section can run on **a single A100 80GB GPU** without PEFT.

Therefore:

> **All experiments in this section start from multiple GPUs.  
> There is no single-GPU run.**

The exact number of GPUs required depends on **how many dimensions of model parallelism are enabled**.

---

### Model Parallelism and GPU Count

Modern large models may use one or more of the following parallelism dimensions:

- **Tensor Model Parallelism (TP)**  
  Splits large weight matrices across GPUs.

- **Expert Parallelism (EP)** (for Mixture-of-Experts models)  
  Splits groups of experts across GPUs.

- **Pipeline Parallelism (PP)** (not used in this workshop)  
  Splits layers across GPUs.

If multiple parallel dimensions are used simultaneously, the **total number of GPUs required** is:

```text
total_gpus = TP × EP × PP
```

For this workshop, PP = 1, so:

```text
total_gpus = TP × EP
```

Examples:

- If **TP = 2** and **EP = 1**, you need **2 GPUs**
- If **TP = 1** and **EP = 4**, you need **4 GPUs**
- If **TP = 2** and **EP = 2**, you need **4 GPUs**
- If **TP = 4** and **EP = 2**, you need **8 GPUs**

This is why we cannot hard-code a “2-GPU version” or “4-GPU version” for all models.  
Different architectures require different combinations of TP and EP to fit into memory.

---

## How the SLURM Script Fits Into This

Regardless of the parallelism scheme, the SLURM script always does the same foundational steps:

1. **Requests the number of GPUs needed**  
   (matching `total_gpus = TP × EP`)

2. **Sets up cache directories**  
   so that everything stays inside the experiment folder.

3. **Loads CUDA and Singularity modules**

4. **Starts background GPU monitoring**

5. **Runs NeMo Factory inside a container**  
   using the GPUs allocated by SLURM

---

### 1. Resource Requests

The top of the script declares the number of GPUs you want SLURM to allocate:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=<total_gpus>
#SBATCH --gpus-per-node=<total_gpus>
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
```

> Whatever TP and EP you configure in NeMo, the SLURM GPU count must match
> total_gpus = TP × EP


### 2. Directory Resolution

The script detects where it is located and uses that as the base for:

- GPU monitoring logs
- Cache directories
- The base model directory (`llama31_8b`)
- The shared `.cache` folder used across runs

This makes the script **portable** and ensures all outputs stay inside the experiment folder.

---

### 3. Cache Setup

All frameworks (NeMo, Hugging Face, PyTorch, Triton) are pointed to a shared on-disk cache directory to avoid quota
problems and repeated downloads:

```bash
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HOME="$CACHE_DIR/hf_cache"
export NEMO_HOME="$CACHE_DIR/nemo_cache"
...
```

The script also creates these directories if they don't exist.

---

### 4. Loading Modules

Ibex requires loading the correct environment modules before container execution:

```bash
module load cuda/12.4.1
module load singularity
```

This ensures that GPU passthrough and CUDA bindings work correctly.

---

### 5. GPU Monitoring

To help participants analyze hardware usage, the script starts a background `nvidia-smi` monitor:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total
--format=csv -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
```

This collects usage data every 5 seconds for later inspection.

---
### 6. Running NeMo Factory

NeMo Factory provides a command-line interface for running standardized fine-tuning workflows using predefined “factory”
recipes. In this workshop, we use it to fine-tune the LLaMA 3.1 8B model with LoRA adapters.

NeMo Factory is always launched through the `nemo` CLI inside a Singularity container to ensure a consistent,
reproducible environment across all participants.

#### The NeMo Factory CLI

The basic structure of a fine-tuning command is:

```commandline
nemo llm finetune \
--factory <factory_name> \
<optional overrides>
```

The `--factory` option selects one of NeMo’s built-in recipe configurations. For example:

```commandline
--factory llama31_8b
```

This loads all the default settings for training LLaMA 3.1 8B with LoRA.

You can then override any configuration parameter directly on the command line, such as the number of training steps or
batch size.

#### Running NeMo inside a Singularity container

To ensure consistent versions of CUDA, PyTorch, and NeMo, we run the CLI inside a container:

```bash
singularity exec --nv <nemo_image.sif> nemo llm finetune ...
```

Using `--nv` exposes the host GPUs to the container.

In the SLURM script, this is wrapped with `srun` so it runs on the allocated GPU node:

```bash
srun singularity exec --nv /path/to/nemo_image.sif \
nemo llm finetune --factory llama31_8b ...
```

#### YAML overrides

NeMo Factory recipes are defined as YAML configurations. Any value in the recipe can be overridden from the command line
using dot notation:


```bash
trainer.max_steps=250 \
data.global_batch_size=8 \
optim.lr_scheduler.warmup_steps=50 \
```

These overrides allow you to quickly modify the training behavior without editing any config files.

**Parallelism Overrides**

The model-parallel configuration is passed via YAML overrides, typically:
```commandline
trainer.strategy.tensor_model_parallel_size=<TP>
```

#### Additional Parameter for Mixtral

MoE models require an expert-parallel dimension, which is specified through:

```bash
trainer.strategy.tensor_model_parallel_size=<TP>
trainer.strategy.expert_model_parallel_size=<EP>
trainer.strategy.sequence_parallel=True \
```
Whatever values you assign:

- SLURM must allocate `TP × EP` GPUs

- NeMo Lightning will launch that many distributed processes

- Each process corresponds to one GPU

> In this workshop, these hyperparameters **are intentionally kept small** so that each run finishes quickly and
> participants can focus on understanding the workflow.
> However, for models as large as **LLaMA 3.1 8B**, or **Mixtral 8x7B**, meaningful fine-tuning typically requires:
>
>- **much larger** batch sizes, and
>
>- **significantly longer** training schedules
>
>In real training scenarios, these values would be scaled up to achieve stable optimization and observable model
> improvements.

---

## Understand the Output

### “Resolved Arguments” Section

When you launch a NeMo Factory command, the first thing it prints is a **dry run** showing the *fully resolved
configuration* that NeMo will use for training. This is extremely useful because it reveals:

Here’s a simplified example of what it looks like and what each section means:

```commandline
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Argument Name    ┃ Resolved Value                                              ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
|data              │ SquadDataModule(seq_length=2048,                            |
|                  |         micro_batch_size=1, global_batch_size=8)            |
|model             │ MixtralModel(config=MixtralConfig8x7B())                    |
|peft              │ LoRA(target_modules=['linear_qkv', 'linear_proj'], dim=32)  |
|optim             │ Adam(lr=1e-4, weight_decay=0.1, bf16=True)                  |
|trainer           │ Trainer(devices=2, max_steps=250)                           |
|strateg           │ MegatronStrategy(tp=1, pp=1, ep=2, sequence_parallel=False) |
|resume            │ path=/ibex/project/.../mixtral/model                        |
```

#### What Each Section Means (quick hints)

- **data** → what dataset is used + batch sizes
- **model** → which LLM architecture is loaded (LLaMA / Mixtral)
- **peft** → whether LoRA is enabled and its rank/targets
- **optim** → optimizer type, learning rate, precision (bf16)
- **trainer** → number of GPUs, training length, logging frequency
- **strategy** → parallelism settings (TP/PP/EP)
- **resume** → path to the pretrained checkpoint
- **tokenizer** (if shown) → which tokenizer NeMo will use

This block is the fastest way to confirm:

- the right model loaded
- LoRA is active
- GPU count is correct
- expert/tensor/pipeline parallel settings
- the dataset and batch sizes

If anything is misconfigured, the issue almost always shows up here.

### Distributed Initialization Logs

You’ll also see a short block confirming that distributed training has started correctly.  
A typical snippet looks like:

```text
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
...
```

#### Quick Hints

- **distributed_backend=nccl**  
  NeMo/Lightning will use **NCCL** for GPU-to-GPU communication (the standard backend on NVIDIA GPUs).

- **Starting with 2 processes**  
  Confirms the **world size = 2** → two ranks, typically one per GPU.

- **[Gloo] Rank X is connected to Y peer ranks**  
  Gloo is used for **control/coordination** (rendezvous, barriers, initialization).  
  These messages show each rank verifying that it can see the expected peers.

- **Expected number of connected peer ranks is : 1**  
  With 2 processes, each rank should see **exactly 1 peer** (the other rank).  
  Matching values indicate that distributed initialization succeeded.

You only need to worry if:

- these messages hang and never progress
- “expected” vs “connected” counts don’t match
- logs stop here with a timeout or error

Otherwise, this simply confirms that multi-GPU communication is ready.

### LoRA in the Model Summary

When LoRA is applied, you’ll see **two** model summaries: one **before** and one **after** the LoRA transform.

#### 1. Before applying LoRA

```text
  | Name   | Type     | Params | Mode  | FLOPs
----------------------------------------------------
0 | module | GPTModel | 24.2 B | train | 0    
----------------------------------------------------
24.2 B    Trainable params
0         Non-trainable params
24.2 B    Total params
96,616.858Total estimated model params size (MB)
1065      Modules in train mode
0         Modules in eval mode
0         Total Flops
```

Quick hints:

- **Trainable params** = 24.2B
  All base model weights are still marked as trainable (LoRA not applied yet).

- **Total params** = 24.2B
  Parameter count of the full Mixtral model.

#### 2. LoRA being injected

You then see a long list like:

```text
[NeMo I ...] Adding lora to: module.decoder.layers.0.self_attention.linear_proj
[NeMo I ...] Adding lora to: module.decoder.layers.0.self_attention.linear_qkv
...
[NeMo I ...] Adding lora to: module.decoder.layers.31.self_attention.linear_proj
[NeMo I ...] Adding lora to: module.decoder.layers.31.self_attention.linear_qkv
```

This means:

- LoRA adapters are being attached to **QKV** and **projection** layers in every decoder block.

- NeMo is wrapping those layers with small trainable low-rank matrices

#### 3. After applying LoRA (After applying model_transform)

```text
  | Name   | Type     | Params | Mode  | FLOPs
----------------------------------------------------
0 | module | GPTModel | 24.2 B | train | 0    
----------------------------------------------------
18.9 M    Trainable params
24.2 B    Non-trainable params
24.2 B    Total params
96,692.355Total estimated model params size (MB)
1385      Modules in train mode
0         Modules in eval mode
0         Total Flops

```

Quick hints:

- **Trainable params = 18.9M**

  Only the LoRA adapters are trainable.
  The original 24.2B base parameters are now frozen.


- **Non-trainable params = 24.2B**

  The full base model is still there, but not updated during training.


- **Modules in train mode increased (1065 → 1385)**

  Extra modules come from the inserted LoRA layers.

This “before vs after” summary is the easiest way to verify that:

- LoRA was actually injected, and

- you are doing **parameter-efficient fine-tuning** instead of full 24B-parameter training.

### Rerun / SIGTERM Messages

You may see lines like:

```text
[rank: 0] Received SIGTERM: 15
[rank: 0] Received SIGTERM: 15
[NeMo W ... rerun_state_machine:1263] Implicit initialization of Rerun State Machine!
[NeMo W ... rerun_state_machine:239] RerunStateMachine initialized in mode RerunMode.DISABLED
```

Quick hints:

- `Received SIGTERM: 15`

  SLURM sends SIGTERM to let the process know about potential requeue/cleanup.
  NeMo catches it so it _can_ support reruns or graceful shutdown.


- `RerunStateMachine initialized in mode RerunMode.DISABLED`

  NeMo is setting up its internal “rerun” mechanism but it is disabled.
  This is just a warning-level log, not an error.

For this workshop, you can safely **ignore** these messages.
They do **not** indicate a problem with training, data, or parallelism—they’re just NeMo’s internal rerun logic being
initialized and left turned off.

### Training Step Logs

During training, NeMo prints a line for each optimization step, for example:

```text
Training epoch 0, iteration 0/249 | lr: 1.961e-06 | global_batch_size: 8 | global_step: 0 | reduced_train_loss: 1.315 | train_step_timing in s: 7.126
Training epoch 0, iteration 1/249 | lr: 3.922e-06 | global_batch_size: 8 | global_step: 1 | reduced_train_loss: 1.653 | train_step_timing in s: 2.505 | consumed_samples: 16
Training epoch 0, iteration 2/249 | lr: 5.882e-06 | global_batch_size: 8 | global_step: 2 | reduced_train_loss: 1.278 | train_step_timing in s: 1.505 | consumed_samples: 24
...
```

Quick hints for the fields:

- **epoch / iteration**  
  Progress within the current epoch (we only run epoch 0 in this workshop).

- **lr**  
  The current learning rate after scheduler warmup.

- **global_batch_size**  
  Total batch size across all GPUs.

- **global_step**  
  The training step counter (0 → 249 in this run).

- **reduced_train_loss**  
  Loss averaged across GPUs.

- **train_step_timing**  
  Time (in seconds) to complete the step.

- **consumed_samples**  
  How many samples have been processed so far.

### Why training may appear to “stall” before logs appear

NeMo prints these logs **in batches**, usually between validation intervals or after several training steps.  
This means:

- If your job seems to be “doing nothing” at first,
- it may simply be inside a training loop **before the first log flush**.

This is normal.

**No logs ≠ no training.**  
Especially on the first few steps (step 0 can take ~5–10 seconds), output may take a while to appear.

As long as GPU utilization is high, the job is running correctly.

---

## Try It Yourself: Run the Experiment and Collect Metrics

In this part of the workshop, you will:

1. Run **LLaMA 3.1 8B + LoRA** with different numbers of GPUs (1, 2, 4, 8).
2. Run **Mixtral 8×7B + LoRA** with different numbers of GPUs (2, 4, 8).
3. Extract a few key metrics from the logs.
4. Fill in scaling tables to see how performance and memory usage change for each model.

### 1. Submitting the Jobs

#### 1.1 LLaMA 3.1 8B 

From the [`model_parallel/llama31_8b/`](./llama31_8b) directory:

##### 2 GPUs

```commandline
cd 2_gpus/
sbatch 2_gpus.slurm   # job-name: l-mp-2g
```

##### 4 GPUs

```commandline
cd 4_gpus/
sbatch 4_gpus.slurm   # job-name: l-mp-4g
```

##### 8 GPUs

```commandline
cd 8_gpus/
sbatch 8_gpus.slurm   # job-name: l-mp-8g
```

---

#### 1.2 Mixtral 8×7B (LoRA + Expert Parallel, starts from 2 GPUs)

From the [`model_parallel/mixtral_8x7b/`](./mixtral_8x7b) directory:

##### 4 GPUs

```commandline
cd 4_gpus/
sbatch 4_gpus.slurm   # job-name: m-lo-4g
```

##### 8 GPUs

```commandline
cd 8_gpus/
sbatch 8_gpus.slurm   # job-name: m-lo-8g
```

Each script uses the same logic, only changing the SLURM GPU directives and the `job-name`.
All SLURM logs are written under the submission directory in:

```commandline
logs/<job-name>-<job-id>.out
```
---

### 2. What to Extract from the Logs (for both models)

#### 2.1 Training Metrics (Final Iteration)

Scroll to the last training line `(249/249)`, which looks like:

```commandline
Training epoch 0, iteration 249/249 | lr: 6.168e-09 | global_batch_size: 8 | global_step: 249 | reduced_train_loss: 0.1058 | train_step_timing in s: 1.608 | consumed_samples: 2000 | val_loss: 0.194
```

Extract:

- `reduced_train_loss`
- `train_step_timing` in s

Use the **last** such line in the file.

#### 2.2 Total Job Runtime

Find the end-of-job block:

```commandline
===============================
 Job finished
 End time   : 2025-12-5 10:34:24
 Total time : 00:09:55
===============================
```

Extract:

- `Total time`

Convert this to seconds for scaling calculations.

#### 2.3 GPU Memory (Peak, Avg, Mode)

At the end of the memory analysis, locate lines such as:

```commandline
[gpu_memory_log - GPU 0] Peak = 16273 MiB, Avg = 1924.11 MiB, Mode = 1 MiB
```

Extract:

- `Peak`
- `Avg`
- `Mode`

Use **GPU 0** consistently for all runs.

---

### 3. Scaling Table

Fill in the tables below using the extracted values.

## 3.1 LLaMA 3.1 8B Scaling Table 

Time scaling = `time_2gpu_seconds` / `time_Ngpu_seconds`  
Memory scaling = `peak_2gpu` / `peak_Ngpu`

| GPUs | Batch per GPU | Total job time (HH:MM:SS) | Train step time (s) | Last reduced_train_loss | GPU 0 Peak (MiB) | GPU 0 Avg (MiB) | GPU 0 Mode (MiB) | Time scaling vs 2 GPU | Peak memory scaling vs 2 GPU |
|------|---------------|---------------------------|---------------------|-------------------------|------------------|-----------------|------------------|-----------------------|------------------------------|
| 2    |               |                           |                     |                         |                  |                 |                  | 1.0                   | 1.0                          |
| 4    |               |                           |                     |                         |                  |                 |                  |                       |                              |
| 8    |               |                           |                     |                         |                  |                 |                  |                       |                              |_**

---

## 3.2 Mixtral 8×7B Scaling Table

Time scaling = `time_4gpu_seconds` / `time_Ngpu_seconds`  
Memory scaling = `peak_4gpu` / `peak_Ngpu`

| GPUs | Batch per GPU | Total job time (HH:MM:SS) | Train step time (s) | Last reduced_train_loss | GPU 0 Peak (MiB) | GPU 0 Avg (MiB) | GPU 0 Mode (MiB) | Time scaling vs 4 GPU | Peak memory scaling vs 4 GPU |
|------|---------------|---------------------------|---------------------|-------------------------|------------------|-----------------|------------------|-----------------------|------------------------------|
| 4    |               |                           |                     |                         |                  |                 |                  | 1.0                   | 1.0                          |
| 8    |               |                           |                     |                         |                  |                 |                  |                       |                              |_**
