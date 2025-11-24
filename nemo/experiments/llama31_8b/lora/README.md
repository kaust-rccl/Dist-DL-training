# Parameter-Efficient Fine-Tuning with LoRA (Default PEFT Method in NeMo Factory)

LoRA (Low-Rank Adaptation) is the **default parameter-efficient fine-tuning (PEFT)** technique used by the NVIDIA NeMo
Factory recipes. Instead of updating all model weights—which is slow, memory-heavy, and expensive—LoRA injects a pair of
small trainable matrices into targeted layers (usually attention and MLP projections).

During training:

- The original model weights are **kept frozen**.

- Only the lightweight low-rank adaptation matrices (**A** and **B**) are trained.

- The final effective weight is `W + ΔW`, where `ΔW = B·A` is the low-rank update.

This approach drastically reduces memory usage and compute requirements, enabling fast and stable fine-tuning even on
limited GPU resources.

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

With the environment prepared and the GPU requirements clarified, we can now look at the SLURM script used in this
workshop to run NeMo Factory LoRA fine-tuning. This script is designed to be simple, reproducible, and optimized for
A100 nodes on Ibex.

The script handles:

1. Requesting the correct resources from SLURM
2. Preparing cache and model directories
3. Loading CUDA and Singularity
4. Monitoring GPU usage
5. Running NeMo Factory inside the container
6. Logging timestamps and runtime

Below is a high-level explanation of each part of the script, taking the [single_gpu.slurm](1_gpu/single_gpu.slurm) as a
reference.

---

### 1. Resource Requests

At the top of the script, the `#SBATCH` directives request the hardware and runtime needed for the experiment:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/%x-%j.out
```

This ensures the job runs on a single A100 GPU with enough memory and CPU resources for NeMo and the container.

---

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

### The NeMo Factory CLI

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

### Running NeMo inside a Singularity container

To ensure consistent versions of CUDA, PyTorch, and NeMo, we run the CLI inside a container:

```commandline
singularity exec --nv <nemo_image.sif> nemo llm finetune ...
```

Using `--nv` exposes the host GPUs to the container.

In the SLURM script, this is wrapped with `srun` so it runs on the allocated GPU node:

```commandline
srun singularity exec --nv /path/to/nemo_image.sif \
nemo llm finetune --factory llama31_8b ...
```

### YAML overrides

NeMo Factory recipes are defined as YAML configurations. Any value in the recipe can be overridden from the command line
using dot notation:

```commandline
trainer.max_steps=250 \
data.global_batch_size=8 \
optim.lr_scheduler.warmup_steps=50 \
```

These overrides allow you to quickly modify the training behavior without editing any config files.

---

## Scaling Up: Data Parallel Training with Multiple GPUs

So far, the script runs LoRA fine-tuning on **a single A100 GPU**.  
To speed things up, we can use **data parallelism** by adding more devices.

In data parallel training:

- Each GPU gets a **copy of the model**.
- The **global batch** is split across GPUs.
- Gradients are synchronized between GPUs at each step.

NeMo (via Lightning) handles this automatically once we increase the number of devices.

---

### 1. Adjust SLURM Resources

For example, to use **2 GPUs on a single node**:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
```

Key points:

- `--gpus-per-node=2` → request 2 GPUs on the node.
- `--ntasks-per-node=2` → Lightning expects one task per GPU.
- Still constrained to A100 nodes.

This is the same pattern you would follow for N GPUs:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=N
#SBATCH --gpus-per-node=N
```

### 2. The Script Automatically Adapts to GPU Count

Our workshop script does **not** hardcode the number of devices.
Instead, it passes:
```bash
trainer.devices="${SLURM_GPUS_PER_NODE}"
```
This value is automatically populated by SLURM based on:
```bash
#SBATCH --gpus-per-node=N
```

### 3.Global Batch Size Behavior

Because we use data parallelism, global batch size works like this:
```commandline
per-GPU batch = global_batch_size / number_of_gpus
```
So for:
```bash
data.global_batch_size=8
```
- 1 GPU → 8 per GPU
- 2 GPUs → 4 per GPU
- 4 GPUs → 2 per GPU

