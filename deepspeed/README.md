# What is this Exercise About?

This exercise explores large language model (LLM) fine-tuning using DeepSpeed, a deep learning optimization library for
large-scale training. The workshop walks through a practical progression:

1. Baseline fine-tuning using Hugging Face Trainer **without** DeepSpeed.

2. Fine-tuning on a single GPU using DeepSpeed with different ZeRO optimization stages.

3. Comparing ZeRO stages with and without CPU offloading.

4. Evaluating memory usage and performance trade-offs across configurations.

5. Scaling up to multi-GPU and multi-node training using DeepSpeed's distributed launcher.

## Why DeepSpeed?

Training such large models is computationally expensive and quickly runs into memory limits — especially on a single
GPU.

DeepSpeed is a deep learning optimization library from Microsoft designed to:

- **Reduce GPU memory usage** via ZeRO optimizations (stage 1–3).

- **Enable distributed training** across GPUs and even nodes.

- Support **model and tensor parallelism**.

- Work seamlessly with **Hugging Face Transformers**.

## What is Hugging Face 🤗?

**Hugging Face** is an open-source ecosystem built around natural language processing (NLP) and machine learning
models — especially transformer-based models like BERT, GPT, and BLOOM.

It provides easy-to-use tools to **download, train, fine-tune, and deploy** state-of-the-art models with just a few
lines of code.

### 🔧 Key Components You'll Use

| Component      | What It Does                                                                                        |
|----------------|-----------------------------------------------------------------------------------------------------|
| `transformers` | Python library for accessing thousands of pre-trained models across NLP, vision, and audio tasks.   |
| `datasets`     | Library for easy loading, sharing, and preprocessing of public datasets like SQuAD, IMDB, and more. |
| `Trainer` API  | High-level training interface to handle training, evaluation, and checkpointing with minimal code.  |
| Model Hub      | Online platform for hosting, sharing, and downloading models — all ready to use.                    |

---
In this workshop, you’ll:

- Use `transformers` to load a pre-trained **BLOOM** model.
- Use `datasets` to load and preprocess a **SQuAD** subset.
- Fine-tune the model on a **question-answering task** using the `Trainer` API.
- Later, enhance scalability using **DeepSpeed** for memory- and compute-efficient training.

## Learning Outcomes

### By the end of this exercise, participants will be able to:

1. Understand the **basics of fine-tuning** transformer models **using Hugging Face and DeepSpeed**.

2. Configure DeepSpeed for **ZeRO Stage 1, 2, and 3**, with and without offloading.

3. Measure and interpret training **performance, memory footprint, and scalability** across setups.

4. Launch **distributed** training jobs on **multiple GPUs and across multiple nodes**.

-----------------------------------

# Environment Setup

We'll use Conda to manage packages and dependencies

**Step 1: Create the Conda Environment:**

```
conda create -n deepspeed-finetune python=3.10 -y
conda activate deepspeed-finetune
```

**Step 2: Install Required Packages**

- Installation through the [requirements.txt](./requirements.txt):
    ```commandline
    pip install -r requirements.txt
    ```

----------------------------------

# Baseline: BLOOM Fine-tuning without DeepSpeed:

## Fine-Tuning Setup

Before exploring `DeepSpeed` optimizations, it’s useful to understand the vanilla `HuggingFace` fine-tuning process
using a smaller LLM like `bigscience/bloom-560m`, and 500 examples subset of `SQuAD` for question-answer format
training.

### Model Loader and Saver

[model.py](baseline/model.py) defines two key functions:

1. `load_model()`: Loads `bigscience/bloom-560m` model and tokenizer.

2. `save_model()`: Saves the trained model and tokenizer to disk.

### Dataset Preprocessing

[data_loader.py](baseline/data_loader.py) Handels:

1. Loading the SQuAD dataset using Hugging Face datasets.

2. Tokenizing each example as:
    ```
    "Question: ... Context: ... Answer: ..."
    ```
3. Padding/truncating to max length (512).

4. Setting labels = input_ids (for causal LM).

5. Optionally subsetting the dataset for faster experiment.

### Training Configuration

[config.py](baseline/config.py) centralizes hyperparameters makes tuning and experimenting easier — change config values
in one file without touching the training script.
This section defines the core training hyperparameters and behaviors using the Hugging Face

- `output_dir`:`./bloom-qa-finetuned`    Directory to store model checkpoints, logs, and evaluation results.
- `eval_strategy`:`epoch`    Evaluation is run at the end of each training epoch.
- `save_strategy`:`epoch`    A model checkpoint is saved at the end of each training epoch.
- `per_device_train_batch_size`:`4`    Batch size to use on each GPU or each CPU worker during training.
- `per_device_eval_batch_size`:`4`    Batch size to use on each device during evaluation.
- `gradient_accumulation_steps`:`4`    Number of steps to accumulate gradients before performing a backward/update pass.
  Simulates a larger effective batch size.
- `num_train_epochs`:`3`    Total number of training epochs (full passes through the dataset).
- `learning_rate`:`5e-5`    Initial learning rate for the optimizer.
- `weight_decay`:`0.01`    Weight decay (L2 penalty) to apply for regularization. Helps prevent overfitting.
- `fp16`:`True`    Enables automatic mixed precision (AMP) training for reduced memory usage and faster computation (
  requires compatible hardware).
- `gradient_checkpointing`:`False`    Disables gradient checkpointing. If set to True, it reduces memory usage by
  trading off compute time.
- `push_to_hub`:`False`    Disables automatic pushing of the model and logs to the Hugging Face Hub. Set to True if you
  want to share your model publicly or privately.

### Training

The [train.py]() script fine-tunes a causal language model (e.g., BLOOM) using Hugging Face's Trainer API.

**Hugging Face Trainer:**

The Trainer class from Hugging Face simplifies the process of training and evaluating models — especially useful when
working with transformers, tokenizers, and large datasets.
Instead of writing your own training loop with forward(), backward(), optimizer steps, and logging — Trainer handles it
for you.

**What Happens When You Call `trainer.train()`?**

- Batches the dataset using the data collator
- Moves data to GPU if available
- Runs model(input_ids) and computes the loss
- Applies backpropagation (loss.backward() + optimizer step)
- Saves checkpoints and logs progress
- Evaluates on the validation set (if provided)

### SLURM Job Submission Script

[baseline.slurm](baseline/baseline.slurm) is used to run the entire experiment on an HPC cluster.

**Key sections:**

- #SBATCH directives define resources: 1 GPU, 32GB RAM, 12 hours.

- Activates your Conda env and loads CUDA module.

- Starts nvidia-smi memory logging in the background.

- Executes python train.py.

- Kills memory logger afterward.

### Bringing It All Together: Running the Baseline Fine-Tuning Experiment

Once all components are in place — model loading, dataset preprocessing, training configuration, and training logic —
you can execute the full fine-tuning workflow with minimal manual steps.

Use the baseline.slurm script to submit the training job on a GPU node:

```commandline
sbatch baseline.slurm
```

### Output Artifacts

After the run finishes, you'll find:

- Fine-tuned model and tokenizer in `./bloom-qa-finetuned/`
- Training logs including evaluation metrics and loss curves
- SLURM log files in the `log` directory or as specified by --output
- GPU memory tracking output with `nvidia-smi` as a `.csv` file

## Exercise: Recreate the Baseline Training & Evaluation Summary Table

As part of this workshop, your task is to **run the baseline fine-tuning experiment** and **recreate the performance
summary table** using your own training logs.

This exercise helps you develop a habit of tracking key metrics like training loss, evaluation loss, and throughput —
which are essential for understanding and debugging model training.

### Objective 1

After running `train.py`, fill in the following table with metrics from your output:

| **Metric**                     | **Your Value**           |
|--------------------------------|--------------------------|
| Train Loss (Final)             | _Fill from final output_ |
| Eval Loss (Epoch 1)            | _From evaluation logs_   |
| Eval Loss (Epoch 2)            | _From evaluation logs_   |
| Eval Loss (Epoch 3)            | _From evaluation logs_   |
| Training Speed (samples/sec)   | _Reported by Trainer_    |
| Evaluation Speed (samples/sec) | _Reported by Trainer_    |
| Steps per Epoch                | _From logs or config_    |

### 🔍 Where to Find These Values

- **Loss values** appear in the `.out` files located in `log` directory , in lines containing `loss=` or `eval_loss=`.
- **Training and evaluation speed** are typically printed after evaluation steps or at the end of training.

### Objective 2

Use your GPU memory log (e.g., `baseline-single-gpu_memory_log.csv`) to calculate and fill in the table below.

| **Metric**               | **Your Value (MiB)**     |
|--------------------------|--------------------------|
| Peak GPU Memory Usage    | _Use max value_          |
| Mean GPU Memory Usage    | _Average across samples_ |
| Minimum GPU Memory Usage | _Lowest recorded_        |

#### Understanding GPU Memory Tracking Output with `nvidia-smi`

A typical GPU memory log file might look like this:

```commandline
  timestamp, index, name, memory.used [MiB], memory.total [MiB]
  2025/05/22 09:42:33.587, 0, Tesla V100-SXM2-32GB, 1, 32768
  2025/05/22 09:42:38.588, 0, Tesla V100-SXM2-32GB, 4, 32768
  2025/05/22 09:42:43.588, 0, Tesla V100-SXM2-32GB, 4, 32768
...
```

#### Column Descriptions

| **Column**           | **Description**                                                     |
|----------------------|---------------------------------------------------------------------|
| `timestamp`          | Time of the memory snapshot.                                        |
| `index`              | GPU index on the node (e.g. `0` for the first GPU).                 |
| `name`               | Full name of the GPU device used.                                   |
| `memory.used [MiB]`  | Amount of GPU memory actively in use at the time of logging.        |
| `memory.total [MiB]` | Total available memory on the GPU. Helpful for calculating % usage. |

### Instructions to FIll The Table

- Extract Peak Memory (MiB)
   ```commandline 
     tail -n +2 baseline-single-gpu_memory_log.csv | cut -d',' -f4 | sort -n | tail -1

   ```
- Extract Minimum Memory (MiB)
  ```commandline
  tail -n +2 baseline-500-1gpu_memory_log.csv | cut -d',' -f4 | sort -n | head -1
  
  ```
- Extract Mean Memory (MiB)
   ```commandline
   tail -n +2 baseline-500-1gpu_memory_log.csv | cut -d',' -f4 | awk '{sum+=$1} END {print sum/NR}'
   
   ```
  ### Explanation

- `tail -n +2` skips the header line.

- `cut -d',' -f4` selects the memory.used [MiB] column.

- `sort -n` sorts numerically.

- `awk` sums and averages the memory values.

---

# DeepSpeed-Zero on Single Node Single GPU:

## DeepSpeed Configuration File:

The DeepSpeed config file [ds_config.json](deepspeed-single-gpu/ds_config.json) is the central interface through which
you control how DeepSpeed integrates with your training pipeline. It acts like a blueprint that tells DeepSpeed how to
optimize and manage.

### Parameter Descriptions

| **Key**                         | **Value**      | **Description**                                                                                 |
|---------------------------------|----------------|-------------------------------------------------------------------------------------------------|
| `train_batch_size`              | `"auto"`       | Automatically sets the largest batch size that fits into GPU memory.                            |
| `gradient_accumulation_steps`   | `"auto"`       | Automatically determines the number of accumulation steps to simulate larger batch sizes.       |
| `gradient_clipping`             | `1.0`          | Caps gradient norms to prevent exploding gradients and ensure training stability.               |
| `optimizer.type`                | `"AdamW"`      | Optimizer used for training; AdamW is standard for transformer models.                          |
| `optimizer.params.lr`           | `5e-5`         | Learning rate used for fine-tuning.                                                             |
| `optimizer.params.betas`        | `[0.9, 0.999]` | Beta values used by Adam optimizer for momentum calculations.                                   |
| `optimizer.params.eps`          | `1e-8`         | Small constant added to prevent division by zero during optimization.                           |
| `optimizer.params.weight_decay` | `0.01`         | Regularization parameter to prevent overfitting.                                                |
| `fp16.enabled`                  | `true`         | Enables Automatic Mixed Precision (AMP) for faster and more memory-efficient training.          |
| `zero_optimization.stage	`      | `0`            | No ZeRO optimization yet — used for compatibility with DeepSpeed's integration in Hugging Face. |

[Official documentation for DeepSpeed Configuration JSON]( https://www.deepspeed.ai/docs/config-json/)

**Note:** `zero_optimization.stage = 0` does not activate ZeRO but satisfies DeepSpeed’s required structure.
More on ZeRO optimization (Stage 1, 2, and 3) will be covered in a later section.

## Turning the Baseline into a DeepSpeed-Enabled Trainer

In the baseline setup, the `Trainer` uses Hugging Face’s standard training loop without any DeepSpeed optimizations.

To integrate DeepSpeed into the training pipeline, the `TrainingArguments` class must reference a DeepSpeed
configuration file `ds_config.json`. This allows Hugging Face's Trainer to apply DeepSpeed's optimization features
during training.

### Modification to `TrainingArguments`

To enable DeepSpeed, a single line is added:

```python
    deepspeed = "./ds_config.json",  # Links the DeepSpeed configuration file
```

**This integration allows DeepSpeed to handle aspects such as:**

- Mixed-precision training (FP16)

- Gradient accumulation and clipping

- Optimizer configuration

## Running the Script with DeepSpeed

Once the `deepspeed` field is added to the `TrainingArguments` configuration, the training process must be launched
using the **DeepSpeed CLI launcher** instead of the standard **Python** command.
This ensures that DeepSpeed initializes properly and applies all runtime optimizations defined in `ds_config.json`.

In the slurm script, replace

```commandline
python train.py
```

with

```commandline
deepspeed train.py
```

## ZeRO in DeepSpeed

**ZeRO** (Zero Redundancy Optimizer) reduces memory usage by partitioning model states across devices.

### Stage 1: Optimizer State Partitioning

**What it does:**

- Reduces memory footprint by not replicating optimizer states on every GPU.
- Each GPU only keeps its own partition of the optimizer state tensors.
- Model parameters and gradients are still fully replicated across devices.

On **single GPU**, there's no sharding — so optimizer state partitioning won't reduce memory use.

#### Integrating ZeRO Stage 1 into Our Implementation:

To enable ZeRO Stage 1, add the following lines to the [ds_config.json](deepspeed-single-gpu/ds_config.json):

```json
 "zero_optimization": {
"stage": 1
}
```

No Python code changes are needed

#### Optimizer State Offloading (Stage 1 with Offload)

In case your model is still hitting memory limits or you'd like to further reduce GPU memory pressure, you can offload
the optimizer state to CPU.

This is particularly useful for:

- Single GPU setups with tight memory
- Larger batch sizes or long sequence lengths

To enable Optimizer State Offloading, update the [ds_config.json](deepspeed-single-gpu/ds_config.json) as follows:

```json
 "zero_optimization": {
"stage": 1,
"offload_optimizer": {
"device": "cpu"
}
}
```

No Python code changes are needed

### Stage 2: + Gradient Partitioning

What it adds:

- Reduces activation memory and gradient memory on each GPU.
- Makes training larger models possible on limited memory.

To enable ZeRO Stage 2, update [ds_config.json](deepspeed-single-gpu/ds_config.json) with:

```python
"zero_optimization": {
    "stage": 2
}
```

No Python code changes are needed

#### Stage 2 + Offloading provides best memory savings by:

- Offloading optimizer states,

- Offloading gradient states,

```json
  "zero_optimization": {
"stage": 2,
"offload_optimizer": {
"device": "cpu"
}
}
```

### Zero 3 Full Parameter Sharding (No Offloading)

**ZeRO Stage 3** is the most memory-efficient stage of DeepSpeed’s ZeRO (Zero Redundancy Optimizer) family. It builds on
Stages 1 and 2 by additionally sharding the **model parameters themselves**, not just optimizer states or gradients.

This stage allows training very large models by distributing:

- **Optimizer states** (as in Stage 1)
- **Gradients** (as in Stage 2)
- **Model parameters** (new in Stage 3)

```json
  "zero_optimization": {
"stage": 3
}
```

#### ZeRO Stage 3: Full Parameter Offloading

ZeRO Stage 3 (Parameter + Gradient + Optimizer Sharding) is the most memory-efficient optimization level in DeepSpeed’s
ZeRO suite. It goes beyond Stage 2 by partitioning model parameters themselves across GPUs, not just the optimizer and
gradient states.

**What It Does**

- Partitions:
    - Optimizer states (like Stage 1)
    - Gradients (like Stage 2)
    - Model parameters (new in Stage 3)
- Reduces peak memory usage drastically during training
- Enables very large models to be trained on fewer GPUs.

```json
  "zero_optimization": {
"stage": 3,
"offload_optimizer": {
"device": "cpu"
},
"offload_param": {
"device": "cpu"
}
}
```

### Offloading: Parameters:`pin_memory: true`

#### In the context of DeepSpeed offloading, setting pin_memory: true enables:

- **Page-locked** (pinned) memory allocation on the host (CPU).
- **Faster transfers** between CPU and GPU because pinned memory allows asynchronous **DMA** (direct memory access),
  which allows devices like GPUs to **read/write memory directly, without CPU intervention.** This means:
    - The **GPU** can **pull/push** offloaded tensors from/to CPU faster.
    - The **CPU** is **not stalled** waiting for transfers to complete.
    - Data can be transferred **while computation** is still happening — overlapping compute and memory I/O.
- Improved data pipeline throughput, especially noticeable in single-GPU training where the offloaded optimizer is
  frequently accessed.

For Both Optimizer and Parameter Offloading (Full CPU Offloading in Stage 3):

```json
  "offload_optimizer": {
"device": "cpu",
"pin_memory": true
}
```

```json
  "offload_param": {
"device": "cpu",
"pin_memory": true
}
```

### Tracking CPU Memory Usage

When using DeepSpeed with ZeRO stages — especially with CPU offloading — it's important to monitor not only GPU memory,
but also CPU memory, since model states or optimizer states can be moved to CPU RAM.

This section demonstrates how to extend a SLURM job script to track both GPU and CPU memory usage using nvidia-smi and
psrecord:

1. **Background the Training job in [slurm script](deepspeed-single-gpu/deepspeed.slurm):**
    ```
   deepspeed train_stage3_offload.py &
   TRAIN_PID=$!
   ```
    - The training command is launched **in the background** (`&`) so the script can capture its PID (`$!`).
    - This is necessary for psrecord to **attach** to and **monitor** the correct process.


2. **Add CPU Memory Monitoring with `psrecord`**:
    ```commandline
    psrecord $TRAIN_PID --include-children --interval 5 --log "deepspeed-cpu-offloading-data-finetune${SLURM_JOB_ID}.txt" &
    MONITOR_CPU_PID=$!
    ```
    - `psrecord` tracks memory usage of the training process.
    - `--include-children` ensures subprocesses like optimizer workers or data loaders are included.
    - `--interval` 5 matches the GPU log sampling rate.
    - `--log writes` memory usage to a timestamped text file.


3. **Wait for the Training to Finish**
    ```
    wait $TRAIN_PID
    ```
    - Ensures the script pauses until training completes.
    - Prevents premature termination of memory logging processes.


4. **Stop All Logging Processes**
    ```
    kill $MONITOR_CPU_PID
    ```
    - Cleans up both GPU and CPU memory logging processes after training ends.

#### Understanding CPU Memory Tracking Output

When psrecord is used to monitor CPU memory usage during training, it generates a log file (e.g.,
`deepspeed-cpu-offloading-data-finetune<job_id>.txt`) that looks like this:

```commandline
# Elapsed time   CPU (%)     Real (MB)   Virtual (MB)
       0.000        0.000       11.250       14.680
       5.018       58.400      500.180     7570.797
       10.036       22.500    1025.770    15192.977
       ...
     321.232       79.100     1091.695    15290.660
```

Each row represents a snapshot taken at a fixed interval (e.g., every 5 seconds).

#### Column Descriptions

| **Column**     | **Description**                                                                 |
|----------------|---------------------------------------------------------------------------------|
| `Elapsed time` | Time in seconds since memory logging began.                                     |
| `CPU (%)`      | Percentage of CPU usage across all threads and child processes.                 |
| `Real (MB)`    | Actual resident memory usage (RAM) of the process in megabytes.                 |
| `Virtual (MB)` | Total virtual memory allocated, including memory that may not be actively used. |

#### Why This Output Matters

- **`Real (MB)`** shows how much CPU RAM is being actively used — crucial when using ZeRO stages with CPU offloading.
- **`Virtual (MB)`** indicates total memory requested by the process, including areas mapped but not loaded into RAM.
- **`CPU (%)`** reveals how much CPU processing is being used, which helps detect bottlenecks or offload activity.

---

## Exercise: Benchmarking ZeRO Stages with and without Offloading

This section walks through how to:

1. Add a second table to compare `pin_memory: true` vs `false`.
2. Recreate the full benchmarking table comparing ZeRO stages.
3. Modify model size and dataset subset for deeper experimentation.

### Part 1: Compare Stage 2 With and Without `pin_memory: true`

This exercise focuses on evaluating the impact of enabling pinned memory in offloading.

#### Setup

- Use `bloom-560m` with `stage 2 + offloading`.
- Prepare **two configs**:
    1. With `pin_memory: true`
    2. With `pin_memory: false` (or omit the key)

Use this table to record memory usage and runtime for ZeRO Stage 2 offloading, with and without pinned memory.

| **Model and Data**     | **Pinned Memory**   | **Peak GPU Memory (MiB)** | **Peak CPU Memory (MiB)** | **Total Runtime** |
|------------------------|---------------------|---------------------------|---------------------------|-------------------|
| Bloom 560M, 500 subset | `pin_memory: true`  |                           |                           |                   |
| Bloom 560M, 500 subset | `pin_memory: false` |                           |                           |                   |

#### Quiz Questions:

**Did enabling `pin_memory: true` reduce runtime in Stage 2 + offloading?**  
→ If so, by how much?
---

### Part 2: Recreate the ZeRO Stage Comparison Table

#### How to Change Model and Dataset Subset Size

This section explains how to modify the model and dataset subset to perform benchmarking on larger-scale scenarios.

#### Changing the Model

To change the model being fine-tuned (e.g., from `bloom-560m` to `bloom-3b` or `bloom-7b`), update the model name in the
script where `load_model()` is defined.

Open [model.py](deepspeed-single-gpu/model.py) or wherever the model is loaded and find this line:

```python
MODEL_NAME = "bigscience/bloom-560m"
```

Replace it with one of the following:

```python
MODEL_NAME = "bigscience/bloom-3b"  # 3 billion parameter version
MODEL_NAME = "bigscience/bloom-7b"  # 7 billion parameter version

```

#### Changing the Dataset Subset Size

To avoid using the full SQuAD dataset (which can be large), the dataset loader supports subsetting.

In [train.py](deepspeed-single-gpu/train.py), locate the `load_squad()` function:

```python
tokenized_datasets = load_squad(subset_size=500)
```

To use a larger dataset, change the value to 10000, for example:

```python
tokenized_datasets = load_squad(subset_size=1000)
```

#### Configurations to Try

| Model        | Subset         | DeepSpeed Configs to Test                                   |
|--------------|----------------|-------------------------------------------------------------|
| `bloom-560m` | 500 samples    | Stage 1, Stage 2, Stage 3, each with and without offloading |
| `bloom-3b`   | 10,000 samples | Stage 2, Stage 3, each with offloading                      |
| `bloom-7b`   | 10,000 samples | Stage 3 with offloading only (Stage 2 likely OOM)           |

#### Fill in the table below using the memory logs (`nvidia-smi`,

`psrecord`) and timing results for each ZeRO stage configuration.

| **Model and Data** | **DeepSpeed Config** | **Peak GPU Memory Used (MiB)** | **Peak CPU Memory Used (MiB)** | **Total RunTime** |
|--------------------|----------------------|-------------------------------:|-------------------------------:|-------------------|
|                    |                      |                                |                                |                   |

#### Quiz Questions:

Use your completed memory and runtime benchmark tables to answer the following questions for each part.

1. Which ZeRO stage gave the lowest peak GPU memory usage on each bloom variation?

2. How does runtime change when offloading is enabled?

---

## DeepSpeed-Zero on Single Node Multi GPUs:

This section describes how to adapt a single-GPU DeepSpeed setup to fine-tune a large model (e.g., BLOOM-1.7B) across
multiple GPUs on one node.

### Setting-up Multi GPUs Training:

To recreate this multi-GPU fine-tuning setup for BLOOM-1.7B using DeepSpeed, you'll begin with your working single-GPU
Hugging Face + DeepSpeed setup and make a few targeted changes. Below is a full explanation of the required
modifications and an overview of how DeepSpeed handles GPU sharding.

#### Step 1: Adjust [SLURM](deepspeed-single-gpu/deepspeed.slurm) Configuration:

```commandline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
```

- `--nodes=1`: Run on one compute node.

- `--ntasks-per-node=1`: Only one task (process) launched per node; DeepSpeed handles spawning processes per GPU
  internally.

- `--gpus-per-node=16`: Allocate 2 GPUs on the node.

#### Step 2: Set up DeepSpeed master address and port:

Place these lines before the training command in the [SLURM](deepspeed-single-gpu/deepspeed.slurm) script:

```commandline
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9999 # Any free port; if 9999 is busy, try 6000 or 2500
export WORLD_SIZE=$SLURM_GPUS_ON_NODE
export RANK=0
export LOCAL_RANK=0
```

These environment variables configure distributed training manually.

- `MASTER_ADDR`: IP or hostname of main node (for NCCL initialization).

- `MASTER_PORT`: Any free port for rendezvous (e.g., 9999; if that is in use, ports like 6000 or 2500 often work).

- `WORLD_SIZE`: Total number of GPUs (set automatically by SLURM).

- `RANK` and LOCAL_RANK: Needed for some custom multi-node setups.

#### Step 3: Use the deepspeed Launcher with `--num_gpus` in the [SLURM](deepspeed-single-gpu/deepspeed.slurm) script:

```commandline
deepspeed --num_gpus=$SLURM_GPUS_ON_NODE train.py &
```

- `--num_gpus=$SLURM_GPUS_ON_NODE` Instructs DeepSpeed to spawn one worker per GPU.

- Make sure [ds_config.json](deepspeed-single-gpu/ds_config.json) has `"train_batch_size": "auto"`  — DeepSpeed will
  scale automatically across GPUs.

---

## Exercise: Benchmarking Multi-GPU DataParallel Training

### Part 1: Benchmarking Weak Scaling on Multi-GPU Setup

This exercise walks through how to measure and tabulate key training metrics for `BLOOM-1.7B` on a 10 000-sample subset,
using DeepSpeed’s DataParallel (no ZeRO) across 2, 4, 6, and 8 GPUs.

Run the same training script with `#SBATCH --gpus-per-node=N` for **N = 2, 4, 6, 8**, then record:

- **Train Runtime** (total seconds)
- **Steps/sec** (extracted from logs or computed)
- **Samples/sec** (`Steps/sec × per_device_train_batch_size × N`)
- **Train Loss** (final)
- **Eval Loss** (final)
- **Eval Speed** (samples/sec during evaluation)
- **Peak GPU Memory** (per GPU)

In a **weak scaling** exercise, the dataset size grows proportionally with the number of GPUs, so that each GPU
processes the same amount of data. For example, if the base is 10 000 samples per GPU:

- **2 GPUs** → 20 000 samples
- **4 GPUs** → 40 000 samples
- **6 GPUs** → 60 000 samples
- **8 GPUs** → 80 000 samples

### Fill in the results for each GPU count below.

| **Metric**               | **2 GPUs** | **4 GPUs** | **6 GPUs** | **8 GPUs** |
|--------------------------|-----------:|-----------:|-----------:|-----------:|
| Train Runtime (sec)      |            |            |            |            |
| Steps/sec                |            |            |            |            |
| Samples/sec              |            |            |            |            |
| Train Loss               |            |            |            |            |
| Eval Loss (final)        |            |            |            |            |
| Eval Speed (samples/sec) |            |            |            |            |
| Peak GPU Memory (MiB)    |            |            |            |            |
| Average GPU Memory (MiB) |            |            |            |            |

#### Quiz Questions

1. Given weak scaling, should **Peak GPU Memory** per GPU remain constant? Explain any deviations you observe.

> **Note on Code Versioning and SLURM Queues**  
> SLURM does **not** snapshot your Python scripts when you call `sbatch`.
> - The job will execute whatever version of `train.py` (or any other `.py` files) is on disk **at the moment the job
    actually starts** running, not when it was submitted.
> - Any edits made to your code while the job is still in the queue will be picked up automatically.
> 
> **Best Practices:** 
> Submit from a dedicated directory that won’t be modified.  
> This ensures reproducibility and avoids unintended changes in long-running or queued jobs.  
---

### part 2: 2-GPU ZeRO Stage Comparison

This exercise guides the measurement and comparison of training metrics for ZeRO Stages 1, 2, 3 on **2 GPUs**, each 
**with** and **without** CPU offloading.
Fill in the results for ZeRO Stages 1, 2, and 3 on **2 GPUs**, both **with** and **without** CPU offloading.

| **Metric**               | **Stage 1** | **Stage 1 + offload** | **Stage 2** | **Stage 2 + offload** | **Stage 3** | **Stage 3 + offload** |
|--------------------------|------------:|----------------------:|------------:|----------------------:|------------:|----------------------:|
| Train Runtime (sec)      |             |                       |             |                       |             |                       |
| Steps/sec                |             |                       |             |                       |             |                       |
| Samples/sec              |             |                       |             |                       |             |                       |
| Train Loss               |             |                       |             |                       |             |                       |
| Eval Loss (final)        |             |                       |             |                       |             |                       |
| Eval Speed (samples/sec) |             |                       |             |                       |             |                       |
| Peak GPU Memory (MiB)    |             |                       |             |                       |             |                       |
| Average GPU Memory (MiB) |             |                       |             |                       |             |                       |
| Peak CPU Memory (MiB)    |             |                       |             |                       |             |                       |
| Average CPU Memory (MiB) |             |                       |             |                       |             |                       |

#### Quiz Questions

1. How does enabling offloading affect **Train Runtime** and **Samples/sec**? Quantify the trade-off between memory savings and speed.  

Fill in the table below to compare each ZeRO stage **with** and **without** offloading. Calculate both absolute and percentage changes.

| **Stage** | **GPU Mem (No Offload)** | **GPU Mem (Offload)** | **Mem Savings (%)** | **Train Runtime (No Offload)** | **Train Runtime (Offload)** | **Runtime Δ (%)** |
|-----------|--------------------------|-----------------------|---------------------|--------------------------------|-----------------------------|-------------------|
| Stage 1   |                          |                       |                     |                                |                             |                   |
| Stage 2   |                          |                       |                     |                                |                             |                   |
| Stage 3   |                          |                       |                     |                                |                             |                   |

- **Mem Savings (%)** = `(GPU Mem No Offload – GPU Mem Offload) / GPU Mem No Offload × 100`  
- **Runtime Δ (%)** = `(Runtime Offload – Runtime No Offload) / Runtime No Offload × 100`
