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

---

# Environment Setup

We'll use Conda to manage packages and dependencies

run these lines:

```bash
conda env create -f environment.yml
conda activate deepspeed-finetune
```

---

#  Repository Structure
```
.
├── scripts/                      # Core Python modules used across all training jobs
│   ├── train.py                  # Launches training with HF Trainer and DeepSpeed
│   ├── data_loader.py            # Loads and tokenizes dataset (e.g., SQuAD)
│   ├── model.py                  # Loads model and tokenizer (e.g., BLOOM)
│   └── config.py                 # Central config for training args and CLI parsing
│
├── ds_configs/                   # DeepSpeed configuration JSON files
│   ├── zero0.json
│   ├── zero1.json
│   ├── zero2_cpu_offload.json
│   ├── zero2_cpu_offload_pinned_memory.json
│   └── ...
│
├── experiments/                  # SLURM job scripts organized by context
│   ├── deepspeed-single-gpu/
│   │   ├── zero_0/
│   │   ├── zero_1/
│   │   ├── zero_2/
│   │   └── ...
│   └── deepspeed-multi-node/
│       ├── 2_nodes/
│       ├── 3_nodes/
│       ├── 4_nodes/
│       └── ...
│
├── log/                          # SLURM job logs (output generated at runtime)
├── gpu_memory/                   # GPU memory logs (generated at runtime)
├── cpu_memory/                   # CPU memory logs (generated at runtime)
│
└── README.md                     # Workshop guide and documentation

```
## What Each Folder Contains

### `scripts/`
Contains the core Python logic:

- train.py: Orchestrates training with Hugging Face Trainer and DeepSpeed.

- data_loader.py: Prepares and tokenizes datasets like SQuAD.

- model.py: Loads the pretrained model and tokenizer (e.g., BLOOM).

- config.py: Parses CLI arguments and sets up TrainingArguments.

### `ds_configs/`

- Holds all DeepSpeed `JSON` configuration files. Each file defines a different ZeRO stage or offloading strategy used in the exercises.

experiments/
Organized by training scenario:

- baseline/: Runs without DeepSpeed — establishes performance and memory usage reference points.

- deepspeed-single-gpu/: Individual subfolders for each ZeRO stage and offloading variant on a single GPU.

- deepspeed-multi-gpu/: Training across multiple GPUs on a single node.

- deepspeed-multi-node/: Weak scaling experiments over 2–6 nodes, each in its own folder.

- Each subfolder contains its SLURM script and output directories (log/, gpu_memory/, cpu_memory/).

### `log/`, `gpu_memory/`, `cpu_memory/`
Automatically created during training runs:

- `log/`: SLURM job output logs.

- `gpu_memory/`: Logs GPU memory usage over time (via nvidia-smi).

- `cpu_memory/`: Logs CPU memory usage over time (via psrecord).

### `README.md`
The central guide for running the workshop, explaining DeepSpeed, ZeRO stages, config tuning, how to extract metrics, and more.

---

# Baseline: BLOOM Fine-tuning without DeepSpeed:

## Fine-Tuning Setup

Before exploring `DeepSpeed` optimizations, it’s useful to understand the vanilla `HuggingFace` fine-tuning process
using a smaller LLM like `bigscience/bloom-560m`, and 500 examples subset of `SQuAD` for question-answer format
training.

### Model Loader and Saver

[model.py](scripts/model.py) defines two key functions:

1. `load_model()`: Loads `bigscience/bloom-560m` model and tokenizer.

2. `save_model()`: Saves the trained model and tokenizer to disk.

### Dataset Preprocessing

[data_loader.py](scripts/data_loader.py) Handels:

1. Loading the SQuAD dataset using Hugging Face datasets.

2. Tokenizing each example as:
    ```
    "Question: ... Context: ... Answer: ..."
    ```
3. Padding/truncating to max length (512).

4. Setting labels = input_ids (for causal LM).

5. Optionally subsetting the dataset for faster experiment.

### Training Configuration

[config.py](scripts/config.py) centralizes hyperparameters makes tuning and experimenting easier — change config values
in one file without touching the training script, it handles different DeepSpeed configurations using the CLI without
touching code.
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
- `deepspeed`:Provided via `--deepspeed` CLI flag; enables DeepSpeed config loading at runtime.

### Training

The [train.py](scripts/train.py) script fine-tunes a causal language model (e.g., BLOOM) using Hugging Face's Trainer
API.

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

[baseline.slurm](experiments/baseline/baseline.slurm) is used to run the entire experiment on an HPC cluster.

**Key sections:**

- `#SBATCH` directives define resources: 1 GPU, 32GB RAM, 12 hours.

- Activates your Conda env and loads CUDA module.

- Starts nvidia-smi memory logging in the background.

- Executes python [train.py](scripts/train.py).

- Kills memory logger afterward.

### Bringing It All Together: Running the Baseline Fine-Tuning Experiment

Once all components are in place — model loading, dataset preprocessing, training configuration, and training logic —
you can execute the full fine-tuning workflow with minimal manual steps.

Use the baseline.slurm script to submit the training job on a GPU node:

```commandline
sbatch baseline.slurm
```

### Output Artifacts

After your SLURM job completes, the following artifacts will be generated:

| **Artifact Type**    | **Location**                             | **Description**                                                |
|----------------------|------------------------------------------|----------------------------------------------------------------|
| **Fine-Tuned Model** | `./bloom-qa-finetuned/`                  | Contains model weights, config, and tokenizer files            |
| **SLURM Log**        | `log/<job-name>-<job_id>.out`            | Console output including training progress and evaluation logs |
| **GPU Memory Log**   | `gpu_memory/gpu_memory_log_<job_id>.csv` | Periodic memory usage from `nvidia-smi`                        |
| **CPU Memory Log**   | `cpu_memory/cpu_memory_log_<job_id>.txt` | RAM usage sampled over time using `psrecord`                   |

---

## Exercise: Run the Baseline Training & Fill Evaluation Summary Table

As part of this workshop, your task is to **run the baseline fine-tuning experiment** and **recreate the performance
summary table** using your own training logs.

This exercise helps you develop a habit of tracking key metrics like training loss, evaluation loss, and throughput —
which are essential for understanding and debugging model training.

### Part 1: Run the Baseline Fine-Tuning Job

#### Steps:

1. Navigate to the [baseline](experiments/baseline) directory
    ```bash
    cd experiments/baseline
    ```
2. Submit the slurm script [baseline.slurm](experiments/baseline/baseline.slurm)
    ```bash
    sbatch baseline.slurm
    ```
3. Once the job is terminated, check for the output artifacts:

   Output logs found in `.out` inside [log](experiments/baseline/logs) directory, it should be tailed with the slurm job
   id.
     ```bash
     cd log
     cat <job_name>_<job_id>.out
     ```
    - You will see lines similar to:
    ```commandline
   100%|██████████| 93/93 [02:05<00:00,  1.35s/it]
    {'eval_loss': 1.2965800762176514, 'eval_runtime': 1.2687, 'eval_samples_per_second': 39.41, ... 'epoch': 1.0}
    {'eval_loss': 1.5810621976852417, 'eval_runtime': 1.269,  'eval_samples_per_second': 39.403, ... 'epoch': 2.0}
    {'eval_loss': 1.7790133953094482, 'eval_runtime': 1.2682,'eval_samples_per_second': 39.427, ... 'epoch': 2.93}
    {'train_runtime': 125.2999, 'train_samples_per_second': 11.971, 'train_steps_per_second': 0.742, 'train_loss': 0.7039181288852486, 'epoch': 2.93}
    Model saved to ./bloom-finetuned
    ```
4. Fill the results table:

   Extract the following metrics from the output log and populate the table below:

   | **Metric**                     | **Log Location & Extraction**                             | **Your Value** |
   |--------------------------------|-----------------------------------------------------------|----------------|
   | Train Loss (Final)             | Last `train_loss` in `{'train_loss': ...}`                |                |
   | Eval Loss (Epoch 1)            | First `eval_loss` where `'epoch': 1.0`                    |                |
   | Eval Loss (Epoch 2)            | `eval_loss` where `'epoch': 2.0`                          |                |
   | Eval Loss (Epoch 3)            | Final `eval_loss` (e.g. where `'epoch': 2.93`)            |                |
   | Training Speed (samples/sec)   | `train_samples_per_second` in the final summary           |                |
   | Evaluation Speed (samples/sec) | `eval_samples_per_second` in any eval line (e.g. epoch 1) |                |
   | Steps per Epoch                | Similar to the `93/93` shown in the progress bar.         |                |

### Part 2: Analyze GPU Memory Usage from Logs

Use your GPU memory log (e.g., `baseline-single-gpu_memory_log.csv`) to calculate and fill in the table below.

#### Steps:

1. Locate the log file that tracks GPU memory:
    ```bash
    cd gpu_memory
    cat gpu_memory_log_<SLURM_JOB_ID>.csv
    ```

2. Understanding GPU memory tracking output with `nvidia-smi`

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

3. FIll in the table

   | **Prompt**               | **Shell Command to Run**                                                                         | **Extracted Value (MiB)** |
   |--------------------------|--------------------------------------------------------------------------------------------------|----------------------------|
   | **Peak** memory used?    | `tail -n +2 gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`                  |                            |
   | **Minimum** memory used? | `tail -n +2 gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| head -1`                  |                            |
   | **Mean** memory usage?   | `tail -n +2 gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| awk '{sum+=\$1} END {print sum/NR}'` |                            |

   #### Explanation

    - `tail -n +2` skips the header line.

    - `cut -d',' -f4` selects the memory.used [MiB] column.

    - `sort -n` sorts numerically.

    - `awk` sums and averages the memory values.

---

# DeepSpeed-Zero on Single Node Single GPU:

## DeepSpeed Configuration File:

The DeepSpeed config file is the central interface through which
you control how DeepSpeed integrates with your training pipeline. It acts like a blueprint that tells DeepSpeed how to
optimize and manage.

A basic example of ds_config.json is available [here](ds_configs/zero0.json) for reference.

### Parameter Descriptions

Below are the key settings used throughout this workshop. These form the foundation for later ZeRO configurations:

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

> **Note:** `zero_optimization.stage = 0` does not activate ZeRO but satisfies DeepSpeed’s required structure.
>
>**More on ZeRO optimization (Stage 1, 2, and 3) will be covered in a later section.**

## Turning the Baseline into a DeepSpeed-Enabled Trainer

In the baseline setup, the `Trainer` uses Hugging Face’s standard training loop without any DeepSpeed optimizations.

To integrate DeepSpeed into the training pipeline, the `TrainingArguments` class must reference a DeepSpeed
configuration file `<ds_config>.json`. This allows Hugging Face's Trainer to apply DeepSpeed's optimization features
during training.

### Running the Script with DeepSpeed

The training process is launched using the **standard Python command**.  
DeepSpeed still initializes and applies all runtime optimizations through its integration with the Hugging Face
`Trainer` API.

To ensure DeepSpeed is activated, pass the config file using the `--deepspeed` argument when launching the script:

```bash
python scripts/train.py --deepspeed ./<ds_config>.json
```

### Modification to `TrainingArguments`

To enable DeepSpeed, a single line is added:

```python
    deepspeed = "./ds_config.json",  # Links the DeepSpeed configuration file
```

> In our setup, we allow the `--deepspeed` path to be passed dynamically at runtime via command-line arguments.
>
>This means no manual editing of the script is needed to switch between configurations — just pass the desired `JSON`
> via the `--deepspeed` and it will be integrated directly into Hugging Face’s training loop.

**This integration allows DeepSpeed to handle aspects such as:**

- Mixed-precision training (FP16)

- Gradient accumulation and clipping

- Optimizer configuration

## ZeRO in DeepSpeed

**ZeRO** (Zero Redundancy Optimizer) reduces memory usage by partitioning model states across devices.
ZeRO is applied in three progressive stages, each targeting a different component of the training state:

- Stage 1: Partitions **optimizer** states

- Stage 2: Partitions **optimizer** states and **gradients**

- Stage 3: Partitions **optimizer** states, **gradients**, and model **parameter**

![Zero Stages Hugging Face Illustration](zero-stages-hf-illustration/img.png)

### Stage 1: Optimizer State Partitioning

**What it does:**

- Reduces memory footprint by not replicating optimizer states on every GPU.
- Each GPU only keeps its own partition of the optimizer state tensors.
- Model parameters and gradients are still fully replicated across devices.

On **single GPU**, there's no sharding — so optimizer state partitioning won't reduce memory use.

#### How to Enable ZeRO Stage 1

To activate ZeRO Stage 1 optimization, update the DeepSpeed config by setting:

```json
  "zero_optimization": {
"stage": 1
}
```

No Python code changes are needed.

#### Optimizer State Offloading (Stage 1 with Offload)

In case your model is still hitting memory limits or you'd like to further reduce GPU memory pressure, you can offload
the optimizer state to CPU.

This is particularly useful for:

- Single GPU setups with tight memory
- Larger batch sizes or long sequence lengths

To enable Optimizer State Offloading, modify your configuration to include:

```json
  "zero_optimization": {
"stage": 1,
"offload_optimizer": {
"device": "cpu"
}
}
```

### Stage 2: + Gradient Partitioning

What it adds:

- Reduces activation memory and gradient memory on each GPU.
- Makes training larger models possible on limited memory.

Configure your DeepSpeed setup with the following parameters, to activate it:

```python
  "zero_optimization": {
    "stage": 2
}
```

#### Stage 2 + Offloading provides best memory savings by:

- Offloading optimizer states,

- Offloading gradient states,

Apply this setting to the DeepSpeed configuration file:

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

Use the configuration snippet below in your DeepSpeed setup.

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

Enable this feature in your DeepSpeed config by including:

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

For Optimizer Offloading:

```json
    "offload_optimizer": {
"device": "cpu",
"pin_memory": true
}
```

For Parameter Offloading:

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

1. **Background the Training job:**
    ```
   # Launch the training script with DeepSpeed in the background
   python scripts/train.py --deepspeed <ds_config>.json &
   # Capture the PID of the DeepSpeed training process for later monitoring or cleanup
   TRAIN_PID=$!
   ```
    - The training command is launched **in the background** (`&`) so the script can capture its PID (`$!`).
    - This is necessary for psrecord to **attach** to and **monitor** the correct process.


2. **Add CPU Memory Monitoring with `psrecord`**:
    ```commandline
    # Start CPU memory logging in the background for the DeepSpeed process (including its children),
    # sampling every 5 seconds and writing to the specified log file
    psrecord $TRAIN_PID --include-children --interval 5 --log "deepspeed-cpu-offloading-data-finetune${SLURM_JOB_ID}.txt" &
    
    # Capture the PID of the psrecord process so it can be terminated after training finishes
    MONITOR_CPU_PID=$!
    ```
    - `psrecord` tracks memory usage of the training process.
    - `--include-children` ensures subprocesses like optimizer workers or data loaders are included.
    - `--interval` 5 matches the GPU log sampling rate.
    - `--log writes` memory usage to a timestamped text file.


3. **Wait for the Training to Finish**
    ```
   # Wait for the DeepSpeed training process to complete before stopping any logging or cleanup
    wait $TRAIN_PID
    ```
    - Ensures the script pauses until training completes.
    - Prevents premature termination of memory logging processes.


4. **Stop Logging Process**
    ```
   # Stop the CPU memory logging process now that training has finished
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

1. Fill a table to compare `pin_memory: true` vs `false`.
2. Fill the full benchmarking table comparing ZeRO stages.
3. Modify model size and dataset subset for deeper experimentation.

### Part 1: Compare Stage 2 With and Without `pin_memory: true`

This exercise focuses on evaluating the impact of enabling pinned memory in offloading.

#### Setup

- Use `bloom-560m` with `stage 2 + offloading`.
- Prepare **two configs**:
    1. With `pin_memory: true`
    2. With `pin_memory: false` (or omit the key)

### Steps:

#### The `pin_memory: true` Experiment

1. Navigate to the [pinned memory](experiments/deepspeed-single-gpu/cpu_offloading/pinned_memory) experiment directory:
    ```commandline
    cd experiments/deepspeed-single-gpu/cpu_offloading/pinned_memory
    ```
2. Submit the job through
   the [slurm script](experiments/deepspeed-single-gpu/cpu_offloading/pinned_memory/deepspeed_zero2_offload_pinned_memory.slurm)
    ```commandline
    sbatch deepspeed_zero2_offload_pinned_memory.slurm
    ```
   This SLURM script launches a DeepSpeed training job using the ZeRO Stage 2 configuration with CPU offloading and
   pinned memory, specified via
   the [zero2_cpu_offload_pinned_memory.json](ds_configs/zero2_cpu_offload_pinned_memory.json) file.


3. Find the output artifacts within same directory, once the job is terminated:
    - The output logs (`.out` file) should be located
      in [log](experiments/deepspeed-single-gpu/cpu_offloading/pinned_memory/log) directory
    - The CPU memory logs in the [cpu_memory](experiments/deepspeed-single-gpu/cpu_offloading/pinned_memory/cpu_memory)
      directory
    - The GPU memory logs in the [cpu_memory](experiments/deepspeed-single-gpu/cpu_offloading/pinned_memory/gpu_memory)
      directory

#### The `pin_memory: false` Experiment

4. Navigate to the [Zero 2 CPU offloading experiment](experiments/deepspeed-single-gpu/cpu_offloading/zero_2):
    ```commandline
    cd experiments/deepspeed-single-gpu/cpu_offloading/zero_2
    ```
5. Submit the job through
   the [slurm script](experiments/deepspeed-single-gpu/cpu_offloading/zero_2/deepspeed_zero2_offload.slurm)
    ```commandline
    sbatch deepspeed_zero2_offload.slurm
    ```

This SLURM script launches a DeepSpeed training job using the ZeRO Stage 2 configuration with CPU offloading, specified
via the [zero2_cpu_offload.jso](ds_configs/zero2_cpu_offload.json) file.

6. Find the output artifacts within same directory, once the job is terminated:
    - The output logs (`.out` file) should be located
      in [log](experiments/deepspeed-single-gpu/cpu_offloading/zero_2/log) directory
    - The CPU memory logs in the [cpu_memory](experiments/deepspeed-single-gpu/cpu_offloading/zero_2/cpu_memory)
      directory
    - The GPU memory logs in the [cpu_memory](experiments/deepspeed-single-gpu/cpu_offloading/zero_2/gpu_memory)
      directory

Use this table to record memory usage and runtime for ZeRO Stage 2 offloading, with and without pinned memory.

| **Model and Data**     | **Pinned Memory**   | **Peak GPU Memory (MiB)**                                                                                                                                    | **Peak CPU Memory (MiB)**                                                                                                                                    | **Train Samples/Seconds**                                                                                              |
|------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Bloom 560M, 500 subset | `pin_memory: true`  | run `tail -n +2 gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> where the GPU memory logs of the `pin_memory: true` run in located  | run `tail -n +2 cpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> where the CPU memory logs of the `pin_memory: false` run in located | `train_samples_per_second` in the final summary <br/> within the `.out` logs of the `pin_memory: TRUE` run in located  |
| Bloom 560M, 500 subset | `pin_memory: false` | run `tail -n +2 gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> where the GPU memory logs of the `pin_memory: false` run in located | run `tail -n +2 cpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> where the CPU memory logs of the `pin_memory: false` run in located | `train_samples_per_second` in the final summary <br/> within the `.out` logs of the `pin_memory: false` run in located |

### Quiz Questions:

**Did enabling `pin_memory: true` reduce runtime in Stage 2 + offloading?**  
→ If so, by how much?

---

### Part 2: Recreate the ZeRO Stage Comparison Table

#### How to Change Model and Dataset Subset Size

This section explains how to modify the model and dataset subset to perform benchmarking on larger-scale scenarios.

#### Changing the Model

To change the model being fine-tuned (e.g., from `bloom-560m` to `bloom-3b` or `bloom-7b`), update the model name in the
script where `load_model()` is defined.

- Open [model.py](scripts/model.py):
    ```commandline
    cd scripts/
    vim model.py
    ```

- Find this line `MODEL_NAME = "bigscience/bloom-560m"`:

  While in vim, press the ":" button then type "12" to find this line inside the model file


- Replace it with the following:

    ```python
    MODEL_NAME = "bigscience/bloom-3b"  # 3 billion parameter version
    ```

#### Changing the Dataset Subset Size

To avoid using the full SQuAD dataset (which can be large), the dataset loader supports subsetting.

In [train.py](experiments/deepspeed-single-gpu/train.py), locate the `load_squad()` function:

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

### ZeRO Stage Comparison Table (Fill in the Missing Values)

| **Model and Data**      | **DeepSpeed Config**      | **Submit Training Job**                                                                              | **Peak GPU Memory Used (MiB)** | **Average GPU Memory Used (MiB)** | **Peak CPU Memory Used (MiB)** | **Average CPU Memory Used (MiB)** | **Train Samples/Second** |
|-------------------------|---------------------------|------------------------------------------------------------------------------------------------------|--------------------------------|-----------------------------------|--------------------------------|-----------------------------------|--------------------------|
| BLOOM 560M, 500 samples | ZeRO Stage 1              | `cd experiments/deepspeed-single-gpu/zero_1/ && sbatch deepspeed_zero1.slurm`                        |                                |                                   |                                |                                   |                          |
| BLOOM 560M, 500 samples | ZeRO Stage 1 + Offloading | `cd experiments/deepspeed-single-gpu/cpu_offloading/zero_1/ && sbatch deepspeed_zero1_offload.slurm` |                                |                                   |                                |                                   |                          |
| BLOOM 560M, 500 samples | ZeRO Stage 2              | `cd experiments/deepspeed-single-gpu/zero_2/ && sbatch deepspeed_zero2.slurm`                        |                                |                                   |                                |                                   |                          |
| BLOOM 560M, 500 samples | ZeRO Stage 2 + Offloading | `cd experiments/deepspeed-single-gpu/cpu_offloading/zero_2/ && sbatch deepspeed_zero2_offload.slurm` |                                |                                   |                                |                                   |                          |
| BLOOM 560M, 500 samples | ZeRO Stage 3              | `cd experiments/deepspeed-single-gpu/zero_3/ && sbatch deepspeed_zero3.slurm`                        |                                |                                   |                                |                                   |                          |
| BLOOM 560M, 500 samples | ZeRO Stage 3 + Offloading | `cd experiments/deepspeed-single-gpu/cpu_offloading/zero_3/ && sbatch deepspeed_zero3_offload.slurm` |                                |                                   |                                |                                   |                          |

#### Running the Experiment and Extracting Metrics

| **Step**               | **Command / Instruction**                                                                                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Check training summary | find `train_samples_per_second` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                                         |
| Peak GPU Memory (MiB)  | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> from the same directory where your SLURM script is located.                   |
| Avg GPU Memory (MiB)   | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| awk '{sum+=\$1} END {print sum/NR}'` <br/> from the same directory where your SLURM script is located. |
| Peak CPU Memory (MiB)  | run `grep -v '^#' cpu_memory/cpu_memory_log_<JOB_ID>.txt \| awk '{print $3}' \| sort -n \| tail -1` <br/> from the same directory where your SLURM script is located.             |
| Avg CPU Memory (MiB)   | run `grep -v '^#' cpu_memory/cpu_memory_log.txt \| awk '{sum+=$3} END {print sum/NR}'`<br/> from the same directory where your SLURM script is located.                           |

### Quiz Questions:

Use your completed memory and runtime benchmark tables to answer the following questions for each part.

1. Which ZeRO stage gave the lowest peak GPU memory usage on each bloom variation?

2. How does runtime change when offloading is enabled?

---

## DeepSpeed-Zero on Single Node Multi GPUs:

This section describes how to adapt a single-GPU DeepSpeed setup to fine-tune a large model across
multiple GPUs on one node.

### Setting-up Multi GPUs Training:

To recreate this multi-GPU fine-tuning setup for BLOOM-560M using DeepSpeed, you'll begin with your working single-GPU
Hugging Face + DeepSpeed setup and make a few targeted changes. Below is a full explanation of the required
modifications and an overview of how DeepSpeed handles GPU sharding.

#### Step 1: Adjust SLURM directives for 2 GPUS per node for example:

```commandline
#SBATCH --gpus=2                               # Request 2 GPU
#SBATCH --gpus-per-node=2                      # 2 GPUs per node
```

#### Step 2: Set up DeepSpeed master address and port:

Place these lines before the training command in
the [SLURM](experiments/deepspeed-single-gpu/zero_0/deepspeed_zero0.slurm) script:

```commandline
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # Hostname or IP of the master node for NCCL initialization
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')    # Dynamically find a free TCP port on the current node to use as the rendezvous port.
export WORLD_SIZE=$SLURM_GPUS_ON_NODE                                        # Total number of GPUs being used on this node
export RANK=0                                                                 # Global rank of this process (0 for single-node jobs)
export LOCAL_RANK=0                                                           # Local GPU index for this process (0–N-1)
```

These environment variables configure distributed training manually.

- `MASTER_ADDR`: IP or hostname of main node (for NCCL initialization).

- `MASTER_PORT`: The Python one-liner creates a socket, binds it to port 0 (let OS choose an available port).

- `WORLD_SIZE`: Total number of GPUs (set automatically by SLURM).

- `RANK` and LOCAL_RANK: Needed for some custom multi-node setups.

#### Step 3: Use the Python Launcher with

```commandline
python -m torch.distributed.run --nproc_per_node=$SLURM_GPUS_ON_NODE  scripts/train.py --deepspeed ds_configs/zero2_cpu_offload_pinned_memory.json &
```

This command launches **distributed training** on **all GPUs in the current node**, using PyTorch’s recommended
launcher (
`torch.distributed.run`).

---

## Exercise: Benchmarking Multi-GPU DataParallel Training

### Part 1: Scaling on Multi-GPU Setup

This exercise walks through how to measure and tabulate key training metrics for `BLOOM-560M` on a 500 sample subset,
using DeepSpeed’s DataParallel (no ZeRO) across 2, 4, 6, and 8 GPUs.

Run the same training script with `#SBATCH --gpus-per-node=N` for **N = 2, 4, 6, 8**, then record:

- **Train Runtime** (total seconds)
- **Steps/sec** (extracted from logs or computed)
- **Samples/sec** (`Steps/sec × per_device_train_batch_size × N`)
- **Train Loss** (final)
- **Eval Loss** (final)
- **Eval Speed** (samples/sec during evaluation)
- **Peak GPU Memory** (per GPU)

**What Is Weak Scaling?**

Weak scaling refers to how well a system handles increasing workload proportional to the number of compute resources.

- You increase the dataset size as you increase the number of GPUs or nodes.
- Each GPU processes the same amount of data, regardless of how many GPUs are used.

For example, if the base is 500 samples per GPU:

- **2 GPUs** → 500 samples
- **4 GPUs** → 1000 samples
- **6 GPUs** → 1500 samples
- **8 GPUs** → 2000 samples

#### Automating Weak Scaling in [train.py](scripts/train.py)

This snippet shows how to compute the dataset subset size automatically based on the number of GPUs available (
weak-scaling mode).

```python
import os

# Define the base number of samples **per GPU**
base_size = 10000

# Detect number of GPUs (DeepSpeed / SLURM will set WORLD_SIZE)
#    Fallback to 1 ifWORLD_SIZE is not set
num_gpus = int(os.environ.get("WORLD_SIZE", 1))

# Compute total subset size = base_size × num_gpus
subset_size = base_size * num_gpus

# Load and tokenize only that many examples
tokenized_datasets = load_squad(subset_size=subset_size)
```

### Fill in the results for each GPU count below.

| **Metric**               |                                                                    **2 GPUs** |                                                                    **4 GPUs** |                                                                    **6 GPUs** |                                                                    **8 GPUs** |
|--------------------------|------------------------------------------------------------------------------:|------------------------------------------------------------------------------:|------------------------------------------------------------------------------:|------------------------------------------------------------------------------:|
| Submit Job               | `cd experiments/deepspeed-multi-gpu/2_gpus/ && sbatch deepspeed_2_gpus.slurm` | `cd experiments/deepspeed-multi-gpu/4_gpus/ && sbatch deepspeed_4_gpus.slurm` | `cd experiments/deepspeed-multi-gpu/6_gpus/ && sbatch deepspeed_6_gpus.slurm` | `cd experiments/deepspeed-multi-gpu/8_gpus/ && sbatch deepspeed_8_gpus.slurm` |
| Train Samples/sec        |                                                                               |                                                                               |                                                                               |                                                                               |
| Train Loss               |                                                                               |                                                                               |                                                                               |                                                                               |
| Peak GPU Memory (MiB)    |                                                                               |                                                                               |                                                                               |                                                                               |
| Average GPU Memory (MiB) |                                                                               |                                                                               |                                                                               |                                                                               |

#### Extracting Metrics

| **Step**                      | **Command / Instruction**                                                                                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Check Training Samples/Second | find `train_samples_per_second` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                                         |
| Check Training loss           | find Last `train_loss` in `{'train_loss': ...}` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                         |
| Peak GPU Memory (MiB)         | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> from the same directory where your SLURM script is located.                   |
| Avg GPU Memory (MiB)          | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| awk '{sum+=\$1} END {print sum/NR}'` <br/> from the same directory where your SLURM script is located. |
| Peak CPU Memory (MiB)         | run `grep -v '^#' cpu_memory/cpu_memory_log_<JOB_ID>.txt \| awk '{print $3}' \| sort -n \| tail -1` <br/> from the same directory where your SLURM script is located.             |
| Avg CPU Memory (MiB)          | run `grep -v '^#' cpu_memory/cpu_memory_log.txt \| awk '{sum+=$3} END {print sum/NR}'`<br/> from the same directory where your SLURM script is located.                           |

### Quiz Questions

1. Given weak scaling, should **Peak GPU Memory** per GPU remain constant? Explain any deviations you observe.

---

### part 2: 2-GPU ZeRO Stage Comparison

This exercise guides the measurement and comparison of training metrics for ZeRO Stages 1, 2, 3 on **2 GPUs**, each
**with** and **without** CPU offloading.
Fill in the results for ZeRO Stages 1, 2, and 3 on **2 GPUs**, both **with** and **without** CPU offloading.

| **Metric**               | **Stage 1**                                                                                                  | **Stage 1 + offload**                                                                                                               | **Stage 2**                                                                                                  | **Stage 2 + offload**                                                                                                               | **Stage 3**                                                                                                  | **Stage 3 + offload**                                                                                                               |
|--------------------------|:-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
| Submit Job               | `cd experiments/deepspeed-multi-gpu/2_gpus_stages_comparison/zero_1/ && sbatch deepspeed_2_gpus_zero1.slurm` | `cd experiments/deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_1/ && sbatch deepspeed_2_gpus_zero1_offload.slurm` | `cd experiments/deepspeed-multi-gpu/2_gpus_stages_comparison/zero_1/ && sbatch deepspeed_2_gpus_zero2.slurm` | `cd experiments/deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_1/ && sbatch deepspeed_2_gpus_zero2_offload.slurm` | `cd experiments/deepspeed-multi-gpu/2_gpus_stages_comparison/zero_1/ && sbatch deepspeed_2_gpus_zero3.slurm` | `cd experiments/deepspeed-multi-gpu/2_gpus_stages_comparison/cpu_offloading/zero_1/ && sbatch deepspeed_2_gpus_zero3_offload.slurm` |
| Train Samples/sec        |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |
| Train Loss               |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |
| Peak GPU Memory (MiB)    |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |
| Average GPU Memory (MiB) |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |
| Peak CPU Memory (MiB)    |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |
| Average CPU Memory (MiB) |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |                                                                                                              |                                                                                                                                     |

#### Extracting Metrics

| **Step**                      | **Command / Instruction**                                                                                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Check Training Samples/Second | find `train_samples_per_second` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                                         |
| Check Training loss           | find Last `train_loss` in `{'train_loss': ...}` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                         |
| Peak GPU Memory (MiB)         | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> from the same directory where your SLURM script is located.                   |
| Avg GPU Memory (MiB)          | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| awk '{sum+=\$1} END {print sum/NR}'` <br/> from the same directory where your SLURM script is located. |
| Peak CPU Memory (MiB)         | run `grep -v '^#' cpu_memory/cpu_memory_log_<JOB_ID>.txt \| awk '{print $3}' \| sort -n \| tail -1` <br/> from the same directory where your SLURM script is located.             |
| Avg CPU Memory (MiB)          | run `grep -v '^#' cpu_memory/cpu_memory_log.txt \| awk '{sum+=$3} END {print sum/NR}'`<br/> from the same directory where your SLURM script is located.                           |

> **Note on Code Versioning and SLURM Queues**  
> SLURM does **not** snapshot your Python scripts when you call `sbatch`.
> - The job will execute whatever version of `train.py` (or any other `.py` files) is on disk **at the moment the job
    actually starts** running, not when it was submitted.
> - Any edits made to your code while the job is still in the queue will be picked up automatically.
>
> **Best Practices:**
> Submit from a dedicated directory that won’t be modified.  
> This ensures reproducibility and avoids unintended changes in long-running or queued jobs.

### Quiz Questions

1. How does enabling offloading affect **Train Runtime** and **Samples/sec**? Quantify the trade-off between memory
   savings and speed.

Fill in the table below to compare each ZeRO stage **with** and **without** offloading. Calculate both absolute and
percentage changes.

| **Stage** | **GPU Mem (No Offload)** | **GPU Mem (Offload)** | **Mem Savings (%)** | **Train Runtime (No Offload)** | **Train Runtime (Offload)** | **Runtime Δ (%)** |
|-----------|--------------------------|-----------------------|---------------------|--------------------------------|-----------------------------|-------------------|
| Stage 1   |                          |                       |                     |                                |                             |                   |
| Stage 2   |                          |                       |                     |                                |                             |                   |
| Stage 3   |                          |                       |                     |                                |                             |                   |

- **Mem Savings (%)** = `(GPU Mem No Offload – GPU Mem Offload) / GPU Mem No Offload × 100`
- **Runtime Δ (%)** = `(Runtime Offload – Runtime No Offload) / Runtime No Offload × 100`

---

## Multi-Node Training with DeepSpeed and torch.distributed

When training large models across multiple nodes, we use PyTorch’s `torch.distributed.run` to launch one training process
per node.

This launcher communicates across nodes using **TCP-based rendezvous**, making it suitable for clusters where direct process
launching across nodes (e.g., via SSH) is restricted or avoided.

Below is a side-by-side of the relevant sections, with **added/modified lines** annotated.

1. SLURM script edits:

- `SBATCH` directives:
    ```commandline
    #SBATCH --nodes=2                                                # Number of nodes to allocate
    #SBATCH --ntasks=2                                               # Number of tasks
    #SBATCH --ntasks-per-node=1                                      # One task (process) per node
    #SBATCH --cpus-per-task=4                                        # Number of CPU cores per task
    #SBATCH --gpus=1                                                 # Request 1 GPU
    #SBATCH --gpus-per-node=1                                        # 1 GPU per node
    ```
- After environment setup lines, determine master node discovery and rendezvous configuration
   ```commandline
   # Getting the node names
   nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
   nodes_array=($nodes)
   echo "Node IDs of participating nodes ${nodes_array[*]}"
        
   # Get the IP address and set port for MASTER node
   head_node="${nodes_array[0]}"
   echo "Getting the IP address of the head node ${head_node}"
   master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
   master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
   echo "head node is ${master_ip}:${master_port}"
   ```
  For multi-node training without passwordless SSH, each process needs to know exactly where and how to connect. This
  snippet discovers the head node’s network address and a free port dynamically, so that all workers—spawned via
  torch.distributed.run—can rendezvous correctly.

    - `scontrol` show hostnames `$SLURM_JOB_NODELIST`, reads the list of node hostnames allocated to the job (via the
      `SLURM` variable `SLURM_JOB_NODELIST`). Returns one hostname per line.
    - `nodes_array=($nodes)` splits the multiline string in nodes into a `Bash` array, `nodes_array`, so that each
      element is a single hostname.
    - `head_node="${nodes_array[0]}"` picks the first element of nodes_array as the master or rendezvous node for
      distributed setup.
    - `srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I`, runs hostname -I on the head node under SLURM (via
      srun), which prints all IP addresses assigned to that machine.
    - `| cut -d " " -f 2` splits the hostname -I output on spaces (`-d " "`), and selects the second field (`-f 2`),
      which is typically the primary network interface IP (e.g. 192.168.1.42).
    - `master_port=$(python -c '…')` launches a short Python one-liner that:
        - Creates a TCP socket.

        - Binds it to port 0 (meaning “choose any free port”).

        - Prints the assigned port number (getsockname()[1]).

        - Closes the socket.

      This ensures a free rendezvous port is picked at runtime.
  

- Detailed Explanation: Per-Node Process Launch and GPU Memory Logging
   ```bash
   logdir=./gpus_usage/4-nodes
   mkdir -p "$logdir"
        
   # Loop over all allocated nodes and launch exactly one training process per node
   for (( i=0; i< ${SLURM_NNODES}; i++ ))
   do
       srun \
         -N1 -n1 \
         -c ${SLURM_CPUS_PER_TASK} \
         --gpus=${SLURM_GPUS_PER_NODE} \
         -w ${nodes_array[$i]} \
         bash -c "
           # Capture the hostname for this node
           hostname=\$(hostname)
        
           # Start GPU memory logging on this node in the background:
           # - Queries timestamp, GPU index, GPU name, used & total memory (MiB)
           # - Samples every 5 seconds
           nvidia-smi \
             --query-gpu=timestamp,index,name,memory.used,memory.total \
             --format=csv,nounits -l 5 \
                  > \"$logdir/gpu_memory_log_\${hostname}.csv\" &
        
           # Save the PID of the nvidia-smi logger so it can be terminated later
           MEMORY_LOG_PID=\$!
        
           # Launch the distributed training process on this node:
           python -m torch.distributed.run \
             --nnodes=$SLURM_JOB_NUM_NODES \           # Total number of nodes in the job
             --nproc_per_node=1 \                       # One process per node (using 1 GPU)
             --node_rank=$i \                           # This node's rank (0..NNODES-1)
             --rdzv_endpoint=$master_ip:$master_port \  # Rendezvous server address
             train.py                                   # Entrypoint script
       " &
   done
        
   # Wait for all backgrounded training and logging tasks to finish
   wait
        
   # After training completes, stop the GPU memory logger
   kill $MEMORY_LOG_PID
   ```
  ### Step-by-Step Breakdown

    1. **`logdir` and `mkdir -p`**
        - Defines `./gpus_usage/4-nodes` as the directory for per-node GPU logs.
        - `mkdir -p` creates it if it doesn’t exist (no error if it already does).

    2. **`for (( i=0; i<${SLURM_NNODES}; i++ ))`**
        - Iterates over each node index `i` (0-based) allocated to the job.

    3. **`srun -N1 -n1 -c ${SLURM_CPUS_PER_TASK} --gpus=${SLURM_GPUS_PER_NODE} -w ${nodes_array[$i]}`**
        - **`-N1 -n1`**: Launch exactly one SLURM task on the specified node.
        - **`-c ${SLURM_CPUS_PER_TASK}`**: Allocate the given number of CPU cores to that task.
        - **`--gpus=${SLURM_GPUS_PER_NODE}`**: Allocate one GPU (or more) per node.
        - **`-w ${nodes_array[$i]}`**: Restrict execution to the `i`th node in the list of allocated hosts.

    4. **Inside the subshell (`bash -c "…"`):**
        - **`hostname=$(hostname)`**: Captures the current node’s hostname.
        - **GPU Logging:**
            - Runs `nvidia-smi` every 5 seconds to record timestamp, GPU index, name, used and total memory.
            - Outputs to a CSV file named after the hostname.
            - Backgrounds this logging process and saves its PID in `MEMORY_LOG_PID`.
        - **Distributed Launch:**
            - Invokes `python -m torch.distributed.run` with:
                - `--nnodes` set to total nodes.
                - `--nproc_per_node=1` (one process per node).
                - `--node_rank=$i` (this node’s rank).
                - `--rdzv_endpoint=$master_ip:$master_port` (rendezvous address).
            - Runs `train.py` under this torch.distributed context.

    5. **Backgrounding and `wait`:**
        - The entire `srun … bash -c "…"` block is backgrounded so the loop continues immediately.
        - A final `wait` pauses the script until **all** backgrounded processes (GPU loggers and training jobs) have
          completed.

    6. **`kill $MEMORY_LOG_PID`**
        - After training finishes, terminates the last recorded GPU memory logger process.
        - (If multiple loggers are active, each PID should be tracked and terminated as nee

2. [train.py](scripts/train_multi_nodes.py) script edits:

- At the top of `train.py`, ensure these modules are imported:
    ```python
    import os
    import torch
    import torch.distributed as dist
    ```
- Log the TCP rendezvous parameters, immediately after the imports:
    ```python
    # =================================================
    # Distributed Initialization Logging (TCP-based)
    # =================================================
    print(f"Initializing process group for rank {os.environ.get('RANK')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    
    ```
  This will print each worker’s rank, and the MASTER_ADDR:MASTER_PORT it connects to.


- Assign the CUDA device, right after logging (and before any NCCL calls), bind the process to its local GPU:
    ```python
    # =============================
    # Device Assignment
    # =============================
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    ```
    - `LOCAL_RANK` is provided by `SLURM` or `torch.distributed.run`.
    - `set_device(local_rank)` ensures each process uses its correct GPU.


- Verify Initialization
  Finally, immediately after the init_process_group call:
    ```python
    # ======================================
    # Verify Process Group Initialization
    # ======================================
    if dist.is_initialized():
        print(f"torch.distributed initialized: "
              f"rank {dist.get_rank()} / world size {dist.get_world_size()}")
    ```

## Exercise: Multi‐Node Scaling

This exercise extends the weak‐scaling concept to **multiple nodes**, automatically computing the total dataset size
based on the number of GPUs allocated across all nodes.

### Objective

Keep **250 samples per GPU** fixed, and increase nodes (and thus GPUs) so the **total dataset** grows proportionally:

- **2 nodes** (1 GPU/node) → 2 GPUs → **500** samples
- **3 nodes** → 3 GPUs → **750** samples
- **4 nodes** → 4 GPUs → **1000** samples
- **6 nodes** → 6 GPUs → **1500** samples

Each GPU processes the same 250-sample “chunk.” Measure how well training time and throughput hold constant as nodes
scale.

#### Step 1: Auto‐Compute Dataset Size in `train.py`

This snippet shows how to compute the dataset subset size automatically based on the number of GPUs and nodes available (
weak-scaling mode).

```python
import os
import torch.distributed as dist

# Base number of samples per GPU
base_size_per_gpu = 10000

# Total number of processes (GPUs) across all nodes
world_size = dist.get_world_size()

# GPUs per node (from SLURM or fallback to all local GPUs)
gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))

# Number of nodes = total GPUs // GPUs per node
num_nodes = world_size // gpus_per_node

print(f"Running on {num_nodes} nodes × {gpus_per_node} GPUs = {world_size} total GPUs")

# Compute total subset size for weak scaling
subset_size = base_size_per_gpu * world_size
print(f"Loading subset_size = {subset_size} examples")

# Load the dataset with the computed subset size
tokenized_datasets = load_squad(subset_size=subset_size)
```

### Table Template: Weak Scaling Across Nodes

Fill in these metrics for **2, 3, 4, and 6 nodes**, where the dataset grows proportionally (10 000 samples per GPU):

| **Metric**               | **2 nodes**                                                                       | **3 nodes**                                                                      | **4 nodes**                                                                      | **5 nodes**                                                                      | **6 nodes**                                                                      |
|--------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|----------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
| Submit Job               | `cd experiments/deepspeed-multi-node/2_nodes/ && sbatch deepspeed_2_nodes.slurm`  | `cd experiments/deepspeed-multi-node/3_nodes/ && sbatch deepspeed_3_nodes.slurm` | `cd experiments/deepspeed-multi-node/4_nodes/ && sbatch deepspeed_4_nodes.slurm` | `cd experiments/deepspeed-multi-node/5_nodes/ && sbatch deepspeed_5_nodes.slurm` | `cd experiments/deepspeed-multi-node/6_nodes/ && sbatch deepspeed_6_nodes.slurm` |
| Train Samples/sec        |                                                                                   |                                                                                  |                                                                                  |                                                                                  |                                                                                  |
| Train Loss               |                                                                                   |                                                                                  |                                                                                  |                                                                                  |                                                                                  |
| Peak GPU Memory (GiB)    |                                                                                   |                                                                                  |                                                                                  |                                                                                  |                                                                                  |
| Average GPU Memory (GiB) |                                                                                   |                                                                                  |                                                                                  |                                                                                  |                                                                                  |

---

#### Extracting Metrics

| **Step**                      | **Command / Instruction**                                                                                                                                                         |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Check Training Samples/Second | find `train_samples_per_second` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                                         |
| Check Training loss           | find Last `train_loss` in `{'train_loss': ...}` in the final summary inside `log/<JOB_NAME>_<JOB_ID>.out`                                                                         |
| Peak GPU Memory (MiB)         | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| sort -n \| tail -1`<br/> from the same directory where your SLURM script is located.                   |
| Avg GPU Memory (MiB)          | run `tail -n +2 gpu_memory/gpu_memory_log_<JOB_ID>.csv \| cut -d',' -f4 \| awk '{sum+=\$1} END {print sum/NR}'` <br/> from the same directory where your SLURM script is located. |
| Peak CPU Memory (MiB)         | run `grep -v '^#' cpu_memory/cpu_memory_log_<JOB_ID>.txt \| awk '{print $3}' \| sort -n \| tail -1` <br/> from the same directory where your SLURM script is located.             |
| Avg CPU Memory (MiB)          | run `grep -v '^#' cpu_memory/cpu_memory_log.txt \| awk '{sum+=$3} END {print sum/NR}'`<br/> from the same directory where your SLURM script is located.                           |

### Quiz Questions:

1. **Expectation Check:**

    In weak scaling, the **Train Samples/sec** metric is expected to remain **roughly constant** as we add more nodes.
    Is this expectation met in your runs? Why or why not?
    > (Hint: Consider communication overhead, data loading, and initialization costs.)

2. Looking at the Average GPU Memory, does it increase, decrease, or stay stable as you scale to more nodes?