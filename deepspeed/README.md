# What is this Exercise About?

This exercise explores large language model (LLM) fine-tuning using DeepSpeed, a deep learning optimization library for large-scale training. The workshop walks through a practical progression:

1. Baseline fine-tuning using Hugging Face Trainer **without** DeepSpeed.

2. Fine-tuning on a single GPU using DeepSpeed with different ZeRO optimization stages.

3. Comparing ZeRO stages with and without CPU offloading.

4. Evaluating memory usage and performance trade-offs across configurations.

5. Scaling up to multi-GPU and multi-node training using DeepSpeed's distributed launcher.

## Why DeepSpeed?

Training such large models is computationally expensive and quickly runs into memory limits — especially on a single GPU.

DeepSpeed is a deep learning optimization library from Microsoft designed to:

- **Reduce GPU memory usage** via ZeRO optimizations (stage 1–3).

- **Enable distributed training** across GPUs and even nodes.

- Support **model and tensor parallelism**.

- Work seamlessly with **Hugging Face Transformers**.

## What is Hugging Face 🤗?

**Hugging Face** is an open-source ecosystem built around natural language processing (NLP) and machine learning models — especially transformer-based models like BERT, GPT, and BLOOM.

It provides easy-to-use tools to **download, train, fine-tune, and deploy** state-of-the-art models with just a few lines of code.
### 🔧 Key Components You'll Use

| Component    | What It Does                                                                 |
|--------------|-------------------------------------------------------------------------------|
|`transformers` | Python library for accessing thousands of pre-trained models across NLP, vision, and audio tasks. |
|`datasets`    | Library for easy loading, sharing, and preprocessing of public datasets like SQuAD, IMDB, and more. |
|`Trainer` API | High-level training interface to handle training, evaluation, and checkpointing with minimal code. |
| Model Hub    | Online platform for hosting, sharing, and downloading models — all ready to use. |

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

Before exploring `DeepSpeed` optimizations, it’s useful to understand the vanilla `HuggingFace` fine-tuning process using a smaller LLM like `bigscience/bloom-560m`, and 500 examples subset of `SQuAD` for question-answer format training.

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
[config.py](baseline/config.py) centralizes hyperparameters makes tuning and experimenting easier — change config values in one file without touching the training script.
This section defines the core training hyperparameters and behaviors using the Hugging Face 
- `output_dir`:`./bloom-qa-finetuned`	Directory to store model checkpoints, logs, and evaluation results.
- `eval_strategy`:`epoch`	Evaluation is run at the end of each training epoch.
- `save_strategy`:`epoch`	A model checkpoint is saved at the end of each training epoch.
- `per_device_train_batch_size`:`4`	Batch size to use on each GPU or each CPU worker during training.
- `per_device_eval_batch_size`:`4`	Batch size to use on each device during evaluation.
- `gradient_accumulation_steps`:`4`	Number of steps to accumulate gradients before performing a backward/update pass. Simulates a larger effective batch size.
- `num_train_epochs`:`3`	Total number of training epochs (full passes through the dataset).
- `learning_rate`:`5e-5`	Initial learning rate for the optimizer.
- `weight_decay`:`0.01`	Weight decay (L2 penalty) to apply for regularization. Helps prevent overfitting.
- `fp16`:`True`	Enables automatic mixed precision (AMP) training for reduced memory usage and faster computation (requires compatible hardware).
- `gradient_checkpointing`:`False`	Disables gradient checkpointing. If set to True, it reduces memory usage by trading off compute time.
- `push_to_hub`:`False`	Disables automatic pushing of the model and logs to the Hugging Face Hub. Set to True if you want to share your model publicly or privately.

### Training 
The [train.py]() script fine-tunes a causal language model (e.g., BLOOM) using Hugging Face's Trainer API.

**Hugging Face Trainer:**

The Trainer class from Hugging Face simplifies the process of training and evaluating models — especially useful when working with transformers, tokenizers, and large datasets.
Instead of writing your own training loop with forward(), backward(), optimizer steps, and logging — Trainer handles it for you.

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

Once all components are in place — model loading, dataset preprocessing, training configuration, and training logic — you can execute the full fine-tuning workflow with minimal manual steps.

Use the baseline.slurm script to submit the training job on a GPU node:

```commandline
sbatch baseline.slurm
```
### Output Artifacts

After the run finishes, you'll find:

- Fine-tuned model and tokenizer in `./bloom-qa-finetuned/`

- Training logs including evaluation metrics and loss curves

- SLURM log files in the `log` directory or as specified by --output

##  Exercise: Recreate the Baseline Training & Evaluation Summary Table

As part of this workshop, your task is to **run the baseline fine-tuning experiment** and **recreate the performance summary table** using your own training logs.

This exercise helps you develop a habit of tracking key metrics like training loss, evaluation loss, and throughput — which are essential for understanding and debugging model training.


### Objective 1

After running `train.py`, fill in the following table with metrics from your output:

| **Metric**                      | **Your Value**           |
|---------------------------------|---------------------------|
| Train Loss (Final)              | _Fill from final output_  |
| Eval Loss (Epoch 1)             | _From evaluation logs_    |
| Eval Loss (Epoch 2)             | _From evaluation logs_    |
| Eval Loss (Epoch 3)             | _From evaluation logs_    |
| Training Speed (samples/sec)    | _Reported by Trainer_     |
| Evaluation Speed (samples/sec)  | _Reported by Trainer_     |
| Steps per Epoch                 | _From logs or config_     |



### 🔍 Where to Find These Values

- **Loss values** appear in the `.out` files located in `log` directory , in lines containing `loss=` or `eval_loss=`.
- **Training and evaluation speed** are typically printed after evaluation steps or at the end of training.

### Objective 2

Use your GPU memory log (e.g., `baseline-single-gpu_memory_log.csv`) to calculate and fill in the table below.

| **Metric**                    | **Your Value (MiB)**     |
|-------------------------------|---------------------------|
| Peak GPU Memory Usage         | _Use max value_           |
| Mean GPU Memory Usage         | _Average across samples_  |
| Minimum GPU Memory Usage      | _Lowest recorded_         |

### 🔍 Instructions

 - Extract Peak Memory (MiB)
    ```commandline 
      tail -n +2 baseline-single-gpu_memory_log.csv | cut -d',' -f4 | sort -n | tail -1

    ```
 -  Extract Minimum Memory (MiB)
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

The DeepSpeed config file [ds_config.json](deepspeed-single-gpu/ds_config.json) is the central interface through which you control how DeepSpeed integrates with your training pipeline. It acts like a blueprint that tells DeepSpeed how to optimize and manage.
### Parameter Descriptions

| **Key**                          | **Value**         | **Description**                                                                 |
|----------------------------------|-------------------|---------------------------------------------------------------------------------|
| `train_batch_size`              | `"auto"`          | Automatically sets the largest batch size that fits into GPU memory.           |
| `gradient_accumulation_steps`   | `"auto"`          | Automatically determines the number of accumulation steps to simulate larger batch sizes. |
| `gradient_clipping`             | `1.0`             | Caps gradient norms to prevent exploding gradients and ensure training stability. |
| `optimizer.type`                | `"AdamW"`         | Optimizer used for training; AdamW is standard for transformer models.         |
| `optimizer.params.lr`           | `5e-5`            | Learning rate used for fine-tuning.                                            |
| `optimizer.params.betas`        | `[0.9, 0.999]`    | Beta values used by Adam optimizer for momentum calculations.                  |
| `optimizer.params.eps`          | `1e-8`            | Small constant added to prevent division by zero during optimization.          |
| `optimizer.params.weight_decay` | `0.01`            | Regularization parameter to prevent overfitting.                               |
| `fp16.enabled`                  | `true`            | Enables Automatic Mixed Precision (AMP) for faster and more memory-efficient training. |

[Official documentation for DeepSpeed Configuration JSON]( https://www.deepspeed.ai/docs/config-json/)

## Turning the Baseline into a DeepSpeed-Enabled Trainer

In the baseline setup, the `Trainer` uses Hugging Face’s standard training loop without any DeepSpeed optimizations.

To integrate DeepSpeed into the training pipeline, the `TrainingArguments` class must reference a DeepSpeed configuration file `ds_config.json`. This allows Hugging Face's Trainer to apply DeepSpeed's optimization features during training.

### Modification to `TrainingArguments`
To enable DeepSpeed, a single line is added:
```python
    deepspeed="./ds_config.json",  # Links the DeepSpeed configuration file
```
**This integration allows DeepSpeed to handle aspects such as:**

- Mixed-precision training (FP16)

- Gradient accumulation and clipping

- Optimizer configuration

## Running the Script with DeepSpeed

Once the `deepspeed` field is added to the `TrainingArguments` configuration, the training process must be launched using the **DeepSpeed CLI launcher** instead of the standard **Python** command. 
This ensures that DeepSpeed initializes properly and applies all runtime optimizations defined in `ds_config.json`.

In the slurm script, replace
```commandline
python train.py
```
with
```commandline
deepspeed train.py
```
