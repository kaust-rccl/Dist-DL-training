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

- Manual installation:
    ```commandline
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.51.2
    pip install deepspeed==0.16.5
    pip install accelerate==1.6.0
    pip install datasets==3.4.1
    ```

- Installation through the [requirements.txt](./requirements.txt):
    ```commandline
    pip install -r requirements.txt
    ```
----------------------------------

# Baseline: BLOOM Fine-tuning without DeepSpeed:

## Fine-Tuning Setup

before exploring `DeepSpeed` optimizations, it’s useful to understand the vanilla `HuggingFace` fine-tuning process using a smaller LLM like `bigscience/bloom-560m`, and 500 examples subset of `SQuAD` for question-answer format training.

### Model Loader and Saver
[model.py](baseline/model.py) Defines two key functions:

1. `load_model()`: Loads `bigscience/bloom-560m` model and tokenizer.

2. `save_model()`: Saves the trained model and tokenizer to disk.

### Dataset Preprocessing

[data_loader.py](baseline/data_loader.py) Handles:

1. Loading the SQuAD dataset using Hugging Face datasets.

2. Tokenizing each example as:
    ```
    "Question: ... Context: ... Answer: ..."
    ```
3. Padding/truncating to max length (512).

4. Setting labels = input_ids (for causal LM).

5. Optionally subsetting the dataset for faster experiment.

### Training Configuration
[config.py](baseline/config.py) Centralizes hyperparameters makes tuning and experimenting easier — change config values in one file without touching the training script.
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

- SLURM log files in the `logs` directory or as specified by --output

##  Exercise: Recreate the Baseline Training & Evaluation Summary Table

As part of this workshop, your task is to **run the baseline fine-tuning experiment** and **recreate the performance summary table** using your own training logs.

This exercise helps you develop a habit of tracking key metrics like training loss, evaluation loss, and throughput — which are essential for understanding and debugging model training.


### Objective

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

- **Loss values** appear in the `.out` files located in `logs` directory , in lines containing `loss=` or `eval_loss=`.
- **Training and evaluation speed** are typically printed after evaluation steps or at the end of training.

---