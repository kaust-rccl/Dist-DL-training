
# What is this Exercise About?

You will fine-tune the **BLOOM-560 M** language model on a subset of **SQuAD v1.1** while gradually introducing **Fully-Sharded Data Parallel (FSDP)**, PyTorch’s native memory-saving and scaling engine. The workshop proceeds in five steps:


 1.   **Baseline** – run Hugging Face `Trainer` on a single GPU with no sharding.
    
 2.   **Single-GPU FSDP** – enable full-parameter sharding to observe the immediate drop in memory usage.
    
 3.   **Sharding Strategies & Off-loading** – compare FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD and optional CPU off-load to understand speed-vs-memory trade-offs.
    
 4.   **Multi-GPU Scaling** – train on two and then eight GPUs within one node, measuring throughput and per-GPU memory.
    
 5.   **Multi-Node Scaling** – launch FSDP across multiple V100 and A100 nodes, after an NCCL bandwidth self-test to ensure healthy interconnects.

## Why FSDP?
-   **Memory relief:** instead of holding the entire model replica on every GPU, FSDP shards parameters, gradients, and optimizer state across all ranks. Each GPU sees only a slice outside the brief moments when full tensors are gathered for compute, freeing a large fraction of VRAM and allowing larger models or batch sizes to fit.
    
-   **Native PyTorch:** FSDP ships with `torch.distributed` — no external runtime, no patched kernels. A single flag inside `TrainingArguments` (`fsdp="full_shard"`) gives you sharding in any Hugging Face training script.
    
-   **Mix-and-match precision and checkpointing:** AMP (FP16/BF16) and activation checkpointing integrate seamlessly with FSDP, letting you trade compute for even deeper memory cuts.
    
-   **Scale from laptop to super-computer:** the same Python file can be launched with `torchrun` on one GPU, eight GPUs in a workstation, or thousands of GPUs across nodes. FSDP relies on NCCL and the PyTorch launch tooling you already know.
    
-   **Flexible sharding strategies:** choose between
    
    -   **FULL_SHARD** – shard parameters, gradients, and optimizer state
        
    -   **SHARD_GRAD_OP** – keep parameters replicated, shard gradients/state
        
    -   **HYBRID_SHARD** – intra-node and inter-node sharding mix
        
    -   optional **CPU off-load** for optimizer state when GPU RAM is scarce.
        
-   **No deep framework lock-in:** because it is part of stock PyTorch, FSDP works with any model architecture, custom optimizer, or callback stack you already use.

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
- Later, enhance scalability using **FSDP** for memory- and compute-efficient training.

## Learning Outcomes

### By the end of this workshop you will be able to:

1-   Fine-tune transformer models with **vanilla Hugging Face Trainer** and then switch to **Fully-Sharded Data Parallel** with only a few extra arguments.
    
2-   Select and justify an **FSDP sharding strategy** (FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD) and decide when to enable **CPU off-load** or **activation checkpointing**.
    
3-   Measure and interpret key signals — **steps per second, samples per second, GPU utilisation, peak/mean memory, validation EM / F1** — using the provided callbacks.
    
4-   Launch distributed jobs with `torchrun` on a single GPU, multiple GPUs within one node, and multiple nodes, including environment variables and NCCL diagnostics.
    
5-   Compare memory footprint and throughput across configurations and articulate the trade-offs in a concise results summary.

-----------------------------------

# Environment Setup

We’ll create a dedicated Conda environment from the [bloom_env.yml](./bloom_env.yml) file that accompanies the workshop. This YAML already pins compatible versions of PyTorch 2.x, CUDA , NCCL, `transformers`, `datasets`, and `wandb`, so you get a reproducible stack for all FSDP scripts.

**Step 1 – Create and activate the environment**
```
conda env create -f bloom_env.yml
conda activate bloom_env
```

**Step 2 – Verify GPU build of PyTorch**

	python - <<'PY'
	import torch, torch.distributed as dist, platform
	print("CUDA available :", torch.cuda.is_available())
	print("GPU name        :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU-only")
	print("PyTorch version :", torch.__version__, " |  CUDA:", torch.version.cuda)
	print("System          :", platform.platform())
	PY
----------------------------------
You should see `CUDA available : True`.
# Baseline: BLOOM Fine-tuning without FSDP:

## Fine-Tuning Setup

Before introducing sharding we warm-up with a plain Hugging Face run on a single A100. Two files matter:

-   **`baseline.py`** – the training script.
    
-   **`baseline.slurm`** – the one-GPU job launcher.

What happens in [baseline.py](./baseline.py):
- **Model and tokenizer**  
`BloomForCausalLM.from_pretrained("bigscience/bloom-560m")` gives a 560 M-parameter causal LM.  
`BloomTokenizerFast` handles tokenisation.

- **Dataset loader**  
`load_squad()` pulls SQuAD v1.1 and turns each row into the prompt  
`"Question: … Context: … Answer:"`.  
Answers are tokenised separately; padding tokens in the label are replaced by `-100` so they are ignored in the loss.
- **Metrics helpers**  
`normalize_answer`, `compute_em_and_f1`, and `evaluate_model` compute Exact-Match and F1 by greedy generation and string comparison.
### Training Configuration
Key knobs are pulled from environment variables: batch size, epochs, gradient-accumulation, FP16/BF16, learning-rate, weight-decay. Nothing related to FSDP is set.
- `output_dir`:`./bloom-qa-finetuned`	Directory to store model checkpoints, logs, and evaluation results.
- `eval_strategy`:`epoch`	Evaluation is run at the end of each training epoch.
- `save_strategy`:`epoch`	A model checkpoint is saved at the end of each training epoch.
- `per_device_train_batch_size`:`BATCH_SIZE`	Batch size to use on each GPU or each CPU worker during training.
- `per_device_eval_batch_size`:`BATCH_SIZE`	Batch size to use on each device during evaluation.
- `gradient_accumulation_steps`:`GRAD_ACC_STEPS`	Number of steps to accumulate gradients before performing a backward/update pass. Simulates a larger effective batch size.
- `num_train_epochs`:`NUM_EPOCHS`	Total number of training epochs (full passes through the dataset).
- `learning_rate`:`LR`	Initial learning rate for the optimizer.
- `weight_decay`:`WEIGHT_DECAY`	Weight decay (L2 penalty) to apply for regularization. Helps prevent overfitting.
- `fp16`:`FP16`	Enables automatic mixed precision (AMP) training for reduced memory usage and faster computation (requires compatible hardware).
-  `bf16`:`BF16`	Enables automatic mixed precision (AMP) training for reduced memory usage and faster computation (requires compatible hardware).
- `gradient_checkpointing`:`False`	Disables gradient checkpointing. If set to True, it reduces memory usage by trading off compute time.
- `push_to_hub`:`False`	Disables automatic pushing of the model and logs to the Hugging Face Hub. Set to True if you want to share your model publicly or privately.

### Training 
The main() in the [baseline.py](./baseline.py) script fine-tunes a causal language model (e.g., BLOOM) using Hugging Face's Trainer API.

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

-   Reserves 1 × A100, 4 CPUs, all GPU memory, 1 h wall-clock.
    
-   Activates `bloom_env`.
    
-   Sets WandB to offline mode and chooses a run name via `$EXPERIMENT_NAME`.
    
-   Runs `python3 baseline.py`.
    
-   After the job it flips WandB back to online and syncs cached runs.

### Bringing It All Together: Running the Baseline Fine-Tuning Experiment

Once all components are in place — model loading, dataset preprocessing, training configuration, and training logic — you can execute the full fine-tuning workflow with minimal manual steps.

Use the baseline.slurm script to submit the training job on a GPU node:

```commandline
sbatch baseline.slurm
```
### Output Artifacts

After the run finishes, you'll find:

- Fine-tuned model and tokenizer in `./bloom-qa-finetuned/`

- Training logs including evaluation metrics and loss curves on wandb

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
| Steps per Epoch                 | _From logs_               |
| Memory Allocated (MB)           | _From logs_     					|
| GPU Utilization (%)             | _From logs_   



### 🔍 ### How to grab the numbers from WandB

> _These instructions assume you ran the job with `WANDB_MODE=offline`, then synchronised the run with `wandb sync --include-offline --sync-all` as shown in the SLURM script._
-   **Open the run page**
    
    -   Go to your WandB workspace and click the run whose name matches `$EXPERIMENT_NAME`.
        
    -   The _Overview_ tab loads by default.
        
-   **Read the numeric metrics**
    
    -   `Trainer` automatically logs
        
        -   `train/loss` (every step) – take the very last value for **Final Train Loss**.
            
        -   `eval/loss` (at the end of each epoch) – copy the first three occurrences for **Eval Loss Epoch **.
            
        -   `train/samples_per_second` – last logged value = **Training Speed**.
            
        -   `eval/samples_per_second` – last logged value = **Evaluation Speed**.
            
        -   `epoch` – the highest integer value is **Steps per Epoch** (Hugging Face writes one log per step, so epoch size is obvious).
            
-   **System metrics captured by WandB Agent**
    
    -   Scroll down to the **System Metrics** panel (right-hand sidebar).
        
        -   `GPU Memory Allocated (Bytes)` (in bytes) → divide by 1 048 576 to get **Memory Allocated (MB)**.
            
        -   `GPU Utilization (%)` (percentage) → mean value shown under the sparkline is **GPU Utilisation (%)**.
            
    -   If you want the true peak memory:
        
        -   Click the metric name so a chart pops up, hover the highest point and note the y-value.

	    

