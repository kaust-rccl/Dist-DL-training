
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

Before introducing sharding we warm-up with a plain Hugging Face run on a single V100. Two files matter:

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

-   Reserves 1 × V100, 4 CPUs, all GPU memory, 30 min wall-clock.
    
-   Activates `bloom_env`.
    
-   Sets WandB to offline mode and chooses a run name via `$EXPERIMENT_NAME`.
    
-   Runs `python baseline.py`.
    
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

##  Exercise: Run the Baseline Training & Fill Evaluation Summary Table

As part of this workshop, your task is to **run the baseline fine-tuning experiment** and **recreate the performance summary table** using your own training logs.

This exercise helps you develop a habit of tracking key metrics like training loss, evaluation loss, and throughput — which are essential for understanding and debugging model training.

## Part 1: Run the Baseline Fine-Tuning Job
### Steps:

 1. Navigate to the baseline directory:  

	```commandline
	cd bloom/baseline
	```
 2. **Modify `env_vars.sh`:**

	Ensure the following variables are correctly set:

	-   **`WANDB_API_KEY`**: Replace with your personal WandB API key.
	    
	-   **`EXPERIMENT_NAME`**: Set a descriptive name for the experiment.
	    
	-   **`LOG_DIR`**: Confirm the path exists and is writable.
    

	Example:
	```commandline
	export WANDB_API_KEY="your_actual_wandb_key"
	export EXPERIMENT_NAME="BLOOM_Baseline"
	export LOG_DIR="/ibex/user/your_username/Dist-DL-training/fsdp/bloom/baseline/logs"
	```
 3. Submit the SLURM job:

	```commandline
	sbatch baseline.slurm
	```
	This script will:

	-   Activate the `bloom_env` Conda environment.
	    
	-   Launch `baseline.py` with the specified configurations.
	    
	-   Log outputs to the designated `LOG_DIR`.

 4. Monitor the job: 
	 After submission, monitor the job's status:
 	```commandline
	squeue --me
	```
	Once completed, proceed to the next step.
	
## 📄 Output Artifacts

Upon job completion, expect the following artifacts:

-   **Model & Tokenizer**: Located in `./bloom-qa-finetuned/`.
    
-   **Training Logs**: Stored in the `logs/` directory, named as `<job_name>_<job_id>.out`.
    
-   **Evaluation Metrics & Loss Curves**: Accessible via WandB (if online mode is enabled) or in the log files.

## 📊 Analyze Logs and Populate Results Table

 1.  **Access the logs:**
	 ```commandline
		cd logs
		cat BLOOM_Baseline_<job_id>.out
		```
	
 2. **Extract Metrics:**

	Look for lines resembling:
	 ```commandline
	{'eval_loss': 1.2965, 'eval_runtime': 1.2687, 'eval_samples_per_second': 39.41, ... 'epoch': 1.0}
	{'eval_loss': 1.5810, 'eval_runtime': 1.2690, 'eval_samples_per_second': 39.403, ... 'epoch': 2.0}
	{'eval_loss': 1.7790, 'eval_runtime': 1.2682, 'eval_samples_per_second': 39.427, ... 'epoch': 2.93}
	{'train_runtime': 125.2999, 'train_samples_per_second': 11.971, 'train_steps_per_second': 0.742, 'train_loss': 0.7039, 'epoch': 2.93}
	```
3. Populate the Results Table:
    | **Metric**                     | **Log Location & Extraction**                             | **Your Value** |
    |--------------------------------|-----------------------------------------------------------|----------------|
    | Train Loss (Final)             | Last `train_loss` in `{'train_loss': ...}`                |                |
    | Eval Loss (Epoch 1)            | First `eval_loss` where `'epoch': 1.0`                    |                |
    | Eval Loss (Epoch 2)            | `eval_loss` where `'epoch': 2.0`                          |                |
    | Eval Loss (Epoch 3)            | Final `eval_loss` (e.g. where `'epoch': 3.0`)            |                |
    | Training Speed (samples/sec)   | `train_samples_per_second` in the final summary           |                |
    | Evaluation Speed (samples/sec) | `eval_samples_per_second` in any eval line (e.g. epoch 1) |                |
    | Steps per Epoch                | From the progress bar.         |                |
    | Memory Allocated (MB)           | _From logs_     					|
	| GPU Utilization (%)             | _From logs_   
	_Note_: Replace the placeholders in the "Your Value" column with the actual values extracted from your logs.




## 📊 Monitoring and Results

### Access Metrics via W&B Dashboard

After completing the training job, you can access detailed metrics through the W&B dashboard:

1.  **Navigate to the W&B Run:**
    
    -   Open your browser and go to [https://wandb.ai](https://wandb.ai).
        
    -   Log in to your account.
        
    -   Navigate to the project associated with your experiment.
        
    -   Click on the run named according to your `EXPERIMENT_NAME`.
        
2.  **View Key Metrics:**
    
    -   In the **Overview** tab, locate the **Summary** section. Here, you'll find aggregated metrics such as:
        
        -   Final training loss
            
        -   Evaluation loss per epoch
            
        -   Samples per second (training and evaluation)
            
        -   Steps per epoch
            
        -   Memory usage
            
        -   GPU utilization
            
3.  **Detailed Metric Analysis:**
    
    -   For a more granular view, explore the **Charts** tab:
        
        -   Select metrics like `train/loss`, `eval/loss`, `train/samples_per_second`, etc.
            
        -   Hover over the plots to see values at specific steps or epochs.
            
        -   Use the zoom and pan tools to focus on particular training phases.
            
4.  **Export Metrics (Optional):**
    
    -   If you wish to analyze metrics outside of W&B:
        
        -   Click on the **Export** button in the **Charts** tab.
            
        -   Choose your preferred format (e.g., CSV) to download the data.
            

By utilizing the W&B dashboard, you gain a comprehensive and interactive view of your training metrics, facilitating easier analysis and comparison across different runs.
# 🚀Multi-GPU Fine-Tuning with FSDP:

## Overview

This section demonstrates how to scale the fine-tuning of the **BLOOM-560M** model on the **SQuAD v1.1** dataset using PyTorch's Fully Sharded Data Parallel (FSDP) across **multiple GPUs** on a single node. By leveraging FSDP, we can efficiently utilize GPU memory and accelerate training.

## Directory Structure

Your project directory is organized as follows:

	multi_gpu/
	├── 2_gpus/
	│   ├── env_vars.sh
	│   ├── multi_gpu.py
	│   ├── multi_gpu.slurm
	├── 4_gpus/
	│   ├── env_vars.sh
	│   ├── multi_gpu.py
	│   ├── multi_gpu.slurm
	├── 8_gpus/
	    ├── env_vars.sh
	    ├── multi_gpu.py
	    ├── multi_gpu.slurm

	
## 🛠️ Step-by-Step Guide

### 1. Navigate to the Desired Node Configuration Directory

For example, to use 2 gpus:

	cd bloom/multi_gpu/2_gpus
### 2. Configure Environment Variables

Edit the `env_vars.sh` file to set up your environment:

	# Conda setup
	export CONDA_SH_PATH="/path/to/miniforge/etc/profile.d/conda.sh"
	export CONDA_ENV="bloom_env"

	# WandB settings
	export EXPERIMENT_NAME="BLOOM_Multi_GPUS_2_GPUs"
	export LOG_DIR="/path/to/multi_gpu/2_gpus/logs"
	export WANDB_API_KEY="your_wandb_api_key"

	# Model and training parameters
	export MODEL_NAME="bigscience/bloom-560m"
	export OUTPUT_DIR="/path/to/multi_gpu/2_gpus/outputs/${EXPERIMENT_NAME}"
	export MAX_LENGTH=512
	export TRAIN_SIZE=500
	export EVAL_SIZE=100
	export NUM_EPOCHS=5
	export BATCH_SIZE=1
	export LEARNING_RATE=5e-5
	export WEIGHT_DECAY=0.01
	export GRAD_ACC=4
	export FP16=True
	export BF16=False
**Note:** Replace `/path/to/` with your actual directory paths and `your_wandb_api_key` with your WandB API key.

### 3. Submit the SLURM Job

Submit the training job using the provided SLURM script:

	sbatch multi_gpu.slurm
	
The `multi_gpu.slurm` script is configured to:

-   Allocate 2 GPUs on a single node.
    
-   Set up the necessary environment variables.
    
-   Launch the training script using `torchrun` for distributed training with FSDP.

## 📄 SLURM Script: `multi_gpu.slurm`

Key configurations in the SLURM script:

	#SBATCH --gpus=8
	#SBATCH --gpus-per-node=8
	#SBATCH --ntasks=2
	#SBATCH --tasks-per-node=2
**Explanation:**

-   `--gpus=8` and `--gpus-per-node=8`: Allocates 8 GPUs on the node. However, we will restrict usage to GPUs 0 and 1.
    
-   `--ntasks=2` and `--tasks-per-node=2`: Launches 2 tasks (processes), one per GPU.


To ensure exclusive usage of the node, set:
	
	export CUDA_VISIBLE_DEVICES=0,1
This restricts the training processes to the specified GPUs.

## ⚙️ FSDP Configuration Parameters

In your training script (`multi_gpu.py`), FSDP is configured as follows:

	fsdp_cfg = {
	    "transformer_layer_cls_to_wrap": ["BloomBlock"],
	    "backward_prefetch": "backward_post",
	    "forward_prefetch": True,
	    "sync_module_states": True
	}

**Explanation of Parameters:**

-   `transformer_layer_cls_to_wrap`: Specifies the transformer layers to wrap with FSDP. In this case, `BloomBlock` layers. 
	- Analysis of the Bloomz-560m architecture reveals that approximately **80%** of the total 560 million parameters are concentrated in the transformer layers (i.e. the BloomBlock layers).
    
-   `backward_prefetch`: Determines when to prefetch the next layer's parameters during the backward pass. `"backward_post"` prefetches after the current layer's backward computation.
    
-   `forward_prefetch`: If set to `True`, enables prefetching of the next layer's parameters during the forward pass.
    
-   `sync_module_states`: If `True`, synchronizes module states (e.g., parameters) from rank 0 to all other ranks at initialization.
    

These configurations optimize memory usage and training efficiency across multiple gpus.

## 🔗 Distributed Training Setup with `torchrun`

Configure the distributed training environment:

	# Distributed setup
	master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
	export MASTER_ADDR=$master_addr
	export MASTER_PORT=29500
	export WORLD_SIZE=2
These environment variables configure distributed training manually.

-   `MASTER_ADDR`: IP or hostname of main node (for NCCL initialization).
    
-   `MASTER_PORT`: Any free port for rendezvous (e.g., 9999; if that is in use, ports like 6000 or 29500 often work).
    
-   `WORLD_SIZE`: Total number of GPUs (set automatically by SLURM).
  
		torchrun \
		  --nnodes=1 \
		  --nproc_per_node=$WORLD_SIZE \
		  --rdzv_backend=c10d \
		  --rdzv_id=$SLURM_JOB_ID \
		  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
		  multi_gpu.py
**Explanation:**

-   `--nnodes=1`: Specifies a single-node setup.
    
-   `--nproc_per_node=$WORLD_SIZE`: Launches one process per GPU.
    
-   `--rdzv_backend=c10d`: Uses the c10d backend for rendezvous.
    
-   `--rdzv_id=$SLURM_JOB_ID`: Unique identifier for the rendezvous.
    
-   `--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}`: Specifies the address and port for rendezvous.




## 🧪 Exercise: Run Multi-GPU Experiments and Populate Results Table
### Steps:

 1.  **Navigate to Each Node Configuration Directory:**
		
			cd bloom/multi_gpu/2_gpus
			sbatch multi_gpu.slurm

			cd ../4_gpus
			sbatch multi_gpu.slurm

			cd ../8_gpus
			sbatch multi_gpu.slurm

 2. **Monitor Jobs:**
 
		squeue --me

 3. **After Completion, Access Logs:**
  
		cd logs
		cat multi_gpu_<gpus>_<job_id>.out
	Replace `<gpus>` with the number of gpus (e.g., `2`, `4`, `8`) and `<job_id>` with the actual SLURM job ID.
	

 4. **Extract Metrics from WandB:**
	Follow the steps outlined in the W&B section to retrieve the necessary metrics.
	
### 📋 Results Table
Populate the following table with the metrics extracted from your experiments:
|          Setup          | Total GPUs | Samples/s | Samples/s Scale Factor | Runtime (s) | Runtime Scale Factor | Loss | GPU Memory Allocated (GB) |
|:-----------------------:|:----------:|:---------:|:----------------------:|:-----------:|:--------------------:|:----:|:-------------------------:|
| Single Node, Single GPU |            |           |                        |             |                      |      |                           |
| Single Node, 2 GPUs     |            |           |                        |             |                      |      |                           |
| Single Node, 4 GPUs     |            |           |                        |             |                      |      |                           |
| Single Node, 8 GPUs     |            |           |                        |             |                      |      |                           |

**Note:** Replace the placeholder values with the actual metrics obtained from your experiments.



## 📝 Notes

-   **FSDP Benefits:** Utilizing FSDP allows for efficient memory usage by sharding model parameters across GPUs, enabling the training of larger models or using larger batch sizes.
    
-   **Scalability:** This setup can be extended to more GPUs or multiple nodes by adjusting the SLURM script and environment variables accordingly.
    
-   **WandB Integration:** Ensure that WandB is properly configured to log training metrics. After training, logs can be synced to the WandB cloud for visualization.

# 🚀Multi-Node Fine-Tuning with FSDP:
## Overview

This section demonstrates how to scale the fine-tuning of the **BLOOM-560M** model on the **SQuAD v1.1** dataset using PyTorch's Fully Sharded Data Parallel (FSDP) across **multiple nodes**. By leveraging FSDP, we can efficiently utilize GPU memory and accelerate training across distributed systems.

## Directory Structure

Your project directory is organized as follows:

	multi_node/
	├── 2_nodes/
	│   ├── env_vars.sh
	│   ├── multi_node.py
	│   ├── multi_node.slurm
	├── 4_nodes/
	│   ├── env_vars.sh
	│   ├── multi_node.py
	│   ├── multi_node.slurm
	├── 8_nodes/
	│   ├── env_vars.sh
	│   ├── multi_node.py
	│   ├── multi_node.slurm

## 🛠️ Step-by-Step Guide

### 1. Navigate to the Desired Node Configuration Directory

For example, to use 2 nodes:

	cd bloom/multi_node/2_nodes

### 2. Configure Environment Variables

Edit the `env_vars.sh` file to set up your environment:

	# Conda setup
	export CONDA_SH_PATH="/path/to/miniforge/etc/profile.d/conda.sh"
	export CONDA_ENV="bloom_env"

	# WandB settings
	export EXPERIMENT_NAME="BLOOM_Multi_Node_2_Nodes"
	export LOG_DIR="/path/to/multi_node/2_nodes/logs"
	export WANDB_API_KEY="your_wandb_api_key"

	# Model and training parameters
	export MODEL_NAME="bigscience/bloom-560m"
	export OUTPUT_DIR="/path/to/multi_node/2_nodes/outputs/${EXPERIMENT_NAME}"
	export MAX_LENGTH=512
	export TRAIN_SIZE=500
	export EVAL_SIZE=100
	export NUM_EPOCHS=5
	export BATCH_SIZE=1
	export LEARNING_RATE=5e-5
	export WEIGHT_DECAY=0.01
	export GRAD_ACC=4
	export FP16=True
	export BF16=False

**Note:** Replace `/path/to/` with your actual directory paths and `your_wandb_api_key` with your WandB API key.

### 3. Submit the SLURM Job

Submit the training job using the provided SLURM script:

	sbatch multi_node.slurm

The `multi_node.slurm` script is configured to:

-   Allocate GPUs across multiple nodes.
    
-   Set up the necessary environment variables.
    
-   Launch the training script using `torchrun` for distributed training with FSDP.

## SLURM Configuration

To allocate resources across multiple nodes, include the following directives in your SLURM script:

	#SBATCH --gpus=2
	#SBATCH --gpus-per-node=1
	#SBATCH --ntasks=2
	#SBATCH --tasks-per-node=1

-   `--gpus=2`: Allocates two gpus for the job.
-   `--gpus-per-node=1`: Allocates one GPU per node.
-   `--tasks-per-node`: Launches one task per node.

## ⚙️ FSDP Configuration Parameters

In your training script (`multi_node.py`), FSDP is configured as follows:

	fsdp_cfg = {
	    "transformer_layer_cls_to_wrap": ["BloomBlock"],
	    "backward_prefetch": "backward_post",
	    "forward_prefetch": True,
	    "sync_module_states": True
	}
**Explanation of Parameters:**

-   `transformer_layer_cls_to_wrap`: Specifies the transformer layers to wrap with FSDP. In this case, `BloomBlock` layers.
    
    -   Analysis of the Bloomz-560m architecture reveals that approximately **80%** of the total 560 million parameters are concentrated in the transformer layers (i.e., the BloomBlock layers).
        
-   `backward_prefetch`: Determines when to prefetch the next layer's parameters during the backward pass. `"backward_post"` prefetches after the current layer's backward computation.
    
-   `forward_prefetch`: If set to `True`, enables prefetching of the next layer's parameters during the forward pass.
    
-   `sync_module_states`: If `True`, synchronizes module states (e.g., parameters) from rank 0 to all other ranks at initialization.
    

These configurations optimize memory usage and training efficiency across multiple nodes.

## Environment Setup

Before launching the training script, set up the environment variables necessary for distributed training:

	master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
	export MASTER_ADDR=$master_addr
	export MASTER_PORT=29500
	export WORLD_SIZE=$SLURM_JOB_NUM_NODES

**Explanation:**

-   `MASTER_ADDR`: Specifies the address of the master node for process synchronization.
    
-   `MASTER_PORT`: Designates the port for communication.
    
-   `WORLD_SIZE`: Indicates the total number of processes across all nodes.
    
-   `NODE_RANK`: Defines the rank of the current node.
## 🔗 Distributed Training Setup

Use `srun` in conjunction with `torchrun` to initiate the distributed training:

	srun torchrun \
	  --nnodes=$WORLD_SIZE \
	  --nproc_per_node=1 \
	  --node_rank=$SLURM_NODEID \
	  --rdzv_backend=c10d \
	  --rdzv_id=$SLURM_JOB_ID \
	  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
	  multi_node.py
**Explanation:**

-   `--nnodes`: Total number of nodes participating in the job.
    
-   `--nproc_per_node`: Number of processes to launch on each node.
    
-   `--node_rank`: Rank of the current node.
    
-   `--rdzv_backend`: Backend used for rendezvous (`c10d` is commonly used).
    
-   `--rdzv_endpoint`: Specifies the rendezvous endpoint using the master address and port.

## 🧪 Exercise: Run Multi-Node Experiments and Populate Results Table

### Steps:

 1.  **Navigate to Each Node Configuration Directory:**

			cd bloom/multi_node/2_nodes
			sbatch multi_node.slurm

			cd ../4_nodes
			sbatch multi_node.slurm

			cd ../8_nodes
			sbatch multi_node.slurm

 2. **Monitor Jobs:**
 
		squeue --me

 3. After Completion, Access Logs:
 
		cd logs
		cat multi_node_<nodes>_<job_id>.out
		
 4.   Replace `<nodes>` with the number of nodes (e.g., `2`, `4`, `8`) and `<job_id>` with the actual SLURM job ID.
    

 5. **Extract Metrics from WandB:**

    
    Follow the steps outlined in the W&B section to retrieve the necessary metrics.
    
### 📋 Results Table
Populate the following table with the metrics extracted from your experiments:
|          Setup          | Total GPUs | Samples/s | Samples/s Scale Factor | Runtime (s) | Runtime Scale Factor | Loss | GPU Memory Allocated (GB) |
|:-----------------------:|:----------:|:---------:|:----------------------:|:-----------:|:--------------------:|:----:|:-------------------------:|
| Single Node|            |           |                        |             |                      |      |                           |
| 2 Node     |            |           |                        |             |                      |      |                           |
| 4 Node     |            |           |                        |             |                      |      |                           |
| 8 Node     |            |           |                        |             |                      |      |                           |

**Note:** Replace the placeholder values with the actual metrics obtained from your experiments.

# Custom Model Fine-Tuning (Single Node)

## Fine-Tuning Setup

This document details the fine-tuning of a custom GPT-like model, implemented using PyTorch, on the SQuAD v1.1 dataset.

### Differences from Hugging Face (Baseline)

-   **Model Implementation:** Instead of using a Hugging Face pretrained model (like BLOOM), this setup uses a custom GPT model (`SimpleGPTModel`) built from scratch.

-   **Architectural Parameters:** Explicitly defined using environment variables (vocabulary size, hidden size, layers, etc.).
    
-   **Training Logic:** Utilizes Hugging Face's `Trainer` API similarly, ensuring consistent training and evaluation methodology.

## Model Parameters Explained

-   `VOCAB_SIZE`: Defines the size of the tokenizer vocabulary.
    
-   `HIDDEN_SIZE`: Sets the dimension of embeddings and hidden states in transformer layers.
    
-   `NUM_LAYERS`: Determines the depth of the transformer stack.
    
-   `NUM_HEADS`: Number of parallel attention mechanisms within each transformer layer.
    
-   `FF_DIM`: Width of the feed-forward network in transformer layers.
    
-   `SEQ_LENGTH`: Maximum length of input sequences.

## Directory Structure
	single_node/
	├── env_vars.sh
	├── single_node.py
	├── single_node.slurm

## Exercise: Run the Custom Model Fine-Tuning Job

### Steps:

 1.  Navigate to the single node directory:

		cd custom_model/single_node

 2.  Modify `env_vars.sh`:
    

		Ensure variables like `WANDB_API_KEY`, `EXPERIMENT_NAME`, `LOG_DIR`, and model parameters are correctly set.

 3. Submit the SLURM job:

		sbatch single_node.slurm

 4. Monitor the job:

		squeue --me
	
### Output Artifacts

Post-training, you'll find:

-   Trained model and logs in `OUTPUT_DIR`
    
-   Training logs (`.out` and `.err` files) in `logs/`
    
-   Evaluation metrics accessible via WandB

### Results Table

Populate after experiment:

   | **Metric**                     | **Log Location & Extraction**                             | **Your Value** |
   |--------------------------------|-----------------------------------------------------------|----------------|
   | Train Loss (Final)             | Last `train_loss` in `{'train_loss': ...}`                |                |
   | Eval Loss (Epoch 1)            | First `eval_loss` where `'epoch': 1.0`                    |                |
   | Eval Loss (Epoch 2)            | `eval_loss` where `'epoch': 2.0`                          |                |
   | Eval Loss (Epoch 3)            | Final `eval_loss` (e.g. where `'epoch': 3.0`)            |                |
   | Training Speed (samples/sec)   | `train_samples_per_second` in the final summary           |                |
   | Evaluation Speed (samples/sec) | `eval_samples_per_second` in any eval line (e.g. epoch 1) |                |
   | Steps per Epoch                | From the progress bar.         |                |
   | Memory Allocated (MB)           | _From logs_     					|
| GPU Utilization (%)             | _From logs_   

# Custom Model Fine-Tuning (Multi-GPU with FSDP)
## Overview

This section demonstrates how to scale the fine-tuning of the **custom GPT-like model** on the **SQuAD v1.1** dataset using PyTorch's Fully Sharded Data Parallel (**FSDP**) across **multiple GPUs** on a single node. By leveraging FSDP, we efficiently utilize GPU memory and accelerate training.

## Directory Structure

Your project directory is organized as follows:

	custom_model/multi_gpu/
	├── 2_gpus/
	│   ├── env_vars.sh
	│   ├── multi_gpu.py
	│   ├── multi_gpu.slurm
	├── 4_gpus/
	│   ├── env_vars.sh
	│   ├── multi_gpu.py
	│   ├── multi_gpu.slurm
	├── 8_gpus/
	    ├── env_vars.sh
	    ├── multi_gpu.py
	    ├── multi_gpu.slurm

## 🛠️ Step-by-Step Guide

### 1. Navigate to the Desired GPU Configuration Directory

Example, using 2 GPUs:

	cd custom_model/multi_gpu/2_gpus

### 2. Configure Environment Variables

Edit `env_vars.sh` to set up your environment:

	# Conda setup
	export CONDA_SH_PATH="/path/to/miniforge/etc/profile.d/conda.sh"
	export CONDA_ENV="bloom_env"

	# WandB settings
	export EXPERIMENT_NAME="Custom_Model_Multi_GPUS_2_GPUs"
	export LOG_DIR="/path/to/multi_gpu/2_gpus/logs"
	export WANDB_API_KEY="your_wandb_api_key"

	# Model and training parameters
	export OUTPUT_DIR="/path/to/multi_gpu/2_gpus/outputs/${EXPERIMENT_NAME}"
	export MAX_LENGTH=512
	export TRAIN_SIZE=500
	export EVAL_SIZE=100
	export NUM_EPOCHS=5
	export BATCH_SIZE=1
	export LEARNING_RATE=5e-5
	export WEIGHT_DECAY=0.01
	export GRAD_ACC=4
	export FP16=True
	export BF16=False

	# Model architecture parameters
	export VOCAB_SIZE=50000          # tokenizer vocabulary size
	export HIDDEN_SIZE=2048          # embedding & transformer hidden size
	export NUM_LAYERS=3              # transformer layers depth
	export NUM_HEADS=16              # attention heads per layer
	export FF_DIM=8192               # transformer feed-forward dimension
	export SEQ_LENGTH=512            # max sequence length
**Note:** Replace `/path/to/` and `your_wandb_api_key` with your actual paths and WandB API key.

### 3. Submit the SLURM Job

Submit the job using:

	sbatch multi_gpu.slurm	

The `multi_gpu.slurm` script will:

-   Allocate the desired GPUs.
    
-   Configure environment variables.
    
-   Launch training with FSDP.

## 📄 SLURM Script (`multi_gpu.slurm`)

Key configurations:

	#SBATCH --gpus-per-node=8
	#SBATCH --ntasks=2
	#SBATCH --tasks-per-node=2

**Explanation:**

-   Allocates GPUs across processes.
    
-   To restrict to specific GPUs (e.g., GPUs 0 and 1):

		export CUDA_VISIBLE_DEVICES=0,1

## ⚙️ FSDP Configuration Parameters

In your training script (`multi_gpu.py`), FSDP configuration is set as:

	fsdp_cfg = {
	    "transformer_layer_cls_to_wrap": ["torch.nn.modules.transformer.TransformerEncoderLayer"],
	    "backward_prefetch": "backward_post",
	    "forward_prefetch": True,
	    "sync_module_states": True,
	}

### Explanation of Parameters:

-   **`transformer_layer_cls_to_wrap`**:
    
    -   Targets transformer layers for sharding (memory optimization).
        
-   **`backward_prefetch`**:
    
    -   Pre-fetches parameters after each layer’s backward pass for efficient GPU communication.
        
-   **`forward_prefetch`**:
    
    -   Pre-fetches parameters ahead of forward computation for faster training.
        
-   **`sync_module_states`**:
    
    -   Synchronizes initial parameters across all GPU ranks.

## 🔗 Distributed Training Setup with `torchrun`

Configure distributed training:

	master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
	export MASTER_ADDR=${master_addr}
	export MASTER_PORT=29500
	export WORLD_SIZE=2  # Number of GPUs/processes

	torchrun \
	  --nnodes=1 \
	  --nproc_per_node=$WORLD_SIZE \
	  --rdzv_backend=c10d \
	  --rdzv_id=$SLURM_JOB_ID \
	  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
	  multi_gpu.py

**Explanation:**

-   Launches distributed training processes on GPUs within a single node.

## 🛎️ Integrated Custom Callbacks

This setup includes custom callbacks for monitoring:

-   **`UtilisationCallback`**: logs GPU memory usage and utilization.
    
-   **`TrainSpeedCallback`**: measures steps and samples per second.
    

Callback metrics (such as GPU utilization, memory allocation, samples/sec) are recorded in the WandB **Run Summary** after job completion.

## 🧪 Exercise: Run Multi-GPU Experiments and Populate Results Table

### Steps:

 1.  Navigate and submit jobs for each GPU configuration:
		
			cd custom_model/multi_gpu/2_gpus
			sbatch multi_gpu.slurm

			cd ../4_gpus
			sbatch multi_gpu.slurm

			cd ../8_gpus
			sbatch multi_gpu.slurm

 2. Monitor job completion:

		squeue --me

 3. Access Logs after Completion:

		cd logs
		cat custom_multi_gpu_<gpus>_<job_id>.out

	Replace `<gpus>` and `<job_id>` accordingly.

4.  **Extract Metrics from WandB**:
    

-   Default metrics: via WandB run dashboard.
    
-   Callback metrics: specifically under WandB **Run Summary** (e.g., `avg_gpu_util_%`, `train/samples_per_second_mean`, `avg_mem_alloc_MB`).

### 📋 Results Table

Fill with metrics obtained from WandB and callbacks:


|          Setup          | Total GPUs | Samples/s (Default) | Samples/s (Callback) | GPU Memory (GB, Default) | GPU Memory (GB, Callback) | GPU Utilization (%) | Loss | Runtime (s) |
|:-----------------------:|:----------:|:-------------------:|:--------------------:|:------------------------:|:-------------------------:|:-------------------:|:----:|:-----------:|
| Single Node, Single GPU | 1          |                     |                      |                          |                           |                     |      |             |
| Single Node, 2 GPUs     | 2          |                     |                      |                          |                           |                     |      |             |
| Single Node, 4 GPUs     | 4          |                     |                      |                          |                           |                     |      |             |
| Single Node, 8 GPUs     | 8          |                     |                      |                          |                           |                     |      |             |

**Note:** Clearly compare values logged by default WandB metrics versus custom callback values.

# 🚀 Multi-Node Fine-Tuning with FSDP (Custom Model):

## Overview

This section demonstrates how to scale fine-tuning of a **custom GPT-like model** on the **SQuAD v1.1** dataset using PyTorch's Fully Sharded Data Parallel (**FSDP**) across **multiple nodes**. By leveraging distributed training with FSDP, we significantly enhance memory efficiency and accelerate training performance.

## Directory Structure

Your project directory is organized as follows:

	custom_model/multi_node/
	├── 2_nodes/
	│   ├── env_vars.sh
	│   ├── multi_node.py
	│   ├── multi_node.slurm
	├── 4_nodes/
	│   ├── env_vars.sh
	│   ├── multi_node.py
	│   ├── multi_node.slurm
	├── 8_nodes/
	    ├── env_vars.sh
	    ├── multi_node.py
	    ├── multi_node.slurm

## 🛠️ Step-by-Step Guide

### 1. Navigate to the Desired Node Configuration Directory

For example, to use 2 nodes:

	cd custom_model/multi_node/2_nodes

### 2. Configure Environment Variables (`env_vars.sh`):

Set up your training environment in the env_vars.sh file

### 3. Submit the SLURM Job (`multi_node.slurm`):

Submit the training job:

	sbatch multi_node.slurm`

The script will:

-   Allocate nodes and GPUs.
    
-   Test inter-node NCCL bandwidth for optimal distributed performance.
    
-   Launch the training script using `torchrun`.


## 📄 SLURM Script (`multi_node.slurm`):

Key configurations:

	#SBATCH --gpus=16
	#SBATCH --gpus-per-node=8
	#SBATCH --ntasks=2
	#SBATCH --tasks-per-node=1

**Explanation**:

-   Allocates 8 GPUs per node, 2 nodes total.
    

### NCCL Bandwidth Test (before training):

Before initiating training, an NCCL bandwidth test validates cluster interconnect:

	module purge
	module load cuda/11.8 nccl/2.17.1-cuda11.8 openmpi/4.1.4/gnu11.2.1-cuda11.8

	# NCCL diagnostic flags (for optimal multi-node performance)
	export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Quickly fail if hangs occur
	export NCCL_ALGO=Tree                       # Stable performance across nodes
	export NCCL_NET_GDR_LEVEL=4                 # Allow GPU Direct RDMA communication
	export NCCL_IB_HCA=mlx5                     # Select Mellanox InfiniBand adapters

	# Run NCCL performance test
	LOG="logs/nccl_${SLURM_JOB_ID}.log"
	srun --cpu-bind=none --ntasks-per-node=1 --gpus-per-task=1 \
	    all_reduce_perf -b 4G -e 4G -f 2 -g 1 -c 0 -n 50 -w 20 > "$LOG" 2>&1

	# Parse NCCL results & validate performance
	BW=$(awk '/^# Avg bus bandwidth/ {print $(NF)}' "$LOG")
	if [[ -z "$BW" ]]; then
	    echo "NCCL bandwidth parse failed – check $LOG"; exit 44
	fi
	echo "Measured AllReduce BW: $BW GB/s"
	if (( $(echo "$BW < 5" | bc -l) )); then
	    echo "Bandwidth below threshold – aborting"; exit 46
	fi

This test ensures high-quality GPU interconnection before starting distributed training.

### Launch Distributed Training (`torchrun`):

Set environment and initiate multi-node distributed training:

	module purge
	export NCCL_SOCKET_IFNAME=ib0
	export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4
	export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
	export TORCH_NCCL_BLOCKING_WAIT=1

	# Distributed setup
	master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
	export MASTER_ADDR=${master_addr}
	export MASTER_PORT=29500
	export WORLD_SIZE=$SLURM_JOB_NUM_NODES

	# Launch training across nodes
	srun --cpu-bind=none --nodes=$SLURM_NNODES --ntasks-per-node=1 \
	    torchrun \
	    --nnodes=$SLURM_JOB_NUM_NODES \
	    --nproc_per_node=1 \
	    --node_rank=$SLURM_NODEID \
	    --rdzv_backend=c10d \
	    --rdzv_id=$SLURM_JOB_ID \
	    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
	    multi_node.py

`torchrun` handles inter-node synchronization and training initialization.

## 🧪 Exercise: Run Multi-Node Experiments and Populate Results Table

### Steps:

 1.  Run experiments for multiple node configurations:

			cd custom_model/multi_node/2_nodes
			sbatch multi_node.slurm

			cd ../4_nodes
			sbatch multi_node.slurm

			cd ../8_nodes
			sbatch multi_node.slurm

 2.  Monitor job progress:

			squeue --me

 3. After completion, check logs:
		
		cd logs
		cat multi_node_custom_<nodes>_<job_id>.out

	Replace placeholders accordingly.

4.  Extract metrics from WandB (default logs and callback summaries).

### 📋 Results Table:

Populate with obtained metrics:

|          Setup          | Total GPUs | Samples/s (Default) | Samples/s (Callback) | GPU Memory (GB, Default) | GPU Memory (GB, Callback) | GPU Utilization (%) | Loss | Runtime (s) |
|:-----------------------:|:----------:|:-------------------:|:--------------------:|:------------------------:|:-------------------------:|:-------------------:|:----:|:-----------:|
| Single Node, Single GPU | 1          |                     |                      |                          |                           |                     |      |             |
| 2 Node    | 2          |                     |                      |                          |                           |                     |      |             |
| 4 Node  | 4          |                     |                      |                          |                           |                     |      |             |
| 8 Node  | 8          |                     |                      |                          |                           |                     |      |             |

**Note:**

-   Compare **WandB default metrics** with **callback metrics** for detailed analysis.
    
-   Callback metrics (`avg_gpu_util_%`, `avg_mem_alloc_MB`, `train/samples_per_second_mean`) are found under WandB **Run Summary**.

## 📝 Notes:

-   **NCCL Flags** explained:
    
    -   `TORCH_NCCL_ASYNC_ERROR_HANDLING`: Ensures immediate detection of communication errors.
        
    -   `NCCL_ALGO=Tree`: Provides stable communication performance across multiple nodes.
        
    -   `NCCL_NET_GDR_LEVEL=4`: Enables direct GPU communication via InfiniBand.
        
    -   `NCCL_IB_HCA=mlx5`: Chooses optimal Mellanox InfiniBand adapters.
        
-   The NCCL bandwidth test helps avoid suboptimal interconnect scenarios that degrade distributed performance.
    
-   Custom callbacks provide insightful, detailed metrics crucial for analyzing GPU utilization and scalability across nodes.


# 🚀 Weak Scaling Experiment with FSDP (Custom Model)

## Overview


This section demonstrates **weak scaling** using the **custom GPT-like model** on the **SQuAD v1.1** dataset. In weak scaling, as the number of GPUs doubles, the **model size (parameters)** doubles by increasing the number of layers, while the **per-device batch size** is halved. This method keeps the total computational load per GPU constant, ideally maintaining the same runtime while scaling up.  

Weak scaling helps identify how efficiently your training setup scales when the workload per GPU is kept constant.

## 🔍 Weak Scaling Setup:

For this weak scaling experiment:

-   **Total Batch Size** remains fixed across experiments:
	
		Total Batch Size=Num GPUs×Per Device Batch Size

	Thus, as GPUs double, the per-device batch size is halved.

-   **Model Complexity** increases proportionally with GPU count by doubling the number of transformer layers:
			
		Num Layers (GPUs)=Base Layers × (Num GPUs​/Base GPUs)
   
	For example, if base layers are 3 at 2 GPUs, the number of layers at 4 GPUs becomes 6, at 8 GPUs becomes 12, and so forth.


## 📁 Directory Structure:

	custom_model/weak_scaling/
	├── 2_gpus/
	│   ├── env_vars.sh
	│   ├── weak_scaling.py
	│   ├── weak_scaling.slurm
	├── 4_gpus/
	│   ├── env_vars.sh
	│   ├── weak_scaling.py
	│   ├── weak_scaling.slurm
	├── 8_gpus/
	│   ├── env_vars.sh
	│   ├── weak_scaling.py
	│   ├── weak_scaling.slurm
	├── 16_gpus/
	    ├── env_vars.sh
	    ├── weak_scaling.py
	    ├── weak_scaling.slurm

## 🛠️ Step-by-Step Guide:

### 1. Navigate to GPU Configuration:

For example, starting with 2 GPUs:

	cd custom_model/weak_scaling/2_gpus`

### 2. Configure Environment Variables (`env_vars.sh`):

Set environment parameters clearly:
	
	export BATCH_SIZE=8                  # Adjust per weak scaling rules
	
	export NUM_LAYERS=3                  # doubled each scaling step

### 3. Submit the SLURM Job (`weak_scaling.slurm`):

Submit the job:

	sbatch weak_scaling.slurm
This SLURM script handles GPU allocation, environment setup, and distributed training initialization.


## ⚙️ Weak Scaling Equations Summary


| GPUs | Per-GPU Batch Size | Number of Layers |   |   |   |   |   |   |
|:----:|:------------------:|:----------------:|:-:|:-:|:-:|:-:|:-:|:-:|
| 2    | 8                  | 3                |   |   |   |   |   |   |
| 4    | 4                  | 6                |   |   |   |   |   |   |
| 8    | 2                  | 12               |   |   |   |   |   |   |
| 16   | 1                  | 24               |

## 🔍 Exercise: Run Weak Scaling Experiments and Populate Results

### Steps:

 1.  Run experiments for each scaling scenario:
		
			cd custom_model/weak_scaling/2_gpus
			sbatch weak_scaling.slurm

			cd ../4_gpus
			sbatch weak_scaling.slurm

			cd ../8_gpus
			sbatch weak_scaling.slurm

			cd ../16_gpus
			sbatch weak_scaling.slurm

 2. Monitor progress:

		squeue --me

 3. After completion, access logs:

		cd logs
		cat custom_weak_scaling_<gpus>_<job_id>.out

Replace placeholders accordingly.

4.  **Extract Metrics from WandB** (default and callbacks):
    

-   WandB default metrics: Dashboard
    
-   Callback metrics: WandB **Run Summary**

### 📋 Results Table for Weak Scaling Experiment

Fill the following table:


| GPUs | Num Layers | Per-GPU Batch | Total Params | Samples/s (Callback) | GPU Memory (GB, Callback) | GPU Utilization (%) | Loss | Runtime (s) |  Efficiency (%) |
|:----:|:----------:|:-------------:|:------------:|:--------------------:|:-------------------------:|:-------------------:|:----:|:-----------:|:---------------:|
| 2    | 3          | 8             |              |                      |                           |                     |      |             | 100% (baseline) |
| 4    | 6          | 4             |              |                      |                           |                     |      |             |                 |
| 8    | 12         | 2             |              |                      |                           |                     |      |             |                 |
| 16   | 24         | 1             |              |                      |                           |                     |      |             |                 |

**Efficiency** calculation (ideal):

	Efficiency(%) = (Current Runtime / Baseline Runtime)​×100

## 📖 Notes:

-   **Weak Scaling Concept** clearly demonstrates scalability and overhead associated with increased inter-GPU communication.
    
-   **Callback Metrics** provide detailed insights into GPU utilization, throughput, and memory usage.
    
-   The ideal efficiency of 100% means perfect scaling (no overhead). Deviations indicate overheads from communication, synchronization, and parameter sharding.

