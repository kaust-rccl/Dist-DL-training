# Parameter-Efficient Fine-Tuning with LoRA (Default PEFT Method in NeMo Factory)

LoRA (Low-Rank Adaptation) is the **default parameter-efficient fine-tuning (PEFT)** technique used by the NVIDIA NeMo Factory recipes. Instead of updating all model weights—which is slow, memory-heavy, and expensive—LoRA injects a pair of small trainable matrices into targeted layers (usually attention and MLP projections).

During training:

- The original model weights are **kept frozen**.

- Only the lightweight low-rank adaptation matrices (**A** and **B**) are trained.

- The final effective weight is `W + ΔW`, where `ΔW = B·A` is the low-rank update.

This approach drastically reduces memory usage and compute requirements, enabling fast and stable fine-tuning even on limited GPU resources.

# Directory Overview
TODO

## Launching NeMo Factory

NeMo Factory provides a command-line interface for running standardized fine-tuning workflows using predefined “factory” recipes. In this workshop, we use it to fine-tune the LLaMA 3.1 8B model with LoRA adapters.

NeMo Factory is always launched through the `nemo` CLI inside a Singularity container to ensure a consistent, reproducible environment across all participants.

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

You can then override any configuration parameter directly on the command line, such as the number of training steps or batch size.

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

NeMo Factory recipes are defined as YAML configurations. Any value in the recipe can be overridden from the command line using dot notation:
```commandline
trainer.max_steps=250 \
data.global_batch_size=8 \
optim.lr_scheduler.warmup_steps=50 \
```
These overrides allow you to quickly modify the training behavior without editing any config files.

### Understanding the `srun` Line

In this experiment, NeMo Factory is launched using the following command:

```commandline
srun singularity exec --nv /ibex/user/x_mohameta/nemo/image/nemo_25.07.sif \
nemo llm finetune \
--factory llama31_8b --yes
trainer.devices="${SLURM_GPUS_PER_NODE}" \
trainer.max_steps=250 \
trainer.limit_val_batches=10 \
optim.lr_scheduler.warmup_steps=50 \
data.global_batch_size=8 \
resume.restore_config.path="$MODEL_DIR" \
```

This single line is the core of the entire script. It performs three layers of execution:

---

### 1. `srun`  
`srun` ensures the command runs **on the specific GPU and CPU resources allocated by SLURM** for this job.

- It binds the process to the allocated node  
- It gives the command access to the 1 GPU requested earlier  
- It handles environment propagation (CUDA paths, SLURM variables, etc.)

Without `srun`, the training might run on the login node or outside the allocated resources.

---

### 2. `singularity exec --nv <image>`  
This starts the NeMo container with GPU access:

- `exec` runs a command inside the container  
- `--nv` forwards NVIDIA libraries and GPUs into the container  
- The container provides a controlled environment with fixed versions of:
  - PyTorch  
  - NeMo  
  - CUDA  
  - Triton kernels  
  - Python dependencies  

This ensures reproducibility across all workshop participants.

---

### 3. `nemo llm finetune ...`  
Inside the container, this launches **NeMo Factory’s LoRA fine-tuning** for LLaMA 3.1 8B.

Key parts:
- `--factory llama31_8b`: loads the official LoRA fine-tuning recipe  
- `trainer.devices="${SLURM_GPUS_PER_NODE}"`: uses the 1 GPU allocated  
- `trainer.max_steps=250`: short demo run  
- `data.global_batch_size=8`: global batch size  
- `resume.restore_config.path="$MODEL_DIR"`: load the pre-downloaded base model  

All training happens *inside this container*, using the factory recipe plus any overrides we specify.

---

### Summary

The `srun` line ties everything together:

- **SLURM** manages the hardware  
- **Singularity** provides a controlled environment  
- **NeMo Factory** performs the LoRA fine-tuning  

It is the single command that launches the entire experiment in a clean, reproducible, GPU-enabled setup.


### Summary

Launching NeMo Factory consists of:

1. Running `nemo llm finetune` inside a Singularity container  
2. Choosing the factory recipe with `--factory`  
3. Optionally overriding recipe parameters on the command line  
4. Using `srun` to ensure your command runs on the SLURM-allocated GPU  
