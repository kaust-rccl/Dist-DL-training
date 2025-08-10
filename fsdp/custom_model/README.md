# Custom GPT-like Model Fine-Tuning Experiments

This directory contains scripts and configurations for fine-tuning a custom GPT-like PyTorch model on the **SQuAD v1.1** dataset with various **Fully-Sharded Data Parallel (FSDP)** configurations and scaling setups.

## Subdirectories:

- **`single_node/`**:  
  Scripts for single-GPU fine-tuning of the custom model without distributed scaling.

- **`multi_gpu/`**:  
  Scripts to fine-tune the custom model across multiple GPUs within a single node using FSDP.

- **`multi_node/`**:  
  Scripts for scaling fine-tuning across multiple nodes using distributed training and FSDP.

- **`weak_scaling/`**:  
  Scripts for performing weak scaling experiments, adjusting model size and batch size as the number of GPUs increases.
