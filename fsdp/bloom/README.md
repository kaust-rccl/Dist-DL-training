# BLOOM-560M Fine-Tuning Experiments

This directory contains scripts and configurations for fine-tuning the **BLOOM-560M** language model on the **SQuAD v1.1** dataset with **Fully-Sharded Data Parallel (FSDP)** scaling.

## Subdirectories:

- **`baseline/`**:  
  Scripts for baseline single-GPU fine-tuning without FSDP.

- **`multi_gpu/`**:  
  Scripts for scaling fine-tuning on multiple GPUs within a single node using FSDP.

- **`multi_node/`**:  
  Scripts for scaling fine-tuning across multiple nodes with FSDP and distributed training setups.
