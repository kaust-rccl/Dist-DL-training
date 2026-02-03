# Dist-DL-training

Welcome to **Dist-DL-training** — a hands-on repository used in the Distributed Deep Learning training sessions by the
KAUST Supercomputing Core Lab (KSL).

This repository is organized as **independent training modules**, each focusing on a specific distributed training
framework or workflow. The goal is to help you understand *how* distributed training works, *when* to use each approach,
and *how* to run it efficiently on IBEX.

---

## Quick start

1. **Complete the prerequisites**
    - Start with the instructions in [`prerequisites/`](./prerequisites)

2. **Choose a training module**
    - New to distributed training? Start with [`ddp/`](ddp)
    - Interested in memory sharding and optimizer scaling? Try [`fsdp/`](./fsdp) or [`deepspeed/`](./deepspeed)
    - Prefer config-driven, recipe-style training? Go to [`nemo/`](./nemo)

---

## Repository navigation

### Training modules

Each directory below represents a self-contained learning module:

- [`ddp/`](./ddp)

  PyTorch Distributed Data Parallel (DDP) examples.  
  Focuses on the core concepts of ranks, world size, data sharding, and gradient synchronization.

- [`deepspeed/`](./deepspeed)

  DeepSpeed training examples.  
  Demonstrates optimizer and parameter sharding (ZeRO stages), memory offloading, and large-model scaling.

- [`fsdp/`](./fsdp)

  Fully Sharded Data Parallel (FSDP) examples.  
  Explores parameter sharding, communication trade-offs, and memory efficiency.

- [`horovod/`](./horovod)  
  Horovod-based distributed training examples.

- [`jax/`](./jax)

  Distributed training examples using JAX.

- [`lightning/`](./lightning)

  PyTorch Lightning distributed workflows with reduced boilerplate.

- [`nemo/`](./nemo)

  NVIDIA NeMo recipe-based training examples, emphasizing configuration-driven workflows and standardized launch
  patterns.

---

## What to expect inside each module

While details vary per framework, most modules contain:

- Runnable training scripts or notebooks
- SLURM job scripts for single-node and multi-node runs
- Configuration files (YAML / JSON)
- Module-specific README files with usage instructions

Each module is designed to be runnable independently once prerequisites are completed.


