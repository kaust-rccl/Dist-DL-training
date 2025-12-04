# GPU Node Sanity Checks

These jobs confirm that participants can:
1. Allocate GPU nodes (V100 and A100)
2. Reach their allocated nodes successfully
3. Detect all available GPUs via `nvidia-smi`

---

## Directory Contents

| File                                                       | Purpose                                     |
|------------------------------------------------------------|---------------------------------------------|
| [`single_v100_gpu.slurm`](single_v100_gpu.slurm)           | Run on 1× V100 GPU (single node)            |
| [`multi_v100_gpus.slurm`](multi_v100_gpus.slurm)           | Run on multiple V100 GPUs (same node)       |
| [`multi_v100_nodes.slurm`](multi_v100_nodes.slurm)         | Run on multiple V100 nodes                  |
| [`single_a100_gpu.slurm`](single_a100_gpu.slurm)           | Run on 1× A100 GPU (single node)            |
| [`multi_a100_gpus.slurm`](multi_a100_gpus.slurm)           | Run on multiple A100 GPUs (same node)       |
| [`multi_a100_nodes.slurm`](multi_a100_nodes.slurm)         | Run on multiple A100 nodes                  |
| [`gpu_nodes_sanity_checks.sh`](gpu_nodes_sanity_checks.sh) | Launcher that submits all of the above jobs |

---
## Run GPU Checks

### Running the Checks
- From inside this directory:
    ```bash
    chmod +x gpu_sanity_checks.sh
    ./gpu_sanity_checks.sh
    ```

- The launcher submits six SLURM jobs sequentially:
    ```commandline
    single_v100_gpu.slurm
    multi_v100_gpus.slurm
    multi_v100_nodes.slurm
    single_a100_gpu.slurm
    multi_a100_gpus.slurm
    multi_a100_nodes.slurm
    ```
  
- Each job logs its output under:
    ```commandline
    log/%x-%j.out
    ``` 
### Reviewing Results

- Monitor jobs while they run:
  ```commandline
  squeue --me
  ```
- Inspect all logs once finished:
    ```
    ls -lt log/
    cat ./*
    ```
- Each log prints a structured progress report:
    ```
  ───────────────────────────────────────────────
   Pre-Workshop Sanity Check: <Check_Name>
  ───────────────────────────────────────────────
  [1/4] Checking SLURM allocation...   [✓]
  [2/4] Per-node hostname verification [✓]
  [3/4] Verifying GPU visibility...    [✓]
  [4/4] Summary: all nodes reachable, GPUs visible
  ───────────────────────────────────────────────
  ```
  
### Success Criteria

A participant passes the sanity checks when:

  - **All logs** show `[✓]` for allocation, reachability, and GPU visibility

  - No `[✗]` markers or SLURM step errors appear

  - `nvidia-smi` lists the expected GPU models (V100 or A100)