# Distributed Data Parallel (DDP) — TinyImageNet + ResNet-50

## Overview

This exercise introduces the most fundamental form of distributed training in PyTorch: **Distributed Data Parallel (DDP)
**.
DDP is the base mechanism that all other distributed training strategies in this workshop build on, including FSDP,
ZeRO,
tensor parallelism, and expert parallelism.

In this experiment, we train a **ResNet-50** model on the **TinyImageNet** dataset using:

- **One process per GPU**
- The **NCCL** backend for GPU-to-GPU communication
- The **DistributedDataParallel (DDP) wrapper** for synchronizing gradients
- **Mixed precision (AMP)** for faster training
- A **StepLR** scheduler for learning-rate decay

This baseline serves as the reference point for understanding:

- How multi-GPU data parallel training works
- How datasets are split across GPUs
- How gradients are synchronized across GPUs and across nodes
- How training speed and memory usage change as GPU count increases

Everything that follows in the workshop builds on these ideas.

---

## How DDP Works (Conceptual Overview)

### Data Parallelism at a Glance

Data Parallelism means every GPU holds a **full copy of the model**, but each GPU trains on a **different slice of the
dataset**.  
Each GPU performs its own forward pass, computes its own loss, and generates its own gradients.  
Afterward, gradients are synchronized so that **all model replicas remain identical**.  
This scales training by using more GPUs without changing model capacity.

### One Process per GPU

DDP launches **one training process per GPU**. Each process:

- Owns **exactly one GPU**
- Loads its **own model replica**
- Receives **its own shard** of the dataset
- Computes **local gradients**

### Distributed Initialization (NCCL + Process Groups)

Before training begins, all processes must join a shared communication group. This is handled by:

- **NCCL** as the backend for GPU-to-GPU communication
- A **master address** and **master port** for rendezvous
- Environment variables like **RANK**, **WORLD_SIZE**, and **LOCAL_RANK**
- A global **process group**, created with `dist.init_process_group`

Once initialized, processes can use collective communication operations such as broadcast, reduce, and all-reduce.

### Gradient Synchronization (AllReduce)

After each backward pass, DDP automatically triggers an **AllReduce** operation across all GPUs:

1. Each GPU contributes its local gradients
2. All gradients are **summed** across GPUs
3. The sum is **divided by the number of GPUs**
4. Every process receives the **same averaged gradients**

This ensures every model replica applies **identical weight updates**, keeping the models synchronized.

### Distributed Samplers & Dataset Sharding

DDP requires datasets to be **sharded** across processes. `DistributedSampler` ensures:

- GPU 0 gets shard 0
- GPU 1 gets shard 1
- and so on

No two GPUs see the same samples within the same epoch.  
On every new epoch, the sampler reshuffles deterministically across processes.

### Per-GPU vs Global Batch Size

DDP defines batch size at two levels:

- **Per-GPU batch size:** number of samples each process sees
- **Global batch size:** `per_gpu_batch_size * world_size`

Global batch size affects learning rate and convergence.  
Per-GPU batch size affects memory and speed.

### How Multi-Node DDP Works

Multi-node DDP extends the same mechanism across nodes:

- Slurm allocates multiple nodes
- Each node launches several processes (one per GPU)
- One process becomes the **master** for rendezvous
- All processes join the same **global process group**
- NCCL synchronizes gradients across nodes using **InfiniBand** and **NVLink** when available

From the training script’s perspective, multi-node DDP behaves exactly like single-node DDP—just with a larger world
size.

---

## Experiment Setup

### Directory Structure

```text
ddp/                        # Root directory for all DDP training materials and scripts
├── environment.yml         # Conda environment file with all required Python dependencies
├── experiments             # Collection of Slurm job scripts for running DDP at different scales
│   ├── baseline            # Single-GPU, single-node baseline experiment
│   │   └── baseline.slurm  # Slurm script for the 1 GPU baseline run
│   ├── multi_gpu           # Experiments scaling up GPUs on a single node
│   │   ├── 2_gpus          # 2-GPU single-node experiment
│   │   │   └── multi_gpu.slurm
│   │   ├── 4_gpus          # 4-GPU single-node experiment
│   │   │   └── multi_gpu.slurm
│   │   └── 8_gpus          # 8-GPU single-node experiment
│   │       └── multi_gpu.slurm
│   └── multi_node          # Experiments scaling across multiple nodes
│       ├── 2_nodes         # 2-node multi-GPU experiment
│       │   └── multi_node.slurm
│       ├── 4_nodes         # 4-node multi-GPU experiment
│       │   └── multi_node.slurm
│       └── 8_nodes         # 8-node multi-GPU experiment
│           └── multi_node.slurm
├── README.md               # Full documentation for running and analyzing all DDP experiments
└── scripts                 # Python training and utility scripts
    ├── analyze_memory.py   # Parses GPU memory logs and prints peak/avg/mode stats
    └── train.py            # PyTorch DDP training script (ResNet-50 on TinyImageNet)


```
### Environment Setup
We'll use Conda to manage packages and dependencies

run these lines:

```commandline
conda env create -f environment.yml
conda activate deepspeed-finetune
```

### Dataset: TinyImageNet
#### Request access for `tinyimagenet` directory

Many training exercises use the TinyImageNet dataset stored in the shared data repository on IBEX.
Please request access before the workshop as follows:

- Log in to [https://my.ibex.kaust.edu.sa/](https://my.ibex.kaust.edu.sa/) using your IBEX username and password.

- From the top menu, go to Reference.

- In the search box, type “tinyimagenet”.

- Click Request Access next to the dataset entry.

- Wait for approval confirmation (usually processed within one working day).

Once approved, the dataset will be accessible under the shared reference directory:

````commandline
/ibex/reference/CV/tinyimagenet
````

---

## Training Script Explanation

### Distributed Initialization (Code)

The script begins by initializing the distributed backend:

```python
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
```

**Key points:**

- `init_process_group` must be called before any distributed operations.
- **NCCL** is the recommended backend for multi-GPU training.
- **RANK** identifies the process globally (0 to world_size-1).
- **LOCAL_RANK** identifies which GPU the process controls on its node.
- Setting the **CUDA device** ensures each process uses the correct GPU.

This step establishes the communication fabric that all DDP processes use.

### Model + AMP + DDP Wrapper

The model is created, moved to CUDA, wrapped in **mixed-precision** utilities, and then wrapped in **DDP**:

```python
model = resnet50(num_classes=200).cuda()
criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)

scaler = amp.GradScaler()

model = DDP(model, device_ids=[local_rank])
```

What this does:

- Every process creates its **own ResNet-50 replica**.
    - TinyImageNet contains **200 distinct classes**, so the final fully connected layer of ResNet-50 must produce 200
      logits.
- We use AMP (autocast + GradScaler) to speed up training and reduce memory.
- Wrapping the model with **DDP** enables:
    - automatic gradient synchronization
    - faster gradient communication using bucketing
    - optimized multi-GPU behavior

### Data Pipeline & DistributedSampler

DDP requires that each process (each GPU) receives its own unique portion of the dataset.  
This is handled using `DistributedSampler`, which divides the dataset evenly across all ranks:

```python
train_sampler = DistributedSampler(train_dataset,
                                   num_replicas=world_size,
                                   rank=rank,
                                   shuffle=True,
                                   drop_last=True)

val_sampler = DistributedSampler(val_dataset,
                                 num_replicas=world_size,
                                 rank=rank,
                                 shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                          sampler=train_sampler, num_workers=args.num_workers)

val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        sampler=val_sampler, num_workers=args.num_workers)
```

**Why this matters:**

- **Dataset sharding**  
  Distributed training requires each process to see a unique portion of the dataset.  
  `DistributedSampler` splits the dataset into `world_size` equal shards and assigns one shard to each GPU.

- **Why sharding is required**  
  Without it, all GPUs would process the same samples, wasting compute and incorrectly increasing the effective batch
  size.

- **Shuffling**  
  When using multiple GPUs, shuffling must be coordinated.  
  Calling `sampler.set_epoch(epoch)` will ensures all processes shuffle consistently while still getting non-overlapping
  data.

- **Validation sharding**  
  Although validation does not require gradient synchronization, sharded validation avoids duplicate compute and keeps
  evaluation consistent across ranks.

- **DataLoader interaction**  
  The DataLoader reads only the shard assigned by the sampler.  
  Each GPU loads and processes only the samples intended for its rank.

Together, these components ensure efficient, correct, and well-distributed data loading across all GPUs.

### Training Loop

```python
model.train()
with amp.autocast(dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

```

- **Model set to train mode**  
  Enables dropout, batchnorm updates, and other training-only behaviors.

- **Mixed precision (autocast + GradScaler)**  
  Autocast runs operations in FP16 where safe and in FP32 where required.  
  GradScaler scales the loss to prevent FP16 underflow.

- **Forward pass**  
  Each GPU processes its own mini-batch independently.  
  No communication occurs during forward.

- **Loss computation**  
  Each GPU computes its own loss using its own shard of data.  
  The losses are not averaged across GPUs.

- **Backward pass**  
  Backprop computes gradients on each GPU individually.  
  Immediately after gradients are computed, DDP intercepts them.

- **Gradient synchronization**  
  DDP automatically performs an AllReduce to average gradients across all processes.  
  After this step, all GPUs have identical gradients.

- **Optimizer step**  
  Every GPU applies the same parameter update, keeping all model replicas identical.

This loop demonstrates the core idea of data-parallel training: parallel computation + synchronized gradients.

### Validation Loop

```python
model.eval()
with torch.no_grad(), amp.autocast(dtype=torch.float16):
    outputs = model(images)
    loss = criterion(outputs, labels)

```

- **Model in evaluation mode**  
  Disables dropout and batchnorm updates for stable evaluation.

- **No gradient computation**  
  `torch.no_grad()` prevents autograd tracking, reducing memory use and speeding up validation.

- **Sharded validation**  
  Each GPU evaluates only its portion of the validation set.  
  This avoids redundant computation across GPUs.

- **Local metric computation**  
  Each process computes validation loss and accuracy on its own shard.  
  In this exercise, metrics are not reduced globally—Rank 0 simply reports its local metrics.

- **No parameter updates**  
  Validation never updates model weights, so gradient synchronization is unnecessary.

This loop verifies the model’s performance without affecting training.

### Metrics & Logging

```python
local_train_loss = epoch_train_loss / epoch_train_samples
local_train_acc = 100.0 * epoch_train_correct / epoch_train_samples
```

Only rank 0 prints:

```python
if is_main_process:
    print(f"... training and validation metrics ...")

```

### Metrics & Logging

- **Local per-rank metrics**  
  Each GPU calculates:
    - training loss on its own shard
    - training accuracy on its own shard
    - validation loss on its own shard
    - validation accuracy on its own shard

- **Rank 0 logging**  
  Only rank 0 prints metrics to avoid duplicate logs.  
  In multi-GPU training, all processes compute metrics, but only rank 0 reports them.

- **Timing metrics**  
  The script measures:
    - total training time
    - per-epoch training progress
    - data loading time indirectly (via speed differences)

- **Memory metrics**  
  `torch.cuda.max_memory_allocated()` reports the peak GPU memory used by tensors.  
  This helps compare:
    - 1 GPU vs multi-GPU runs
    - DDP vs future methods like FSDP or ZeRO

- **Purpose of local metrics**  
  Although only rank 0 logs them, each GPU contributes equally to model updates via gradient averaging.  
  Local metrics are sufficient for demonstrating how DDP scales.

The collected metrics allow clear comparison of speed, memory usage, and behavior across different GPU counts.

>#### Disclaimer
>
> For the sake of keeping the workshop practical and fast to run on shared cluster resources, all experiments in this module use **only 3 training epochs**.  
> This shortened schedule is intentional: it keeps runtime low while still showing clear improvements in training loss, training accuracy, throughput, and scaling behavior.  
>
> These results should be interpreted as a **toy example** for understanding Distributed Data Parallelism—not as a meaningful training run for ResNet-50 or TinyImageNet.  
>In real research or production settings, you would train for many more epochs to achieve competitive accuracy.

---

## Slurm Job Structure

### Resources Requested

The Slurm directives at the top of the script request:

```bash
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=v100
#SBATCH --time=00:30:00
```

- **CPUs, GPUs, and nodes**
    - A certain number of **nodes** (e.g., 1 or more)
    - A fixed number of **tasks per node** (usually 1 launcher task per node)
    - A fixed number of **GPUs per node** (e.g., 1, 2, 4, or 8)
    - A specific number of **CPUs per task** (for data loading and Python overhead)

  Conceptually:
    - Each node runs one launcher process.
    - Each launcher process starts multiple worker processes (one per GPU).
    - The total number of DDP processes = `num_nodes * gpus_per_node`.

- **Memory and time**
  The script also requests a certain amount of **system memory** (in GB) and a **wall-clock time limit**.  
  These must respect the partition/QOS limits on the cluster.

- **GPU type / constraint**
  A `constraint` (e.g., `v100` or `a100`) ensures the job runs on nodes with the desired GPU type.

Together, these directives define the hardware envelope for the DDP job.

---

### Key Environment Variables

During setup, the script defines and/or uses several critical environment variables:

```bash
module load dl pytorch

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export PYTHONFAULTHANDLER=1
EXPERIMENT_DIR=$PWD

export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory/$JOB_ID"
CPU_LOG_DIR="$EXPERIMENT_DIR/cpu_memory/$JOB_ID"

mkdir -p "$GPU_LOG_DIR" "$CPU_LOG_DIR"
```

- **Module Environment**  
  The script loads a PyTorch or deep-learning module stack so that:
    - `python` refers to the correct interpreter.
        - `torch`, `torchvision`, and other libraries are available.

- **NCCL-related settings (e.g., NCCL_SOCKET_IFNAME)**  
  These control which network interface NCCL uses for communication (e.g., `ib0` for InfiniBand).  
  Setting this explicitly can avoid using the wrong interface and improve stability.

- **Misc debugging variables (e.g., NCCL_DEBUG, PYTHONFAULTHANDLER)**  
  These enable detailed debugging logs for NCCL and Python exceptions, which is useful for workshop troubleshooting.

- **DATA_DIR**  
  Points to the root directory containing the dataset (TinyImageNet in this exercise).  
  The Python script reads this path to locate the `train` and `val` subfolders.

- **SCRIPTS_DIR**  
  Points to the directory containing the training and analysis scripts.  
  This lets the Slurm script call the training Python file and any post-processing utilities.

These environment variables ensure that both the Python script and NCCL run in a configured, reproducible environment.

---

### GPU Memory Logging

```bash
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total \
  --format=csv,nounits -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
GPU_LOG_PID=$!
```

- **Global nvidia-smi logger**
  At job start, the script launches `nvidia-smi` in the background with:
    - A query for each GPU’s memory usage and total memory.
    - A fixed sampling interval (e.g., every 5 seconds).
    - Output redirected to a CSV file under a `gpu_memory` directory, often organized by job ID.

- **Purpose**
  This logger provides:
    - A time series of **GPU memory usage** during the entire job.
    - The ability to visualize how memory changes over epochs and across different GPU counts.
    - A simple way to compare DDP runs with later experiments (FSDP, ZeRO, etc.).

- **Lifetime**
  The logger runs in the background while training is active and is explicitly killed near the end of the script.

This gives participants a low-friction way to inspect how “full” each GPU was during training.

---

### CPU Memory Logging

```bash
psrecord $TRAIN_PID --include-children --interval 5 \
  --log "$CPU_LOG_DIR/cpu_memory_log.txt" &
CPU_LOG_PID=$!
```

- **psrecord-based logging**
  The script uses `psrecord` to track:
    - CPU memory usage of the main training process (and its children).
    - CPU utilization over time.

  The typical pattern is:
    - Identify the PID of the main training launcher.
    - Start `psrecord` in the background, with:
        - `--include-children` to capture all worker processes.
        - A fixed logging interval (e.g., every few seconds).
        - Output to a CPU log file in a `cpu_memory` directory.

- **Purpose**
  This provides a timeline of:
    - CPU RAM usage.
    - CPU load.
    - Effects of different configurations (more GPUs, more workers, different batch sizes).

- **Cleanup**
  Like the GPU logger, the CPU logger is terminated when the training process finishes.

This lets you pair GPU and CPU utilization curves for a complete view of resource behavior during the experiment.

---

### Multi-Node Launch Logic

- **Node discovery**
    ```bash
    head_node="${nodes_array[0]}"
    echo "Getting the IP address of the head node ${head_node}"
    ```
  The script obtains the list of allocated nodes from Slurm:
    - It queries the node list associated with the job.
    - Converts that into an array of hostnames.

- **Choosing a master node**
    ```bash
    master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
    master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
    ```
  The first node in the list is designated as the **master node**:
    - Its hostname or IP address becomes the `MASTER_ADDR`.
    - A fixed or dynamically chosen port becomes the `MASTER_PORT`.

- **Per-node launcher**
    ```bash
    export OMP_NUM_THREADS=1
    for (( i=0; i< ${SLURM_NNODES}; i++ ))
    do
         srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
          python -m torch.distributed.launch --use_env \
         --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
         --master_addr=${master_ip} --master_port=${master_port} \
         $SCRIPTS_DIR/train.py --epochs=3 --num-workers=${SLURM_CPUS_PER_TASK} --lr=0.001 --batch-size=512 &
    done
    wait
    ```
  For each node in the allocation:
    - The script starts a single `srun` launcher task on that node.
    - That launcher calls PyTorch’s distributed launcher (e.g., `torch.distributed.launch`) or `torchrun`.
    - The launcher is given:
        - The total number of nodes.
        - The number of processes per node (i.e., GPUs per node).
        - The node’s rank in the cluster (node_rank).
        - The master address and master port.

- **World size and ranks**
  Across all nodes, the total number of processes (`WORLD_SIZE`) equals:
    - `num_nodes * gpus_per_node`.
      Each process is assigned:
    - A unique global `RANK` (0 to WORLD_SIZE - 1).
    - A `LOCAL_RANK` determining which GPU it controls on its node.

- **Synchronization**
  All processes connect to the master address and port and join the same process group.  
  Once this group is formed, DDP can run across all nodes as if they were a single large machine.

This multi-node logic allows the same training script to scale from:

- 1 GPU on 1 node,
- to many GPUs across multiple nodes,
  without changing the Python code—only the Slurm resources and launch parameters change.

---

## Running the Experiments

### 1 GPU (single process)

Goal: establish a simple **baseline** for accuracy, runtime, and memory usage.

What this run represents:

- Only **one GPU** is used.
- DDP still works, but effectively behaves like standard single-GPU training (`WORLD_SIZE = 1`).
- No real speedup from parallelism, but:
    - The code path is identical to multi-GPU runs.
    - It confirms that the script, dataset path, and environment are correct.

How to run (conceptually):

- Request 1 node with 1 GPU in your Slurm script.
- Submit the job (e.g., `sbatch ddp_1g1n.slurm`).
- Record:
    - Total training time
    - Final train/val accuracy
    - Max GPU memory usage
    - CPU usage from the logs

You will later compare all other runs against this 1-GPU baseline.

---

### 2 GPUs (single node)

Goal: demonstrate **intra-node scaling** and verify that DDP really uses both GPUs.

What this run represents:

- **2 processes** are started on the same node, each bound to a different GPU.
- The dataset is split into 2 shards (one per GPU).
- Gradients are averaged across both GPUs at each step.

Expected behavior:

- Training time per epoch should decrease compared to 1 GPU.
- GPU memory per process may be similar, but total global batch size doubles (if per-GPU batch size stays the same).
- Accuracy and loss curves should look similar to the 1-GPU case (same effective training, just faster).

How to run (conceptually):

- Request 1 node with 2 GPUs.
- Use the same training script and Slurm logic, but with `gpus-per-node=2`.
- Submit the job (e.g., `sbatch ddp_2g1n.slurm`).
- Compare:
    - Runtime vs 1 GPU
    - GPU memory profile
    - Speedup (1G vs 2G)

This run shows participants how adding a second GPU affects performance and resource use.

---

### 4 GPUs (single node)

Goal: explore **stronger scaling** on a single node and show diminishing returns / overheads.

What this run represents:

- **4 processes** on one node, each mapped to a different GPU.
- The dataset is now split into 4 shards.
- Gradients are synchronized across all 4 GPUs after every backward pass.

Expected behavior:

- Training time per epoch should further decrease, but not by a perfect factor of 4.
- Communication overhead (AllReduce) becomes more visible.
- Global batch size = `4 × (per-GPU batch size)`, unless you adjust per-GPU batch.

How to run (conceptually):

- Request 1 node with 4 GPUs.
- Keep the same script and launch logic but adjust Slurm to `gpus-per-node=4`.
- Submit the job (e.g., `sbatch ddp_4g1n.slurm`).
- Compare against 1-GPU and 2-GPU runs:
    - Runtime and speedup
    - GPU utilization
    - Memory usage per GPU

This run illustrates the trend: **more GPUs → faster**, but not perfectly linear due to communication costs.

---

### Multi-Node (2 nodes × N GPUs)

Goal: show that exactly the same training script can scale **across nodes**, not just across GPUs on a single node.

What this run represents:

- Slurm allocates **2 nodes**.
- Each node has **N GPUs** (e.g., 2 or 4).
- One node is chosen as the **master** (rendezvous point).
- Total number of DDP processes = `2 × N`.
- NCCL uses the cluster’s high-speed network (e.g., InfiniBand) to synchronize gradients across nodes.

Expected behavior:

- Further reduction in training time, assuming the per-GPU batch size stays constant.
- Inter-node communication overhead appears, so scaling is less ideal than within a single node.
- Metrics (loss, accuracy) should match the behavior of single-node runs with the same global batch size.

How to run (conceptually):

- Request 2 nodes in your Slurm directives.
- Keep `gpus-per-node = N` (e.g., 2 or 4).
- Ensure the launch logic:
    - Discovers all nodes from `SLURM_JOB_NODELIST`.
    - Picks a `MASTER_ADDR` (from the first node) and a `MASTER_PORT`.
    - Assigns `node_rank` = 0 for the first node, 1 for the second.
- Submit the job (e.g., `sbatch ddp_ng2n.slurm` for “N GPUs × 2 nodes”).

What to compare:

- Single-node vs multi-node runtime for the same total number of GPUs.
- GPU and CPU utilization patterns.
- Any new issues (e.g., NCCL timeouts, port conflicts, misconfigured interfaces).

This run completes the story: from single-GPU DDP to multi-GPU single-node, then to **multi-node DDP**, using one
unified training script and only changing the Slurm resource request.

---

## Running the Experiments

Each experiment is launched simply by navigating to the correct directory and submitting the corresponding Slurm
script.  
No Python code changes are required — only the Slurm resource request changes.

### Job Submissions

#### 1 GPU (single process)

```commandline
cd experiments/baseline
sbatch baseline.slurm
```

--- 

#### 2 GPUs (single node)

```commandline
cd experiments/multi_gpu/2_gpus
sbatch multi_gpu.slurm
```

#### 4 GPUs (single node)

```commandline
cd experiments/multi_gpu/4_gpus
sbatch multi_gpu.slurm
```

#### 8 GPUs (single node)

```commandline
cd experiments/multi_gpu/8_gpus
sbatch multi_gpu.slurm
```

---

#### Multi-Node (2 nodes × 2 GPUs)

```commandline
cd experiments/multi_node/2_nodes
sbatch multi_node.slurm
```

#### Multi-Node (4 nodes × 2 GPUs)

```commandline
cd experiments/multi_node/4_nodes
sbatch multi_node.slurm
```

#### Multi-Node (8 nodes × 2 GPUs)

```commandline
cd experiments/multi_node/8_nodes
sbatch multi_node.slurm
```

Each script automatically:

- Detects the allocated Slurm nodes
- Sets up MASTER_ADDR / MASTER_PORT
- Launches the correct number of processes per node
- Starts GPU and CPU logging
- Runs the DDP training script with the appropriate world size

This keeps the training code unchanged while letting you scale from 1 GPU → multi-GPU → multi-node with only a directory
change and one `sbatch` command.

### Expected Output and Metrics Extraction

#### Training Output

The log file for every job will be located at:

```commandline
cd <submission_dir>/log
```

It will be under the name:

```commandline
<job_name>_<job_id>.out
```

For each run, the training script will print one summary line per epoch on rank 0.  
You are interested in the **last epoch** (epoch 3 in this example):

```text
...
Epoch [3/3] LR: 0.001000 | Loss (train, val): 5.149, 5.664 | Accuracy (train, val): 2.16%, 0.98% | Throughput: 704.9 img/s
...
```

From this line, extract:

- **Train loss** (epoch 3) → 5.149

- **Train accuracy** (epoch 3) → 2.16%%

- **Throughput** (epoch 3) → 704.9 img/s

You will copy these values into the tables below for each configuration (1, 2, 4, 8 GPUs or nodes).

#### GPU Memory Output (analyze_memory.py)

At the end of the job, the memory analysis script prints a summary for each GPU.
You will use the line corresponding to **GPU 0**:

```text
[gpu_memory_log - GPU 0] Peak = 26532 MiB, Avg = 22458.63 MiB, Mode = 26532 MiB
```

From this line, extract:

- **GPU 0 Peak** memory → 26532 MiB
- **GPU 0 Average** memory → 22458.63 MiB
- **GPU 0 Mode** memory → 26532 MiB

You will copy these values into the same tables, under GPU 0 (Peak / Avg / Mode).

### Multi-GPU Scaling (1 Node)

Use this table to compare **1, 2, 4, 8 GPUs on a single node**, using the same Slurm script family:

- `baseline.slurm` → [1 GPU](./experiments/baseline)

- `multi_gpu.slurm` in [`2_gpus`](./experiments/multi_gpu/2_gpus), [`4_gpus`](./experiments/multi_gpu/4_gpus), [
  `8_gpus`](./experiments/multi_gpu/8_gpus)

Fill the table with values from:

- The **epoch 3** training line printed by the script.

- The **GPU 0** memory summary from analyze_memory.py.

Compute scaling factors relative to 1 GPU, 1 node:

- **Throughput Scaling** = `Throughput_config / Throughput_1G`

- **Memory Avg Scaling** = `GPU0_Avg_config / GPU0_Avg_1G`

| Config                   | Train Loss (Epoch 3) | Train Acc (Epoch 3) | Throughput (img/s) | GPU 0 Peak (MiB) | GPU 0 Avg (MiB) | GPU 0 Mode (MiB) | Throughput Scaling vs 1G | Memory Avg Scaling vs 1G |
|--------------------------|----------------------|---------------------|--------------------|------------------|-----------------|------------------|--------------------------|--------------------------|
| 1 GPU, 1 Node (baseline) |                      |                     |                    |                  |                 |                  | 1.0×                     | 1.0×                     |
| 2 GPUs, 1 Node           |                      |                     |                    |                  |                 |                  |                          |                          |
| 4 GPUs, 1 Node           |                      |                     |                    |                  |                 |                  |                          |                          |
| 8 GPUs, 1 Node           |                      |                     |                    |                  |                 |                  |                          |                          |

### Multi-Node Scaling (Fixed GPUs per Node)

Use this table to compare **1 node vs 2, 4, 8 nodes,** keeping the same number of GPUs per node (as defined in your
multi-node Slurm scripts).

You will use:

- `baseline.slurm` → [1 GPU](./experiments/baseline)

- `multi_node.slurm` in [`2_nodes`](./experiments/multi_node/2_nodes), [`4_nodes`](./experiments/multi_node/4_nodes), [
  `8_nodes`](./experiments/multi_node/8_nodes)

Again, extract values from **epoch 3** and **GPU 0**’ s memory stats.

Compute scaling factors relative to 1 GPU, 1 node:

- **Throughput Scaling** = `Throughput_config / Throughput_1G`

- **Memory Avg Scaling** = `GPU0_Avg_config / GPU0_Avg_1G`

| Config            | Train Loss (Epoch 3) | Train Acc (Epoch 3) | Throughput (img/s) | GPU 0 Peak (MiB) | GPU 0 Avg (MiB) | GPU 0 Mode (MiB) | Throughput Scaling vs 1N | Memory Avg Scaling vs 1N |
|-------------------|----------------------|---------------------|--------------------|------------------|-----------------|------------------|--------------------------|--------------------------|
| 1 Node (baseline) |                      |                     |                    |                  |                 |                  | 1.0×                     | 1.0×                     |
| 2 Nodes           |                      |                     |                    |                  |                 |                  |                          |                          |
| 4 Nodes           |                      |                     |                    |                  |                 |                  |                          |                          |
| 8 Nodes           |                      |                     |                    |                  |                 |                  |                          |                          |
