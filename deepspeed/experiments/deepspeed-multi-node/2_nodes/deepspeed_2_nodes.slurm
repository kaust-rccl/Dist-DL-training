#!/bin/bash
#SBATCH --job-name=2-nodes-zero0-deepspeed-bloom-finetune        # SLURM job name
#SBATCH --nodes=2                                                # Number of nodes to allocate
#SBATCH --ntasks=4                                               # Number of tasks
#SBATCH --ntasks-per-node=2                                      # 2 tasks (process) per node
#SBATCH --cpus-per-task=4                                        # Number of CPU cores per task
#SBATCH --gres=gpu:v100:2                                        # 2 GPU per node
#SBATCH --mem=32G                                                # Memory per node
#SBATCH --constraint=v100                                        # Constrain allocation to V100-equipped nodes
#SBATCH --time=01:00:00                                          # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                          # Standard output log (%x=job name, %j=job ID)

# ------------------------------
# Resolve directory paths
# ------------------------------
SCRIPT_DIR=$(pwd)

cd "$SCRIPT_DIR/../../.."   # → Go to project root: Dist-DL-training/deepspeed

ROOT_DIR=$(pwd)

JOB_ID=${SLURM_JOB_ID:-manual}
EXPERIMENT_DIR="$SCRIPT_DIR"

# Log directories (all adjacent to SLURM script)
LOG_OUT_DIR="$EXPERIMENT_DIR/log"
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory/$JOB_ID"

mkdir -p "$LOG_OUT_DIR" "$GPU_LOG_DIR"

# ------------------------------
# Environment Setup
# ------------------------------
source /ibex/user/$USER/miniforge/etc/profile.d/conda.sh
conda activate deepspeed-finetune
module load cuda/12.4.1

# ----------------------------
# Discover participating nodes
# ----------------------------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")         # Get list of hostnames in this job
nodes_array=($nodes)                                           # Split the list into a Bash array
echo "Node IDs of participating nodes: ${nodes_array[*]}"      # Print out the node hostnames

# -----------------------------------------
# Determine MASTER node IP and open port
# -----------------------------------------
head_node="${nodes_array[0]}"                                           # Choose the first node as the master
echo "Head node hostname: ${head_node}"
master_ip=$(srun -n1 -N1 -w ${head_node} hostname -I | cut -d " " -f2)  # Fetch the IP address of the head node
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')  # Find a free TCP port dynamically
echo "Master endpoint: ${master_ip}:${master_port}"


# ----------------------------
# Launch one process per node
# ----------------------------
for (( i=0; i<${SLURM_NNODES}; i++ )); do
    srun -N1 -n1 \
         -c ${SLURM_CPUS_PER_TASK} \
         --gpus=${SLURM_GPUS_PER_NODE}\
         -w ${nodes_array[$i]} \
         bash -c "
        hostname=\$(hostname)                                                      # Capture this node's hostname
        nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total \
                   --format=csv,nounits -l 5 \
                   > \"$GPU_LOG_DIR/gpu_memory_log_\${hostname}.csv\" &                # Start GPU memory logging in background
        MEMORY_LOG_PID=\$!                                                         # Save PID of nvidia-smi logger

        python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=$SLURM_GPUS_PER_NODE \
            --node_rank=$i \
            --rdzv_endpoint=$master_ip:$master_port \
            scripts/train_multi_nodes.py --deepspeed $ROOT_DIR/ds_configs/zero2_cpu_offload_pinned_memory.json                           # Launch distributed training
    " &
done

wait   # Wait for all backgrounded tasks to complete

# ----------------------------
# Cleanup
# ----------------------------
kill $MEMORY_LOG_PID    # Stop the GPU memory logger

echo "Analyzing memory logs for job ${SLURM_JOB_ID:-manual}"
python scripts/analyze_memory.py ${SLURM_JOB_ID:-manual} --path "$SCRIPT_DIR" --gpu-only
