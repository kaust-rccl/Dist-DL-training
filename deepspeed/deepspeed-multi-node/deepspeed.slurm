#!/bin/bash
#SBATCH --job-name=multinode-deepspeed-torchdist-1b7-bloom-finetune  # SLURM job name
#SBATCH --nodes=4                                                # Number of nodes to allocate
#SBATCH --ntasks-per-node=1                                      # One task (process) per node
#SBATCH --gres=gpu:v100:1                                        # Request one V100 GPU per node
#SBATCH --cpus-per-task=4                                        # Number of CPU cores per task
#SBATCH --mem=32G                                                # Memory per node
#SBATCH --constraint=v100                                        # Constrain allocation to V100-equipped nodes
#SBATCH --time=12:00:00                                          # Maximum runtime (HH:MM:SS)
#SBATCH --output=logs/4-nodes/%x-%j.out                          # Standard output log (%x=job name, %j=job ID)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True         # Enable expandable segments for PyTorch CUDA allocator
source /ibex/user/x_mohameta/miniforge/etc/profile.d/conda.sh   # Load Conda environment script
conda activate bloom-finetune                                   # Activate the Conda environment
module load cuda/12.1                                           # Load the CUDA 12.1 module

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

# -----------------------------------------
# Prepare GPU memory logging directory
# -----------------------------------------
logdir=./gpus_usage/4-nodes                                    # Directory to store GPU logs
mkdir -p "$logdir"

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
                   > \"$logdir/gpu_memory_log_\${hostname}.csv\" &                # Start GPU memory logging in background
        MEMORY_LOG_PID=\$!                                                         # Save PID of nvidia-smi logger

        python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=1 \
            --node_rank=$i \
            --rdzv_endpoint=$master_ip:$master_port \
            train.py                                                              # Launch distributed training
    " &
done

wait   # Wait for all backgrounded tasks to complete

# ----------------------------
# Cleanup
# ----------------------------
kill $MEMORY_LOG_PID    # Stop the GPU memory logger
