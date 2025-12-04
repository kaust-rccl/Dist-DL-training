#!/bin/bash

#SBATCH --job-name=ddp_4G1N
#SBATCH --output=log/%x-%j.out
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --constraint=v100
#SBATCH --time=00:30:00

scontrol show job $SLURM_JOBID

#source /ibex/ai/home/$USER/miniconda3/bin/activate dist-pytorch
module load dl pytorch

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export PYTHONFAULTHANDLER=1
EXPERIMENT_DIR=$PWD

export SCRIPTS_DIR="../../../scripts"
#export DATA_DIR=/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet
GPU_LOG_DIR="$EXPERIMENT_DIR/gpu_memory/${SLURM_JOB_ID}"

mkdir -p "$GPU_LOG_DIR"

# ------------------------------
# GPU Memory Logging
# ------------------------------
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total \
  --format=csv,nounits -l 5 > "$GPU_LOG_DIR/gpu_memory_log.csv" &
GPU_LOG_PID=$!


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${master_port}"
echo "world size:  $WORLD_SIZE"
echo "local rank:  $LOCAL_RANK"
echo "rank:  $RANK"

declare -a SRUN_PIDS


export OMP_NUM_THREADS=1
for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
     srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
      python -m torch.distributed.launch --use_env \
     --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
     --master_addr=${master_ip} --master_port=${master_port} \
     $SCRIPTS_DIR/train.py --epochs=3 --num-workers=${SLURM_CPUS_PER_TASK}  --lr=0.001 --batch-size=512 &
     SRUN_PIDS+=($!)
done

for pid in "${SRUN_PIDS[@]}"; do
    wait "$pid"
done

# ------------------------------
# Wait & Cleanup
# ------------------------------
kill $GPU_LOG_PID

echo "Analyzing memory logs for job ${SLURM_JOB_ID:-manual}"
python $SCRIPTS_DIR/analyze_memory.py ${SLURM_JOB_ID:-manual} --path "$SCRIPT_DIR" --gpu-only

