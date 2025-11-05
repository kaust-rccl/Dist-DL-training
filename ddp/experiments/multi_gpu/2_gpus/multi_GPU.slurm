#!/bin/bash

#SBATCH --job-name=ddp_N1G2
#SBATCH --output=logs/ddp_N1G2_%j.out
#SBATCH --error=logs/ddp_N1G2_%j.err
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --constraint=v100
#SBATCH --time=00:30:00

scontrol show job $SLURM_JOBID 

#source /ibex/ai/home/$USER/miniconda3/bin/activate dist-pytorch
module load dl pytorch

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export PYTHONFAULTHANDLER=1

#export DATA_DIR=/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"

# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
master_port=10121
echo "head node is ${master_ip}:${master_port}"
echo "world size: " $WORLD_SIZE
echo "local rank: " $LOCAL_RANK
echo "rank: " $RANK



export OMP_NUM_THREADS=1
for (( i=0; i< ${SLURM_NNODES}; i++ ))
do
     srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} -w ${nodes_array[i]} --gpus=${SLURM_GPUS_PER_NODE}  \
      python -m torch.distributed.launch --use_env \
     --nproc_per_node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --node_rank=${i} \
     --master_addr=${master_ip} --master_port=${master_port} \
     multi_GPU.py --epochs=10 --lr=0.001 --num-workers=10 &
done
wait

