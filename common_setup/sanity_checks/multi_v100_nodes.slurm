#!/bin/bash
#SBATCH --job-name=multi-v100-nodes-access-check     # Name of the job shown in SLURM queue
#SBATCH --ntasks=4                                   # Number of tasks
#SBATCH --tasks-per-node=2                           # One task per node
#SBATCH --cpus-per-task=4                            # Number of CPUs allocated per task
#SBATCH --gpus=4                                     # Request 1 GPU
#SBATCH --gpus-per-node=2                      	     # GPUs per node
#SBATCH --constraint=v100                     	     # Request specific GPU type (V100)
#SBATCH --time=00:05:00                              # Maximum runtime (HH:MM:SS)
#SBATCH --output=log/%x-%j.out                       # Redirect SLURM .out log to log/ directory


echo "────────────────────────────────────────────────────────"
echo "  Pre-Workshop Sanity Check: Multi-V100-Node Multi-GPU"
echo "────────────────────────────────────────────────────────"
echo

echo "[1/4] Checking SLURM allocation..."
echo "---------------------------------"
echo "→ Job ID:              ${SLURM_JOB_ID:-N/A}"
echo "→ Node list:           ${SLURM_JOB_NODELIST:-N/A}"
echo "→ Nodes allocated:     ${SLURM_JOB_NUM_NODES:-N/A}"
echo "→ GPUs per node (req): 2"
echo

# Expand node hostnames up front
NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
nodes_array=($NODES)

echo "[2/4] Per-node hostname verification..."
echo "---------------------------------------"
any_host_fail=0

for (( i=0; i<${SLURM_NNODES}; i++ )); do
	printf "• %s ... " "${nodes_array[$i]}"
	if srun -N1 -n1 -c ${SLURM_CPUS_PER_TASK} --gpus=${SLURM_GPUS_PER_NODE} -w ${nodes_array[$i]} bash -lc 'hostname >/dev/null'; then
	  echo "[✓] reachable"
	else
	  echo "[✗] failed"
	  any_host_fail=1
	fi
done
echo

echo "[3/4] Verifying GPU visibility on each node..."
echo "----------------------------------------------"
any_gpu_fail=0
for (( i=0; i<${SLURM_NNODES}; i++ )); do
        echo "Node: ${nodes_array[$i]}"
	echo "------"
	if srun -N1 -n1 -c ${SLURM_CPUS_PER_TASK} --gpus=${SLURM_GPUS_PER_NODE} -w ${nodes_array[$i]} bash -lc '
		if command -v nvidia-smi &>/dev/null; then
        	  nvidia-smi -L
        	  cnt=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " ")
        	  echo "Detected GPUs: ${cnt}"
        	  exit 0
      		else
        	  echo "[✗] nvidia-smi not found on this node."
        	  exit 2
  	        fi
		'; then
		echo "[✓] GPU query succeeded."
	else
	  echo "[✗] GPU query failed."
    	  any_gpu_fail=1
	fi
done
echo

echo "[4/4] Summary"
echo "--------------"
echo "SLURM allocation:         [✓] confirmed"
if (( any_host_fail == 0 )); then
  echo "Per-node reachability:    [✓] all nodes reachable"
else
  echo "Per-node reachability:    [✗] some nodes unreachable"
fi
if (( any_gpu_fail == 0 )); then
  echo "GPU visibility (per node):[✓] nvidia-smi OK on all nodes"
else
  echo "GPU visibility (per node):[✗] issues detected"
fi
echo
echo "Done."
echo "───────────────────────────────────────────────"
