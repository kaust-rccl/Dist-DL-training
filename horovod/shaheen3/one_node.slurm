#!/bin/bash 
#SBATCH --job-name=single
#SBATCH --time=05:0:0
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread

module swap PrgEnv-cray PrgEnv-gnu
module load pytorch/2.2.1
module load horovod/0.28.1-torch221
module list

echo "Hostnames: $SLURM_NODELIST"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export RUNDIR=${PWD}/result_${SLURM_JOB_NAME}_${SLURM_JOBID}
mkdir -p $RUNDIR

# ImageNet dataset 1000 classes
  ## local storage
export DATA_DIR="/scratch/shaima0d/bandwidth/datasets"

batch_size=32
epochs=5
workers=${SLURM_CPUS_PER_TASK}

echo "Hostname: $(/bin/hostname)"
echo "Data source: $DATA_DIR"
echo "Using Batch size : $batch_size"
echo "Epochs : $epochs"
echo "CPU workers: $workers"

cd $RUNDIR
main_exe="train_resnet50.py"
cmd="python3 ${main_exe} --epochs ${epochs} --batch-size ${batch_size} --num_workers=$workers --root-dir=${DATA_DIR} --train-dir ${DATA_DIR}/train --val-dir ${DATA_DIR}/val ${NODE_LOCAL_STORAGE}"
echo "time -p srun -u -n ${SLURM_NTASKS}  -c ${SLURM_CPUS_PER_TASK} ${cmd} --log-dir=log.${SLURM_JOBID} --warmup-epochs=0.0 --no-cuda"
time -p srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} \
	-c ${SLURM_CPUS_PER_TASK} \
	--cpu-bind=verbose,cores \
	${cmd} &> output.txt
    
