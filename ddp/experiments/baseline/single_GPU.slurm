#!/bin/bash 

#SBATCH --time=00:30:00
#SBATCH --job-name=ddp-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100
#SBATCH --cpus-per-task=10
#SBATCH --error=logs/ddp-base-%j.err             # Standard error
#SBATCH --output=logs/ddp-base-%j.out            # Standard output


 
#ad environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

# setup the environment

# You can use the machine learning module 
#module load machine_learning/2022.11

# or you can activate the conda environment directly by uncommenting the following lines
#source /ibex/ai/home/$USER/miniconda3/bin/activate dist-pytorch

module load dl pytorch/2.0.0

export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet/
#export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet

time -p python single_GPU.py
