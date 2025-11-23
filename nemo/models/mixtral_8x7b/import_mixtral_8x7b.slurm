#!/bin/bash
#SBATCH --job-name=m-imp
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --constraint=a100
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out

SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
SCRIPT_DIR="$( cd "$( dirname "$SCRIPT_PATH" )" && pwd )"
MODEL_DIR="$SCRIPT_DIR/model"
mkdir -p "$MODEL_DIR"
NEMO_ROOT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
CACHE_DIR="$NEMO_ROOT_DIR/.cache"

export TORCH_HOME="$CACHE_DIR/torch"
export NEMO_DATASETS_CACHE="$CACHE_DIR/datasets"
export HF_HOME="$CACHE_DIR/hf_cache"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets_cache"
export TMPDIR="$CACHE_DIR/tmp"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export NEMO_HOME="$CACHE_DIR/nemo_cache"
mkdir -p "$TORCH_HOME" "$NEMO_DATASETS_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$TRITON_CACHE_DIR" "$NEMO_HOME"

module load cuda/12.4.1
module load singularity

export HF_TOKEN="<YOUR_HF_TOKEN>"

srun singularity exec --nv /ibex/user/x_mohameta/nemo/image/nemo_25.07.sif \
nemo llm import \
model=mixtral_8x7b \
source="hf://mistralai/Mixtral-8x7B-v0.1" \
output_path="$MODEL_DIR" \
overwrite=True \
--yes