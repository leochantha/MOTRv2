#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=runs/run%j.out

module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

conda activate pytorch

# Set environment variables for single-GPU training
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# CUDA optimization flags for H100
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

./tools/train.sh configs/motrv2_volleyball.args


## ./tools/simple_inference.sh ./motrv2_dancetrack.pth
## python create_video.py