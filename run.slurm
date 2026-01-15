#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=runs/run%j.out

module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

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

conda activate pytorch

./tools/train.sh configs/motrv2_volleyball.args


## ./tools/simple_inference.sh ./motrv2_dancetrack.pth
## python create_video.py data/Dataset/mot/Volleyball-Activity-Dataset-3/train/match1/img1 output.mp4 --tracking-files data/Dataset/mot/Volleyball-Activity-Dataset-3/train/match1/gt/gt.txt --model-names "New Visualization Script"