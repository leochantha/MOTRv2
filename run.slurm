#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=runs/run%j.out

module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

conda activate pytorch

./tools/simple_inference.sh ./motrv2_dancetrack.pth

python create_video.py