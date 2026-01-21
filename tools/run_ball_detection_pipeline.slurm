#!/bin/bash
#SBATCH --partition=gpu8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --output=runs/detection_run%j.out

# ============================================
# Configuration - ADJUST THESE PATHS
# ============================================

# Input dataset
SPORTSMOT_DIR="data/Dataset/mot/SportsMOT_Volleyball"

# Output directories
DETECTIONS_DIR="data/Dataset/mot/SportsMOT_Volleyball/ball_detections"

# D-FINE repository and model paths
DFINE_PATH="/home/es/es_es/es_lechit00/FP/D-FINE"
DFINE_CONFIG="configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml"  # relative to DFINE_PATH
DFINE_WEIGHTS="/home/es/es_es/es_lechit00/FP/D-FINE/dfine_l_obj2coco_e25.pth"

# Inference settings
BATCH_SIZE=8
CONF_THRESHOLD=0.15
DEVICE="cuda:0"

# Merge settings
BALL_ID_START=100
MAX_BALLS=1

# ============================================
# Setup environment
# ============================================

# Load modules (adjust for your HPC)
module load devel/miniforge/24.11.0-python-3.12
module load devel/cuda/12.4

# Activate virtual environment
conda activate pytorch

# ============================================
# Step 1: Run D-FINE inference
# ============================================

echo "=========================================="
echo "Step 1: Running D-FINE inference"
echo "=========================================="

python tools/dfine_volleyball_inference.py \
    --dfine_path "$DFINE_PATH" \
    --config "$DFINE_CONFIG" \
    --weights "$DFINE_WEIGHTS" \
    --sportsmot_dir "$SPORTSMOT_DIR" \
    --output_dir "$DETECTIONS_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --conf_threshold "$CONF_THRESHOLD"

if [ $? -ne 0 ]; then
    echo "Error in inference step!"
    exit 1
fi

# ============================================
# Step 2: Merge detections with GT
# ============================================

# echo "=========================================="
# echo "Step 2: Merging ball detections with GT"
# echo "=========================================="

# python tools/merge_ball_detections.py \
#     --sportsmot_dir "$SPORTSMOT_DIR" \
#     --detections_dir "$DETECTIONS_DIR" \
#     --output_dir "$SPORTSMOT_DIR" \
#     --ball_id_start "$BALL_ID_START" \
#     --max_balls_per_frame "$MAX_BALLS"

# if [ $? -ne 0 ]; then
#     echo "Error in merge step!"
#     exit 1
# fi

# echo "=========================================="
# echo "Pipeline complete!"
# echo "=========================================="
# echo "Extended dataset saved to: $SPORTSMOT_DIR"
