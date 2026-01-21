#!/bin/bash
#SBATCH --job-name=volleyball_ball_det
#SBATCH --output=volleyball_ball_det_%j.out
#SBATCH --error=volleyball_ball_det_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ============================================
# Configuration - ADJUST THESE PATHS
# ============================================

# Input dataset
SPORTSMOT_DIR="/path/to/SportsMOT_volleyball"

# Output directories
DETECTIONS_DIR="/path/to/ball_detections"
OUTPUT_DIR="/path/to/SportsMOT_volleyball_extended"

# D-FINE repository and model paths
DFINE_PATH="/path/to/D-FINE"
DFINE_CONFIG="configs/dfine/dfine_hgnetv2_l_coco.yml"  # relative to DFINE_PATH
DFINE_WEIGHTS="/path/to/dfine_l_coco.pth"

# Inference settings
BATCH_SIZE=8
CONF_THRESHOLD=0.3
DEVICE="cuda:0"

# Merge settings
BALL_ID_START=100
MAX_BALLS=1

# ============================================
# Setup environment
# ============================================

# Load modules (adjust for your HPC)
module load python/3.10
module load cuda/11.8

# Activate virtual environment
source /path/to/your/venv/bin/activate

# Or use conda
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env

# ============================================
# Step 1: Run D-FINE inference
# ============================================

echo "=========================================="
echo "Step 1: Running D-FINE inference"
echo "=========================================="

python dfine_volleyball_inference.py \
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

echo "=========================================="
echo "Step 2: Merging ball detections with GT"
echo "=========================================="

python merge_ball_detections.py \
    --sportsmot_dir "$SPORTSMOT_DIR" \
    --detections_dir "$DETECTIONS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --ball_id_start "$BALL_ID_START" \
    --max_balls_per_frame "$MAX_BALLS"

if [ $? -ne 0 ]; then
    echo "Error in merge step!"
    exit 1
fi

echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
echo "Extended dataset saved to: $OUTPUT_DIR"
