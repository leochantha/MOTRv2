#!/usr/bin/env python3
"""
Merge Ball Detections with Player Ground Truth

Combines D-FINE ball detections with existing SportsMOT player annotations
into a unified MOT format ground truth file.

MOT Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

Class mapping:
    1 = player (existing)
    2 = ball (new)

Usage:
    python merge_ball_detections.py \
        --sportsmot_dir /path/to/SportsMOT_volleyball \
        --detections_dir /path/to/ball_detections \
        --output_dir /path/to/SportsMOT_volleyball_extended \
        --ball_id_start 100 \
        --max_balls_per_frame 1
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np


# Class IDs for extended ground truth
CLASS_PLAYER = 1
CLASS_BALL = 2


def load_mot_gt(gt_path):
    """
    Load MOT format ground truth file.
    
    Returns:
        dict: {frame_id: [(id, x, y, w, h, conf, class, vis), ...]}
    """
    gt = {}
    
    if not gt_path.exists():
        return gt
    
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            cls = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            
            if frame_id not in gt:
                gt[frame_id] = []
            
            gt[frame_id].append({
                'id': obj_id,
                'bbox': [x, y, w, h],
                'conf': conf,
                'class': cls,
                'visibility': vis
            })
    
    return gt


def load_ball_detections(det_path):
    """
    Load ball detections from JSON file.
    
    Returns:
        dict: {frame_id: [{'bbox': [x,y,w,h], 'confidence': float}, ...]}
    """
    if not det_path.exists():
        return {}
    
    with open(det_path, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to int
    return {int(k): v for k, v in data.items()}


def select_best_ball(detections, max_balls=1):
    """
    Select the best ball detection(s) from a list.
    Uses confidence score and optionally NMS.
    
    Args:
        detections: List of detection dicts
        max_balls: Maximum number of balls to keep
        
    Returns:
        List of selected detections
    """
    if not detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    if max_balls == 1:
        return [sorted_dets[0]]
    
    # Simple NMS for multiple balls
    selected = []
    for det in sorted_dets:
        if len(selected) >= max_balls:
            break
        
        # Check IoU with already selected
        dominated = False
        for sel in selected:
            iou = compute_iou(det['bbox'], sel['bbox'])
            if iou > 0.5:
                dominated = True
                break
        
        if not dominated:
            selected.append(det)
    
    return selected


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to x1, y1, x2, y2
    b1 = [x1, y1, x1 + w1, y1 + h1]
    b2 = [x2, y2, x2 + w2, y2 + h2]
    
    # Intersection
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_gt_and_balls(player_gt, ball_dets, ball_id_start=100, max_balls=1):
    """
    Merge player ground truth with ball detections.
    
    Args:
        player_gt: Dict of player annotations per frame
        ball_dets: Dict of ball detections per frame
        ball_id_start: Starting ID for ball tracks
        max_balls: Maximum balls per frame
        
    Returns:
        Dict of merged annotations per frame
    """
    merged = {}
    
    # Get all frame IDs
    all_frames = set(player_gt.keys()) | set(ball_dets.keys())
    
    # Track ball ID assignment for temporal consistency
    # Simple approach: use fixed ID for single ball tracking
    current_ball_id = ball_id_start
    
    for frame_id in sorted(all_frames):
        merged[frame_id] = []
        
        # Add player annotations (unchanged)
        if frame_id in player_gt:
            for ann in player_gt[frame_id]:
                merged[frame_id].append({
                    'id': ann['id'],
                    'bbox': ann['bbox'],
                    'conf': ann['conf'],
                    'class': CLASS_PLAYER,
                    'visibility': ann['visibility']
                })
        
        # Add ball detections
        if frame_id in ball_dets:
            selected_balls = select_best_ball(ball_dets[frame_id], max_balls)
            
            for i, ball in enumerate(selected_balls):
                merged[frame_id].append({
                    'id': ball_id_start + i,  # Fixed ID for tracking
                    'bbox': ball['bbox'],
                    'conf': ball['confidence'],
                    'class': CLASS_BALL,
                    'visibility': 1.0
                })
    
    return merged


def write_mot_gt(merged_gt, output_path):
    """Write merged ground truth to MOT format file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    for frame_id in sorted(merged_gt.keys()):
        for ann in merged_gt[frame_id]:
            x, y, w, h = ann['bbox']
            line = f"{frame_id},{ann['id']},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{ann['conf']:.4f},{ann['class']},{ann['visibility']:.2f}"
            lines.append(line)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def process_sequence(seq_dir, det_path, output_seq_dir, ball_id_start, max_balls):
    """Process a single sequence."""
    
    # Load existing ground truth
    gt_path = seq_dir / 'gt' / 'gt.txt'
    player_gt = load_mot_gt(gt_path)
    
    # Load ball detections
    ball_dets = load_ball_detections(det_path)
    
    # Merge
    merged = merge_gt_and_balls(player_gt, ball_dets, ball_id_start, max_balls)
    
    # Copy sequence structure
    output_seq_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images
    src_img_dir = seq_dir / 'img1'
    dst_img_dir = output_seq_dir / 'img1'
    if src_img_dir.exists() and not dst_img_dir.exists():
        shutil.copytree(src_img_dir, dst_img_dir)
    
    # Copy seqinfo.ini
    src_seqinfo = seq_dir / 'seqinfo.ini'
    dst_seqinfo = output_seq_dir / 'seqinfo.ini'
    if src_seqinfo.exists():
        shutil.copy(src_seqinfo, dst_seqinfo)
    
    # Write new ground truth
    output_gt_dir = output_seq_dir / 'gt'
    output_gt_dir.mkdir(exist_ok=True)
    write_mot_gt(merged, output_gt_dir / 'gt_extended.txt')
    
    # Statistics
    n_frames = len(merged)
    n_frames_with_ball = sum(1 for f in merged.values() if any(a['class'] == CLASS_BALL for a in f))
    
    return {
        'n_frames': n_frames,
        'n_frames_with_ball': n_frames_with_ball
    }


def main():
    parser = argparse.ArgumentParser(description='Merge ball detections with player GT')
    
    parser.add_argument('--sportsmot_dir', type=str, required=True,
                        help='Path to SportsMOT volleyball dataset')
    parser.add_argument('--detections_dir', type=str, required=True,
                        help='Path to ball detections from inference script')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save extended dataset')
    parser.add_argument('--ball_id_start', type=int, default=100,
                        help='Starting ID for ball tracks (should be higher than max player ID)')
    parser.add_argument('--max_balls_per_frame', type=int, default=1,
                        help='Maximum number of balls to keep per frame')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to process')
    parser.add_argument('--symlink_images', action='store_true',
                        help='Use symlinks for images instead of copying (saves space)')
    
    args = parser.parse_args()
    
    sportsmot_dir = Path(args.sportsmot_dir)
    detections_dir = Path(args.detections_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    total_stats = {
        'sequences': 0,
        'frames': 0,
        'frames_with_ball': 0
    }
    
    for split in args.splits:
        split_dir = sportsmot_dir / split
        det_split_dir = detections_dir / split
        output_split_dir = output_dir / split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        sequences = [d for d in split_dir.iterdir() if d.is_dir()]
        
        print(f"\nProcessing {split} split ({len(sequences)} sequences)...")
        
        for seq_dir in tqdm(sequences, desc=f"Processing {split}"):
            seq_name = seq_dir.name
            det_path = det_split_dir / f"{seq_name}_ball_detections.json"
            
            if not det_path.exists():
                print(f"  Warning: No detections for {seq_name}, skipping...")
                continue
            
            output_seq_dir = output_split_dir / seq_name
            
            stats = process_sequence(
                seq_dir, det_path, output_seq_dir,
                args.ball_id_start, args.max_balls_per_frame
            )
            
            total_stats['sequences'] += 1
            total_stats['frames'] += stats['n_frames']
            total_stats['frames_with_ball'] += stats['n_frames_with_ball']
    
    # Copy splits_txt if it exists
    splits_txt_src = sportsmot_dir / 'splits_txt'
    if splits_txt_src.exists():
        shutil.copytree(splits_txt_src, output_dir / 'splits_txt', dirs_exist_ok=True)
    
    # Write metadata
    metadata = {
        'source_dataset': str(sportsmot_dir),
        'detections_source': str(detections_dir),
        'ball_id_start': args.ball_id_start,
        'max_balls_per_frame': args.max_balls_per_frame,
        'class_mapping': {
            '1': 'player',
            '2': 'ball'
        },
        'statistics': total_stats
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("Merge complete!")
    print(f"{'='*50}")
    print(f"Sequences processed: {total_stats['sequences']}")
    print(f"Total frames: {total_stats['frames']}")
    print(f"Frames with ball detection: {total_stats['frames_with_ball']} "
          f"({100*total_stats['frames_with_ball']/max(1, total_stats['frames']):.1f}%)")
    print(f"\nOutput saved to: {output_dir}")
    print(f"\nClass mapping in GT:")
    print(f"  1 = player")
    print(f"  2 = ball")
    print(f"\nBall track ID starts at: {args.ball_id_start}")


if __name__ == '__main__':
    main()
