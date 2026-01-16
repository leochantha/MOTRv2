#!/usr/bin/env python3
"""
Prepare Roboflow-exported volleyball dataset for MOTRv2 training.

This script:
1. Renames Roboflow images (1_jpg.rf.xxxxx.jpg) to standard format (000001.jpg)
2. Updates the detection JSON file to use correct paths
3. Creates MOT-format gt.txt from the JSON detections

Usage:
    python tools/prepare_roboflow_dataset.py \
        --dataset-dir data/Dataset/mot/Volleyball-Activity-Dataset-3 \
        --sequence train/match1 \
        --det-json finetune_gt_dfine.json \
        --output-json det_db_dfine.json
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from collections import defaultdict


def extract_frame_number_from_roboflow(filename):
    """Extract frame number from Roboflow filename like '1_jpg.rf.xxxxx.jpg' or '10_jpg.rf.xxxxx.jpg'."""
    match = re.match(r'^(\d+)_jpg\.rf\.', filename)
    if match:
        return int(match.group(1))
    # Also try pattern without _jpg
    match = re.match(r'^(\d+)\.rf\.', filename)
    if match:
        return int(match.group(1))
    # Try just leading numbers
    match = re.match(r'^(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def rename_roboflow_images(img_dir, dry_run=False):
    """Rename Roboflow images to standard format (000001.jpg).

    Returns:
        dict: Mapping from old filename to new filename
    """
    img_path = Path(img_dir)
    if not img_path.exists():
        print(f"Error: Image directory {img_dir} not found")
        return {}

    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(img_path.glob(ext))

    if not image_files:
        print(f"No image files found in {img_dir}")
        return {}

    # Extract frame numbers and sort
    frames = []
    for img_file in image_files:
        frame_num = extract_frame_number_from_roboflow(img_file.name)
        if frame_num is not None:
            frames.append((frame_num, img_file))
        else:
            print(f"Warning: Could not extract frame number from {img_file.name}")

    frames.sort(key=lambda x: x[0])

    # Create rename mapping
    rename_map = {}
    print(f"\nFound {len(frames)} images to rename:")

    for frame_num, img_file in frames:
        ext = img_file.suffix.lower()
        new_name = f"{frame_num:06d}{ext}"
        rename_map[img_file.name] = new_name

        if frame_num <= 3 or frame_num >= len(frames) - 2:
            print(f"  {img_file.name} -> {new_name}")
        elif frame_num == 4:
            print(f"  ...")

    if dry_run:
        print("\n[DRY RUN] No files were renamed")
        return rename_map

    # Perform renaming (use temp names first to avoid conflicts)
    print("\nRenaming files...")
    temp_dir = img_path / "_temp_rename"
    temp_dir.mkdir(exist_ok=True)

    # Move to temp directory
    for old_name, new_name in rename_map.items():
        old_path = img_path / old_name
        temp_path = temp_dir / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(temp_path))

    # Move back from temp directory
    for new_name in rename_map.values():
        temp_path = temp_dir / new_name
        new_path = img_path / new_name
        if temp_path.exists():
            shutil.move(str(temp_path), str(new_path))

    # Remove temp directory
    temp_dir.rmdir()

    print(f"Renamed {len(rename_map)} files")
    return rename_map


def convert_json_detections(input_json, output_json, sequence_path, mot_path):
    """Convert detection JSON to use correct paths for MOTRv2.

    Args:
        input_json: Path to input JSON (e.g., finetune_gt_dfine.json)
        output_json: Path to output JSON (e.g., det_db_dfine.json)
        sequence_path: Sequence path relative to mot_path (e.g., 'train/match1')
        mot_path: Base dataset path
    """
    with open(input_json, 'r') as f:
        data = json.load(f)

    print(f"\nConverting {len(data)} entries from {input_json}")

    converted = {}
    for old_key, detections in data.items():
        # Extract frame number from old key
        # Old format: "volleyball/combined/gt/img1/000002"
        parts = old_key.split('/')
        frame_str = parts[-1]  # Get last part (e.g., "000002")

        # Try to extract frame number
        numbers = re.findall(r'\d+', frame_str)
        if numbers:
            frame_num = int(numbers[-1])
        else:
            print(f"Warning: Could not extract frame number from {old_key}")
            continue

        # Create new key with correct path
        # New format: "train/match1/img1/000001.txt" (MOTRv2 det_db format)
        new_key = f"{sequence_path}/img1/{frame_num:08d}.txt"

        converted[new_key] = detections

    # Save converted JSON
    output_path = os.path.join(mot_path, output_json) if not os.path.isabs(output_json) else output_json
    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"Saved converted detections to {output_path}")
    print(f"  Total entries: {len(converted)}")
    if converted:
        sample_key = list(converted.keys())[0]
        print(f"  Sample key: {sample_key}")

    return converted


def create_mot_gt_from_json(input_json, output_gt, sequence_path, mot_path,
                            iou_threshold=0.5, max_frames_to_skip=5):
    """Create MOT-format gt.txt from detection JSON using IoU tracking.

    This generates track IDs by associating detections across frames.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    def compute_iou(box1, box2):
        """Compute IoU between two boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    # Load JSON
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Parse detections by frame
    frames = defaultdict(list)
    for key, detections in data.items():
        # Extract frame number
        parts = key.split('/')
        frame_str = parts[-1]
        numbers = re.findall(r'\d+', frame_str)
        if numbers:
            frame_num = int(numbers[-1])
        else:
            continue

        for det_str in detections:
            det_str = det_str.strip().strip('\n')
            if not det_str:
                continue
            parts = det_str.split(',')
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                score = float(parts[4]) if len(parts) > 4 else 1.0
                frames[frame_num].append({
                    'box': [x, y, w, h],
                    'score': score,
                    'track_id': None
                })

    if not frames:
        print("No detections found in JSON")
        return

    print(f"\nGenerating track IDs for {len(frames)} frames...")

    # Sort frames
    sorted_frames = sorted(frames.keys())

    # Track management
    next_track_id = 1
    active_tracks = {}  # track_id -> {'box': [x,y,w,h], 'last_seen': frame_num}

    # Process frames in order
    for frame_num in sorted_frames:
        dets = frames[frame_num]

        if not active_tracks:
            # First frame or no active tracks - assign new IDs
            for det in dets:
                det['track_id'] = next_track_id
                active_tracks[next_track_id] = {
                    'box': det['box'],
                    'last_seen': frame_num
                }
                next_track_id += 1
        else:
            # Match detections to existing tracks using Hungarian algorithm
            track_ids = list(active_tracks.keys())
            track_boxes = [active_tracks[tid]['box'] for tid in track_ids]

            if dets and track_boxes:
                # Compute cost matrix (negative IoU)
                cost_matrix = np.zeros((len(dets), len(track_boxes)))
                for i, det in enumerate(dets):
                    for j, track_box in enumerate(track_boxes):
                        cost_matrix[i, j] = -compute_iou(det['box'], track_box)

                # Hungarian matching
                det_indices, track_indices = linear_sum_assignment(cost_matrix)

                matched_dets = set()
                matched_tracks = set()

                for det_idx, track_idx in zip(det_indices, track_indices):
                    iou = -cost_matrix[det_idx, track_idx]
                    if iou >= iou_threshold:
                        track_id = track_ids[track_idx]
                        dets[det_idx]['track_id'] = track_id
                        active_tracks[track_id]['box'] = dets[det_idx]['box']
                        active_tracks[track_id]['last_seen'] = frame_num
                        matched_dets.add(det_idx)
                        matched_tracks.add(track_idx)

                # Assign new IDs to unmatched detections
                for i, det in enumerate(dets):
                    if i not in matched_dets:
                        det['track_id'] = next_track_id
                        active_tracks[next_track_id] = {
                            'box': det['box'],
                            'last_seen': frame_num
                        }
                        next_track_id += 1
            else:
                # No tracks to match against
                for det in dets:
                    det['track_id'] = next_track_id
                    active_tracks[next_track_id] = {
                        'box': det['box'],
                        'last_seen': frame_num
                    }
                    next_track_id += 1

        # Remove stale tracks
        stale_tracks = [
            tid for tid, track in active_tracks.items()
            if frame_num - track['last_seen'] > max_frames_to_skip
        ]
        for tid in stale_tracks:
            del active_tracks[tid]

    # Write gt.txt
    gt_dir = os.path.join(mot_path, sequence_path, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    gt_path = os.path.join(gt_dir, 'gt.txt')

    with open(gt_path, 'w') as f:
        for frame_num in sorted_frames:
            for det in frames[frame_num]:
                x, y, w, h = det['box']
                track_id = det['track_id']
                # MOT format: frame, id, x, y, w, h, mark, label
                f.write(f"{frame_num},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1\n")

    print(f"Saved gt.txt to {gt_path}")
    print(f"  Total tracks: {next_track_id - 1}")
    print(f"  Total detections: {sum(len(frames[f]) for f in frames)}")


def main():
    parser = argparse.ArgumentParser(description='Prepare Roboflow dataset for MOTRv2')
    parser.add_argument('--dataset-dir', required=True,
                        help='Path to dataset directory (e.g., data/Dataset/mot/Volleyball-Activity-Dataset-3)')
    parser.add_argument('--sequence', required=True,
                        help='Sequence path relative to dataset (e.g., train/match1)')
    parser.add_argument('--det-json', required=True,
                        help='Input detection JSON file (relative to sequence/gt/)')
    parser.add_argument('--output-json', default='det_db_dfine.json',
                        help='Output detection database JSON (default: det_db_dfine.json)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--skip-rename', action='store_true',
                        help='Skip image renaming step')
    parser.add_argument('--skip-gt', action='store_true',
                        help='Skip gt.txt generation')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                        help='IoU threshold for track matching (default: 0.3)')

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    sequence = args.sequence
    img_dir = os.path.join(dataset_dir, sequence, 'img1')
    gt_dir = os.path.join(dataset_dir, sequence, 'gt')
    input_json = os.path.join(gt_dir, args.det_json)

    print("="*60)
    print("Roboflow Dataset Preparation for MOTRv2")
    print("="*60)
    print(f"Dataset: {dataset_dir}")
    print(f"Sequence: {sequence}")
    print(f"Image dir: {img_dir}")
    print(f"Detection JSON: {input_json}")
    print("="*60)

    # Step 1: Rename images
    if not args.skip_rename:
        print("\n[Step 1] Renaming Roboflow images...")
        rename_map = rename_roboflow_images(img_dir, dry_run=args.dry_run)
    else:
        print("\n[Step 1] Skipping image rename")

    # Step 2: Convert detection JSON
    if os.path.exists(input_json):
        print("\n[Step 2] Converting detection JSON...")
        if not args.dry_run:
            convert_json_detections(
                input_json,
                args.output_json,
                sequence,
                dataset_dir
            )
        else:
            print("[DRY RUN] Would convert JSON")
    else:
        print(f"\n[Step 2] Detection JSON not found: {input_json}")

    # Step 3: Generate gt.txt with track IDs
    if not args.skip_gt and os.path.exists(input_json):
        print("\n[Step 3] Generating gt.txt with track IDs...")
        if not args.dry_run:
            create_mot_gt_from_json(
                input_json,
                os.path.join(gt_dir, 'gt.txt'),
                sequence,
                dataset_dir,
                iou_threshold=args.iou_threshold
            )
        else:
            print("[DRY RUN] Would generate gt.txt")
    else:
        print("\n[Step 3] Skipping gt.txt generation")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    main()
