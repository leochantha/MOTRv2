#!/usr/bin/env python3
"""
Convert JSON detections to MOT format ground truth with track IDs.

This script:
1. Loads detections from JSON format: {"path/img1/000001": ["x,y,w,h,score\n", ...]}
2. Associates detections across frames using IoU-based Hungarian matching
3. Outputs MOT-style gt.txt files with track IDs

Usage:
    python tools/convert_detections_to_mot.py \
        --input detections.json \
        --output_dir /path/to/output \
        --iou_threshold 0.3 \
        --max_age 30 \
        --min_hits 3

Output structure:
    {output_dir}/
    ├── train/
    │   └── combined/
    │       ├── img1/
    │       │   └── (symlinks or copy images here)
    │       └── gt/
    │           └── gt.txt
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args():
    parser = argparse.ArgumentParser(description='Convert JSON detections to MOT format with track IDs')
    parser.add_argument('--input', '-i', required=True, type=str,
                        help='Path to input JSON detection file')
    parser.add_argument('--output_dir', '-o', required=True, type=str,
                        help='Output directory for MOT-format data')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for matching detections (default: 0.3)')
    parser.add_argument('--max_age', type=int, default=30,
                        help='Maximum frames to keep a track alive without detection (default: 30)')
    parser.add_argument('--min_hits', type=int, default=3,
                        help='Minimum detections before a track is confirmed (default: 3)')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Minimum detection score to consider (default: 0.5)')
    parser.add_argument('--split', type=str, default='train',
                        help='Split name for output directory (default: train)')
    return parser.parse_args()


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    # Convert to [x1, y1, x2, y2]
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Intersection
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_iou_matrix(boxes1, boxes2):
    """Compute IoU matrix between two sets of boxes."""
    n1, n2 = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((n1, n2))

    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            iou_matrix[i, j] = compute_iou(b1, b2)

    return iou_matrix


class Track:
    """Simple track class for IoU-based tracking."""

    _id_counter = 0

    def __init__(self, box, score, frame_id):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.box = box  # [x, y, w, h]
        self.score = score
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = [(frame_id, box, score)]

    def update(self, box, score, frame_id):
        """Update track with new detection."""
        self.box = box
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.history.append((frame_id, box, score))

    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.age += 1
        self.time_since_update += 1

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0


class SimpleTracker:
    """
    Simple IoU-based tracker (similar to SORT without Kalman filter).

    Associates detections across frames using Hungarian algorithm with IoU cost.
    """

    def __init__(self, iou_threshold=0.3, max_age=30, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0

    def update(self, detections, frame_id):
        """
        Update tracker with new detections.

        Args:
            detections: List of [x, y, w, h, score]
            frame_id: Current frame number

        Returns:
            List of (track_id, x, y, w, h, score) for confirmed tracks
        """
        self.frame_count += 1

        # Get predicted locations from existing tracks
        if len(self.tracks) == 0:
            # No existing tracks, create new ones
            for det in detections:
                self.tracks.append(Track(det[:4], det[4], frame_id))
            return []

        # Compute IoU matrix between tracks and detections
        track_boxes = [t.box for t in self.tracks]
        det_boxes = [d[:4] for d in detections]

        if len(det_boxes) == 0:
            # No detections, mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
            return self._get_confirmed_tracks(frame_id)

        iou_matrix = compute_iou_matrix(track_boxes, det_boxes)

        # Hungarian algorithm for assignment (minimize cost = 1 - IoU)
        cost_matrix = 1 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Process matches
        matched_tracks = set()
        matched_dets = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                self.tracks[row].update(detections[col][:4], detections[col][4], frame_id)
                matched_tracks.add(row)
                matched_dets.add(col)

        # Mark unmatched tracks as missed
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks.append(Track(det[:4], det[4], frame_id))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]

        return self._get_confirmed_tracks(frame_id)

    def _get_confirmed_tracks(self, frame_id):
        """Get tracks that have been confirmed (enough hits)."""
        results = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                results.append((
                    track.id,
                    track.box[0],  # x
                    track.box[1],  # y
                    track.box[2],  # w
                    track.box[3],  # h
                    track.score
                ))
        return results

    def get_all_track_history(self):
        """Get full history of all tracks (including unconfirmed)."""
        all_history = []
        for track in self.tracks:
            for frame_id, box, score in track.history:
                all_history.append((frame_id, track.id, box, score))
        return all_history


def parse_json_key(key):
    """
    Parse JSON key to extract video path and frame number.

    Key format: "volleyball/combined/gt/img1/000001"
    Returns: (video_path, frame_number)
    """
    parts = key.split('/')

    # Find 'img1' in path
    try:
        img1_idx = parts.index('img1')
        # Video path is everything before img1, but we want to use a cleaner path
        # e.g., "volleyball/combined/gt" -> "volleyball/combined"
        vid_parts = parts[:img1_idx]
        # Remove 'gt' if it's at the end
        if vid_parts and vid_parts[-1] == 'gt':
            vid_parts = vid_parts[:-1]
        vid_path = '/'.join(vid_parts) if vid_parts else 'video'

        frame_str = parts[img1_idx + 1]
        frame_num = int(frame_str)
    except (ValueError, IndexError):
        # Fallback: assume last part is frame number
        vid_path = '/'.join(parts[:-1]) if len(parts) > 1 else 'video'
        frame_num = int(parts[-1])

    return vid_path, frame_num


def parse_detection_string(det_str):
    """
    Parse detection string to extract box and score.

    Format: "x,y,w,h,score\n"
    Returns: [x, y, w, h, score]
    """
    det_str = det_str.strip()
    parts = det_str.split(',')

    if len(parts) >= 5:
        x, y, w, h, score = map(float, parts[:5])
    elif len(parts) >= 4:
        x, y, w, h = map(float, parts[:4])
        score = 1.0
    else:
        return None

    return [x, y, w, h, score]


def main():
    args = parse_args()

    print(f"Loading detections from: {args.input}")
    with open(args.input) as f:
        det_data = json.load(f)

    print(f"Found {len(det_data)} frames in JSON")

    # Group detections by video
    videos = defaultdict(dict)  # video_path -> {frame_num: [detections]}

    for key, det_list in det_data.items():
        vid_path, frame_num = parse_json_key(key)

        detections = []
        for det_str in det_list:
            det = parse_detection_string(det_str)
            if det is not None and det[4] >= args.score_threshold:
                detections.append(det)

        if detections:
            videos[vid_path][frame_num] = detections

    print(f"Found {len(videos)} video sequences")

    # Process each video
    for vid_path, frames in videos.items():
        print(f"\nProcessing: {vid_path}")

        # Sort frames by frame number
        frame_nums = sorted(frames.keys())
        print(f"  Frames: {frame_nums[0]} to {frame_nums[-1]} ({len(frame_nums)} frames)")

        # Reset tracker for each video
        Track.reset_id_counter()
        tracker = SimpleTracker(
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            min_hits=args.min_hits
        )

        # Collect all tracking results
        all_results = []  # List of (frame_id, track_id, x, y, w, h, score)

        # Process frames in order
        for frame_num in frame_nums:
            detections = frames[frame_num]

            # Update tracker
            tracked = tracker.update(detections, frame_num)

            # Store results
            for track_id, x, y, w, h, score in tracked:
                all_results.append((frame_num, track_id, x, y, w, h, score))

        # Also get tracks that didn't meet min_hits but have history
        # (useful for short sequences)
        if len(all_results) == 0:
            print("  Warning: No confirmed tracks. Using all track history...")
            for track in tracker.tracks:
                for frame_id, box, score in track.history:
                    all_results.append((
                        frame_id, track.id,
                        box[0], box[1], box[2], box[3],
                        score
                    ))

        # Sort results by frame
        all_results.sort(key=lambda x: (x[0], x[1]))

        # Count unique track IDs
        unique_tracks = len(set(r[1] for r in all_results))
        print(f"  Generated {len(all_results)} annotations with {unique_tracks} unique tracks")

        # Create output directory structure
        # vid_path might be "volleyball/combined", we want to create:
        # output_dir/train/combined/gt/gt.txt
        vid_name = vid_path.split('/')[-1] if '/' in vid_path else vid_path
        output_vid_dir = Path(args.output_dir) / args.split / vid_name
        gt_dir = output_vid_dir / 'gt'
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Write gt.txt
        gt_path = gt_dir / 'gt.txt'
        with open(gt_path, 'w') as f:
            for frame_id, track_id, x, y, w, h, score in all_results:
                # MOT format: frame_id, track_id, x, y, w, h, mark, label
                # mark=1 means valid, label=1 means pedestrian/player
                line = f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1\n"
                f.write(line)

        print(f"  Saved: {gt_path}")

        # Create img1 directory placeholder
        img1_dir = output_vid_dir / 'img1'
        img1_dir.mkdir(parents=True, exist_ok=True)

        # Write a README about image setup
        readme_path = img1_dir / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write("Place your images here or create symlinks.\n")
            f.write(f"Expected format: {frame_nums[0]:08d}.jpg to {frame_nums[-1]:08d}.jpg\n")
            f.write("Or use 6-digit format: 000001.jpg\n")

    # Create data path file
    data_path_dir = Path(args.output_dir).parent / 'datasets' / 'data_path'
    data_path_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_path_dir / 'volleyball.train'
    with open(train_file, 'w') as f:
        for vid_path in videos.keys():
            vid_name = vid_path.split('/')[-1] if '/' in vid_path else vid_path
            f.write(f"{vid_name}\n")

    print(f"\nCreated data path file: {train_file}")

    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"1. Copy/symlink your images to: {args.output_dir}/{args.split}/*/img1/")
    print(f"2. Update your config to use:")
    print(f"   --mot_path {args.output_dir}")
    print(f"   --dataset_file e2e_volleyball")
    print(f"\nTo train:")
    print(f"   python main.py --args_file configs/motrv2_volleyball.args")


if __name__ == '__main__':
    main()
