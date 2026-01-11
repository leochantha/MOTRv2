#!/usr/bin/env python3
"""
Unified MOTRv2 Inference and Visualization Pipeline.

This script:
1. Runs MOTRv2 inference on video sequences
2. Automatically saves tracking results in MOT format
3. Optionally generates visualization videos

Usage:
    # Basic inference
    python tools/infer_and_visualize.py \
        --checkpoint exps/motrv2_volleyball/run1/checkpoint.pth \
        --data_dir /data/Dataset/volleyball/test \
        --det_db det_db_motrv2.json

    # With visualization
    python tools/infer_and_visualize.py \
        --checkpoint exps/motrv2_volleyball/run1/checkpoint.pth \
        --data_dir /data/Dataset/volleyball/test \
        --det_db det_db_motrv2.json \
        --visualize

    # Compare multiple models
    python tools/infer_and_visualize.py \
        --compare \
        --checkpoints model1.pth model2.pth \
        --names "MOTRv2-DFine" "MOTRv2-YOLOX" \
        --data_dir /data/Dataset/volleyball/test

Output structure:
    {output_dir}/
    ├── {exp_name}/
    │   ├── {seq_name}.txt          # MOT format tracking results
    │   ├── {seq_name}_video.mp4    # Visualization (if --visualize)
    │   └── config.json             # Run configuration
    └── comparisons/
        └── {seq_name}_comparison.mp4  # Side-by-side comparison
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_default_exp_name(checkpoint_path):
    """Generate experiment name from checkpoint path."""
    path = Path(checkpoint_path)
    # Try to extract meaningful name from path
    # e.g., exps/motrv2_volleyball/run1/checkpoint.pth -> motrv2_volleyball_run1
    parts = path.parts

    # Find 'exps' in path and take next parts
    if 'exps' in parts:
        idx = parts.index('exps')
        name_parts = parts[idx+1:-1]  # Exclude 'exps' and filename
        return '_'.join(name_parts)

    # Fallback: use parent directory name + timestamp
    return f"{path.parent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_model_name_from_path(path):
    """Extract a readable model name from file path."""
    path = Path(path)

    # For checkpoint: exps/motrv2_volleyball/run1/checkpoint.pth
    if path.suffix == '.pth':
        parts = path.parts
        if 'exps' in parts:
            idx = parts.index('exps')
            return '/'.join(parts[idx+1:-1])
        return path.parent.name

    # For tracking results: tracker/test1.txt
    if path.suffix == '.txt':
        return path.stem

    # For JSON: det_db_motrv2_DFINE.json
    if path.suffix == '.json':
        name = path.stem
        # Clean up common prefixes
        for prefix in ['det_db_', 'detections_']:
            if name.startswith(prefix):
                name = name[len(prefix):]
        return name

    return path.stem


def find_tracking_results(search_dirs, pattern="*.txt"):
    """Auto-discover tracking result files."""
    results = []

    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue

        # Search for .txt files (MOT format)
        for txt_file in search_path.rglob(pattern):
            # Skip non-tracking files
            if txt_file.name in ['config.txt', 'git_status', 'git_diff', 'desc']:
                continue
            results.append(str(txt_file))

    return sorted(results)


def find_sequences(data_dir):
    """Find all video sequences in data directory."""
    data_path = Path(data_dir)
    sequences = []

    for seq_dir in sorted(data_path.iterdir()):
        if seq_dir.is_dir() and (seq_dir / 'img1').exists():
            sequences.append(seq_dir.name)

    return sequences


def run_inference(args, checkpoint, seq_dir, output_dir, exp_name):
    """Run MOTRv2 inference on a sequence."""
    import torch
    import cv2
    from tqdm import tqdm
    from copy import deepcopy
    from torch.utils.data import DataLoader

    from models import build_model
    from util.tool import load_model
    from submit_dance import ListImgDataset, RuntimeTrackerBase
    from main import get_args_parser

    # Build model
    parser = get_args_parser()
    model_args = parser.parse_args([])

    # Override with our args
    model_args.resume = checkpoint
    model_args.mot_path = str(Path(seq_dir).parent.parent)
    model_args.det_db = args.det_db
    model_args.output_dir = output_dir
    model_args.exp_name = exp_name

    # Load model
    detr, _, _ = build_model(model_args)
    detr.track_embed.score_thr = args.score_threshold
    detr.track_base = RuntimeTrackerBase(
        args.score_threshold,
        args.score_threshold,
        args.miss_tolerance
    )
    detr = load_model(detr, checkpoint)
    detr.eval()
    detr = detr.cuda()

    # Get sequence info
    seq_name = Path(seq_dir).name
    img_list = sorted([
        str(Path(seq_dir) / 'img1' / f)
        for f in os.listdir(Path(seq_dir) / 'img1')
        if f.endswith('.jpg')
    ])

    # Load detection database
    det_db_path = Path(model_args.mot_path) / args.det_db
    with open(det_db_path) as f:
        det_db = json.load(f)

    # Run inference
    track_instances = None
    lines = []

    loader = DataLoader(
        ListImgDataset(model_args.mot_path, img_list, det_db),
        batch_size=1,
        num_workers=2
    )

    print(f"Running inference on {seq_name}...")
    for i, data in enumerate(tqdm(loader, desc=seq_name)):
        cur_img, ori_img, proposals = [d[0] for d in data]
        cur_img, proposals = cur_img.cuda(), proposals.cuda()

        if track_instances is not None:
            track_instances.remove('boxes')
            track_instances.remove('labels')

        seq_h, seq_w, _ = ori_img.shape
        res = detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
        track_instances = res['track_instances']

        dt_instances = deepcopy(track_instances)

        # Filter by score and area
        keep = dt_instances.scores > args.score_threshold
        keep &= dt_instances.obj_idxes >= 0
        dt_instances = dt_instances[keep]

        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > args.area_threshold
        dt_instances = dt_instances[keep]

        # Save results
        save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
        for xyxy, track_id in zip(dt_instances.boxes.tolist(), dt_instances.obj_idxes.tolist()):
            if track_id < 0:
                continue
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            lines.append(save_format.format(frame=i + 1, id=int(track_id), x1=x1, y1=y1, w=w, h=h))

    # Save tracking results
    result_dir = Path(output_dir) / exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f'{seq_name}.txt'

    with open(result_file, 'w') as f:
        f.writelines(lines)

    print(f"Saved tracking results to: {result_file}")
    return str(result_file)


def create_visualization(frames_dir, tracking_file, output_video, model_name,
                         fps=30, show_ids=True):
    """Create visualization video for a single model."""
    import cv2
    import numpy as np
    from collections import defaultdict

    # Import from create_video.py
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from create_video import (
        parse_tracking_data,
        draw_bounding_boxes,
        add_panel_label
    )

    # Parse tracking data
    tracking_data = parse_tracking_data(tracking_file)

    # Get frame files
    frame_files = sorted([
        str(f) for f in Path(frames_dir).glob('*.jpg')
    ])

    if not frame_files:
        frame_files = sorted([str(f) for f in Path(frames_dir).glob('*.png')])

    if not frame_files:
        print(f"No frames found in {frames_dir}")
        return None

    # Get dimensions
    first_frame = cv2.imread(frame_files[0])
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height + 40))

    print(f"Creating visualization: {output_video}")
    for frame_file in tqdm(frame_files, desc="Rendering"):
        frame = cv2.imread(frame_file)

        # Extract frame number
        import re
        numbers = re.findall(r'\d+', Path(frame_file).stem)
        frame_num = int(numbers[-1]) if numbers else 1

        # Draw detections
        detections = tracking_data.get(frame_num, [])
        if detections:
            frame = draw_bounding_boxes(frame, detections, show_ids=show_ids)

        # Add frame info
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracks: {len(detections)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add model label
        frame = add_panel_label(frame, model_name, 'top')

        out.write(frame)

    out.release()
    print(f"Saved visualization to: {output_video}")
    return output_video


def main():
    parser = argparse.ArgumentParser(
        description='MOTRv2 Inference and Visualization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--infer', action='store_true', default=True,
                           help='Run inference (default)')
    mode_group.add_argument('--visualize-only', action='store_true',
                           help='Only create visualization from existing results')
    mode_group.add_argument('--compare', action='store_true',
                           help='Compare multiple tracking results')

    # Inference options
    infer_group = parser.add_argument_group('Inference Options')
    infer_group.add_argument('--checkpoint', '-c', type=str,
                            help='Path to model checkpoint')
    infer_group.add_argument('--checkpoints', nargs='+',
                            help='Multiple checkpoints for comparison')
    infer_group.add_argument('--det_db', type=str, required=True,
                            help='Detection database JSON file')
    infer_group.add_argument('--data_dir', '-d', type=str,
                            help='Directory containing test sequences')
    infer_group.add_argument('--sequences', nargs='+',
                            help='Specific sequences to process (default: all)')
    infer_group.add_argument('--score_threshold', type=float, default=0.5,
                            help='Detection score threshold (default: 0.5)')
    infer_group.add_argument('--miss_tolerance', type=int, default=20,
                            help='Frames to keep lost tracks (default: 20)')
    infer_group.add_argument('--area_threshold', type=float, default=100,
                            help='Minimum detection area (default: 100)')

    # Visualization options
    vis_group = parser.add_argument_group('Visualization Options')
    vis_group.add_argument('--visualize', '-v', action='store_true',
                          help='Generate visualization video after inference')
    vis_group.add_argument('--tracking_files', nargs='+',
                          help='Tracking result files (for --visualize-only or --compare)')
    vis_group.add_argument('--names', nargs='+',
                          help='Model names for visualization (auto-generated if not provided)')
    vis_group.add_argument('--fps', type=int, default=30,
                          help='Video FPS (default: 30)')
    vis_group.add_argument('--no-ids', action='store_true',
                          help='Hide track IDs in visualization')
    vis_group.add_argument('--frames_dir', type=str,
                          help='Frames directory (for --visualize-only)')

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', '-o', type=str, default='./results',
                             help='Output directory (default: ./results)')
    output_group.add_argument('--exp_name', type=str,
                             help='Experiment name (auto-generated if not provided)')

    # Discovery options
    discover_group = parser.add_argument_group('Auto-Discovery Options')
    discover_group.add_argument('--discover', action='store_true',
                               help='Auto-discover tracking results in common locations')
    discover_group.add_argument('--search_dirs', nargs='+',
                               default=['./tracker', './results', './exps'],
                               help='Directories to search for tracking results')

    args = parser.parse_args()

    # Handle different modes
    if args.discover:
        # Auto-discover mode
        print("Searching for tracking results...")
        tracking_files = find_tracking_results(args.search_dirs)

        if not tracking_files:
            print("No tracking results found. Searched in:")
            for d in args.search_dirs:
                print(f"  {d}")
            return

        print(f"Found {len(tracking_files)} tracking result files:")
        for i, f in enumerate(tracking_files):
            print(f"  [{i+1}] {f}")

        # Let user select
        print("\nEnter file numbers to compare (comma-separated), or 'all':")
        selection = input("> ").strip()

        if selection.lower() == 'all':
            args.tracking_files = tracking_files
        else:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            args.tracking_files = [tracking_files[i] for i in indices]

        args.names = [get_model_name_from_path(f) for f in args.tracking_files]
        args.compare = True

    if args.compare or args.visualize_only:
        # Visualization/comparison mode
        if not args.tracking_files:
            parser.error("--tracking_files required for visualization")

        if not args.frames_dir:
            parser.error("--frames_dir required for visualization")

        # Auto-generate names if not provided
        if not args.names:
            args.names = [get_model_name_from_path(f) for f in args.tracking_files]

        # Import and run visualization
        from create_video import create_comparison_video

        output_video = Path(args.output_dir) / 'comparison.mp4'
        output_video.parent.mkdir(parents=True, exist_ok=True)

        create_comparison_video(
            frames_dir=args.frames_dir,
            tracking_files=args.tracking_files,
            model_names=args.names,
            output_video=str(output_video),
            fps=args.fps,
            show_ids=not args.no_ids
        )

    elif args.checkpoint or args.checkpoints:
        # Inference mode
        checkpoints = args.checkpoints if args.checkpoints else [args.checkpoint]

        if not args.data_dir:
            parser.error("--data_dir required for inference")

        # Find sequences
        sequences = args.sequences if args.sequences else find_sequences(args.data_dir)

        if not sequences:
            print(f"No sequences found in {args.data_dir}")
            return

        print(f"Found {len(sequences)} sequences: {sequences}")

        # Process each checkpoint
        all_results = {}
        for checkpoint in checkpoints:
            exp_name = args.exp_name or get_default_exp_name(checkpoint)
            model_name = get_model_name_from_path(checkpoint)

            print(f"\n{'='*60}")
            print(f"Processing: {model_name}")
            print(f"Checkpoint: {checkpoint}")
            print(f"{'='*60}")

            results = []
            for seq in sequences:
                seq_dir = Path(args.data_dir) / seq
                result_file = run_inference(
                    args, checkpoint, str(seq_dir),
                    args.output_dir, exp_name
                )
                results.append(result_file)

                # Optional visualization
                if args.visualize:
                    video_file = str(Path(result_file).with_suffix('.mp4'))
                    create_visualization(
                        str(seq_dir / 'img1'),
                        result_file,
                        video_file,
                        model_name,
                        fps=args.fps,
                        show_ids=not args.no_ids
                    )

            all_results[model_name] = results

        # Save config
        config = {
            'timestamp': datetime.now().isoformat(),
            'checkpoints': checkpoints,
            'sequences': sequences,
            'args': vars(args),
            'results': all_results
        }

        config_file = Path(args.output_dir) / 'run_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\nConfig saved to: {config_file}")
        print(f"Results saved to: {args.output_dir}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
