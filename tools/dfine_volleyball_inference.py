#!/usr/bin/env python3
"""
D-FINE Inference Script for Volleyball Ball Detection
Uses the official D-FINE repository: https://github.com/Peterande/D-FINE

Setup:
    1. Clone D-FINE repo: git clone https://github.com/Peterande/D-FINE.git
    2. Install requirements: pip install -r D-FINE/requirements.txt
    3. Download pretrained weights from the D-FINE releases

Usage:
    python dfine_volleyball_inference.py \
        --dfine_path /path/to/D-FINE \
        --sportsmot_dir /path/to/SportsMOT_volleyball \
        --output_dir /path/to/ball_detections \
        --config configs/dfine/dfine_hgnetv2_l_coco.yml \
        --weights dfine_l_coco.pth \
        --device cuda:0 \
        --conf_threshold 0.3
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FrameDataset(Dataset):
    """Dataset for loading frames from SportsMOT structure."""
    
    def __init__(self, sportsmot_dir, splits=None):
        self.frames = []
        
        if splits is None:
            splits = ['train', 'val', 'test']
        
        sportsmot_path = Path(sportsmot_dir)
        
        for split in splits:
            split_dir = sportsmot_path / split
            if not split_dir.exists():
                print(f"Warning: {split_dir} does not exist, skipping...")
                continue
                
            for seq_dir in sorted(split_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                    
                img_dir = seq_dir / 'img1'
                if not img_dir.exists():
                    print(f"Warning: {img_dir} does not exist, skipping...")
                    continue
                
                for img_path in sorted(img_dir.glob('*.jpg')):
                    frame_id = int(img_path.stem)
                    self.frames.append({
                        'path': str(img_path),
                        'split': split,
                        'sequence': seq_dir.name,
                        'frame_id': frame_id
                    })
        
        print(f"Found {len(self.frames)} frames across {splits}")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_info = self.frames[idx]
        return frame_info


class DFINEDetector:
    """Wrapper for official D-FINE model inference."""
    
    # COCO class IDs for ball-like objects
    # 32: sports ball
    BALL_CLASS_IDS = [32]
    
    def __init__(self, dfine_path, config_path, weights_path, device='cuda:0', conf_threshold=0.3):
        self.device = device
        self.conf_threshold = conf_threshold
        self.dfine_path = Path(dfine_path)
        
        # Add D-FINE to path
        sys.path.insert(0, str(self.dfine_path))
        
        # Import D-FINE modules
        from src.core import YAMLConfig
        from src.solver import TASKS
        
        # Load config
        cfg = YAMLConfig(config_path, resume=weights_path)
        
        # Build model
        self.model = cfg.model.to(device)
        self.model.eval()
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location=device)
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        
        # Store postprocessor
        self.postprocessor = cfg.postprocessor.to(device)
        self.postprocessor.eval()
        
        print(f"D-FINE model loaded on {device}")
        print(f"Config: {config_path}")
        print(f"Weights: {weights_path}")
        print(f"Ball class IDs (COCO): {self.BALL_CLASS_IDS}")
    
    def preprocess(self, image_path):
        """Preprocess a single image for D-FINE."""
        # Read image
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Resize to 640x640 (D-FINE default)
        img_resized = cv2.resize(img, (640, 640))
        
        # Normalize
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # HWC -> CHW -> BCHW
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0)
        
        return img_tensor.to(self.device), (orig_w, orig_h)
    
    @torch.no_grad()
    def detect_single(self, image_path):
        """
        Run detection on a single image.
        
        Returns:
            List of detections: [{'bbox': [x, y, w, h], 'confidence': float, 'class_id': int}, ...]
        """
        img_tensor, orig_size = self.preprocess(image_path)
        
        # Run inference
        outputs = self.model(img_tensor)
        
        # Post-process - pass original sizes as [w, h] without reversal
        # orig_size from preprocess is (w, h)
        orig_sizes = torch.tensor([list(orig_size)]).to(self.device)  # [w, h]
        results = self.postprocessor(outputs, orig_sizes)
        
        detections = []
        
        # Results format: list of dicts with 'boxes', 'scores', 'labels'
        if len(results) > 0:
            result = results[0]
            boxes = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score < self.conf_threshold:
                    continue
                
                if int(label) not in self.BALL_CLASS_IDS:
                    continue
                
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'confidence': float(score),
                    'class_id': int(label)
                })
        
        return detections
    
    @torch.no_grad()
    def detect_batch(self, image_paths):
        """
        Run detection on a batch of images.
        
        Returns:
            List of detection lists
        """
        if len(image_paths) == 0:
            return []
        
        # Preprocess all images
        tensors = []
        orig_sizes = []
        
        for img_path in image_paths:
            tensor, orig_size = self.preprocess(img_path)
            tensors.append(tensor)
            orig_sizes.append(list(orig_size))  # [w, h] - no reversal needed
        
        # Stack into batch
        batch = torch.cat(tensors, dim=0)
        sizes = torch.tensor(orig_sizes).to(self.device)
        
        # Run inference
        outputs = self.model(batch)
        
        # Post-process
        results = self.postprocessor(outputs, sizes)
        
        all_detections = []
        
        for result in results:
            detections = []
            boxes = result['boxes'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score < self.conf_threshold:
                    continue
                
                if int(label) not in self.BALL_CLASS_IDS:
                    continue
                
                x1, y1, x2, y2 = box
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'confidence': float(score),
                    'class_id': int(label)
                })
            
            all_detections.append(detections)
        
        return all_detections


def run_inference(args):
    """Main inference loop."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = DFINEDetector(
        dfine_path=args.dfine_path,
        config_path=args.config,
        weights_path=args.weights,
        device=args.device,
        conf_threshold=args.conf_threshold
    )
    
    # Create dataset
    dataset = FrameDataset(args.sportsmot_dir, splits=args.splits)
    
    # Dictionary to store detections per sequence
    # Structure: {split: {sequence: {frame_id: [detections]}}}
    all_results = {}
    
    # Process frames
    if args.batch_size > 1:
        # Batch processing
        print(f"Processing with batch size {args.batch_size}...")
        
        # Group frames by sequence for better cache locality
        seq_frames = {}
        for frame_info in dataset.frames:
            key = (frame_info['split'], frame_info['sequence'])
            if key not in seq_frames:
                seq_frames[key] = []
            seq_frames[key].append(frame_info)
        
        for (split, seq), frames in tqdm(seq_frames.items(), desc="Processing sequences"):
            if split not in all_results:
                all_results[split] = {}
            if seq not in all_results[split]:
                all_results[split][seq] = {}
            
            # Process in batches
            for i in range(0, len(frames), args.batch_size):
                batch_frames = frames[i:i + args.batch_size]
                batch_paths = [f['path'] for f in batch_frames]
                
                try:
                    batch_dets = detector.detect_batch(batch_paths)
                    
                    for frame_info, dets in zip(batch_frames, batch_dets):
                        all_results[split][seq][frame_info['frame_id']] = dets
                except Exception as e:
                    print(f"Error processing batch in {seq}: {e}")
                    # Fall back to single image processing
                    for frame_info in batch_frames:
                        try:
                            dets = detector.detect_single(frame_info['path'])
                            all_results[split][seq][frame_info['frame_id']] = dets
                        except Exception as e2:
                            print(f"Error processing {frame_info['path']}: {e2}")
                            all_results[split][seq][frame_info['frame_id']] = []
    else:
        # Single image processing
        print("Processing single images...")
        for frame_info in tqdm(dataset.frames, desc="Running inference"):
            split = frame_info['split']
            seq = frame_info['sequence']
            frame_id = frame_info['frame_id']
            
            if split not in all_results:
                all_results[split] = {}
            if seq not in all_results[split]:
                all_results[split][seq] = {}
            
            try:
                dets = detector.detect_single(frame_info['path'])
                all_results[split][seq][frame_id] = dets
            except Exception as e:
                print(f"Error processing {frame_info['path']}: {e}")
                all_results[split][seq][frame_id] = []
    
    # Save results per sequence
    for split, sequences in all_results.items():
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        
        for seq, frames in sequences.items():
            # Convert int keys to strings for JSON
            frames_str_keys = {str(k): v for k, v in frames.items()}
            
            seq_file = split_dir / f"{seq}_ball_detections.json"
            with open(seq_file, 'w') as f:
                json.dump(frames_str_keys, f, indent=2)
            print(f"Saved {seq_file}")
    
    # Save summary
    total_frames = len(dataset)
    total_detections = sum(
        len(dets) 
        for seqs in all_results.values() 
        for frames in seqs.values() 
        for dets in frames.values()
    )
    frames_with_detection = sum(
        1 
        for seqs in all_results.values() 
        for frames in seqs.values() 
        for dets in frames.values() 
        if len(dets) > 0
    )
    
    summary = {
        'total_frames': total_frames,
        'total_detections': total_detections,
        'frames_with_detection': frames_with_detection,
        'detection_rate': frames_with_detection / max(1, total_frames),
        'sequences': {
            split: list(seqs.keys()) 
            for split, seqs in all_results.items()
        },
        'config': {
            'conf_threshold': args.conf_threshold,
            'dfine_config': args.config,
            'dfine_weights': args.weights,
            'ball_class_ids': DFINEDetector.BALL_CLASS_IDS
        }
    }
    
    with open(output_dir / 'inference_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Inference complete!")
    print(f"{'='*50}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with ball detection: {frames_with_detection} ({100*frames_with_detection/max(1,total_frames):.1f}%)")
    print(f"Total detections: {total_detections}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='D-FINE inference for volleyball ball detection')
    
    # D-FINE paths
    parser.add_argument('--dfine_path', type=str, required=True,
                        help='Path to D-FINE repository')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to D-FINE config file (relative to dfine_path or absolute)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to D-FINE weights file')
    
    # Dataset paths
    parser.add_argument('--sportsmot_dir', type=str, required=True,
                        help='Path to SportsMOT volleyball dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save detection results')
    
    # Inference settings
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference on')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (1 for safest)')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to process')
    
    args = parser.parse_args()
    
    # Resolve config path
    if not os.path.isabs(args.config):
        args.config = os.path.join(args.dfine_path, args.config)
    
    run_inference(args)


if __name__ == '__main__':
    main()