# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Volleyball dataset for MOTRv2 training.
Supports tracking players in volleyball matches.

Supports two data formats:
1. MOT-style: gt/gt.txt files with frame_id, track_id, x, y, w, h, mark, label
2. JSON-style: Single JSON file with frame paths as keys and detection lists as values
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances

from random import choice, randint


class DetVolleyballDetection:
    """
    Volleyball dataset for multi-object tracking.

    Supports two annotation formats:

    FORMAT 1 - MOT-style (recommended for tracking):
    ================================================
    {mot_path}/
    ├── train/
    │   ├── match1/
    │   │   ├── img1/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   └── gt/
    │   │       └── gt.txt  (frame_id, track_id, x, y, w, h, mark, label)
    │   └── ...

    FORMAT 2 - JSON-style (your format):
    =====================================
    Single JSON file (specified via --gt_json) with structure:
    {
        "volleyball/combined/gt/img1/000001": [
            "x,y,w,h,score\\n",
            ...
        ],
        ...
    }

    For JSON format, you also need --gt_json_has_track_ids=True if your format includes track IDs:
        "track_id,x,y,w,h,score\\n"

    Without track IDs, the dataset will generate pseudo-IDs per frame (NOT suitable for tracking training,
    but can be used with detection proposals for inference).
    """

    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path
        self.data_txt_path = data_txt_path

        self.labels_full = defaultdict(lambda: defaultdict(list))
        self.frame_paths = {}  # Maps (vid, frame_idx) -> image path

        # Check which format to use
        gt_json_path = getattr(args, 'gt_json', None)
        if gt_json_path:
            self._load_json_format(gt_json_path, args)
        else:
            self._load_mot_format(data_txt_path)

        vid_files = list(self.labels_full.keys())

        # Initialize existing_frames if not set (e.g., for JSON format)
        if not hasattr(self, 'existing_frames'):
            self.existing_frames = {}

        self.indices = []
        self.vid_tmax = {}
        self.vid_frames = {}  # Store sorted list of valid frames per video

        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            if not self.labels_full[vid]:
                continue

            # Get frames that have both images AND annotations
            annotated_frames = set(self.labels_full[vid].keys())
            if vid in self.existing_frames:
                # Intersect with existing images
                valid_frames = sorted(annotated_frames & set(self.existing_frames[vid]))
            else:
                valid_frames = sorted(annotated_frames)

            if len(valid_frames) < self.num_frames_per_batch:
                print(f"Warning: {vid} has only {len(valid_frames)} valid frames, need {self.num_frames_per_batch}")
                continue

            self.vid_frames[vid] = valid_frames
            self.vid_tmax[vid] = valid_frames[-1]

            # Create indices - use position in valid_frames list, not raw frame numbers
            for idx in range(len(valid_frames) - self.num_frames_per_batch + 1):
                self.indices.append((vid, idx))

        print(f"Found {len(vid_files)} volleyball videos, {len(self.indices)} valid training samples")

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print(f"sampler_steps={self.sampler_steps} lengths={self.lengths}")
        self.period_idx = 0

        # Load detection database if provided (for proposals)
        if args.det_db:
            det_db_path = os.path.join(args.mot_path, args.det_db)
            if os.path.exists(det_db_path):
                with open(det_db_path) as f:
                    self.det_db = json.load(f)
                print(f"Loaded detection database from {det_db_path}")
            else:
                print(f"Warning: Detection database {det_db_path} not found, using empty")
                self.det_db = defaultdict(list)
        else:
            self.det_db = defaultdict(list)

    def _load_json_format(self, gt_json_path, args):
        """
        Load annotations from JSON format.

        Expected JSON structure:
        {
            "volleyball/combined/gt/img1/000001": [
                "x,y,w,h,score\\n",  # without track_id
                OR
                "track_id,x,y,w,h,score\\n",  # with track_id
                ...
            ],
            ...
        }
        """
        full_path = os.path.join(self.mot_path, gt_json_path)
        if not os.path.exists(full_path):
            # Try as absolute path
            full_path = gt_json_path

        print(f"Loading JSON annotations from: {full_path}")

        with open(full_path) as f:
            gt_data = json.load(f)

        has_track_ids = getattr(args, 'gt_json_has_track_ids', False)
        print(f"JSON format has track IDs: {has_track_ids}")

        if not has_track_ids:
            print("WARNING: JSON format without track IDs. Generating pseudo-IDs.")
            print("         This is NOT recommended for tracking training!")
            print("         Consider adding track IDs to your annotations.")

        # Parse JSON keys to extract video and frame info
        # Key format: "volleyball/combined/gt/img1/000001"
        for key, detections in gt_data.items():
            # Parse the key to get video path and frame number
            parts = key.split('/')
            # Find 'img1' in path and extract frame number
            try:
                img1_idx = parts.index('img1')
                vid_path = '/'.join(parts[:img1_idx])  # e.g., "volleyball/combined/gt"
                frame_str = parts[img1_idx + 1]  # e.g., "000001"
                frame_idx = int(frame_str)
            except (ValueError, IndexError):
                # Alternative parsing: assume last part is frame number
                vid_path = '/'.join(parts[:-1])
                frame_str = parts[-1]
                frame_idx = int(frame_str)

            # Store image path for this frame
            # Assuming images are at: {mot_path}/{vid_path}/img1/{frame:06d}.jpg
            img_key = (vid_path, frame_idx)
            self.frame_paths[img_key] = key

            # Parse detections
            for det_idx, det_str in enumerate(detections):
                det_str = det_str.strip()
                if not det_str:
                    continue

                parts = det_str.split(',')

                if has_track_ids:
                    # Format: track_id, x, y, w, h, score
                    if len(parts) >= 6:
                        track_id = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:5])
                        score = float(parts[5]) if len(parts) > 5 else 1.0
                    else:
                        continue
                else:
                    # Format: x, y, w, h, score (no track_id)
                    if len(parts) >= 5:
                        x, y, w, h, score = map(float, parts[:5])
                        # Generate pseudo track ID (unique per detection per frame)
                        # This won't provide cross-frame association!
                        track_id = det_idx + 1
                    elif len(parts) >= 4:
                        x, y, w, h = map(float, parts[:4])
                        score = 1.0
                        track_id = det_idx + 1
                    else:
                        continue

                # Store: x, y, w, h, track_id, is_crowd, score
                self.labels_full[vid_path][frame_idx].append([x, y, w, h, track_id, False, score])

        print(f"Loaded {len(gt_data)} frames from JSON")

    def _load_mot_format(self, data_txt_path):
        """Load annotations from MOT-style gt.txt files.

        Reads sequences from data_txt_path file. Each line can be:
        - Just sequence name: "match1" (will look in train/, valid/, test/)
        - Full path: "train/match1"
        """

        def get_existing_frames(vid_path):
            """Scan img1 folder to find which frames actually exist."""
            img_dir = os.path.join(self.mot_path, vid_path, 'img1')
            if not os.path.exists(img_dir):
                return set()

            existing = set()
            import re
            for fname in os.listdir(img_dir):
                # Extract frame number from filename
                numbers = re.findall(r'\d+', fname)
                if numbers:
                    existing.add(int(numbers[0]))
            return existing

        def load_sequence(vid_path):
            """Load annotations for a single sequence."""
            gt_path = os.path.join(self.mot_path, vid_path, 'gt', 'gt.txt')

            if not os.path.exists(gt_path):
                print(f"Warning: GT file not found: {gt_path}")
                return False

            # First, scan which images actually exist
            existing_frames = get_existing_frames(vid_path)
            if not existing_frames:
                print(f"Warning: No images found for {vid_path}")
                return False

            print(f"  Loading: {vid_path} ({len(existing_frames)} images found)")

            loaded_frames = 0
            skipped_frames = 0

            for l in open(gt_path):
                parts = l.strip().split(',')
                if len(parts) < 6:
                    continue

                # Support both 8-column and 6-column formats
                if len(parts) >= 8:
                    t, i, *xywh, mark, label = parts[:8]
                    t, i, mark, label = map(int, (t, i, mark, label))
                    if mark == 0:
                        continue
                else:
                    # 6-column format: frame_id, track_id, x, y, w, h
                    t, i = int(parts[0]), int(parts[1])
                    xywh = parts[2:6]

                # Skip if image doesn't exist
                if t not in existing_frames:
                    skipped_frames += 1
                    continue

                x, y, w, h = map(float, xywh)
                # Store: x, y, w, h, track_id, is_crowd, score
                self.labels_full[vid_path][t].append([x, y, w, h, i, False, 1.0])
                loaded_frames += 1

            if skipped_frames > 0:
                print(f"    Skipped {skipped_frames} annotations (missing images)")
            print(f"    Loaded {len(self.labels_full[vid_path])} frames with annotations")

            # Store existing frames for this video (for sampling)
            self.existing_frames[vid_path] = sorted(existing_frames)

            return True

        # Initialize existing frames tracker
        self.existing_frames = {}

        # Read sequences from data_txt_path file
        print(f"Loading sequences from: {data_txt_path}")

        if not os.path.exists(data_txt_path):
            print(f"Warning: Data file {data_txt_path} not found")
            return

        with open(data_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Check if it's a full path (contains /) or just sequence name
                if '/' in line:
                    vid_path = line
                    if load_sequence(vid_path):
                        continue

                # Try common split directories
                for split in ['train', 'valid', 'val', 'test']:
                    vid_path = os.path.join(split, line)
                    full_path = os.path.join(self.mot_path, vid_path, 'gt', 'gt.txt')
                    if os.path.exists(full_path):
                        load_sequence(vid_path)
                        break
                else:
                    print(f"Warning: Sequence '{line}' not found in any split directory")

    def set_epoch(self, epoch):
        """Update sampling parameters based on current epoch."""
        self.current_epoch = epoch
        if self.sampler_steps is None or len(self.sampler_steps) == 0:
            return

        for i in range(len(self.sampler_steps)):
            if epoch >= self.sampler_steps[i]:
                self.period_idx = i + 1
        print(f"Set epoch: epoch {epoch} period_idx={self.period_idx}")
        self.num_frames_per_batch = self.lengths[self.period_idx]

    def step_epoch(self):
        """Called when one epoch finishes."""
        print(f"Dataset: epoch {self.current_epoch} finishes")
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        """Convert target dict to Instances object."""
        gt_instances = Instances(tuple(img_shape))
        n_gt = len(targets['labels'])
        gt_instances.boxes = targets['boxes'][:n_gt]
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        return gt_instances

    def _get_image_path(self, vid, idx):
        """Get image path for a given video and frame index."""
        # Check if we have a stored path from JSON format
        img_key = (vid, idx)
        if img_key in self.frame_paths:
            # JSON format - reconstruct image path
            # The JSON key might be gt path, we need to find the actual image
            json_key = self.frame_paths[img_key]
            # Try common image locations
            possible_paths = [
                os.path.join(self.mot_path, vid, 'img1', f'{idx:06d}.jpg'),
                os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg'),
                os.path.join(self.mot_path, vid.replace('/gt', ''), 'img1', f'{idx:06d}.jpg'),
                os.path.join(self.mot_path, vid.replace('/gt', ''), f'{idx:06d}.jpg'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path

        # Standard MOT format path
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')
        if os.path.exists(img_path):
            return img_path

        # Try 6-digit format
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:06d}.jpg')
        if os.path.exists(img_path):
            return img_path

        # Try alternative extensions
        for digits in [8, 6]:
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:0{digits}d}{ext}')
                if os.path.exists(img_path):
                    return img_path

        # Fallback
        return os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')

    def _pre_single_frame(self, vid, idx: int):
        """Load and preprocess a single frame."""
        img_path = self._get_image_path(vid, idx)

        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, f"Invalid image {img_path} with shape {w}x{h}"

        # Unique ID offset per video to ensure global uniqueness
        obj_idx_offset = self.video_dict[vid] * 100000

        targets['dataset'] = 'Volleyball'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])

        # Load ground truth annotations
        for ann in self.labels_full[vid][idx]:
            if len(ann) >= 7:
                x, y, w_box, h_box, track_id, crowd, score = ann[:7]
            elif len(ann) >= 6:
                x, y, w_box, h_box, track_id, crowd = ann[:6]
                score = 1.0
            else:
                continue

            targets['boxes'].append([x, y, w_box, h_box])
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)  # All objects mapped to class 0 for tracking
            targets['obj_ids'].append(track_id + obj_idx_offset)
            targets['scores'].append(score)

        # Load detection proposals if available
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        for line in self.det_db.get(txt_key, []):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                *box, s = map(float, parts[:5])
                targets['boxes'].append(box)
                targets['scores'].append(s)

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)

        # Convert from xywh to xyxy format
        if targets['boxes'].numel() > 0:
            targets['boxes'][:, 2:] += targets['boxes'][:, :2]

        return img, targets

    def _get_sample_range(self, start_idx):
        """Calculate frame sampling range."""
        assert self.sample_mode in ['fixed_interval', 'random_interval'], \
            f'Invalid sample mode: {self.sample_mode}'

        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)

        default_range = (
            start_idx,
            start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1,
            sample_interval
        )
        return default_range

    def pre_continuous_frames(self, vid, frame_numbers):
        """Load multiple frames by their actual frame numbers."""
        return zip(*[self._pre_single_frame(vid, frame_num) for frame_num in frame_numbers])

    def sample_indices(self, vid, start_pos):
        """Sample frame indices from valid frames list.

        Args:
            vid: Video identifier
            start_pos: Starting position in vid_frames list (not frame number)

        Returns:
            List of actual frame numbers to load
        """
        valid_frames = self.vid_frames[vid]
        max_pos = len(valid_frames) - 1

        if self.sample_mode == 'random_interval':
            # Random interval sampling from valid frames
            rate = randint(1, min(self.sample_interval + 1, (max_pos - start_pos) // self.num_frames_per_batch + 1))
        else:
            rate = 1

        # Sample positions in the valid_frames list
        positions = [min(start_pos + rate * i, max_pos) for i in range(self.num_frames_per_batch)]

        # Convert positions to actual frame numbers
        return [valid_frames[pos] for pos in positions]

    def __getitem__(self, idx):
        """Get a training sample (sequence of frames with annotations)."""
        vid, start_pos = self.indices[idx]
        frame_numbers = self.sample_indices(vid, start_pos)
        images, targets = self.pre_continuous_frames(vid, frame_numbers)

        if self.transform is not None:
            images, targets = self.transform(images, targets)

        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['labels'])
            proposals.append(torch.cat([
                targets_i['boxes'][n_gt:],
                targets_i['scores'][n_gt:, None],
            ], dim=1))

        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'proposals': proposals,
        }

    def __len__(self):
        return len(self.indices)


class DetVolleyballValidation(DetVolleyballDetection):
    """Validation dataset for volleyball."""

    def __init__(self, args, seqs_folder, transform):
        # Override to load validation split
        original_mot_path = args.mot_path
        super().__init__(args, args.data_txt_path_val, seqs_folder, transform)


def make_transforms_for_volleyball(image_set, args=None):
    """
    Create data augmentation transforms for volleyball dataset.

    Volleyball-specific considerations:
    - Court is typically horizontal, so horizontal flip is appropriate
    - Players move fast, so good motion blur handling helps
    - Court markings should be preserved
    """
    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Multi-scale training sizes
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'Unknown image_set: {image_set}')


def build_transform(args, image_set):
    """Build transforms for the specified image set."""
    if image_set == 'train':
        return make_transforms_for_volleyball('train', args)
    elif image_set == 'val':
        return make_transforms_for_volleyball('val', args)
    else:
        raise NotImplementedError(f"Unknown image_set: {image_set}")


def build(image_set, args):
    """
    Build volleyball dataset.

    Args:
        image_set: 'train' or 'val'
        args: Training arguments
            - mot_path: Base path to dataset
            - gt_json: (optional) Path to JSON annotation file
            - gt_json_has_track_ids: (optional) Whether JSON has track IDs

    Returns:
        DetVolleyballDetection dataset instance
    """
    root = Path(args.mot_path)
    assert root.exists(), f'Provided MOT path {root} does not exist'

    transform = build_transform(args, image_set)

    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetVolleyballDetection(
            args,
            data_txt_path=data_txt_path,
            seqs_folder=root,
            transform=transform
        )
    elif image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetVolleyballDetection(
            args,
            data_txt_path=data_txt_path,
            seqs_folder=root,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown image_set: {image_set}")

    return dataset
