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

    Expected data structure:
    {mot_path}/
    ├── train/
    │   ├── match1/
    │   │   ├── img1/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   └── gt/
    │   │       └── gt.txt
    │   └── match2/
    │       └── ...
    └── val/
        └── ...

    Ground truth format (gt.txt):
    frame_id, track_id, x, y, w, h, mark, label
    - frame_id: 1-indexed frame number
    - track_id: unique ID for each player
    - x, y, w, h: bounding box (top-left corner + width/height)
    - mark: 1 = valid, 0 = ignore
    - label: object class (1 = player, can extend for ball, referee, etc.)

    Label mapping:
    - 1: Player
    - 2: Ball (optional)
    - 3: Referee (optional)
    """

    # Class labels for volleyball
    VALID_LABELS = [1, 2, 3]  # Player, Ball, Referee
    LABEL_NAMES = {1: 'player', 2: 'ball', 3: 'referee'}

    def __init__(self, args, data_txt_path: str, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = args.mot_path

        self.labels_full = defaultdict(lambda: defaultdict(list))

        def add_volleyball_folder(split_dir):
            """Load annotations from a volleyball dataset split directory."""
            split_path = os.path.join(self.mot_path, split_dir)
            if not os.path.exists(split_path):
                print(f"Warning: Split directory {split_path} does not exist")
                return

            print(f"Adding volleyball data from: {split_dir}")
            for vid in os.listdir(split_path):
                if vid.startswith('.') or vid == 'seqmap':
                    continue

                vid_path = os.path.join(split_dir, vid)
                gt_path = os.path.join(self.mot_path, vid_path, 'gt', 'gt.txt')

                if not os.path.exists(gt_path):
                    print(f"Warning: GT file not found for {vid_path}")
                    continue

                for l in open(gt_path):
                    parts = l.strip().split(',')
                    if len(parts) < 8:
                        continue

                    t, i, *xywh, mark, label = parts[:8]
                    t, i, mark, label = map(int, (t, i, mark, label))

                    # Skip ignored annotations
                    if mark == 0:
                        continue

                    # Filter by valid labels (default: keep all valid volleyball labels)
                    if label not in self.VALID_LABELS:
                        continue

                    x, y, w, h = map(float, xywh)
                    # Store: x, y, w, h, track_id, is_crowd
                    self.labels_full[vid_path][t].append([x, y, w, h, i, False])

        # Load training data
        add_volleyball_folder("train")

        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch):
                self.indices.append((vid, t))

        print(f"Found {len(vid_files)} volleyball videos, {len(self.indices)} valid frames")

        self.sampler_steps: list = args.sampler_steps
        self.lengths: list = args.sampler_lengths
        print(f"sampler_steps={self.sampler_steps} lengths={self.lengths}")
        self.period_idx = 0

        # Load detection database if provided (from external detector like YOLOX)
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

    def _pre_single_frame(self, vid, idx: int):
        """Load and preprocess a single frame."""
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:08d}.jpg')

        # Try alternative image formats if jpg doesn't exist
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                alt_path = img_path.replace('.jpg', ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break

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
        for *xywh, track_id, crowd in self.labels_full[vid][idx]:
            targets['boxes'].append(xywh)
            targets['iscrowd'].append(crowd)
            targets['labels'].append(0)  # All objects mapped to class 0 for tracking
            targets['obj_ids'].append(track_id + obj_idx_offset)
            targets['scores'].append(1.0)

        # Load detection proposals if available
        txt_key = os.path.join(vid, 'img1', f'{idx:08d}.txt')
        for line in self.det_db.get(txt_key, []):
            *box, s = map(float, line.split(','))
            targets['boxes'].append(box)
            targets['scores'].append(s)

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'], dtype=torch.float64)
        targets['scores'] = torch.as_tensor(targets['scores'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)

        # Convert from xywh to xyxy format
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

    def pre_continuous_frames(self, vid, indices):
        """Load multiple consecutive frames."""
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        """Sample frame indices with random interval."""
        assert self.sample_mode == 'random_interval'
        rate = randint(1, self.sample_interval + 1)
        tmax = self.vid_tmax[vid]
        ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]

    def __getitem__(self, idx):
        """Get a training sample (sequence of frames with annotations)."""
        vid, f_index = self.indices[idx]
        indices = self.sample_indices(vid, f_index)
        images, targets = self.pre_continuous_frames(vid, indices)

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
