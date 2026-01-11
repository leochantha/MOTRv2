# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from copy import deepcopy
import json
import time
from datetime import datetime
from collections import defaultdict

import os
import argparse
import torchvision.transforms.functional as F
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader
import re


def get_next_submit_folder(base_dir, exp_name):
    """Find the next available numbered submit folder (submit_1, submit_2, etc.).

    Args:
        base_dir: Base output directory
        exp_name: Experiment name (used as prefix)

    Returns:
        Path to the next available submit folder
    """
    exp_base = os.path.join(base_dir, exp_name)
    os.makedirs(exp_base, exist_ok=True)

    # Find existing submit folders
    existing = []
    if os.path.exists(exp_base):
        for name in os.listdir(exp_base):
            match = re.match(r'submit_(\d+)$', name)
            if match:
                existing.append(int(match.group(1)))

    # Get next number
    next_num = max(existing, default=0) + 1
    submit_dir = os.path.join(exp_base, f'submit_{next_num}')
    os.makedirs(submit_dir, exist_ok=True)

    return submit_dir, next_num


class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        # for line in self.det_db[f_path[:-4] + '.txt']:
        for line in self.det_db[f_path[:-4]]:
            l, t, w, h, s = list(map(float, line.split(',')))
            proposals.append([(l + w / 2) / im_w,
                                (t + h / 2) / im_h,
                                w / im_w,
                                h / im_h,
                                s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)


class Detector(object):
    def __init__(self, args, model, vid, predict_path):
        self.args = args
        self.detr = model

        self.vid = vid
        self.seq_num = os.path.basename(vid)
        img_list = os.listdir(os.path.join(self.args.mot_path, vid, 'img1'))
        img_list = [os.path.join(vid, 'img1', i) for i in img_list if 'jpg' in i]

        self.img_list = sorted(img_list)
        self.img_len = len(self.img_list)

        self.predict_path = predict_path

        # Statistics tracking
        self.stats = {
            'sequence': self.seq_num,
            'video_path': vid,
            'num_frames': self.img_len,
        }

    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, prob_threshold=0.5, area_threshold=25, vis=False):
        start_time = time.time()
        total_dts = 0
        total_occlusion_dts = 0

        # Per-frame statistics
        detections_per_frame = []
        all_track_ids = set()
        frame_times = []

        track_instances = None
        with open(os.path.join(self.args.mot_path, self.args.det_db)) as f:
            det_db = json.load(f)
        loader = DataLoader(ListImgDataset(self.args.mot_path, self.img_list, det_db), 1, num_workers=2)
        lines = []
        for i, data in enumerate(tqdm(loader, desc=f"Processing {self.seq_num}")):
            frame_start = time.time()
            cur_img, ori_img, proposals = [d[0] for d in data]
            cur_img, proposals = cur_img.cuda(), proposals.cuda()

            # track_instances = None
            if track_instances is not None:
                track_instances.remove('boxes')
                track_instances.remove('labels')
            seq_h, seq_w, _ = ori_img.shape

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), track_instances, proposals)
            track_instances = res['track_instances']

            dt_instances = deepcopy(track_instances)

            # filter det instances by score.
            dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            num_dets = len(dt_instances)
            total_dts += num_dets
            detections_per_frame.append(num_dets)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_idxes.tolist()

            save_format = '{frame},{id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n'
            for xyxy, track_id in zip(bbox_xyxy, identities):
                if track_id < 0 or track_id is None:
                    continue
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                lines.append(save_format.format(frame=i + 1, id=int(track_id), x1=x1, y1=y1, w=w, h=h))
                all_track_ids.add(int(track_id))

            frame_times.append(time.time() - frame_start)

        # Save tracking results
        result_file = os.path.join(self.predict_path, f'{self.seq_num}.txt')
        with open(result_file, 'w') as f:
            f.writelines(lines)

        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_fps = len(self.img_list) / elapsed_time if elapsed_time > 0 else 0

        self.stats.update({
            'total_detections': total_dts,
            'unique_tracks': len(all_track_ids),
            'avg_detections_per_frame': total_dts / len(self.img_list) if self.img_list else 0,
            'min_detections_per_frame': min(detections_per_frame) if detections_per_frame else 0,
            'max_detections_per_frame': max(detections_per_frame) if detections_per_frame else 0,
            'elapsed_time_seconds': round(elapsed_time, 2),
            'avg_fps': round(avg_fps, 2),
            'avg_frame_time_ms': round(sum(frame_times) / len(frame_times) * 1000, 2) if frame_times else 0,
            'result_file': result_file,
        })

        print(f"\n{'='*50}")
        print(f"Sequence: {self.seq_num}")
        print(f"  Frames: {self.stats['num_frames']}")
        print(f"  Total detections: {total_dts}")
        print(f"  Unique tracks: {len(all_track_ids)}")
        print(f"  Avg detections/frame: {self.stats['avg_detections_per_frame']:.1f}")
        print(f"  Processing time: {elapsed_time:.1f}s ({avg_fps:.1f} FPS)")
        print(f"  Results saved to: {result_file}")
        print(f"{'='*50}\n")

        return self.stats

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


def save_run_metadata(args, all_stats, submit_dir, submit_num):
    """Save run metadata and statistics to JSON file.

    Args:
        args: Command line arguments
        all_stats: List of per-sequence statistics
        submit_dir: Path to the submit folder (e.g., output/exp_name/submit_1)
        submit_num: Submit folder number (e.g., 1)
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'submit_number': submit_num,
        'submit_dir': submit_dir,
        'parameters': {
            'checkpoint': args.resume,
            'det_db': args.det_db,
            'mot_path': args.mot_path,
            'score_threshold': args.score_threshold,
            'update_score_threshold': args.update_score_threshold,
            'miss_tolerance': args.miss_tolerance,
            'exp_name': args.exp_name,
        },
        'summary': {
            'num_sequences': len(all_stats),
            'total_frames': sum(s['num_frames'] for s in all_stats),
            'total_detections': sum(s['total_detections'] for s in all_stats),
            'total_unique_tracks': sum(s['unique_tracks'] for s in all_stats),
            'total_time_seconds': sum(s['elapsed_time_seconds'] for s in all_stats),
        },
        'sequences': all_stats,
    }

    # Calculate overall averages
    if all_stats:
        metadata['summary']['avg_detections_per_frame'] = round(
            metadata['summary']['total_detections'] / metadata['summary']['total_frames'], 2
        ) if metadata['summary']['total_frames'] > 0 else 0
        metadata['summary']['overall_fps'] = round(
            metadata['summary']['total_frames'] / metadata['summary']['total_time_seconds'], 2
        ) if metadata['summary']['total_time_seconds'] > 0 else 0

    # Save metadata
    metadata_file = os.path.join(submit_dir, 'inference_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Submit folder: submit_{submit_num}")
    print(f"  Sequences processed: {metadata['summary']['num_sequences']}")
    print(f"  Total frames: {metadata['summary']['total_frames']}")
    print(f"  Total detections: {metadata['summary']['total_detections']}")
    print(f"  Total unique tracks: {metadata['summary']['total_unique_tracks']}")
    print(f"  Overall FPS: {metadata['summary'].get('overall_fps', 'N/A')}")
    print(f"  Results saved to: {submit_dir}")
    print(f"{'='*60}\n")

    return metadata


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=20, type=int)
    parser.add_argument('--area_threshold', default=25, type=float,
                       help='Minimum detection area in pixels (default: 25)')
    parser.add_argument('--sub_dir', default='volleyball/test', type=str,
                       help='Subdirectory containing test sequences (default: volleyball/test)')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Create auto-incrementing submit folder
    submit_dir, submit_num = get_next_submit_folder(args.output_dir, args.exp_name)

    print(f"\n{'='*60}")
    print("MOTRv2 INFERENCE")
    print(f"{'='*60}")
    print(f"  Output: {submit_dir}")
    print(f"  Checkpoint: {args.resume}")
    print(f"  Detection DB: {args.det_db}")
    print(f"  Score threshold: {args.score_threshold}")
    print(f"  Miss tolerance: {args.miss_tolerance}")
    print(f"  Area threshold: {args.area_threshold}")
    print(f"{'='*60}\n")

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_embed.score_thr = args.update_score_threshold
    detr.track_base = RuntimeTrackerBase(args.score_threshold, args.score_threshold, args.miss_tolerance)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    # Get sequences to process
    sub_dir = args.sub_dir
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in sorted(seq_nums)]

    rank = int(os.environ.get('RLAUNCH_REPLICA', '0'))
    ws = int(os.environ.get('RLAUNCH_REPLICA_TOTAL', '1'))
    vids = vids[rank::ws]

    print(f"Processing {len(vids)} sequences: {[os.path.basename(v) for v in vids]}\n")

    # Process all sequences and collect stats
    all_stats = []
    for vid in vids:
        det = Detector(args, model=detr, vid=vid, predict_path=submit_dir)
        stats = det.detect(args.score_threshold, args.area_threshold)
        all_stats.append(stats)

    # Save metadata
    save_run_metadata(args, all_stats, submit_dir, submit_num)
