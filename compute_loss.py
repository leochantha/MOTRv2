# ------------------------------------------------------------------------
# Compute MOTRv2 losses from saved prediction files vs ground truth.
#
# Compares already-generated MOT-format prediction .txt files against
# gt/gt.txt ground truth annotations, computing the same losses used
# during training (L1 box loss, GIoU loss) after Hungarian matching.
#
# Usage:
#   python compute_loss.py \
#       --pred_dir output/exp_name/submit_1 \
#       --gt_dir /path/to/data/DanceTrack/val \
#       --output_dir output/loss_results
#
#   # With image size for normalized L1 (optional):
#   python compute_loss.py \
#       --pred_dir output/exp_name/submit_1 \
#       --gt_dir /path/to/data/DanceTrack/val \
#       --img_width 1920 --img_height 1080
# ------------------------------------------------------------------------

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def tlwh_to_xyxy(tlwh):
    """Convert [x, y, w, h] (top-left + size) to [x1, y1, x2, y2]."""
    x, y, w, h = tlwh
    return [x, y, x + w, y + h]


def xyxy_to_cxcywh(xyxy):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = xyxy
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def compute_iou_matrix(boxes1_xyxy, boxes2_xyxy):
    """Compute pairwise IoU between two sets of xyxy boxes.

    Args:
        boxes1_xyxy: np.array of shape (N, 4)
        boxes2_xyxy: np.array of shape (M, 4)

    Returns:
        iou: np.array of shape (N, M)
    """
    x1 = np.maximum(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    y1 = np.maximum(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    x2 = np.minimum(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    y2 = np.minimum(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

    union = area1[:, None] + area2[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def compute_giou(boxes1_xyxy, boxes2_xyxy):
    """Compute element-wise GIoU between matched box pairs.

    Args:
        boxes1_xyxy: np.array of shape (N, 4) - [x1, y1, x2, y2]
        boxes2_xyxy: np.array of shape (N, 4) - [x1, y1, x2, y2]

    Returns:
        giou: np.array of shape (N,)
    """
    # Intersection
    x1 = np.maximum(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
    y1 = np.maximum(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
    x2 = np.minimum(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
    y2 = np.minimum(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    union = area1 + area2 - inter

    iou = np.where(union > 0, inter / union, 0.0)

    # Enclosing box
    ex1 = np.minimum(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
    ey1 = np.minimum(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
    ex2 = np.maximum(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
    ey2 = np.maximum(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
    enclose_area = (ex2 - ex1) * (ey2 - ey1)

    giou = iou - np.where(enclose_area > 0, (enclose_area - union) / enclose_area, 0.0)
    return giou


def read_mot_file(filepath, is_gt=False):
    """Read a MOT-format text file.

    Args:
        filepath: Path to the .txt file.
        is_gt: If True, parse as GT (frame,id,x,y,w,h,mark,label).
               If False, parse as prediction (frame,id,x,y,w,h,score,...).

    Returns:
        dict: {frame_id: [(tlwh, track_id), ...]}
    """
    results = defaultdict(list)
    if not os.path.isfile(filepath):
        return results

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            if fid < 1:
                continue
            track_id = int(float(parts[1]))
            tlwh = tuple(map(float, parts[2:6]))

            if is_gt:
                if len(parts) >= 7:
                    mark = int(float(parts[6]))
                    if mark == 0:
                        continue
                if len(parts) >= 8:
                    label = int(float(parts[7]))
                    if label in [3, 4, 5, 6, 9, 10, 11]:
                        continue

            results[fid].append((tlwh, track_id))

    return results


def match_frame(gt_boxes_xyxy, pred_boxes_xyxy, iou_threshold=0.5):
    """Hungarian match predictions to GT using IoU cost.

    Args:
        gt_boxes_xyxy: np.array (N, 4)
        pred_boxes_xyxy: np.array (M, 4)
        iou_threshold: Minimum IoU to accept a match.

    Returns:
        matched_gt_idx: np.array of matched GT indices
        matched_pred_idx: np.array of matched prediction indices
        num_unmatched_gt: int (false negatives)
        num_unmatched_pred: int (false positives)
    """
    if len(gt_boxes_xyxy) == 0 or len(pred_boxes_xyxy) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), len(gt_boxes_xyxy), len(pred_boxes_xyxy)

    iou_matrix = compute_iou_matrix(gt_boxes_xyxy, pred_boxes_xyxy)
    cost_matrix = 1 - iou_matrix  # Hungarian minimizes cost

    gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

    # Filter by IoU threshold
    keep = iou_matrix[gt_idx, pred_idx] >= iou_threshold
    gt_idx = gt_idx[keep]
    pred_idx = pred_idx[keep]

    num_unmatched_gt = len(gt_boxes_xyxy) - len(gt_idx)
    num_unmatched_pred = len(pred_boxes_xyxy) - len(pred_idx)

    return gt_idx, pred_idx, num_unmatched_gt, num_unmatched_pred


def compute_losses_for_sequence(gt_file, pred_file, img_width=None, img_height=None, iou_threshold=0.5):
    """Compute per-frame losses for one sequence.

    Returns a dict with aggregate loss info for the sequence.
    """
    gt_frames = read_mot_file(gt_file, is_gt=True)
    pred_frames = read_mot_file(pred_file, is_gt=False)

    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))

    total_l1 = 0.0
    total_giou_loss = 0.0
    total_matched = 0
    total_gt = 0
    total_pred = 0
    total_fn = 0
    total_fp = 0
    iou_values = []

    normalize = img_width is not None and img_height is not None

    for fid in all_frames:
        gt_entries = gt_frames.get(fid, [])
        pred_entries = pred_frames.get(fid, [])

        gt_tlwhs = [e[0] for e in gt_entries]
        pred_tlwhs = [e[0] for e in pred_entries]

        gt_xyxy = np.array([tlwh_to_xyxy(b) for b in gt_tlwhs]).reshape(-1, 4) if gt_tlwhs else np.zeros((0, 4))
        pred_xyxy = np.array([tlwh_to_xyxy(b) for b in pred_tlwhs]).reshape(-1, 4) if pred_tlwhs else np.zeros((0, 4))

        total_gt += len(gt_xyxy)
        total_pred += len(pred_xyxy)

        gt_idx, pred_idx, fn, fp = match_frame(gt_xyxy, pred_xyxy, iou_threshold)
        total_fn += fn
        total_fp += fp
        total_matched += len(gt_idx)

        if len(gt_idx) == 0:
            continue

        matched_gt = gt_xyxy[gt_idx]
        matched_pred = pred_xyxy[pred_idx]

        # IoU for matched pairs
        frame_giou = compute_giou(matched_gt, matched_pred)
        frame_iou = compute_iou_matrix(matched_gt, matched_pred)
        frame_iou_diag = np.diag(frame_iou) if frame_iou.size > 0 else np.array([])
        iou_values.extend(frame_iou_diag.tolist())

        # GIoU loss: 1 - GIoU (same as training)
        giou_loss = 1.0 - frame_giou
        total_giou_loss += giou_loss.sum()

        # L1 loss on cxcywh (optionally normalized by image size)
        gt_cxcywh = np.array([xyxy_to_cxcywh(b) for b in matched_gt])
        pred_cxcywh = np.array([xyxy_to_cxcywh(b) for b in matched_pred])

        if normalize:
            scale = np.array([img_width, img_height, img_width, img_height], dtype=np.float64)
            gt_cxcywh = gt_cxcywh / scale
            pred_cxcywh = pred_cxcywh / scale

        l1 = np.abs(gt_cxcywh - pred_cxcywh).sum(axis=1)  # per-box L1
        total_l1 += l1.sum()

    num_boxes = max(total_matched, 1)
    return {
        'loss_bbox': total_l1 / num_boxes,
        'loss_giou': total_giou_loss / num_boxes,
        'total_frames': len(all_frames),
        'total_gt': total_gt,
        'total_pred': total_pred,
        'total_matched': total_matched,
        'false_negatives': total_fn,
        'false_positives': total_fp,
        'precision': total_matched / max(total_pred, 1),
        'recall': total_matched / max(total_gt, 1),
        'mean_iou': float(np.mean(iou_values)) if iou_values else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute MOTRv2 losses from saved prediction files vs ground truth.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_loss.py \\
      --pred_dir output/exp_name/submit_1 \\
      --gt_dir /path/to/data/DanceTrack/val

  python compute_loss.py \\
      --pred_dir output/exp_name/submit_1 \\
      --gt_dir /path/to/data/DanceTrack/val \\
      --img_width 1920 --img_height 1080 \\
      --output_dir output/loss_results
        """,
    )
    parser.add_argument('--pred_dir', required=True,
                        help='Directory containing prediction .txt files (MOT format)')
    parser.add_argument('--gt_dir', required=True,
                        help='Directory containing sequence folders with gt/gt.txt')
    parser.add_argument('--iou_threshold', default=0.5, type=float,
                        help='Minimum IoU to accept a match (default: 0.5)')
    parser.add_argument('--img_width', default=None, type=int,
                        help='Image width for normalized L1 loss (optional)')
    parser.add_argument('--img_height', default=None, type=int,
                        help='Image height for normalized L1 loss (optional)')
    parser.add_argument('--output_dir', default='',
                        help='Directory to save loss_results.json (optional)')
    args = parser.parse_args()

    # Discover sequences: each .txt in pred_dir should match a folder in gt_dir
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith('.txt')])
    if not pred_files:
        print(f"ERROR: No .txt files found in {args.pred_dir}")
        sys.exit(1)

    normalize = args.img_width is not None and args.img_height is not None

    print(f"\n{'='*60}")
    print("MOTRv2 LOSS COMPUTATION (Predictions vs Ground Truth)")
    print(f"{'='*60}")
    print(f"  Prediction dir: {args.pred_dir}")
    print(f"  Ground truth dir: {args.gt_dir}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  Normalize L1: {normalize}" + (f" ({args.img_width}x{args.img_height})" if normalize else " (pixel-space)"))
    print(f"  Sequences found: {len(pred_files)}")
    print(f"{'='*60}\n")

    all_results = {}
    totals = defaultdict(float)
    total_sequences = 0

    for pred_file in pred_files:
        seq_name = pred_file.replace('.txt', '')
        pred_path = os.path.join(args.pred_dir, pred_file)
        gt_path = os.path.join(args.gt_dir, seq_name, 'gt', 'gt.txt')

        if not os.path.isfile(gt_path):
            print(f"  SKIP {seq_name}: GT not found at {gt_path}")
            continue

        result = compute_losses_for_sequence(
            gt_path, pred_path,
            img_width=args.img_width, img_height=args.img_height,
            iou_threshold=args.iou_threshold,
        )
        all_results[seq_name] = result
        total_sequences += 1

        for k, v in result.items():
            totals[k] += v

        print(f"  {seq_name:30s}  loss_bbox={result['loss_bbox']:.4f}  "
              f"loss_giou={result['loss_giou']:.4f}  "
              f"IoU={result['mean_iou']:.3f}  "
              f"P={result['precision']:.3f}  R={result['recall']:.3f}  "
              f"matched={result['total_matched']}/{result['total_gt']}")

    if total_sequences == 0:
        print("\nERROR: No sequences matched between predictions and GT.")
        sys.exit(1)

    # Aggregate
    avg = {k: v / total_sequences for k, v in totals.items()}

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"  Sequences evaluated:  {total_sequences}")
    print(f"  Total GT boxes:       {int(totals['total_gt'])}")
    print(f"  Total pred boxes:     {int(totals['total_pred'])}")
    print(f"  Total matched:        {int(totals['total_matched'])}")
    print(f"  Total FN:             {int(totals['false_negatives'])}")
    print(f"  Total FP:             {int(totals['false_positives'])}")
    print(f"")
    print(f"  Avg loss_bbox (L1):   {avg['loss_bbox']:.6f}" + ("  (normalized)" if normalize else "  (pixels)"))
    print(f"  Avg loss_giou:        {avg['loss_giou']:.6f}")
    print(f"  Avg IoU:              {avg['mean_iou']:.4f}")
    print(f"  Avg precision:        {avg['precision']:.4f}")
    print(f"  Avg recall:           {avg['recall']:.4f}")
    print(f"{'='*60}\n")

    # Optionally save
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output = {
            'config': {
                'pred_dir': args.pred_dir,
                'gt_dir': args.gt_dir,
                'iou_threshold': args.iou_threshold,
                'img_width': args.img_width,
                'img_height': args.img_height,
                'normalized_l1': normalize,
            },
            'aggregate': {k: float(v) for k, v in avg.items()},
            'per_sequence': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()},
        }
        out_path = os.path.join(args.output_dir, 'loss_results.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
