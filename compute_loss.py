# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Compute MOTRv2 loss on data with ground truth annotations.
#
# This script loads a trained MOTRv2 checkpoint, runs it on dataset clips
# in training mode (so the criterion's Hungarian matching + loss computation
# is triggered), and reports per-clip and aggregate losses.
#
# Usage:
#   python compute_loss.py @configs/motrv2.args \
#       --resume path/to/checkpoint.pth \
#       --mot_path /path/to/data \
#       --max_samples 50
# ------------------------------------------------------------------------

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import build_dataset
from engine import train_one_epoch_mot
from main import get_args_parser
from models import build_model
from util.tool import load_model
import util.misc as utils
from datasets.data_prefetcher import data_dict_to_cuda


def compute_loss(args):
    device = torch.device(args.device)

    # Build model and criterion
    model, criterion, _ = build_model(args)
    model.to(device)
    criterion.to(device)

    # Load checkpoint
    if not args.resume:
        print("ERROR: --resume must point to a checkpoint file.")
        sys.exit(1)

    model = load_model(model, args.resume)
    print(f"Loaded checkpoint: {args.resume}")

    # Build dataset (uses training set by default since it has GT)
    dataset = build_dataset(image_set='train', args=args)
    dataset.set_epoch(0)

    # Optionally limit number of samples
    max_samples = getattr(args, 'max_samples', 0)
    if max_samples > 0 and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
        print(f"Using {max_samples} / {len(dataset)} samples")
    else:
        print(f"Using all {len(dataset)} samples")

    collate_fn = utils.mot_collate_fn
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Put model in training mode so the criterion matching is triggered,
    # but disable gradient computation since we only want loss values.
    model.train()
    criterion.train()

    all_losses = defaultdict(list)
    total_weighted_losses = []

    weight_dict = criterion.weight_dict

    print(f"\nComputing loss over {len(data_loader)} clips...\n")
    print(f"Loss weights: { {k: v for k, v in sorted(weight_dict.items()) if 'frame_0' in k} }")
    print()

    for batch_idx, data_dict in enumerate(data_loader):
        data_dict = data_dict_to_cuda(data_dict, device)

        with torch.no_grad():
            outputs = model(data_dict)
            loss_dict = criterion(outputs, data_dict)

        # Compute weighted total
        weighted_loss = sum(
            loss_dict[k].item() * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )
        total_weighted_losses.append(weighted_loss)

        # Accumulate individual losses (aggregate by loss type across frames)
        per_type = defaultdict(float)
        per_type_count = defaultdict(int)
        for k, v in loss_dict.items():
            val = v.item()
            all_losses[k].append(val)
            # Extract the loss type (e.g., "loss_ce" from "frame_0_loss_ce")
            parts = k.split('_')
            # Find the loss type: everything after "frame_N_" or "frame_N_auxM_" or "frame_N_psM_"
            loss_type = None
            for i, p in enumerate(parts):
                if p == 'loss':
                    loss_type = '_'.join(parts[i:])
                    break
            if loss_type:
                per_type[loss_type] += val
                per_type_count[loss_type] += 1

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            print(f"  [{batch_idx+1}/{len(data_loader)}] weighted_loss={weighted_loss:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("LOSS SUMMARY")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {len(total_weighted_losses)}")

    avg_total = np.mean(total_weighted_losses)
    std_total = np.std(total_weighted_losses)
    print(f"  Total weighted loss: {avg_total:.4f} +/- {std_total:.4f}")
    print(f"  Min weighted loss:   {np.min(total_weighted_losses):.4f}")
    print(f"  Max weighted loss:   {np.max(total_weighted_losses):.4f}")

    # Aggregate by loss type
    print(f"\n  Per-loss-type averages (across all frames and samples):")
    type_aggregates = defaultdict(list)
    for k, vals in all_losses.items():
        parts = k.split('_')
        loss_type = None
        for i, p in enumerate(parts):
            if p == 'loss':
                loss_type = '_'.join(parts[i:])
                break
        if loss_type:
            type_aggregates[loss_type].extend(vals)

    for loss_type in sorted(type_aggregates.keys()):
        vals = type_aggregates[loss_type]
        weight = weight_dict.get(f'frame_0_{loss_type}', 0)
        avg = np.mean(vals)
        print(f"    {loss_type:20s}: {avg:.6f}  (weight={weight}, weighted={avg*weight:.6f})")

    print(f"{'='*60}\n")

    # Optionally save to JSON
    output_dir = getattr(args, 'output_dir', '')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results = {
            'checkpoint': args.resume,
            'num_samples': len(total_weighted_losses),
            'avg_weighted_loss': float(avg_total),
            'std_weighted_loss': float(std_total),
            'min_weighted_loss': float(np.min(total_weighted_losses)),
            'max_weighted_loss': float(np.max(total_weighted_losses)),
            'per_loss_type': {
                lt: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for lt, v in type_aggregates.items()
            },
            'per_sample_weighted_loss': [float(x) for x in total_weighted_losses],
        }
        out_path = os.path.join(output_dir, 'loss_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    return avg_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MOTRv2 loss computation',
        parents=[get_args_parser()],
    )
    parser.add_argument('--max_samples', default=0, type=int,
                        help='Max number of dataset clips to evaluate (0 = all)')
    args = parser.parse_args()
    compute_loss(args)
