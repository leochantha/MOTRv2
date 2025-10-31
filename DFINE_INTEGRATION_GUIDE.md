# DFine Integration and Multi-Class Tracking Guide

## Overview

This document describes the modifications made to MOTRv2 to support:
1. **DFine detector** as an alternative to YOLOX
2. **Multi-class tracking** (persons + volleyball)
3. **Class-specific tracking parameters** for different object types

## Key Changes

### 1. Multi-Class Support (num_classes = 2)

**Modified:** `models/motr.py:714`
- Changed `'e2e_dance': 1` to `'e2e_dance': 2`
- Class 0 = Person, Class 1 = Ball

### 2. Detection Format Update

**Old Format (5 dimensions):**
```
left, top, width, height, score
```

**New Format (6 dimensions):**
```
left, top, width, height, score, class_id
```

**Backward Compatible:** The code automatically handles both formats:
- 5-value format: defaults to class 0 (person)
- 6-value format: uses provided class_id

**Modified Files:**
- `submit_dance.py:50-65` - Inference detection loading
- `datasets/dance.py:186-199` - Training detection loading
- `datasets/dance.py:244-249` - Proposal creation

### 3. Score Extraction for Multi-Class

**Modified:** `models/motr.py:585-590`

**Before:**
```python
track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()  # Only class 0
```

**After:**
```python
pred_logits_sigmoid = frame_res['pred_logits'][0, :].sigmoid()
track_scores, pred_classes = pred_logits_sigmoid.max(dim=-1)  # All classes
track_instances.pred_classes = pred_classes
```

### 4. TrackerPostProcess Update

**Modified:** `models/motr.py:345-357`

**Before:**
```python
scores = out_logits[..., 0].sigmoid()  # Only class 0
track_instances.labels = torch.full_like(scores, 0)  # All zeros
```

**After:**
```python
scores, pred_classes = out_logits.sigmoid().max(dim=-1)  # Max across classes
track_instances.labels = pred_classes  # Actual predicted classes
```

### 5. Query Initialization with Class Information

**Modified:** `models/motr.py:466-469`

Proposals are now 6D: `[cx, cy, w, h, score, class]`

The query positional embedding now encodes both score and class:
```python
track_instances.query_pos = torch.cat([
    self.query_embed.weight,
    pos2posemb(proposals[:, 4:6], d_model) + self.yolox_embed.weight
])  # proposals[:, 4:6] = [score, class]
```

### 6. Class-Aware RuntimeTrackerBase

**Modified:** `models/motr.py:302-360` and `submit_dance.py:161-219`

**New Parameters:**
- `ball_score_thresh` (default: 0.3) - Lower threshold for ball detection
- `ball_miss_tolerance` (default: 5) - Shorter tolerance for ball disappearance

**Logic:**
- Separate thresholds for persons vs balls
- Different miss tolerance (balls disappear/reappear more frequently)
- Class-specific track creation and deletion

**Command Line Arguments:**
```bash
--score_threshold 0.5          # Person score threshold
--miss_tolerance 20            # Person miss tolerance
--ball_score_threshold 0.3     # Ball score threshold
--ball_miss_tolerance 5        # Ball miss tolerance
```

## DFine Confidence Score Analysis

Based on your DFine output:
- **Min:** 0.790
- **Max:** 0.999
- **Mean:** 0.969
- **Median:** 0.978
- **Std:** 0.028

**Key Insight:** DFine produces very high confidence scores (0.79-0.99) compared to YOLOX (0.3-0.9). This means:
- The default thresholds (0.5-0.6) are not the issue
- All DFine detections pass the confidence threshold
- Track loss is likely due to detection quality or association issues, not low confidence

## Creating a Multi-Class Detection Database

### Format Your DFine Output

For each frame, create a `.txt` file with detections in format:
```
x, y, w, h, score, class_id
100.5, 200.3, 50.2, 100.1, 0.95, 0    # Person
450.2, 300.1, 15.5, 15.5, 0.87, 1     # Ball
```

Where:
- `x, y` = top-left corner
- `w, h` = width and height
- `score` = confidence (0-1)
- `class_id` = 0 for person, 1 for ball

### Create Detection Database

Use the updated `tools/make_detdb.py` script to create a JSON database:

```python
# The script will automatically handle the new 6-value format
python tools/make_detdb.py
```

## Running Inference with DFine Detections

### Basic Command

```bash
python submit_dance.py \
    --resume /path/to/checkpoint.pth \
    --mot_path /path/to/dataset \
    --det_db det_db_dfine.json \
    --score_threshold 0.5 \
    --ball_score_threshold 0.3 \
    --miss_tolerance 20 \
    --ball_miss_tolerance 5
```

### Parameter Tuning Guide

#### For Players (Persons)
- `--score_threshold`: Increase if too many false positive tracks (default: 0.5)
- `--miss_tolerance`: Increase if players are lost too quickly (default: 20)

#### For Ball
- `--ball_score_threshold`: Lower if ball is not being tracked (default: 0.3)
- `--ball_miss_tolerance`: Increase if ball track is lost during brief occlusions (default: 5)

### Recommended Starting Values

Given DFine's high confidence scores:

```bash
# Conservative (fewer false positives)
--score_threshold 0.7
--ball_score_threshold 0.5

# Aggressive (catch more detections)
--score_threshold 0.5
--ball_score_threshold 0.3
```

## Training with Multi-Class

Update your config file (e.g., `configs/motrv2_volleyball.args`):

```
--meta_arch motr
--dataset_file e2e_dance  # Now uses num_classes=2
--num_classes 2  # Not needed - inferred from dataset_file
--det_db det_db_dfine.json
```

The training pipeline will automatically:
- Load 6D proposals (with class labels)
- Match predictions to GT with class information
- Learn class-specific representations

## Troubleshooting

### Issue: Ball Not Tracked

**Possible Causes:**
1. Ball detections not in detection database
2. Ball confidence too low (< `ball_score_threshold`)
3. Ball detections have wrong class_id (should be 1)

**Solutions:**
- Check detection database contains class_id=1 entries
- Lower `--ball_score_threshold` to 0.2 or 0.3
- Verify DFine outputs ball detections

### Issue: Players Lost Too Quickly

**Possible Causes:**
1. DFine detections missing in some frames
2. Bbox jitter causing association failures
3. Miss tolerance too low

**Solutions:**
- Increase `--miss_tolerance` to 30-50
- Check DFine detection consistency frame-by-frame
- Lower `--score_threshold` slightly

### Issue: Too Many False Positive Tracks

**Possible Causes:**
1. Low confidence detections creating tracks
2. Background objects being tracked

**Solutions:**
- Increase `--score_threshold` to 0.6-0.7
- Increase `--ball_score_threshold` to 0.4-0.5
- Add area filtering (already implemented)

## Output Format

The tracker outputs `.txt` files in MOT format:
```
frame, id, x1, y1, w, h, 1, -1, -1, -1
```

**Class Information:**
- Class labels are NOT included in MOT format output
- If you need class information, modify `submit_dance.py:148` to save labels

## Validation

To verify multi-class tracking works:

1. **Check predictions have multiple classes:**
   ```python
   # Add after line 592 in models/motr.py
   print(f"Classes: {track_instances.pred_classes.unique()}")
   # Should print: tensor([0, 1])
   ```

2. **Check track counts per class:**
   ```python
   # In submit_dance.py after line 144
   person_tracks = (dt_instances.labels == 0).sum()
   ball_tracks = (dt_instances.labels == 1).sum()
   print(f"Persons: {person_tracks}, Balls: {ball_tracks}")
   ```

## Performance Considerations

### Memory
- Multi-class increases model output from `[N, 1]` to `[N, 2]`
- Minimal impact (< 1% memory increase)

### Speed
- No significant impact on inference speed
- Score extraction uses `.max(dim=-1)` which is fast

### Accuracy
- Separate class embeddings may improve person tracking
- Lower thresholds for ball may increase ball recall

## Next Steps

1. **Generate DFine detection database** with class labels
2. **Test with default parameters** first
3. **Tune class-specific thresholds** based on results
4. **Monitor track statistics** (creation/deletion rates)
5. **Visualize** to verify ball tracking works

## References

- MOTRv2 Paper: https://arxiv.org/abs/2211.09791
- Original MOTRv2 Repo: https://github.com/megvii-research/MOTRv2
- DFine Confidence Analysis: See histogram provided
