"""
Improved Data Pipeline Design Document
======================================

This document describes the redesigned data collection and augmentation system
for enemy position detection in gameplay footage.

## Pipeline Overview

### Stage 1: Smart Frame Sampling
- Replace fixed FRAME_SKIP with intelligent sampling
- Sample frames around engagements (-30, -20, -10, 0, +10, +20 frames)
- Include random exploration frames throughout gameplay
- Explicitly sample edges, corners, and off-center regions
- Capture negative samples (no enemies)

### Stage 2: Detection + Confidence-Based Review
- YOLO detection with confidence scoring
- Three-tier system:
  * HIGH_CONF (>0.7): Auto-label, no human review
  * MED_CONF (0.4-0.7): Quick human confirmation (Y/N)
  * LOW_CONF (<0.4): Full manual selection or skip
- Multi-detection handling: user selects correct target

### Stage 3: Multi-Object Tracking (MOT)
- After human confirmation, initialize tracker (SORT/DeepSORT/OpenCV KCF)
- Propagate labels forward (next 10-30 frames)
- Quality checks:
  * Motion consistency (reject large jumps)
  * Confidence decay (stop when <0.3)
  * Drift detection (compare YOLO re-detection vs tracker)

### Stage 4: Enhanced Label Format
CSV columns:
- video_id: Source video identifier
- frame_index: Frame number in video
- has_enemy: 0 or 1
- x_center, y_center: Normalized 0-1
- width, height: Bounding box (normalized)
- confidence: Detection confidence (0-1)
- source_type: auto / reviewed / tracked / manual
- occluded: 0 or 1
- multiple_targets: 0 or 1
- blur_score: Image quality metric (0-1)

### Stage 5: Real-Time Balancing
- Grid-based position tracking (3x3 or 5x5 grid)
- Quota system per grid cell
- Prefer underrepresented regions during collection
- Skip overrepresented regions (with probability)

### Stage 6: Quality Control & Filtering
- Duplicate detection: Perceptual hashing (pHash) or SSIM
- Blur detection: Laplacian variance threshold
- Jump detection: Reject if position delta > threshold between frames
- Box validation: Reject invalid/out-of-bounds boxes
- Save debug overlays for visual inspection

### Stage 7: Deduplication
- Frame difference (pixel-level)
- Perceptual hashing for near-duplicate detection
- Keep best quality frame from duplicates

### Stage 8: Train/Val Split
- Split by video_id (never same video in both splits)
- Temporal: Ensure no overlap even between videos
- Recommended: 80/20 or 70/30 split by video count

### Stage 9: Bias-Aware Augmentation
- Skip augmentation for overrepresented regions
- Heavy augmentation for underrepresented regions
- Only pixel-level augmentation (brightness, noise, blur) for center samples
- Spatial augmentation (shift, zoom, rotate) for edge/corner samples
- Coordinates must be recalculated for all spatial transforms

### Stage 10: Model Architecture Recommendation
- **RECOMMENDED: Heatmap-based approach**
  * Output: 64x64 or 32x32 heatmap with Gaussian peaks
  * Loss: MSE or BCE with focal loss
  * Advantages:
    - Handles 'no enemy' naturally (empty heatmap)
    - Multiple enemies supported
    - Spatial uncertainty encoded in peak spread
    - More stable training
- Alternative: Object detection (YOLO/Faster R-CNN style)
  * Direct bbox prediction
  * Built-in confidence score
  * Existing anchors can be tuned for game enemies

## Key Algorithms

### 1. Frame Sampling Strategy
```
for each engagement_timestamp:
    for offset in [-30, -20, -10, 0, 10, 20]:
        frame = video[engagement_timestamp + offset]
        process(frame)

# Random exploration samples
while samples_needed > 0:
    frame = video[random.randint(0, total_frames)]
    if not is_near_existing_sample(frame):
        process(frame)
        samples_needed -= 1
```

### 2. Confidence-Based Review Flow
```
detections = yolo(frame)
for det in detections:
    if det.conf > 0.7 and not multiple_detections:
        auto_label(det)
    elif det.conf > 0.4:
        user_confirm(det)  # Y/N
    else:
        user_select_or_skip(frame)
```

### 3. Tracking with Validation
```
tracker = init_tracker(frame, bbox)
for i in range(10, 30):
    next_frame = video[frame_idx + i]
    tracked_bbox = tracker.update(next_frame)
    
    # Validation
    yolo_redetection = yolo(next_frame, region=tracked_bbox)
    if iou(tracked_bbox, yolo_redetection) < 0.5:
        break  # Drift detected
    
    if motion_delta(prev_pos, current_pos) > threshold:
        break  # Unnatural jump
    
    save_label(tracked_bbox, source='tracked')
```

### 4. Grid-Based Balancing
```
GRID = 5x5
counts = zeros(GRID)
max_per_cell = total_desired_samples / (GRID * 0.6)  # allow some imbalance

def should_accept_sample(x, y):
    cell_x = int(x * 5)
    cell_y = int(y * 5)
    if counts[cell_x, cell_y] < max_per_cell:
        counts[cell_x, cell_y] += 1
        return True
    else:
        return random() < 0.1  # 10% chance to accept anyway
```

### 5. Duplicate Detection
```
# Perceptual hash approach
hash1 = pHash(frame1)
hash2 = pHash(frame2)
hamming_distance = count_different_bits(hash1, hash2)
if hamming_distance < 10:
    mark_duplicate()

# Or SSIM approach
if ssim(frame1, frame2) > 0.95:
    mark_duplicate()
```

## File Structure

```
dataset/
├── raw_videos/              # Original gameplay recordings
├── sampled_frames/          # Stage 1 output (before labeling)
├── labeled/
│   ├── auto/               # High-confidence auto labels
│   ├── reviewed/           # Human-confirmed labels
│   ├── tracked/            # Tracker-generated labels
│   └── manual/             # Fully manual labels
├── quality_control/
│   ├── duplicates/         # Flagged duplicates
│   ├── rejected/           # Failed quality checks
│   └── debug_overlays/     # Visual inspection images
├── train/                  # Final train split
├── val/                    # Final validation split
└── metadata/
    ├── labels.csv          # Master label file
    ├── video_manifest.json # Video info and splits
    └── collection_stats.json # Grid counts, quality metrics
```

## Implementation Priorities

1. **P0 (Critical)**: Smart sampling, confidence workflow, enhanced labels
2. **P1 (High)**: Tracking integration, grid balancing, quality control
3. **P2 (Medium)**: Deduplication, augmentation, train/val split
4. **P3 (Low)**: Heatmap model transition, advanced filters

## Trade-offs

1. **Auto-labeling vs Accuracy**: Higher auto-label threshold = less work but more noise
   - Recommended: 0.7+ for auto, manual review for rest

2. **Tracking length vs Drift**: Longer tracking = more samples but risk of drift
   - Recommended: 10-20 frames max, with drift detection

3. **Balancing vs Dataset size**: Strict balancing = smaller dataset
   - Recommended: Allow 2x overrepresentation in center

4. **Grid granularity**: Finer grid = better balance but more complex
   - Recommended: 5x5 grid for 1920x1080
"""
