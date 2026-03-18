# Improved Data Collection Pipeline - Command Guide

## Quick Start

```bash
# Basic usage
python src/process_video_improved.py \
    --videos_dir src/videos \
    --output_dir dataset/labeled

# With all options
python src/process_video_improved.py \
    --videos_dir src/videos \
    --output_dir dataset/labeled \
    --yolo_model yolov8n.pt \
    --engagement_file engagements.json
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--videos_dir` | ✅ | - | Directory containing input video files (.mp4, .avi) |
| `--output_dir` | ✅ | - | Output directory for labeled dataset |
| `--yolo_model` | ❌ | `yolov8n.pt` | Path to YOLO model for detection |
| `--engagement_file` | ❌ | `None` | JSON file with engagement timestamps |

## Interactive Review Controls

During processing, a CV2 window shows detections. Use these keys:

| Key | Action | When to Use |
|-----|--------|-------------|
| `y` | **Accept** detection | Confirm the highlighted enemy is correct |
| `n` | **Reject** detection | Detection is wrong (false positive) |
| `s` | **Skip** frame | No enemy present or unclear scene |
| `p` | **Skip to next video** | Skip remaining frames in current video |
| `1-9` | **Select detection #** | Multiple detections shown, pick correct one |
| `c` + click | **Manual correction** | Click on correct enemy position |
| `t` | **Start tracking** | Track this enemy forward N frames |
| `ESC` | **Quit pipeline** | Stop all processing and save progress |

### Mode-Specific Controls

**Auto-Label Mode** (high confidence detections):
- Press any key to accept
- Press `s` to skip if incorrect

**Confirm Mode** (medium confidence):
- `y` = Yes, this is an enemy
- `n` = No, reject this detection
- `s` = Skip this frame

**Select Mode** (multiple detections):
- Press number `1-9` corresponding to detection number shown
- `c` then click for manual correction
- `s` to skip entirely

**Manual Mode** (no detections or very low confidence):
- Click on enemy position
- `n` if no enemy present
- `s` to skip

## Output Files

```
dataset/labeled/
├── images/
│   ├── video1_frame_000123.png      # Saved frame images
│   ├── video1_frame_000456.png
│   └── ...
├── labels_enhanced.csv              # Main label file
└── collection_stats.json            # Processing statistics
```

### CSV Format (labels_enhanced.csv)

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | string | Source video filename |
| `frame_index` | int | Frame number in video |
| `has_enemy` | 0/1 | 1 if enemy present, 0 for negative sample |
| `x_center` | float | Normalized X coordinate (0-1) |
| `y_center` | float | Normalized Y coordinate (0-1) |
| `width` | float | Bounding box width (normalized) |
| `height` | float | Bounding box height (normalized) |
| `confidence` | float | Detection confidence (0-1) |
| `source_type` | string | `auto`, `reviewed`, `manual`, `negative`, `tracked` |
| `occluded` | 0/1 | 1 if enemy is partially occluded |
| `multiple_targets` | 0/1 | 1 if multiple enemies in frame |
| `blur_score` | float | Image quality score (0-1) |
| `image_hash` | string | Perceptual hash for deduplication |
| `timestamp` | ISO8601 | When label was created |

## Configuration (Edit script to change)

### Detection Thresholds
```python
HIGH_CONF_THRESHOLD = 0.70    # Auto-label without review
MED_CONF_THRESHOLD = 0.40     # Requires human confirmation
MIN_ENEMY_AREA_PX = 100       # Minimum detection size
MAX_ENEMY_AREA_RATIO = 0.25   # Maximum detection size (25% of image)
```

### Self-Player Filtering (Third-Person Games)
```python
SELF_PLAYER_BOTTOM_REGION_PCT = 0.35   # Bottom 35% exclusion zone
SELF_PLAYER_CENTER_ZONE_X = (0.3, 0.7) # Horizontal range (30-70%)
SELF_PLAYER_MAX_SIZE_RATIO = 0.15      # Max size (15% of image)
SELF_PLAYER_IOU_THRESHOLD = 0.3         # Overlap threshold
SELF_PLAYER_TEMPORAL_FRAMES = 10       # Persistence check frames
```

### Spatial Balancing
```python
GRID_SIZE = 5                 # 5x5 grid for position tracking
MAX_SAMPLES_PER_CELL = 100    # Quota per grid region
CENTER_CELL_PENALTY = 0.3     # Reduce center quota by 70%
```

### Frame Sampling
```python
ENGAGEMENT_WINDOW = [-30, -20, -10, 0, 10, 20]  # Frame offsets
RANDOM_SAMPLE_RATE = 0.10     # 10% random exploration frames
```

### Quality Control
```python
BLUR_THRESHOLD = 100          # Laplacian variance threshold
DUPLICATE_HASH_THRESHOLD = 10 # Hamming distance for pHash
TRACK_FRAMES = 20             # Frames to track forward
```

## Pipeline Stages

1. **Smart Frame Sampling**
   - Engagement windows (around combat timestamps)
   - Random exploration samples
   - Edge/corner targeted pre-scan

2. **Detection + Self-Player Filtering**
   - YOLO inference
   - Geometric filtering (size, position)
   - IoU-based overlap filtering
   - Temporal persistence check

3. **Confidence-Based Review**
   - High confidence (>0.7): Auto-label
   - Medium confidence (0.4-0.7): Human confirm
   - Low confidence (<0.4): Manual selection

4. **Spatial Balancing**
   - Tracks samples per grid cell
   - Prefers edge/corner regions
   - Limits center region over-sampling

5. **Label Saving**
   - Saves image + metadata
   - Updates CSV
   - Generates statistics

## Stats Output (collection_stats.json)

```json
{
  "processing_stats": {
    "processed": 1500,
    "auto_labeled": 450,
    "confirmed": 320,
    "manual": 80,
    "tracked": 0,
    "rejected": 350,
    "skipped": 300,
    "self_player_filtered": 42
  },
  "spatial_distribution": {
    "grid_counts": [[10, 15, 20, 12, 8], ...],
    "total_samples": 850,
    "center_20_pct": 0.35,
    "edge_pct": 0.40
  },
  "configuration": {
    "high_conf_threshold": 0.7,
    "self_player_filter": {
      "enabled": true,
      "bottom_region_pct": 0.35,
      "iou_threshold": 0.3
    }
  }
}
```

## Keyboard Shortcuts Reference

```
┌─────────────────────────────────────────────────────────┐
│  REVIEW MODE CONTROLS                                   │
├─────────────────────────────────────────────────────────┤
│  y         = Accept detection                           │
│  n         = Reject detection                           │
│  s         = Skip frame                                 │
│  p         = Skip to next video                       │
│  1-9       = Select detection number                  │
│  c + click = Manual correction (click on enemy)       │
│  t         = Start tracking this enemy                │
│  ESC       = Quit pipeline                              │
└─────────────────────────────────────────────────────────┘
```

## Visual Indicators in Review Window

- **Green box**: High confidence (>0.7) - auto-label candidate
- **Yellow box**: Medium confidence (0.4-0.7) - needs confirmation
- **Red box**: Low confidence (<0.4) - likely false positive
- **Red shaded region**: Self-player exclusion zone (bottom-center)
- **Gray grid**: Spatial balancing grid (5x5)
- **Blue crosshair**: Screen center (aim reference)

## Troubleshooting

### No detections appearing
- Lower `HIGH_CONF_THRESHOLD` and `MED_CONF_THRESHOLD`
- Check YOLO model path is correct
- Verify video file is readable

### Too many false positives
- Increase `SELF_PLAYER_MAX_SIZE_RATIO` to filter large objects
- Adjust `SELF_PLAYER_CENTER_ZONE_*` to match your game
- Increase `HIGH_CONF_THRESHOLD` for stricter auto-labeling

### Too many self-player detections
- Expand `SELF_PLAYER_CENTER_ZONE_X` (e.g., 0.2-0.8)
- Increase `SELF_PLAYER_BOTTOM_REGION_PCT` (e.g., 0.40)
- Lower `SELF_PLAYER_IOU_THRESHOLD` (e.g., 0.2)

### Center bias still high
- Decrease `CENTER_CELL_PENALTY` (e.g., 0.2)
- Increase edge/corner sampling in pre-scan
- Manually select more edge/corner enemies during review

### Processing too slow
- Increase `sample_interval` in pre-scan function
- Reduce `RANDOM_SAMPLE_RATE`
- Use smaller YOLO model (yolov8n.pt instead of yolov8s.pt)

## Engagement File Format (Optional)

Create a JSON file with frame indices of combat/engagement moments:

```json
{
  "video1.mp4": [120, 450, 890, 1234],
  "video2.mp4": [230, 567, 999]
}
```

The pipeline will sample frames around these timestamps (±30, ±20, ±10, 0, +10, +20).

## Example Workflow

1. **Prepare videos**: Place .mp4 files in `src/videos/`
2. **Configure**: Edit thresholds in script if needed
3. **Run**: `python src/process_video_improved.py --videos_dir src/videos --output_dir dataset/labeled`
4. **Review**: Use controls to accept/reject/select detections
5. **Check stats**: Review `collection_stats.json` for bias metrics
6. **Augment**: Run `augment_dataset_improved.py` for bias-aware augmentation
7. **Split**: Run `split_dataset.py` for train/val split

## Next Steps After Collection

```bash
# Bias-aware augmentation
python src/augment_dataset_improved.py \
    --input_csv dataset/labeled/labels_enhanced.csv \
    --input_dir dataset/labeled/images \
    --output_dir dataset/augmented

# Train/val split by video
python src/split_dataset.py \
    --csv dataset/augmented/augmented_labels.csv \
    --img_dir dataset/augmented/images \
    --output_dir dataset/final \
    --val_ratio 0.2 \
    --stratified

# Train model (see docs/model_architecture_recommendation.py)
python train_heatmap_model.py --data dataset/final
```
