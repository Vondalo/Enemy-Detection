# Game Enemy Detection - Object Detection Pipeline

This project now trains an **object detector** instead of regressing a single `(x, y)` point. The data pipeline exports **bounding boxes** in YOLO format, the trainer fine-tunes a detector backbone with `ultralytics`, and inference returns full detections instead of a red-dot coordinate guess.

## Environment Setup

To set up the Python environment required for this project, run the following command in PowerShell:

```powershell
.\setup_venv.ps1
```

This script will:
1. Create a virtual environment in the `.venv` directory.
2. Install all necessary dependencies from `requirements.txt`.
3. Detect an NVIDIA GPU and replace the default CPU-only PyTorch build with the official CUDA-enabled wheel.
4. Print the final `torch` / CUDA runtime so you can confirm whether training will use the GPU.

If you have an NVIDIA GPU, the trainer should report a CUDA-enabled runtime such as `torch=2.9.1+cu128`, `cuda_available=True`, and your GPU name.
If it still shows a CPU-only build, training will fall back to CPU unless you reinstall the CUDA wheel.

## What Changed

- The old ResNet18 coordinate regressor was replaced with a YOLO-style detector workflow.
- Video processing now writes:
  - `images/` for captured frames
  - `labels/` with YOLO `.txt` annotations
  - `labels_enhanced.csv` with bbox metadata (`x_center`, `y_center`, `width`, `height`)
- Training now produces `models/best_model.pt`.
- Inference returns bounding boxes, classes, confidences, and an annotated image.

## Model Choices

The trainer supports these detector baselines from `src/model.py`:

- `yolov8n`: best default for fast iteration and weaker GPUs
- `yolov8s`: better small-target recall when you can afford more compute
- `yolov8m`: stronger accuracy ceiling with higher VRAM/training cost
- `rtdetr-l`: good comparison baseline when latency matters less than global scene reasoning

Default choice: `yolov8n`

Why that default:
- It is already bundled in this repo as `yolov8n.pt`
- It keeps iteration fast while you stabilize the new bbox labels
- It is much easier to debug data issues with a fast detector than a heavy one

## Typical Workflow

### 1. Collect bbox training data

```bash
python src/process_video_improved.py --videos_dir src/videos --output_dir dataset/labeled --yolo_model yolov8n.pt --auto_skip
```

### 2. Augment collected labels

```bash
python src/augment_dataset_improved.py --input_csv dataset/labeled/labels_enhanced.csv --input_dir dataset/labeled/images --output_dir dataset/augmented
```

### 3. Clean and rebalance

```bash
python src/clean_dataset_remove_bias.py --csv dataset/augmented/augmented_labels.csv --img_dir dataset/augmented/images --output_dir dataset/cleaned
```

### 4. Split into YOLO train/val folders

```bash
python src/split_dataset.py --csv dataset/cleaned/labels_cleaned.csv --img_dir dataset/cleaned/images --output_dir dataset/final --val_ratio 0.2 --stratified
```

This writes:

- `dataset/final/train/images`
- `dataset/final/train/labels`
- `dataset/final/val/images`
- `dataset/final/val/labels`
- `dataset/final/data.yaml`

### 5. Train the detector

```bash
python src/train.py --dataset_dir dataset/final --model yolov8n --epochs 30 --batch_size 16
```

Or train directly from a collected CSV and let the script build the split for you:

```bash
python src/train.py --train_csv data_sets/my_dataset/labels_enhanced.csv --train_dir data_sets/my_dataset/images --model yolov8n --epochs 30
```

### 6. Run inference

```bash
python src/predict_cli.py path/to/frame.png --save_path annotated.png
```

## Output Format

The bbox CSV now uses these core columns:

- `filename`
- `class_id`
- `class_name`
- `x_center`
- `y_center`
- `width`
- `height`
- `video_id`
- `frame_idx`
- `confidence`
- `auto_labeled`

Coordinates are normalized to `[0, 1]` in YOLO format.
