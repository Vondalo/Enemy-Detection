# Game Enemy Localization – CNN Regression Project

A CNN-based system that predicts the **(x, y)** pixel coordinate of an enemy player in game screenshots. Built with **ResNet18** (pretrained on ImageNet) and a custom regression head.

## Project Structure

```
Enemy-Detection/
├── data/
│   ├── raw/              # Raw screenshots (1920x1080)
│   ├── processed/        # Processed/cleaned images
│   └── labels/           # Ground truth CSVs
├── src/
│   ├── dataset.py        # PyTorch Dataset & DataLoaders
│   ├── model.py          # ResNet18 regression model
│   ├── train.py          # Training loop (MSE + Adam)
│   ├── evaluate.py       # Pixel error metrics & visual overlays
│   ├── process_video.py  # Video → labeled dataset pipeline
│   ├── preprocess_images.py  # YOLO auto-labeling
│   ├── screenshot_tool.py    # In-game screenshot capture
│   └── test_gpu.py       # GPU availability check
├── models/               # Saved model checkpoints & plots
├── inference_demo.py     # Live inference demo
├── SETUP.py              # Project scaffolding
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup

1. **Clone** the repo and create a virtual environment:
   ```bash
   git clone <repo-url>
   cd Enemy-Detection
   python -m venv venv
   venv\Scripts\activate       # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU** (optional but recommended):
   ```bash
   python src/test_gpu.py
   ```

## Workflow

### Phase 1 – Data Collection & Labeling

1. **Capture screenshots** during gameplay:
   ```bash
   python src/screenshot_tool.py
   # Press F10 to capture, ESC to quit
   ```

2. **Extract frames from video** with YOLO-assisted labeling:
   ```bash
   python src/process_video.py
   # Click on enemies or press number keys, 'a' to accept, 'd' to skip
   ```

3. **Output**: Images in `src/dataset/uncleaned/` and `labels.csv` with columns `filename, x_norm, y_norm`.

### Phase 2 – Training

4. **Train the model**:
   ```bash
   python src/train.py --csv src/dataset/uncleaned/labels.csv --img_dir src/dataset/uncleaned --epochs 30
   ```
   - Saves best model to `models/best_model.pth`
   - Saves loss curve to `models/loss_curve.png`

### Phase 3 – Evaluation & Inference

5. **Evaluate** accuracy:
   ```bash
   python src/evaluate.py --model models/best_model.pth --csv src/dataset/uncleaned/labels.csv --img_dir src/dataset/uncleaned
   ```
   - Prints Average Pixel Error, accuracy thresholds
   - Saves error histogram and visual overlays to `models/evaluation/`

6. **Run inference demo**:
   ```bash
   # Single image
   python inference_demo.py --image path/to/screenshot.png

   # Browse a directory
   python inference_demo.py --dir src/dataset/uncleaned
   # Controls: N = Next, P = Previous, Q = Quit
   ```

## Model Architecture

```
ResNet18 (pretrained) → AvgPool → Flatten
  → Linear(512, 128) → ReLU → Dropout(0.3)
  → Linear(128, 2) → Sigmoid
  → Output: (x_norm, y_norm) ∈ [0, 1]
```

## Training Config

| Parameter     | Value          |
|---------------|----------------|
| Loss Function | MSE            |
| Optimizer     | Adam (lr=1e-4) |
| Epochs        | 30             |
| Batch Size    | 16             |
| Input Size    | 256 × 256      |
| Train/Val     | 80% / 20%      |

## Tech Stack

- **Python 3.10+**
- **PyTorch** + torchvision (ResNet18)
- **OpenCV** (image processing & display)
- **YOLOv8** (ultralytics, for assisted labeling)
- **Matplotlib** (loss curves & plots)