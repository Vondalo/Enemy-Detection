"""
Unified Fortnite Enemy Detection Pipeline
=========================================

Executes the end-to-end machine learning workflow:
1. Data Extraction (process_video_improved.py)
2. Augmentation & Bias Correction (augment_dataset_improved.py)
3. Dataset Splitting & Analysis (split_dataset.py, visualize_dataset.py)
4. Model Training (train.py)

Usage:
    python run_pipeline.py --videos_dir src/videos --auto_skip --epochs 30
"""

import os
import sys
import subprocess
import argparse
import time

# Ensure utf-8 output to avoid charmap encode errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_command(cmd, step_name):
    """Run a shell command and check for errors."""
    print(f"\n{'-'*60}")
    print(f"🚀 STEP: {step_name}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'-'*60}\n")
    
    start_time = time.time()
    try:
        # We use check_call to pipe output directly to console
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Pipeline failed at step '{step_name}' with exit code {e.returncode}")
        print("Pipeline aborted.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ User interrupted pipeline at step '{step_name}'.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"\n✅ Step '{step_name}' completed in {duration:.1f} seconds.\n")

def main():
    parser = argparse.ArgumentParser(description="End-to-End Enemy Detection Pipeline")
    parser.add_argument("--videos_dir", type=str, default="src/videos", help="Directory with input videos")
    parser.add_argument("--auto_skip", action="store_true", help="Auto-skip review in data collection")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--skip_collection", action="store_true", help="Skip video processing phase")
    args = parser.parse_args()
    
    # 1. DATA COLLECTION
    if not args.skip_collection:
        cmd_collect = [
            sys.executable, "src/process_video_improved.py",
            "--videos_dir", args.videos_dir,
            "--output_dir", "dataset/labeled",
            "--yolo_model", "yolov8n.pt"
        ]
        if args.auto_skip:
            cmd_collect.append("--auto_skip")
        run_command(cmd_collect, "1. Video Processing & Collection")
    else:
        print("\n⏭️ Skipping Video Processing (using existing data in dataset/labeled/)")

    # 2. DATA AUGMENTATION & RELOCATION
    cmd_augment = [
        sys.executable, "src/augment_dataset_improved.py",
        "--input_csv", "dataset/labeled/labels_enhanced.csv",
        "--input_dir", "dataset/labeled/images",
        "--output_dir", "dataset/augmented"
    ]
    run_command(cmd_augment, "2. Bias-Aware Augmentation & Relocation")
    
    # 3. DATASET CLEANING & REBALANCING (Remove Player Model & Fix Bias)
    cmd_clean = [
        sys.executable, "src/clean_dataset_remove_bias.py",
        "--csv", "dataset/augmented/augmented_labels.csv",
        "--img_dir", "dataset/augmented/images",
        "--output_dir", "dataset/cleaned"
    ]
    run_command(cmd_clean, "3. Dataset Cleaning & Rebalancing")
    
    # 4. SPATIAL ANALYSIS (Visualize Center Bias / Blind spot fixes)
    cmd_visualize = [
        sys.executable, "src/visualize_dataset.py",
        "--csv", "dataset/cleaned/labels_cleaned.csv",
        "--output", "dataset/cleaned/center_bias_heatmap.png"
    ]
    run_command(cmd_visualize, "4a. Generate Bias Heatmap")
    
    # 5. TRAIN/VAL SPLITTING
    cmd_split = [
        sys.executable, "src/split_dataset.py",
        "--csv", "dataset/cleaned/labels_cleaned.csv",
        "--img_dir", "dataset/cleaned/images",
        "--output_dir", "dataset/final",
        "--val_ratio", "0.2",
        "--stratified"
    ]
    run_command(cmd_split, "4b. Stratified Video Split")
    
    # 6. MODEL TRAINING
    cmd_train = [
        sys.executable, "src/train.py",
        # Use relative path from script location if needed, 
        # but train.py usually looks at dataset/final/
        "--epochs", str(args.epochs),
        "--batch_size", "16",
        "--lr", "1e-4"
    ]
    run_command(cmd_train, "5. Model Training")
    
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("Best model saved to: models/best_model.pth")
    print("Bias visualization saved to: dataset/cleaned/center_bias_heatmap.png")

if __name__ == "__main__":
    main()
