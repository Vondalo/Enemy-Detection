#!/usr/bin/env python3
"""
Reset script for Enemy Detection project.
Clears training data, model checkpoints, and statistics to start fresh.

Usage:
    python reset_project.py [--all] [--data] [--models] [--logs] [--yes]

Options:
    --all       Reset everything (data, models, logs)
    --data      Reset only dataset directories
    --models    Reset only model checkpoints
    --logs      Reset only log files and stats
    --yes       Skip confirmation prompts
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    """Get project root directory (parent of script location)."""
    return Path(__file__).parent


def reset_dataset(project_root: Path, backup: bool = True) -> None:
    """Reset all dataset directories."""
    dataset_dirs = [
        project_root / "dataset" / "labeled",
        project_root / "dataset" / "augmented",
        project_root / "dataset" / "final",
        project_root / "dataset" / "uncleaned",
        project_root / "src" / "dataset" / "uncleaned",
        project_root / "src" / "dataset" / "augmented",
    ]
    
    for dir_path in dataset_dirs:
        if dir_path.exists():
            if backup:
                # Backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = Path(str(dir_path) + f"_backup_{timestamp}")
                shutil.move(str(dir_path), str(backup_path))
                print(f"  ✓ Moved {dir_path.name} to backup: {backup_path.name}")
            else:
                shutil.rmtree(dir_path)
                print(f"  ✓ Deleted {dir_path.name}")
        else:
            print(f"  - {dir_path.name} (not found, skipping)")
    
    # Recreate empty directories
    (project_root / "dataset" / "labeled" / "images").mkdir(parents=True, exist_ok=True)
    (project_root / "dataset" / "uncleaned" / "images").mkdir(parents=True, exist_ok=True)
    print("  ✓ Created fresh dataset directories")


def reset_models(project_root: Path, backup: bool = True) -> None:
    """Reset model checkpoints and weights."""
    model_paths = [
        project_root / "models",
        project_root / "checkpoints",
        project_root / "weights",
        project_root / "runs",  # YOLO default output directory
        project_root / "src" / "models",
        project_root / "src" / "checkpoints",
    ]
    
    # Also find .pt, .pth, .onnx files
    model_files = []
    for pattern in ["**/*.pt", "**/*.pth", "**/*.onnx", "**/*.engine"]:
        model_files.extend(project_root.glob(pattern))
    
    # Exclude YOLO base models (yolov8*.pt)
    model_files = [f for f in model_files if not f.name.startswith("yolov8")]
    
    # Remove directories
    for path in model_paths:
        if path.exists():
            if backup:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = Path(str(path) + f"_backup_{timestamp}")
                shutil.move(str(path), str(backup_path))
                print(f"  ✓ Moved {path.name} to backup")
            else:
                shutil.rmtree(path)
                print(f"  ✓ Deleted {path.name}")
    
    # Remove individual model files
    for file_path in model_files:
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            backup_path = file_path.parent / backup_name
            shutil.move(str(file_path), str(backup_path))
            print(f"  ✓ Moved {file_path.name} to backup")
        else:
            file_path.unlink()
            print(f"  ✓ Deleted {file_path.name}")
    
    if not model_paths and not model_files:
        print("  - No model files found")


def reset_logs(project_root: Path) -> None:
    """Reset log files and statistics."""
    log_patterns = [
        "**/*.log",
        "**/*_stats.json",
        "**/collection_stats.json",
        "**/training_history.json",
        "**/labels.csv",
        "**/labels_enhanced.csv",
        "**/augmented_labels.csv",
        "**/train_labels.csv",
        "**/val_labels.csv",
    ]
    
    removed_count = 0
    for pattern in log_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                removed_count += 1
                print(f"  ✓ Deleted {file_path.name}")
    
    if removed_count == 0:
        print("  - No log/stats files found")
    else:
        print(f"  ✓ Removed {removed_count} log/stats files")


def reset_cache(project_root: Path) -> None:
    """Reset cache and temporary files."""
    cache_dirs = [
        project_root / "__pycache__",
        project_root / ".pytest_cache",
        project_root / ".mypy_cache",
        project_root / ".cache",
    ]
    
    # Find all __pycache__ directories
    for pycache in project_root.rglob("__pycache__"):
        cache_dirs.append(pycache)
    
    for dir_path in cache_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  ✓ Cleared {dir_path.relative_to(project_root)}")


def confirm_reset(what: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"\nAre you sure you want to reset {what}? [y/N]: ").lower()
    return response in ['y', 'yes']


def main():
    parser = argparse.ArgumentParser(
        description="Reset Enemy Detection project data and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python reset_project.py --all --yes          # Reset everything without prompts
    python reset_project.py --data               # Reset only dataset
    python reset_project.py --models --logs      # Reset models and logs, keep data
    python reset_project.py --all                # Interactive mode (will ask for confirmation)
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Reset everything (data, models, logs, cache)')
    parser.add_argument('--data', action='store_true',
                        help='Reset only dataset directories')
    parser.add_argument('--models', action='store_true',
                        help='Reset only model checkpoints')
    parser.add_argument('--logs', action='store_true',
                        help='Reset only log files and statistics')
    parser.add_argument('--cache', action='store_true',
                        help='Clear cache and temporary files')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts')
    parser.add_argument('--no-backup', action='store_true',
                        help='Delete without creating backups (DANGEROUS)')
    
    args = parser.parse_args()
    
    # If no specific options given, show help
    if not (args.all or args.data or args.models or args.logs or args.cache):
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK RESET OPTIONS:")
        print("="*60)
        print("1. Full reset (everything): python reset_project.py --all")
        print("2. Just data:              python reset_project.py --data")
        print("3. Just models:            python reset_project.py --models")
        print("4. Just logs:              python reset_project.py --logs")
        print("="*60)
        sys.exit(0)
    
    project_root = get_project_root()
    backup = not args.no_backup
    
    print("="*60)
    print("ENEMY DETECTION PROJECT RESET")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Backup mode: {'ENABLED' if backup else 'DISABLED'}")
    print("="*60)
    
    # Confirm reset
    if not args.yes:
        reset_items = []
        if args.all or args.data:
            reset_items.append("dataset (all labeled images and CSV files)")
        if args.all or args.models:
            reset_items.append("model checkpoints and weights")
        if args.all or args.logs:
            reset_items.append("log files and statistics")
        if args.all or args.cache:
            reset_items.append("cache files")
        
        if not confirm_reset(" and ".join(reset_items)):
            print("\nReset cancelled.")
            sys.exit(0)
    
    # Perform reset
    print("\n" + "="*60)
    
    if args.all or args.data:
        print("\n🗑️  Resetting Dataset...")
        reset_dataset(project_root, backup=backup)
    
    if args.all or args.models:
        print("\n🗑️  Resetting Models...")
        reset_models(project_root, backup=backup)
    
    if args.all or args.logs:
        print("\n🗑️  Resetting Logs & Stats...")
        reset_logs(project_root)
    
    if args.all or args.cache:
        print("\n🗑️  Clearing Cache...")
        reset_cache(project_root)
    
    print("\n" + "="*60)
    print("✅ RESET COMPLETE")
    print("="*60)
    print("\nProject is now clean and ready for fresh training!")
    print("\nNext steps:")
    print("1. Collect new data:   python src/process_video_improved.py --videos_dir src/videos --output_dir dataset/labeled")
    print("2. Augment data:       python src/augment_dataset_improved.py --input_csv dataset/labeled/labels_enhanced.csv --input_dir dataset/labeled/images --output_dir dataset/augmented")
    print("3. Split dataset:      python src/split_dataset.py --csv dataset/augmented/augmented_labels.csv --img_dir dataset/augmented/images --output_dir dataset/final")
    print("4. Train model:        See docs/model_architecture_recommendation.py")
    print("="*60)


if __name__ == "__main__":
    main()
