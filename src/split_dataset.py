"""
Video-Based Train/Validation Split
===================================

Creates train/val splits by video to prevent temporal leakage.
Ensures no frames from the same video appear in both splits.

Usage:
    python split_dataset.py --csv dataset/labeled/labels_enhanced.csv \
                            --output_dir dataset \
                            --val_ratio 0.2

Output:
    - train/labels.csv & train/images/
    - val/labels.csv & val/images/
    - split_manifest.json (metadata about split)
"""

import os
import csv
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np


def load_labels_by_video(csv_path: str) -> Dict[str, List[Dict]]:
    """Load labels grouped by video_id."""
    video_labels = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row['video_id']
            video_labels[video_id].append(row)
    
    return dict(video_labels)


def analyze_spatial_distribution(labels: List[Dict]) -> Dict:
    """Analyze spatial distribution of labels."""
    has_enemy_count = sum(1 for l in labels if int(l['has_enemy']) == 1)
    no_enemy_count = len(labels) - has_enemy_count
    
    if has_enemy_count == 0:
        return {'has_enemy': 0, 'no_enemy': 0, 'center_ratio': 0}
    
    # Count center vs edge
    center_count = 0
    for label in labels:
        if int(label['has_enemy']) == 1:
            x = float(label['x_center'])
            y = float(label['y_center'])
            # Center 20%
            if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6:
                center_count += 1
    
    return {
        'has_enemy': has_enemy_count,
        'no_enemy': no_enemy_count,
        'center_ratio': center_count / has_enemy_count if has_enemy_count > 0 else 0
    }


def stratified_video_split(video_labels: Dict[str, List[Dict]], 
                            val_ratio: float = 0.2,
                            random_seed: int = 42) -> Tuple[Set[str], Set[str]]:
    """
    Split videos into train/val while attempting to balance:
    - Number of samples
    - Spatial distribution (center vs edge)
    - Negative sample ratio
    
    Uses a greedy approach to minimize distribution shift.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    video_ids = list(video_labels.keys())
    
    # Analyze each video
    video_stats = {}
    for vid in video_ids:
        stats = analyze_spatial_distribution(video_labels[vid])
        video_stats[vid] = {
            'total_samples': len(video_labels[vid]),
            'has_enemy': stats['has_enemy'],
            'no_enemy': stats['no_enemy'],
            'center_ratio': stats['center_ratio']
        }
    
    # Shuffle videos
    random.shuffle(video_ids)
    
    # Greedy assignment to balance val set
    val_videos = set()
    train_videos = set()
    
    val_stats = {'total': 0, 'has_enemy': 0, 'no_enemy': 0}
    target_val_samples = sum(len(labels) for labels in video_labels.values()) * val_ratio
    
    for vid in video_ids:
        stats = video_stats[vid]
        
        # If adding this video to val would exceed target, skip
        if val_stats['total'] + stats['total_samples'] > target_val_samples * 1.2:
            train_videos.add(vid)
            continue
        
        # Try to balance: prefer videos with lower center_ratio (more edge samples)
        # if val already has high center_ratio
        if len(val_videos) == 0:
            val_videos.add(vid)
            val_stats['total'] += stats['total_samples']
            val_stats['has_enemy'] += stats['has_enemy']
            val_stats['no_enemy'] += stats['no_enemy']
        else:
            current_center_ratio = val_stats['has_enemy'] / max(val_stats['total'], 1)
            
            # If this video reduces center bias, prefer it for val
            if stats['center_ratio'] < current_center_ratio:
                val_videos.add(vid)
                val_stats['total'] += stats['total_samples']
                val_stats['has_enemy'] += stats['has_enemy']
                val_stats['no_enemy'] += stats['no_enemy']
            else:
                train_videos.add(vid)
    
    # Move remaining to train
    for vid in video_ids:
        if vid not in val_videos:
            train_videos.add(vid)
    
    return train_videos, val_videos


def simple_random_split(video_labels: Dict[str, List[Dict]], 
                        val_ratio: float = 0.2,
                        random_seed: int = 42) -> Tuple[Set[str], Set[str]]:
    """Simple random split by video."""
    random.seed(random_seed)
    
    video_ids = list(video_labels.keys())
    random.shuffle(video_ids)
    
    n_val = max(1, int(len(video_ids) * val_ratio))
    val_videos = set(video_ids[:n_val])
    train_videos = set(video_ids[n_val:])
    
    return train_videos, val_videos


def copy_files_for_split(video_labels: Dict[str, List[Dict]],
                         train_videos: Set[str], val_videos: Set[str],
                         input_img_dir: str, output_dir: str,
                         manifest: Dict):
    """Copy image files and create label CSVs for each split."""
    input_img_dir = Path(input_img_dir)
    output_dir = Path(output_dir)
    
    # Create directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / 'images').mkdir(exist_ok=True)
    (val_dir / 'images').mkdir(exist_ok=True)
    
    # Get fieldnames from first label
    fieldnames = list(list(video_labels.values())[0][0].keys())
    
    # Create CSV writers
    train_csv_path = train_dir / 'labels.csv'
    val_csv_path = val_dir / 'labels.csv'
    
    train_count = 0
    val_count = 0
    train_copied = 0
    val_copied = 0
    
    with open(train_csv_path, 'w', newline='') as f_train, \
         open(val_csv_path, 'w', newline='') as f_val:
        
        train_writer = csv.DictWriter(f_train, fieldnames=fieldnames)
        val_writer = csv.DictWriter(f_val, fieldnames=fieldnames)
        
        train_writer.writeheader()
        val_writer.writeheader()
        
        for video_id, labels in video_labels.items():
            is_val = video_id in val_videos
            writer = val_writer if is_val else train_writer
            dest_dir = val_dir / 'images' if is_val else train_dir / 'images'
            
            for label in labels:
                # Write label
                writer.writerow(label)
                
                # Copy image
                img_name = label.get('filename', f"{video_id}_frame_{int(label['frame_index']):06d}.png")
                src_path = input_img_dir / img_name
                
                # Try alternate paths
                if not src_path.exists():
                    src_path = input_img_dir / Path(img_name).name
                if not src_path.exists():
                    src_path = input_img_dir / video_id / img_name
                
                if src_path.exists():
                    dest_path = dest_dir / img_name
                    shutil.copy2(src_path, dest_path)
                    
                    if is_val:
                        val_copied += 1
                    else:
                        train_copied += 1
                else:
                    print(f"Warning: Image not found: {src_path}")
                
                if is_val:
                    val_count += 1
                else:
                    train_count += 1
    
    manifest['train_samples'] = train_count
    manifest['val_samples'] = val_count
    manifest['train_copied'] = train_copied
    manifest['val_copied'] = val_copied
    
    print(f"\nTrain set: {train_count} labels, {train_copied} images copied")
    print(f"Val set: {val_count} labels, {val_copied} images copied")


def print_split_statistics(video_labels: Dict[str, List[Dict]], 
                          train_videos: Set[str], val_videos: Set[str]):
    """Print detailed statistics about the split."""
    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    
    # Count by split
    train_labels = []
    val_labels = []
    
    for vid, labels in video_labels.items():
        if vid in train_videos:
            train_labels.extend(labels)
        else:
            val_labels.extend(labels)
    
    # Overall counts
    print(f"\nVideos: {len(train_videos)} train, {len(val_videos)} val")
    print(f"Samples: {len(train_labels)} train, {len(val_labels)} val")
    print(f"Split ratio: {len(val_labels) / (len(train_labels) + len(val_labels)):.1%} val")
    
    # Spatial distribution
    def get_center_ratio(labels):
        enemy_labels = [l for l in labels if int(l['has_enemy']) == 1]
        if not enemy_labels:
            return 0
        center = sum(1 for l in enemy_labels 
                    if 0.4 <= float(l['x_center']) <= 0.6 and 0.4 <= float(l['y_center']) <= 0.6)
        return center / len(enemy_labels)
    
    train_center = get_center_ratio(train_labels)
    val_center = get_center_ratio(val_labels)
    
    print(f"\nCenter bias (enemy samples in center 20%):")
    print(f"  Train: {train_center:.1%}")
    print(f"  Val: {val_center:.1%}")
    print(f"  Difference: {abs(train_center - val_center):.1%}")
    
    # Negative samples
    train_neg = sum(1 for l in train_labels if int(l['has_enemy']) == 0)
    val_neg = sum(1 for l in val_labels if int(l['has_enemy']) == 0)
    
    print(f"\nNegative samples (no enemy):")
    print(f"  Train: {train_neg} ({train_neg / len(train_labels):.1%})")
    print(f"  Val: {val_neg} ({val_neg / len(val_labels):.1%})")


def main():
    parser = argparse.ArgumentParser(description="Video-Based Train/Val Split")
    parser.add_argument("--csv", type=str, required=True,
                       help="Path to labels CSV file")
    parser.add_argument("--img_dir", type=str, required=True,
                       help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for train/val splits")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                       help="Proportion of videos for validation (0-1)")
    parser.add_argument("--stratified", action='store_true',
                       help="Use stratified split (balances spatial distribution)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("Loading labels...")
    video_labels = load_labels_by_video(args.csv)
    print(f"Found {len(video_labels)} videos with {sum(len(l) for l in video_labels.values())} total labels")
    
    # Perform split
    if args.stratified:
        print("Using stratified split...")
        train_videos, val_videos = stratified_video_split(
            video_labels, args.val_ratio, args.seed
        )
    else:
        print("Using random split...")
        train_videos, val_videos = simple_random_split(
            video_labels, args.val_ratio, args.seed
        )
    
    # Print statistics
    print_split_statistics(video_labels, train_videos, val_videos)
    
    # Create manifest
    manifest = {
        'train_videos': sorted(list(train_videos)),
        'val_videos': sorted(list(val_videos)),
        'val_ratio': args.val_ratio,
        'stratified': args.stratified,
        'random_seed': args.seed,
        'source_csv': args.csv
    }
    
    # Copy files
    print("\nCopying files...")
    copy_files_for_split(
        video_labels, train_videos, val_videos,
        args.img_dir, args.output_dir, manifest
    )
    
    # Save manifest
    manifest_path = Path(args.output_dir) / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n[Manifest] Saved to {manifest_path}")
    print(f"\nDataset split complete!")
    print(f"Train: {args.output_dir}/train/")
    print(f"Val: {args.output_dir}/val/")


if __name__ == "__main__":
    main()
