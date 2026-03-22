import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    center_ratio,
    export_split_dataset,
    group_annotations_by_video,
    load_annotations,
    split_annotations_by_video,
    write_data_yaml,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_split_statistics(train_annotations, val_annotations, train_videos, val_videos):
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)
    print(f"Videos: {len(train_videos)} train, {len(val_videos)} val")
    print(f"Samples: {len(train_annotations)} train, {len(val_annotations)} val")
    total = max(1, len(train_annotations) + len(val_annotations))
    print(f"Split ratio: {len(val_annotations) / total:.1%} val")
    print(f"Center bias: {center_ratio(train_annotations):.1%} train vs {center_ratio(val_annotations):.1%} val")


def main():
    parser = argparse.ArgumentParser(description="Create YOLO train/val splits from bbox CSV annotations")
    parser.add_argument("--csv", type=str, required=True, help="Path to labels CSV file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train/val splits")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--stratified", action="store_true", help="Balance the split by positional bias")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    annotations = load_annotations(args.csv)
    if not annotations:
        raise RuntimeError(f"No valid annotations found in {args.csv}")

    train_annotations, val_annotations, train_videos, val_videos = split_annotations_by_video(
        annotations,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratified=args.stratified,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(annotations)} annotations from {args.csv}")
    print_split_statistics(train_annotations, val_annotations, train_videos, val_videos)

    print("\nCopying train split...")
    train_stats = export_split_dataset(train_annotations, args.img_dir, output_dir / "train")
    print("Copying validation split...")
    val_stats = export_split_dataset(val_annotations, args.img_dir, output_dir / "val")

    data_yaml = write_data_yaml(output_dir)
    video_groups = group_annotations_by_video(annotations)
    manifest = {
        "source_csv": args.csv,
        "source_videos": sorted(video_groups.keys()),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "val_ratio": args.val_ratio,
        "stratified": args.stratified,
        "seed": args.seed,
        "train_annotations": len(train_annotations),
        "val_annotations": len(val_annotations),
        "train_images": train_stats["images"],
        "val_images": val_stats["images"],
        "data_yaml": str(data_yaml),
    }

    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nData YAML: {data_yaml}")
    print(f"Manifest:  {manifest_path}")
    print("Split complete.")


if __name__ == "__main__":
    main()
