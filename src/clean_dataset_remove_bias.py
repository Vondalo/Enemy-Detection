from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    PLAYER_CLASS_ID,
    DetectionAnnotation,
    find_image_path,
    group_annotations_by_filename,
    load_annotations,
    write_annotations_csv,
    write_yolo_label_file,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ENEMY_BUCKETS = ["Corner", "Edge", "Center", "Normal"]
SIDE_BUCKETS = ["PlayerOnly", "Negative"]


def get_spatial_category(x: float, y: float) -> str:
    is_corner_x = x < 0.15 or x > 0.85
    is_corner_y = y < 0.15 or y > 0.85
    if is_corner_x and is_corner_y:
        return "Corner"

    if x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9:
        return "Edge"

    if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6:
        return "Center"

    return "Normal"


def choose_primary_enemy(annotations: List[DetectionAnnotation]) -> DetectionAnnotation | None:
    enemies = [
        annotation
        for annotation in annotations
        if annotation.has_enemy and annotation.class_id != PLAYER_CLASS_ID
    ]
    if not enemies:
        return None
    return max(enemies, key=lambda annotation: annotation.width * annotation.height)


def categorize_image(annotations: List[DetectionAnnotation]) -> str:
    primary_enemy = choose_primary_enemy(annotations)
    if primary_enemy is not None:
        return get_spatial_category(primary_enemy.x_center, primary_enemy.y_center)

    positives = [annotation for annotation in annotations if annotation.has_enemy]
    if positives:
        return "PlayerOnly"
    return "Negative"


def build_output_name(source_filename: str) -> str:
    source_path = Path(source_filename)
    digest = hashlib.sha1(source_filename.encode("utf-8")).hexdigest()[:8]
    return f"{source_path.stem}_{digest}{source_path.suffix or '.png'}"


def clean_and_balance(input_csv: str, img_dir: str, output_dir: str) -> None:
    input_csv = str(input_csv)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)
    out_img_dir = output_dir / "images"
    out_label_dir = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset: {input_csv}")
    annotations = load_annotations(input_csv)
    grouped = group_annotations_by_filename(annotations)
    print(f"Loaded {len(grouped)} images and {len(annotations)} annotation rows")

    buckets: Dict[str, List[tuple[str, List[DetectionAnnotation]]]] = {
        key: []
        for key in ENEMY_BUCKETS + SIDE_BUCKETS
    }
    for filename, image_annotations in grouped.items():
        category = categorize_image(image_annotations)
        buckets.setdefault(category, []).append((filename, image_annotations))

    print("\nInitial distribution by image:")
    for category in ENEMY_BUCKETS + SIDE_BUCKETS:
        print(f"  - {category}: {len(buckets.get(category, []))}")

    enemy_counts = [len(buckets[category]) for category in ENEMY_BUCKETS if len(buckets[category]) > 0]
    if enemy_counts:
        target_count = min(enemy_counts)
        print(f"\nTarget enemy-image count per non-empty spatial bucket: {target_count}")
    else:
        target_count = 0
        print("\nNo enemy-labeled images were found. Keeping only side buckets.")

    selected: List[tuple[str, List[DetectionAnnotation]]] = []
    for category in ENEMY_BUCKETS:
        items = buckets[category]
        if not items:
            print(f"  - {category}: SKIPPED (0 images)")
            continue
        if target_count == 0:
            sampled = items
        elif len(items) > target_count:
            sampled = random.sample(items, target_count)
        else:
            sampled = items
        selected.extend(sampled)
        print(f"  - {category}: kept {len(sampled)} image(s)")

    for category in SIDE_BUCKETS:
        items = buckets[category]
        selected.extend(items)
        print(f"  - {category}: kept {len(items)} image(s)")

    if not selected:
        raise RuntimeError("No samples remained after cleaning and balancing.")

    random.shuffle(selected)

    final_annotations: List[DetectionAnnotation] = []
    copied_count = 0
    for source_filename, image_annotations in selected:
        src_path = find_image_path(img_dir, source_filename)
        if src_path is None:
            print(f"Warning: Image not found for {source_filename}")
            continue

        destination_name = build_output_name(source_filename)
        destination_path = out_img_dir / destination_name
        shutil.copy2(src_path, destination_path)
        copied_count += 1

        rewritten_annotations: List[DetectionAnnotation] = []
        for annotation in image_annotations:
            rewritten_annotations.append(
                DetectionAnnotation(
                    filename=destination_name,
                    class_id=annotation.class_id,
                    class_name=annotation.class_name,
                    has_enemy=annotation.has_enemy,
                    x_center=annotation.x_center,
                    y_center=annotation.y_center,
                    width=annotation.width,
                    height=annotation.height,
                    video_id=annotation.video_id,
                    frame_idx=annotation.frame_idx,
                    timestamp=annotation.timestamp,
                    confidence=annotation.confidence,
                    auto_labeled=annotation.auto_labeled,
                    bbox_source=annotation.bbox_source,
                    aug_type=annotation.aug_type,
                )
            )

        write_yolo_label_file(out_label_dir / f"{Path(destination_name).stem}.txt", rewritten_annotations)
        final_annotations.extend(rewritten_annotations)

    if not final_annotations:
        raise RuntimeError("No annotations could be exported to the cleaned dataset.")

    output_csv = output_dir / "labels_cleaned.csv"
    write_annotations_csv(output_csv, final_annotations)

    print("\nDataset cleaned and balanced.")
    print(f"Total output images: {copied_count}")
    print(f"Total output annotations: {len(final_annotations)}")
    print(f"Output CSV: {output_csv}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset and rebalance enemy spatial bias for YOLO datasets")
    parser.add_argument("--csv", type=str, required=True, help="Input labels CSV")
    parser.add_argument("--img_dir", type=str, required=True, help="Input images directory")
    parser.add_argument("--output_dir", type=str, default="dataset/cleaned", help="Output directory")
    args = parser.parse_args()

    clean_and_balance(args.csv, args.img_dir, args.output_dir)
