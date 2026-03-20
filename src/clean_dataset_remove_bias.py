import os
import csv
import random
import shutil
import argparse
from pathlib import Path

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Player model region (normalized xyxy)
PLAYER_BOX = (0.281, 0.389, 0.656, 0.944)

def is_in_player_region(x, y):
    """Check if a point is within the player model box."""
    x1, y1, x2, y2 = PLAYER_BOX
    return x1 <= x <= x2 and y1 <= y <= y2

def get_spatial_category(x, y):
    """Categorize a point into Corner, Edge, Center, or Normal."""
    # Corners (15% margin)
    is_corner_x = x < 0.15 or x > 0.85
    is_corner_y = y < 0.15 or y > 0.85
    if is_corner_x and is_corner_y:
        return "Corner"
    
    # Edges (10% margin, excluding corners)
    if x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9:
        return "Edge"
    
    # Center (20% box)
    if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6:
        return "Center"
    
    return "Normal"

def clean_and_balance(input_csv, img_dir, output_dir):
    """Filter player model and balance spatial distribution."""
    os.makedirs(output_dir, exist_ok=True)
    out_img_dir = os.path.join(output_dir, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    # Buckets for each category
    buckets = {
        "Corner": [],
        "Edge": [],
        "Center": [],
        "Normal": []
    }

    print(f"Reading dataset: {input_csv}")
    total_processed = 0
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_processed += 1
            try:
                # Handle both 'x_norm' and 'x_center' formats
                x = float(row.get('x_norm', row.get('x_center')))
                y = float(row.get('y_norm', row.get('y_center')))
                
                # 1. Filter out player model
                if is_in_player_region(x, y):
                    continue
                
                # 2. Categorize
                category = get_spatial_category(x, y)
                buckets[category].append(row)
            except (ValueError, TypeError):
                continue
    print(f"Finished reading. Total rows processed: {total_processed}")

    # Summary before balancing
    print("\nInitial distribution (after player filtering):")
    for cat, items in buckets.items():
        print(f"  - {cat}: {len(items)}")

    # 3. Balance categories (Undersampling)
    # Only consider categories that have at least one sample
    counts = [len(items) for items in buckets.values() if len(items) > 0]
    
    if not counts:
        print("\n❌ ERROR: All categories are empty after filtering!")
        return

    min_count = min(counts)
    print(f"\nTarget count per category (balancing non-empty categories): {min_count}")
    
    final_rows = []
    for cat in buckets:
        if len(buckets[cat]) >= min_count:
            sampled = random.sample(buckets[cat], min_count)
            final_rows.extend(sampled)
            print(f"  - {cat}: sampled {min_count}")
        elif len(buckets[cat]) == 0:
            print(f"  - {cat}: SKIPPED (0 samples)")
        else:
            # This case shouldn't happen with min_count logic above, but for safety:
            final_rows.extend(buckets[cat])
            print(f"  - {cat}: took all {len(buckets[cat])} samples")

    # Shuffle to mix categories
    random.shuffle(final_rows)

    # 4. Save results
    output_csv = os.path.join(output_dir, "labels_cleaned.csv")
    fieldnames = list(final_rows[0].keys()) if final_rows else []
    
    if not fieldnames:
        print("❌ ERROR: No samples left after filtering and balancing!")
        return

    print(f"Writing {len(final_rows)} samples to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    # 5. Copy images
    print("Copying images...")
    copied_count = 0
    for row in final_rows:
        img_name = row['filename']
        src_path = os.path.join(img_dir, img_name)
        dst_path = os.path.join(out_img_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"⚠️ Image not found: {img_name}")

    print(f"\n✅ Dataset cleaned and balanced!")
    print(f"Total samples: {len(final_rows)}")
    print(f"Images copied: {copied_count}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean dataset and remove positional bias")
    parser.add_argument("--csv", type=str, required=True, help="Input labels CSV")
    parser.add_argument("--img_dir", type=str, required=True, help="Input images directory")
    parser.add_argument("--output_dir", type=str, default="dataset/cleaned", help="Output directory")
    args = parser.parse_args()
    
    clean_and_balance(args.csv, args.img_dir, args.output_dir)
