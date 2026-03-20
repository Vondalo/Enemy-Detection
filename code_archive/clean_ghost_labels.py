import os
import csv
from pathlib import Path

def clean_dataset(output_dir="dataset/labeled"):
    """
    Cleans up the dataset by removing rows from labels_enhanced.csv
    that point to missing image files in the images/ directory.
    This happens when cv2.imwrite silently fails due to a full disk.
    """
    labels_csv = Path(output_dir) / "labels_enhanced.csv"
    images_dir = Path(output_dir) / "images"
    
    if not labels_csv.exists():
        print(f"File not found: {labels_csv}")
        return
        
    print(f"Scanning dataset in {output_dir}...")
    
    # Read all labels
    with open(labels_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
        
    initial_count = len(rows)
    valid_rows = []
    missing_count = 0
    
    # Check if images actually exist
    for row in rows:
        filename = row.get('filename')
        img_path = images_dir / filename
        
        if img_path.exists() and img_path.stat().st_size > 0:
            valid_rows.append(row)
        else:
            missing_count += 1
            
    print(f"Total entries in CSV: {initial_count}")
    print(f"Missing image files: {missing_count} (Ghost labels from full disk)")
    print(f"Valid image entries: {len(valid_rows)}")
    
    # Write only valid ones back
    if missing_count > 0:
        backup_csv = Path(output_dir) / "labels_enhanced.csv.backup"
        labels_csv.rename(backup_csv)
        print(f"Backed up original CSV to {backup_csv.name}")
        
        with open(labels_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_rows)
            
        print("Successfully cleaned up the dataset CSV!")
    else:
        print("Dataset is clean! No missing images found.")

if __name__ == "__main__":
    clean_dataset()
