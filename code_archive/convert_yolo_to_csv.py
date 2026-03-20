import os
import csv
from pathlib import Path

def convert_dir(base_dir):
    labels_dir = base_dir / "labels"
    images_dir = base_dir / "images"
    
    if not labels_dir.exists() or not images_dir.exists():
        print(f"Skipping {base_dir} (missing labels or images dir)")
        return

    # Map valid image files (resolving any jpg/png mismatches automatically)
    image_files = {}
    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files[img_path.stem] = img_path.name

    csv_path = base_dir / "labels.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Match exactly what src/dataset.py expects
        writer.writerow(['filename', 'x_norm', 'y_norm'])
        
        count = 0
        for txt_file in labels_dir.glob("*.txt"):
            stem = txt_file.stem
            if stem not in image_files:
                continue
                
            with open(txt_file, 'r') as tf:
                lines = tf.readlines()
                if not lines:
                    continue
                
                # Assume the highest priority target is the first row
                parts = lines[0].strip().split()
                if len(parts) >= 5:
                    x_norm = float(parts[1])
                    y_norm = float(parts[2])
                    
                    writer.writerow([image_files[stem], x_norm, y_norm])
                    count += 1
                    
    print(f"✅ Generated coordinate mapping for {count} images in {base_dir.name} -> {csv_path}")

def main():
    dataset_root = Path("src/fn-dataset")
    print(f"Scanning YOLO format dataset at: {dataset_root}")
    for split in ["train", "valid", "test"]:
        convert_dir(dataset_root / split)

if __name__ == "__main__":
    main()
