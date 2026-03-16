import cv2
import pandas as pd
from ultralytics import YOLO
import albumentations as A
import os

# 1. Auto-Labeling Setup
model = YOLO('yolov8n.pt') # Nutzt ein vortrainiertes Standard-Modell
raw_dir = "data/raw/"
output_csv = "data/labels/ground_truth.csv"

# Load existing labels so we append instead of overwriting
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    data = existing_df.values.tolist()
    already_processed = set(existing_df['filename'].tolist())
else:
    data = []
    already_processed = set()

# 2. Augmentation Pipeline (Beispiel)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.1)
], keypoint_params=A.KeypointParams(format='xy'))

# 3. Loop durch die Bilder
for img_name in os.listdir(raw_dir):
    # Skip images already processed in a previous run
    if img_name in already_processed:
        continue
    img = cv2.imread(os.path.join(raw_dir, img_name))
    results = model(img) # YOLO erkennt Gegner
    
    for r in results:
        boxes = r.boxes.xywh.cpu().numpy() # x_center, y_center, w, h
        if len(boxes) > 0:
            # Wir nehmen den ersten erkannten Gegner
            x_c, y_c, w, h = boxes[0]
            
            # Hier könnten wir jetzt Augmentations anwenden:
            # transformed = transform(image=img, keypoints=[(x_c, y_c)])
            
            data.append([img_name, x_c, y_c])
            break # Nur ein Gegner pro Bild (wie in eurem Plan)

# 4. Speichern als CSV
df = pd.DataFrame(data, columns=['filename', 'x', 'y'])
df.to_csv(output_csv, index=False)
print(f"Auto-Labeling beendet. {len(df)} Bilder gelabelt.")