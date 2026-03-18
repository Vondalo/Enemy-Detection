import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
CSV_PATH = "src/dataset/uncleaned/labels.csv"
IMG_DIR = "src/dataset/uncleaned"
JITTER_PAD = 64  # Pixels to pad on each side
NUM_VARIANTS = 2 # How many jittered versions per original image

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    new_rows = []
    
    print(f"Generating synthetic jitter for {len(df)} images...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['filename']
        img_path = os.path.join(IMG_DIR, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        # Current normalized center
        enemy_x = row['x_norm'] * w
        enemy_y = row['y_norm'] * h
        
        for i in range(NUM_VARIANTS):
            # Pad the image
            # We use BORDER_REPLICATE to avoid hard black edges which might confuse the model
            padded = cv2.copyMakeBorder(
                img, JITTER_PAD, JITTER_PAD, JITTER_PAD, JITTER_PAD, 
                cv2.BORDER_REPLICATE
            )
            
            # New enemy position in padded image
            px = enemy_x + JITTER_PAD
            py = enemy_y + JITTER_PAD
            
            # Randomly crop a 256x256 window from the padded image
            # The window must contain the enemy (mostly)
            # Max offset we can shift the crop while keeping enemy inside is JITTER_PAD
            off_x = np.random.randint(-JITTER_PAD + 10, JITTER_PAD - 10)
            off_y = np.random.randint(-JITTER_PAD + 10, JITTER_PAD - 10)
            
            # Top-left of crop
            # Original center was at (w/2, h/2). In padded it's (w/2 + pad, h/2 + pad)
            # If we want it to stay exactly in center, crop starts at (pad, pad)
            c_x1 = JITTER_PAD + off_x
            c_y1 = JITTER_PAD + off_y
            
            crop = padded[c_y1:c_y1+h, c_x1:c_x1+w]
            
            # New enemy position relative to crop
            new_enemy_x = px - c_x1
            new_enemy_y = py - c_y1
            
            new_name = f"{os.path.splitext(img_name)[0]}_jitter{i}.png"
            cv2.imwrite(os.path.join(IMG_DIR, new_name), crop)
            
            new_rows.append({
                'filename': new_name,
                'x_norm': new_enemy_x / w,
                'y_norm': new_enemy_y / h
            })

    # Append new rows to dataframe and save
    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(CSV_PATH, index=False)
    
    print(f"\nDone! Added {len(new_rows)} jittered images.")
    print(f"Total dataset size: {len(updated_df)}")

if __name__ == "__main__":
    main()
