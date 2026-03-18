import os
import csv
import cv2
import random
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: assuming input directory is uncleaned and output is train_data
INPUT_DIR = os.path.join(SCRIPT_DIR, "dataset", "uncleaned") 
CSV_PATH_IN = os.path.join(INPUT_DIR, "labels.csv")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset", "augmented")
CSV_PATH_OUT = os.path.join(OUTPUT_DIR, "augmented_labels.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define which augmentations to run
# flip is already done in process_video.py, but we can do more
# Options: "brightness", "noise", "shift", "zoom_crop"
AUGMENTATIONS = ["brightness", "noise", "shift", "zoom_crop"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def adjust_brightness(image, factor):
    """Adjusts brightness by a factor (e.g. 0.8 for darker, 1.2 for brighter)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Scale V channel and clip to 255
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def add_noise(image, intensity=0.05):
    """Adds small amount of Gaussian noise"""
    row, col, ch = image.shape
    mean = 0
    sigma = intensity * 255
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def random_shift(image, x_norm, y_norm, max_shift_pct=0.15):
    """Shifts the image randomly and pads with black. Updates coordinates."""
    h, w = image.shape[:2]
    
    # Random shift amounts in pixels
    shift_x = int(random.uniform(-max_shift_pct, max_shift_pct) * w)
    shift_y = int(random.uniform(-max_shift_pct, max_shift_pct) * h)
    
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    # Pad with mean color (e.g. gray) or black. Using black here.
    shifted_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Calculate new normalized coordinates
    # Original pixel coords
    px = x_norm * w
    py = y_norm * h
    
    # New pixel coords
    new_px = px + shift_x
    new_py = py + shift_y
    
    new_x_norm = new_px / w
    new_y_norm = new_py / h
    
    return shifted_img, new_x_norm, new_y_norm

def random_zoom_crop(image, x_norm, y_norm, scale_range=(0.6, 1.4)):
    """
    Randomly crops a sub-region (zoom in) or pads (zoom out), ensuring the 
    target point remains in the frame. Resizes back to original aspect ratio.
    This effectively breaks center bias.
    """
    h, w = image.shape[:2]
    px = x_norm * w
    py = y_norm * h
    
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # New dimensions we want to crop (or pad to)
    new_w, new_h = int(w / scale), int(h / scale)
    
    # We must ensure the crop box contains the target point (px, py).
    # The crop box is defined by top-left corner (crop_x1, crop_y1).
    # Minimum valid crop_x1 is where the right edge of crop touches px: px - new_w
    # Maximum valid crop_x1 is where left edge touches px: px
    
    # To keep it safe and not directly on the edge, we add a margin
    margin = 5
    min_x = max(0, int(px - new_w + margin))
    max_x = min(w - new_w, int(px - margin))
    
    min_y = max(0, int(py - new_h + margin))
    max_y = min(h - new_h, int(py - margin))
    
    # If the new box is larger than the original (zoom out), min_x will be > max_x
    # We handle Zoom in (scale > 1.0) and Zoom out (scale < 1.0) slightly differently
    if scale >= 1.0: # ZOOM IN (Crop)
        if min_x >= max_x: crop_x1 = min_x
        else: crop_x1 = random.randint(min_x, max_x)
            
        if min_y >= max_y: crop_y1 = min_y
        else: crop_y1 = random.randint(min_y, max_y)
        
        crop_x2 = crop_x1 + new_w
        crop_y2 = crop_y1 + new_h
        
        cropped_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
    else: # ZOOM OUT (Pad)
        # For zoom out, we just place the original image randomly inside a larger black canvas.
        # new_w and new_h are LARGER than w, h
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        
        # Random top-left corner to place the original image
        pad_x1 = random.randint(0, new_w - w)
        pad_y1 = random.randint(0, new_h - h)
        
        canvas[pad_y1:pad_y1+h, pad_x1:pad_x1+w] = image
        cropped_img = canvas
        
        # Update point relative to the new canvas
        crop_x1 = -pad_x1
        crop_y1 = -pad_y1
    
    # Resize back to original size (640x360)
    resized_img = cv2.resize(cropped_img, (w, h))
    
    # Recalculate normalized coordinates
    # 1. Coordinate relative to the crop box
    new_px_crop = px - crop_x1
    new_py_crop = py - crop_y1
    
    # 2. Normalize based on the crop box size (which then gets resized to w,h)
    new_x_norm = new_px_crop / new_w
    new_y_norm = new_py_crop / new_h
    
    return resized_img, new_x_norm, new_y_norm


# -----------------------------
# MAIN AUGMENTATION LOOP
# -----------------------------
def main():
    if not os.path.exists(CSV_PATH_IN):
        print(f"Error: Input CSV not found at {CSV_PATH_IN}")
        return

    # Read original labels
    original_data = []
    with open(CSV_PATH_IN, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            original_data.append(row)

    print(f"Found {len(original_data)} images in starting dataset.")
    
    # Prepare output CSV
    with open(CSV_PATH_OUT, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["filename", "x_norm", "y_norm"])
        
        pbar = tqdm(original_data, desc="Applying Augmentations")
        for idx, row in enumerate(pbar):
            filename, x_norm_str, y_norm_str = row
            x_norm = float(x_norm_str)
            y_norm = float(y_norm_str)
            
            img_path = os.path.join(INPUT_DIR, filename)
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # 1. Save original to the new folder
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
            writer.writerow([filename, x_norm, y_norm])
            
            base_name = os.path.splitext(filename)[0]

            # 2. Apply Brightness (Pixel-level, coords stay same)
            if "brightness" in AUGMENTATIONS:
                for factor, label in [(0.7, "dark"), (1.3, "bright")]:
                    aug_img = adjust_brightness(img, factor)
                    aug_name = f"{base_name}_{label}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, aug_name), aug_img)
                    writer.writerow([aug_name, x_norm, y_norm])
                    
            # 3. Apply Noise (Pixel-level, coords stay same)
            if "noise" in AUGMENTATIONS:
                aug_img = add_noise(img, intensity=0.03)
                aug_name = f"{base_name}_noise.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, aug_name), aug_img)
                writer.writerow([aug_name, x_norm, y_norm])
                
            # 4. Apply Random Shift (Spatial, coords change)
            if "shift" in AUGMENTATIONS:
                # generate 2 random shifts per image
                for i in range(2):
                    aug_img, nx, ny = random_shift(img, x_norm, y_norm, max_shift_pct=0.2)
                    
                    # Ensure coordinates are still valid (0 to 1)
                    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                        aug_name = f"{base_name}_shift_{i}.png"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, aug_name), aug_img)
                        writer.writerow([aug_name, nx, ny])

            # 5. Apply Random Zoom & Crop (Spatial, coords change)
            if "zoom_crop" in AUGMENTATIONS:
                # generate 3 zoom variations per image (e.g. 1 zoom out, 2 zoom ins)
                for i, scale_range in enumerate([(0.7, 0.9), (1.1, 1.4), (1.3, 1.6)]):
                    aug_img, nx, ny = random_zoom_crop(img, x_norm, y_norm, scale_range=scale_range)
                    
                    # Ensure coordinates are still valid (0 to 1) 
                    # (they might fall slightly out of bounds if margin calculations fail at extreme edges)
                    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                        aug_name = f"{base_name}_zoom_{i}.png"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, aug_name), aug_img)
                        writer.writerow([aug_name, nx, ny])

    print(f"\nAugmentation complete! Check the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
