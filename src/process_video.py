import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from tqdm import tqdm

# -----------------------------
# CONFIGURATION
# -----------------------------
# Get paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "dataset", "uncleaned")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

FRAME_SKIP = 5                   # process every 5th frame
CONF_THRESHOLD = 0.3
CROSSHAIR_MODE = "center"
MAX_BOTTOM_Y_RATIO = 0.7
MAX_BOX_AREA_RATIO = 0.25
PATCH_SIZE = 50                  # pixels around click for precise center
CROP_SIZE = 256                  # size of cropped training images
AUGMENTATIONS = ["flip", "brightness"]

# Maximum preview window size (screen resolution)
SCREEN_W, SCREEN_H = 1280, 720

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
print(f"Loading YOLOv8 model... This might take a few seconds.")
model = YOLO("yolov8n.pt")  # replace with fine-tuned enemy model if available
print(f"Model loaded successfully! Scanning directory for videos: {VIDEO_DIR}")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_crosshair_position(frame):
    h, w = frame.shape[:2]
    if CROSSHAIR_MODE == "center":
        return (w//2, h//2)
    return CROSSHAIR_MODE

def augment_image(img, x, y):
    aug_data = []

    if "flip" in AUGMENTATIONS:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_x = img.width - x
        aug_data.append((flipped, flipped_x, y))

    if "brightness" in AUGMENTATIONS:
        enhancer = ImageEnhance.Brightness(img)
        for factor in [0.8, 1.2]:
            bright = enhancer.enhance(factor)
            aug_data.append((bright, x, y))

    return aug_data

def get_precise_center(frame, click_x, click_y, patch_size=PATCH_SIZE):
    h, w = frame.shape[:2]
    x1 = max(click_x - patch_size//2, 0)
    y1 = max(click_y - patch_size//2, 0)
    x2 = min(click_x + patch_size//2, w)
    y2 = min(click_y + patch_size//2, h)

    patch = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    if M["m00"] == 0:
        return click_x, click_y
    cx = int(M["m10"]/M["m00"]) + x1
    cy = int(M["m01"]/M["m00"]) + y1
    return cx, cy

# -----------------------------
# MOUSE CALLBACK
# -----------------------------
clicked_point = None
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (int(x/param["zoom_factor"]), int(y/param["zoom_factor"]))

# -----------------------------
# MAIN VIDEO LOOP
# -----------------------------
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
print(f"Found {len(video_files)} video(s) to process.")

# Determine starting image_index from existing data so we don't overwrite
if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
    import pandas as pd
    existing = pd.read_csv(CSV_PATH)
    # Filter for frame_N.png pattern
    mask = existing['filename'].str.match(r'frame_\d+\.png')
    if mask.any():
        max_idx = existing.loc[mask, 'filename'].str.extract(r'frame_(\d+)').astype(float).max().values[0]
        image_index = int(max_idx) + 1 if not np.isnan(max_idx) else 0
    else:
        image_index = 0
else:
    image_index = 0

for video_name in video_files:
    current_video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(current_video_path)
    frame_index = 0
    
    print(f"\nProcessing Video: {video_name}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frame")

    # Append to CSV instead of overwriting; write header only if file is new
    write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
    with open(CSV_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["filename","x_norm","y_norm"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pbar.update(1)

            if frame_index % FRAME_SKIP != 0:
                frame_index += 1
                continue

            h, w = frame.shape[:2]
            crosshair = get_crosshair_position(frame)

            # DETECTION
            results = model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                cx = float((x1+x2)/2)
                cy = float((y1+y2)/2)
                area = (x2-x1)*(y2-y1)

                if cy > MAX_BOTTOM_Y_RATIO*h or area > MAX_BOX_AREA_RATIO*h*w:
                    continue
                if conf < CONF_THRESHOLD:
                    continue
                detections.append((cx, cy))

            if len(detections) == 0:
                frame_index += 1
                continue

            # -----------------------------
            # PREVIEW IMAGE WITH SCALING
            # -----------------------------
            preview = frame.copy()
            for idx, (cx, cy) in enumerate(detections):
                cv2.circle(preview, (int(cx), int(cy)), 8, (0,0,255), -1)
                cv2.putText(preview, str(idx+1), (int(cx)+5,int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # Draw crosshair
            cv2.line(preview, (w//2-10,h//2),(w//2+10,h//2),(255,0,0),2)
            cv2.line(preview, (w//2,h//2-10),(w//2,h//2+10),(255,0,0),2)

            # Auto-fit to screen
            zoom_w = SCREEN_W / w
            zoom_h = SCREEN_H / h
            zoom_factor = min(zoom_w, zoom_h, 1.0)
            preview_zoom = cv2.resize(preview, (int(w*zoom_factor), int(h*zoom_factor)))

            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
            cv2.imshow("Preview", preview_zoom)
            cv2.setMouseCallback("Preview", mouse_callback, {"zoom_factor": zoom_factor})
            clicked_point = None

            # -----------------------------
            # SELECTION LOOP
            # -----------------------------
            selected = None
            print("Click roughly on enemy or press number key for detection, then 'a' to accept, 'd' to skip frame.")

            while selected is None:
                cv2.imshow("Preview", preview_zoom)
                key = cv2.waitKey(1) & 0xFF

                # Number key selects detection
                if key >= ord('1') and key <= ord(str(min(len(detections),9))):
                    index = key - ord('1')
                    selected = detections[index]

                # 'a' accepts manually clicked point
                elif key == ord('a') and clicked_point is not None:
                    selected = get_precise_center(frame, clicked_point[0], clicked_point[1])

                # 'd' skips frame
                elif key == ord('d'):
                    selected = False
                    break

            if not selected:
                frame_index += 1
                continue

            # -----------------------------
            # SAVE FULL FRAME FOR TRAINING
            # -----------------------------
            px, py = selected
            
            # Save the full frame instead of a crop to avoid center bias
            x_norm = px / w
            y_norm = py / h
            
            # Save resized frame (640x360 is still 16:9, but much smaller than 1080p)
            save_img = cv2.resize(frame, (640, 360))
            
            filename = f"frame_{image_index}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(save_path, save_img)
            
            writer.writerow([filename, x_norm, y_norm])
            csvfile.flush() # Ensure it's saved to disk immediately

            # (Optional) Basic augmentations could be added back here if needed
            # for the full frame, but standard horizontal flip is most common.
            if "flip" in AUGMENTATIONS:
                flipped = cv2.flip(save_img, 1)
                aug_filename = f"frame_{image_index}_flip.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, aug_filename), flipped)
                writer.writerow([aug_filename, 1.0 - x_norm, y_norm])
                csvfile.flush() # Ensure it's saved to disk immediately

            image_index += 1
            frame_index += 1
    
    pbar.close()
    cap.release()

cv2.destroyAllWindows()
print("\nAll videos processed. Dataset generation complete.")