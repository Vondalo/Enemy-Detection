import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEO_PATH = r"C:\Users\Eray\Documents\GitHub\Enemy-Detection\src\videos\video1.mp4"
OUTPUT_DIR = r"dataset/uncleaned"
CSV_PATH = r"dataset/uncleaned/labels.csv"
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
os.makedirs("dataset/uncleaned", exist_ok=True)

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("yolov8n.pt")  # replace with fine-tuned enemy model if available

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
cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0
image_index = 0

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename","x_norm","y_norm"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        px, py = selected

        # -----------------------------
        # CROP & RESIZE FOR TRAINING
        # -----------------------------
        x1_crop = max(int(px - CROP_SIZE//2), 0)
        y1_crop = max(int(py - CROP_SIZE//2), 0)
        x2_crop = min(x1_crop + CROP_SIZE, w)
        y2_crop = min(y1_crop + CROP_SIZE, h)
        cropped_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop]

        # Adjust center for cropped image
        cx_crop = px - x1_crop
        cy_crop = py - y1_crop

        # Save image
        filename = f"frame_{image_index}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, cv2.resize(cropped_frame, (CROP_SIZE,CROP_SIZE)))

        # Normalized coordinates
        x_norm = cx_crop / CROP_SIZE
        y_norm = cy_crop / CROP_SIZE
        writer.writerow([filename, x_norm, y_norm])

        # AUGMENTATIONS
        pil_img = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
        aug_data = augment_image(pil_img, cx_crop, cy_crop)
        for i,(aug_img,ax,ay) in enumerate(aug_data):
            aug_filename = f"frame_{image_index}_aug{i}.png"
            aug_path = os.path.join(OUTPUT_DIR, aug_filename)
            aug_img = aug_img.resize((CROP_SIZE,CROP_SIZE))  # ensure 256x256
            aug_img.save(aug_path)
            writer.writerow([aug_filename, ax/CROP_SIZE, ay/CROP_SIZE])

        image_index += 1
        frame_index += 1

cap.release()
cv2.destroyAllWindows()
print("Dataset generation complete.")