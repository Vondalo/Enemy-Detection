import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import EnemyLocalizationModel

# Configuration
MODEL_PATH = "models/best_model.pth"
CSV_PATH = "src/dataset/uncleaned/labels.csv"
BASE_OVERLAY_DIR = "overlays"

# Image preprocessing
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model(device):
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return None
    model = EnemyLocalizationModel(pretrained=False).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def get_ground_truth(filename):
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        match = df[df['filename'] == os.path.basename(filename)]
        if not match.empty:
            return [float(match.iloc[0]['x_norm']), float(match.iloc[0]['y_norm'])]
    except Exception:
        pass
    return None

@torch.no_grad()
def predict(model, image_path, device):
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(device)
        output = model(tensor).cpu().numpy()[0]
        return [float(output[0]), float(output[1])]
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/overlay_cli.py <path_to_image>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    if model is None: return

    # Perform inference
    pred = predict(model, image_path, device)
    if pred is None: return
    
    # Get ground truth
    truth = get_ground_truth(image_path)

    # Prepare overlay directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_OVERLAY_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Draw overlay
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ERROR: OpenCV could not read image: {image_path}")
        return
        
    h, w = image_bgr.shape[:2]
    
    # Draw Prediction (Red)
    px, py = int(pred[0] * w), int(pred[1] * h)
    cv2.circle(image_bgr, (px, py), 10, (0, 0, 255), -1)
    cv2.circle(image_bgr, (px, py), 12, (255, 255, 255), 2)
    cv2.putText(image_bgr, "Predicted", (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw Truth (Green) if available
    if truth:
        tx, ty = int(truth[0] * w), int(truth[1] * h)
        cv2.circle(image_bgr, (tx, ty), 8, (0, 255, 0), -1)
        cv2.circle(image_bgr, (tx, ty), 10, (255, 255, 255), 2)
        cv2.putText(image_bgr, "True", (tx + 15, ty + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Line between if they are different
        cv2.line(image_bgr, (px, py), (tx, ty), (255, 255, 0), 1)

    # Save result
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, image_bgr)
    
    print(f"\nDetection successful!")
    print(f"Prediction: {pred[0]:.4f}, {pred[1]:.4f}")
    if truth:
        print(f"Ground Truth: {truth[0]:.4f}, {truth[1]:.4f}")
        error = np.sqrt((pred[0]-truth[0])**2 + (pred[1]-truth[1])**2)
        print(f"L2 Error: {error:.6f}")
    
    print(f"\nOverlay saved to: {output_path}")

if __name__ == "__main__":
    main()
