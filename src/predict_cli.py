import os
import sys
import json
import torch
import cv2
import pandas as pd
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import EnemyLocalizationModel

# Configuration
MODEL_PATH = "models/best_model.pth"
CSV_PATH = "src/dataset/uncleaned/labels.csv"

# Image preprocessing (same as training)
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
        # Search for exact filename or base name
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
        return None

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"File not found: {image_path}"}))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    
    if model is None:
        print(json.dumps({"error": "Model not found. Train it first!"}))
        return

    pred = predict(model, image_path, device)
    truth = get_ground_truth(image_path)

    result = {
        "prediction": pred,
        "truth": truth
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
