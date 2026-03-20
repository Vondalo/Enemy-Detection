import sys
import json
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import os
import warnings
import argparse

# Suppress warnings that might corrupt JSON output
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import EnemyLocalizationModel

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--save_path", help="Path to save the result image with a red dot.", default=None)
    args = parser.parse_args()

    image_path = args.image_path

    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
        
        if not os.path.exists(model_path):
            print(json.dumps({"error": f"Model not found at {model_path}. Please complete training first."}))
            sys.exit(1)

        # Initialize and load model
        model = EnemyLocalizationModel(pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()

        # Image preprocessing
        original_image = Image.open(image_path).convert("RGB")
        img_w, img_h = original_image.size
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            pred_x = float(output[0, 0].item())
            pred_y = float(output[0, 1].item())

        # Save result image with red dot if requested
        saved_image_path = None
        if args.save_path:
            draw = ImageDraw.Draw(original_image)
            radius = max(5, int(min(img_w, img_h) * 0.01)) # 1% of the smallest dimension or at least 5px
            
            # Map normalized coordinates back to pixel space
            px = pred_x * img_w
            py = pred_y * img_h
            
            # Draw red circle
            draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill="red", outline="white", width=2)
            
            original_image.save(args.save_path)
            saved_image_path = os.path.abspath(args.save_path)

        # Output explicit JSON for the React App to parse
        print(json.dumps({
            "prediction": [pred_x, pred_y],
            "truth": None,
            "saved_image_path": saved_image_path
        }))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
