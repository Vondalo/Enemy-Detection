# Woche 8: Inference Demo & Live Pipeline
# ==========================================
# Load image → Resize 256x256 → Predict → Denormalize → Draw Dot
# Runs fully offline, no internet required.

import os
import sys
import argparse
import glob
import numpy as np
import torch
import cv2
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.model import EnemyLocalizationModel


# Image preprocessing (same as training)
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    model = EnemyLocalizationModel(pretrained=False).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded from {model_path} (epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f})")
    return model


@torch.no_grad()
def predict(model, image_bgr, device):
    """
    Run inference on a single BGR image.

    Args:
        model: Trained EnemyLocalizationModel
        image_bgr: OpenCV BGR image (any size)
        device: torch device

    Returns:
        (x_pixel, y_pixel): Predicted enemy position in original image coordinates
        (x_norm, y_norm): Normalized prediction [0, 1]
    """
    h, w = image_bgr.shape[:2]

    # Convert BGR → RGB and preprocess
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(image_rgb).unsqueeze(0).to(device)

    # Predict
    output = model(tensor).cpu().numpy()[0]
    x_norm, y_norm = float(output[0]), float(output[1])

    # Denormalize to original image coordinates
    x_pixel = int(x_norm * w)
    y_pixel = int(y_norm * h)

    return (x_pixel, y_pixel), (x_norm, y_norm)


def draw_prediction(image_bgr, x, y, label="Enemy"):
    """Draw prediction dot and crosshair on image."""
    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    # Outer circle (white border)
    cv2.circle(overlay, (x, y), 14, (255, 255, 255), 2)
    # Inner circle (red filled)
    cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)
    # Crosshair lines
    cv2.line(overlay, (x - 20, y), (x + 20, y), (0, 255, 255), 1)
    cv2.line(overlay, (x, y - 20), (x, y + 20), (0, 255, 255), 1)

    # Label text
    text = f"{label} ({x}, {y})"
    cv2.putText(overlay, text, (x + 15, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Instructions
    cv2.putText(overlay, "N=Next | Q=Quit", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return overlay


def run_single_image(model, image_path, device):
    """Run inference on a single image and display the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    (x, y), (xn, yn) = predict(model, image, device)
    print(f"  Prediction: ({x}, {y})  |  Normalized: ({xn:.4f}, {yn:.4f})")

    result = draw_prediction(image, x, y)

    # Fit to screen
    screen_h, screen_w = 900, 1400
    h, w = result.shape[:2]
    scale = min(screen_w / w, screen_h / h, 1.0)
    display = cv2.resize(result, (int(w * scale), int(h * scale)))

    cv2.imshow("Enemy Detection - Inference Demo", display)
    print("Press any key to continue, 'q' to quit...")
    key = cv2.waitKey(0) & 0xFF
    return key != ord('q')


def run_directory(model, dir_path, device):
    """Run inference on all images in a directory with keyboard navigation."""
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(dir_path, ext)))

    image_files.sort()

    if len(image_files) == 0:
        print(f"ERROR: No images found in {dir_path}")
        return

    print(f"\n[Demo] Found {len(image_files)} images in {dir_path}")
    print("[Controls] N = Next | P = Previous | Q = Quit\n")

    idx = 0
    while 0 <= idx < len(image_files):
        img_path = image_files[idx]
        filename = os.path.basename(img_path)
        print(f"[{idx + 1}/{len(image_files)}] {filename}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  Skipping unreadable image: {filename}")
            idx += 1
            continue

        (x, y), (xn, yn) = predict(model, image, device)
        print(f"  Prediction: ({x}, {y})  |  Normalized: ({xn:.4f}, {yn:.4f})")

        result = draw_prediction(image, x, y)

        # Add file info
        cv2.putText(result, f"[{idx+1}/{len(image_files)}] {filename}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Fit to screen
        screen_h, screen_w = 900, 1400
        h, w = result.shape[:2]
        scale = min(screen_w / w, screen_h / h, 1.0)
        display = cv2.resize(result, (int(w * scale), int(h * scale)))

        cv2.imshow("Enemy Detection - Inference Demo", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p') and idx > 0:
            idx -= 1
        else:
            idx += 1

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Enemy Detection Inference Demo",
        epilog="Examples:\n"
               "  python inference_demo.py --image screenshot.png\n"
               "  python inference_demo.py --dir src/dataset/uncleaned\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image for inference")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to a directory of images for batch inference")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")

    # Load model
    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("Train the model first with: python src/train.py")
        sys.exit(1)

    model = load_model(args.model, device)

    # Run inference
    if args.image:
        run_single_image(model, args.image, device)
        cv2.destroyAllWindows()
    elif args.dir:
        run_directory(model, args.dir, device)
    else:
        print("Please specify --image or --dir. Use --help for usage info.")
        sys.exit(1)

    print("\nInference demo complete.")


if __name__ == "__main__":
    main()