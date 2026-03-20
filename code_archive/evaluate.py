# Woche 7: Evaluation & Metrics
# ===============================
# Evaluates the trained model: Average Pixel Error, visual overlays,
# and error case analysis.

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import get_dataloaders, EnemyDataset
from src.model import EnemyLocalizationModel


# Original image resolution (for denormalization)
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080


@torch.no_grad()
def evaluate_model(model, loader, device, orig_w=ORIGINAL_WIDTH, orig_h=ORIGINAL_HEIGHT):
    """
    Run predictions and compute pixel errors.

    Returns:
        errors: list of Euclidean pixel errors
        predictions: list of (x_pred, y_pred) in normalized coords
        ground_truths: list of (x_true, y_true) in normalized coords
    """
    model.eval()
    errors = []
    predictions = []
    ground_truths = []

    for images, labels in loader:
        images = images.to(device)
        preds = model(images).cpu().numpy()
        labels = labels.numpy()

        for pred, gt in zip(preds, labels):
            # Denormalize to pixel coordinates
            px_pred = pred[0] * orig_w
            py_pred = pred[1] * orig_h
            px_true = gt[0] * orig_w
            py_true = gt[1] * orig_h

            # Euclidean pixel error
            error = np.sqrt((px_pred - px_true)**2 + (py_pred - py_true)**2)
            errors.append(error)
            predictions.append(pred)
            ground_truths.append(gt)

    return errors, predictions, ground_truths


def plot_error_histogram(errors, save_path):
    """Save histogram of pixel errors."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, edgecolor="black", alpha=0.7, color="#4A90D9")
    plt.axvline(np.mean(errors), color="red", linestyle="--", linewidth=2,
                label=f"Mean: {np.mean(errors):.1f}px")
    plt.axvline(np.median(errors), color="green", linestyle="--", linewidth=2,
                label=f"Median: {np.median(errors):.1f}px")
    plt.xlabel("Pixel Error (Euclidean distance)")
    plt.ylabel("Count")
    plt.title("Distribution of Prediction Errors")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Error histogram saved to {save_path}")


def create_visual_overlays(dataset, model, device, save_dir, num_samples=10,
                           orig_w=ORIGINAL_WIDTH, orig_h=ORIGINAL_HEIGHT):
    """
    Create overlay images showing predicted (red) vs ground-truth (green) dots.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # Sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        image_tensor, label = dataset[idx]

        # Predict
        with torch.no_grad():
            pred = model(image_tensor.unsqueeze(0).to(device)).cpu().numpy()[0]

        # Denormalize image for visualization
        # Reverse ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = image_tensor.numpy().transpose(1, 2, 0)  # [H, W, C]
        img_np = img_np * std + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Scale coordinates to 256x256 display
        img_h, img_w = img_bgr.shape[:2]
        gt_x, gt_y = int(label[0] * img_w), int(label[1] * img_h)
        pred_x, pred_y = int(pred[0] * img_w), int(pred[1] * img_h)

        # Draw ground truth (green) and prediction (red)
        cv2.circle(img_bgr, (gt_x, gt_y), 6, (0, 255, 0), -1)      # Green = truth
        cv2.circle(img_bgr, (pred_x, pred_y), 6, (0, 0, 255), -1)  # Red = prediction
        cv2.line(img_bgr, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)  # Line between

        # Pixel error (in original resolution)
        error = np.sqrt((pred[0]*orig_w - label[0].item()*orig_w)**2 +
                        (pred[1]*orig_h - label[1].item()*orig_h)**2)
        cv2.putText(img_bgr, f"Error: {error:.1f}px", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        save_path = os.path.join(save_dir, f"overlay_{i:03d}.png")
        cv2.imwrite(save_path, img_bgr)

    print(f"[Visual] Saved {len(indices)} overlay images to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enemy Localization Model")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--csv", type=str, default="src/dataset/uncleaned/labels.csv",
                        help="Path to labels CSV")
    parser.add_argument("--img_dir", type=str, default="src/dataset/uncleaned",
                        help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="models/evaluation",
                        help="Output directory for evaluation results")
    parser.add_argument("--orig_w", type=int, default=ORIGINAL_WIDTH,
                        help="Original image width for denormalization")
    parser.add_argument("--orig_h", type=int, default=ORIGINAL_HEIGHT,
                        help="Original image height for denormalization")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of visual overlay samples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")

    # ---- Load Model ----
    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Train the model first with: python src/train.py")
        sys.exit(1)

    model = EnemyLocalizationModel(pretrained=False).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Model] Loaded from {args.model} (epoch {checkpoint['epoch']})")

    # ---- Load Data ----
    _, val_loader, dataset = get_dataloaders(args.csv, args.img_dir)

    # ---- Compute Metrics ----
    errors, predictions, ground_truths = evaluate_model(
        model, val_loader, device, args.orig_w, args.orig_h
    )

    if len(errors) == 0:
        print("ERROR: No validation samples found.")
        sys.exit(1)

    # ---- Print Results ----
    errors_arr = np.array(errors)
    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  Samples evaluated:    {len(errors)}")
    print(f"  Average Pixel Error:  {errors_arr.mean():.2f} px")
    print(f"  Median Pixel Error:   {np.median(errors_arr):.2f} px")
    print(f"  Std Pixel Error:      {errors_arr.std():.2f} px")
    print(f"  Min Pixel Error:      {errors_arr.min():.2f} px")
    print(f"  Max Pixel Error:      {errors_arr.max():.2f} px")
    print(f"  < 50px accuracy:      {(errors_arr < 50).mean()*100:.1f}%")
    print(f"  < 100px accuracy:     {(errors_arr < 100).mean()*100:.1f}%")
    print(f"{'='*50}")

    # ---- Save Error Histogram ----
    hist_path = os.path.join(args.output_dir, "error_histogram.png")
    plot_error_histogram(errors, hist_path)

    # ---- Visual Overlays ----
    overlay_dir = os.path.join(args.output_dir, "overlays")
    create_visual_overlays(dataset, model, device, overlay_dir,
                           num_samples=args.num_samples,
                           orig_w=args.orig_w, orig_h=args.orig_h)

    # ---- Save metrics to text file ----
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Average Pixel Error: {errors_arr.mean():.2f} px\n")
        f.write(f"Median Pixel Error:  {np.median(errors_arr):.2f} px\n")
        f.write(f"Std Pixel Error:     {errors_arr.std():.2f} px\n")
        f.write(f"Min Pixel Error:     {errors_arr.min():.2f} px\n")
        f.write(f"Max Pixel Error:     {errors_arr.max():.2f} px\n")
        f.write(f"< 50px accuracy:     {(errors_arr < 50).mean()*100:.1f}%\n")
        f.write(f"< 100px accuracy:    {(errors_arr < 100).mean()*100:.1f}%\n")
    print(f"[Metrics] Saved to {metrics_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
