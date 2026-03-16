# Woche 6: Training Loop
# =======================
# Trains the ResNet18 regression model using MSE loss and Adam optimizer.
# Saves best model checkpoint and loss curves.

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import get_dataloaders
from src.model import EnemyLocalizationModel, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / count


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns average loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = criterion(predictions, labels)

        total_loss += loss.item() * images.size(0)
        count += images.size(0)

    return total_loss / count


def plot_losses(train_losses, val_losses, save_path):
    """Save train/val loss curves to a PNG file."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Loss curve saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Enemy Localization Model")
    parser.add_argument("--csv", type=str, default="src/dataset/uncleaned/labels.csv",
                        help="Path to labels CSV")
    parser.add_argument("--img_dir", type=str, default="src/dataset/uncleaned",
                        help="Path to image directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model and plots")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained ResNet18 weights")
    args = parser.parse_args()

    # ---- Setup ----
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] Device: {device}")
    print(f"[Config] Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    # ---- Data ----
    train_loader, val_loader, dataset = get_dataloaders(
        csv_path=args.csv,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
    )

    if len(dataset) == 0:
        print("ERROR: No samples found. Check your CSV and image directory paths.")
        sys.exit(1)

    # ---- Model ----
    model = EnemyLocalizationModel(pretrained=args.pretrained).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"[Model] Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # ---- Training Setup ----
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler: reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ---- Training Loop ----
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_epoch = 0
    model_save_path = os.path.join(args.output_dir, "best_model.pth")

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Best':>6} | {'Time':>6}")
    print(f"{'='*60}")

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, model_save_path)

        elapsed = time.time() - epoch_start
        marker = " ★" if is_best else ""
        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {'  ★' if is_best else '   '} | {elapsed:>5.1f}s")

    total_time = time.time() - total_start
    print(f"{'='*60}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best model at epoch {best_epoch} with val_loss={best_val_loss:.6f}")
    print(f"Model saved to: {model_save_path}")

    # ---- Plot Loss Curves ----
    plot_path = os.path.join(args.output_dir, "loss_curve.png")
    plot_losses(train_losses, val_losses, plot_path)


if __name__ == "__main__":
    main()