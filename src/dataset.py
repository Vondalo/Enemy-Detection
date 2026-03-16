# Woche 4: PyTorch Dataset Loader & Preprocessing Pipeline
# =========================================================
# Loads images + CSV labels, resizes to 256x256, normalizes
# coordinates to [0,1], and applies ImageNet normalization.

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class EnemyDataset(Dataset):
    """
    PyTorch Dataset for enemy localization.

    Expects a CSV file with columns: filename, x_norm, y_norm
    where x_norm and y_norm are already normalized to [0, 1].

    Each sample returns:
        image: Tensor [3, 256, 256] (ImageNet-normalized)
        label: Tensor [2] -> [x_norm, y_norm]
    """

    IMG_SIZE = 256

    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path: Path to CSV with columns (filename, x_norm, y_norm)
            img_dir:  Directory containing the image files
            transform: Optional torchvision transforms (applied AFTER resize)
        """
        self.img_dir = img_dir
        self.samples = []

        # Default transform: resize + to tensor + ImageNet normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]     # ImageNet stds
                ),
            ])
        else:
            self.transform = transform

        # Parse CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["filename"]
                x_norm = float(row["x_norm"])
                y_norm = float(row["y_norm"])

                # Only include if the image file actually exists
                img_path = os.path.join(img_dir, filename)
                if os.path.isfile(img_path):
                    self.samples.append((filename, x_norm, y_norm))

        print(f"[EnemyDataset] Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, x_norm, y_norm = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)

        # Load image as RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transforms (resize + normalize)
        image = self.transform(image)

        # Label as float32 tensor [x, y]
        label = torch.tensor([x_norm, y_norm], dtype=torch.float32)

        return image, label


def get_dataloaders(csv_path, img_dir, batch_size=16, val_split=0.2, num_workers=0):
    """
    Creates train and validation DataLoaders with an 80/20 split.

    Args:
        csv_path:    Path to labels CSV
        img_dir:     Path to image directory
        batch_size:  Batch size (default: 16)
        val_split:   Fraction for validation (default: 0.2)
        num_workers: DataLoader workers (default: 0 for Windows compatibility)

    Returns:
        (train_loader, val_loader, dataset)
    """
    dataset = EnemyDataset(csv_path, img_dir)

    # Calculate split sizes
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    # Deterministic split for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"[DataLoaders] Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, dataset


# ----- Quick self-test -----
if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "src/dataset/uncleaned/labels.csv"
    img_dir  = sys.argv[2] if len(sys.argv) > 2 else "src/dataset/uncleaned"

    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Usage: python src/dataset.py <csv_path> <img_dir>")
        sys.exit(1)

    train_loader, val_loader, ds = get_dataloaders(csv_path, img_dir)

    # Show one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: images={images.shape}, labels={labels.shape}")
    print(f"Label range: x=[{labels[:,0].min():.3f}, {labels[:,0].max():.3f}], "
          f"y=[{labels[:,1].min():.3f}, {labels[:,1].max():.3f}]")