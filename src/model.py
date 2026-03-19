# Woche 5: ResNet18 Regression Model
# ====================================
# Pretrained ResNet18 backbone with a custom regression head
# that outputs 2 continuous values (x_norm, y_norm).

import torch
import torch.nn as nn
from torchvision import models

# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')



class EnemyLocalizationModel(nn.Module):
    """
    ResNet18-based regression model for enemy localization.

    Architecture:
        - ResNet18 backbone (pretrained on ImageNet)
        - Classification head replaced with regression head:
          Linear(512 → 128) → ReLU → Dropout(0.3) → Linear(128 → 2)

    Input:  Tensor [B, 3, 256, 256]
    Output: Tensor [B, 2] → (x_norm, y_norm) in [0, 1]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove the original fully connected classification layer
        # Keep everything up to avgpool → outputs [B, 512]
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Custom regression head
        self.regression_head = nn.Sequential(
            nn.Flatten(),               # [B, 512, 1, 1] → [B, 512]
            nn.Linear(512, 128),        # Fully connected with 128 neurons
            nn.ReLU(inplace=True),      # ReLU activation
            nn.Dropout(0.3),            # Regularization
            nn.Linear(128, 2),          # Output: 2 values (x, y)
            nn.Sigmoid(),               # Clamp output to [0, 1]
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, 256, 256]

        Returns:
            Tensor [B, 2] with (x_norm, y_norm) ∈ [0, 1]
        """
        features = self.features(x)     # [B, 512, 1, 1]
        output = self.regression_head(features)  # [B, 2]
        return output


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----- Quick self-test -----
if __name__ == "__main__":
    model = EnemyLocalizationModel(pretrained=False)

    # Test forward pass
    dummy_input = torch.randn(4, 3, 256, 256)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    total, trainable = count_parameters(model)
    print(f"Parameters:   {total:,} total | {trainable:,} trainable")

    assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"
    print("\n✓ Model test passed!")
