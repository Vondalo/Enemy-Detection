"""
Model Architecture Recommendation: Heatmap-Based Approach
===========================================================

Current System: Coordinate Regression (ResNet18 → Linear(2))
Recommended System: Heatmap Prediction (Encoder-Decoder → 64×64 Heatmap)

## Why Heatmap Over Regression?

### 1. Handles "No Enemy" Naturally

**Regression Problem:**
- Must predict coordinates even when no enemy exists
- Requires separate classification head (has_enemy?)
- Ambiguous: what should (x,y) be when no enemy?

**Heatmap Solution:**
- Empty heatmap = no enemy
- Natural confidence: sum of heatmap values
- Single unified output for detection + localization

### 2. Multiple Enemy Support

**Regression Problem:**
- Single (x,y) output can't represent multiple enemies
- Must choose which enemy to detect
- Misses secondary targets

**Heatmap Solution:**
- Multiple peaks = multiple enemies
- Can detect 2, 3, 5+ enemies simultaneously
- No architectural change needed

### 3. Spatial Uncertainty Encoding

**Regression Problem:**
- Point estimate provides no uncertainty
- Model can't express "enemy is somewhere in this area"
- Overconfident wrong predictions

**Heatmap Solution:**
- Peak spread represents uncertainty
- Sharp peak = confident, centered detection
- Diffuse blob = uncertain, broad area
- Natural calibration of model confidence

### 4. Training Stability

**Regression Problem:**
- MSE loss is unforgiving of small errors
- Large gradients when far from target
- Can cause training instability

**Heatmap Solution:**
- Gaussian-smoothed targets are soft
- Model learns gradually from rough to precise localization
- More forgiving loss landscape
- Better convergence

### 5. Positional Bias Resistance

**Regression Problem:**
- Easy to learn center bias (always predict 0.5, 0.5)
- Weak signal from actual visual features
- Your current 9.93× center bias demonstrates this

**Heatmap Solution:**
- Must predict spatial distribution
- Harder to "cheat" with center-only predictions
- Visual features drive peak location naturally
- Built-in spatial reasoning

## Proposed Architecture

```
Input: 256×256×3 (game screenshot)
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
Backbone          Backbone          Backbone
(ResNet18)       (EfficientNet)    (MobileNet)
    ↓                 ↓                 ↓
Feature Map      Feature Map       Feature Map
(8×8×512)        (8×8×1280)        (8×8×1280)
    ↓                 ↓                 ↓
    └─────────────────┴─────────────────┘
                      ↓
         Decoder (Upsampling)
        Conv2D → Upsample → Conv2D
                ↓
         Feature (64×64×64)
                ↓
         Conv2D (1×1 kernel)
                ↓
         Output: 64×64 Heatmap
```

## Implementation Details

### Target Generation

Convert (x, y) bbox center to heatmap:

```python
def generate_target_heatmap(x_norm, y_norm, has_enemy, size=64):
    """Create Gaussian heatmap target."""
    heatmap = np.zeros((size, size))
    
    if has_enemy == 0:
        return heatmap  # Empty = no enemy
    
    # Convert normalized to heatmap coordinates
    cx = int(x_norm * size)
    cy = int(y_norm * size)
    
    # Gaussian with σ=2 (adjustable)
    sigma = 2.0
    for i in range(size):
        for j in range(size):
            dist = (i - cy)**2 + (j - cx)**2
            heatmap[i, j] = np.exp(-dist / (2 * sigma**2))
    
    return heatmap
```

### Loss Function

```python
class HeatmapLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        # Focal loss variant for heatmap
        self.alpha = alpha  # Focus parameter
        self.beta = beta    # Down-weight easy negatives
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) - sigmoid activated
        target: (B, 1, H, W) - Gaussian targets
        """
        # Modified focal loss for heatmaps
        pos_mask = target.eq(1)
        neg_mask = target.lt(1)
        
        # Positive locations
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_mask
        
        # Negative locations (down-weighted near positives)
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * \
                   torch.pow(1 - target, self.beta) * neg_mask
        
        loss = -(pos_loss.sum() + neg_loss.sum()) / pos_mask.sum()
        return loss
```

### Inference

```python
def extract_peak(heatmap, threshold=0.3):
    """Extract (x, y) from predicted heatmap."""
    # Find local maxima
    from scipy.ndimage import maximum_filter
    
    # Suppress non-maximal values
    max_filtered = maximum_filter(heatmap, size=3)
    peaks = (heatmap == max_filtered) & (heatmap > threshold)
    
    # Extract coordinates
    peak_coords = np.argwhere(peaks)
    
    # Convert to normalized coordinates
    results = []
    h, w = heatmap.shape
    for cy, cx in peak_coords:
        conf = heatmap[cy, cx]
        x_norm = cx / w
        y_norm = cy / h
        results.append((x_norm, y_norm, conf))
    
    # Sort by confidence
    results.sort(key=lambda x: x[2], reverse=True)
    return results

# Alternative: Soft-argmax for single enemy
# Differentiable, more precise than hard argmax
def soft_argmax(heatmap):
    """Differentiable coordinate extraction."""
    h, w = heatmap.shape
    
    # Create coordinate grids
    y_coords = torch.arange(h, device=heatmap.device).float()
    x_coords = torch.arange(w, device=heatmap.device).float()
    
    # Softmax over flattened heatmap
    flat_heatmap = heatmap.view(-1)
    weights = F.softmax(flat_heatmap, dim=0)
    
    # Weighted sum of coordinates
    y_pred = (weights.view(h, w) * y_coords.view(-1, 1)).sum()
    x_pred = (weights.view(h, w) * x_coords.view(1, -1)).sum()
    
    return x_pred / w, y_pred / h
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet18 (frozen first 2 layers) |
| Decoder | 2× ConvTranspose (2× upsample) |
| Output | 64×64 heatmap |
| Loss | Modified Focal Loss |
| Optimizer | Adam (lr=1e-3) |
| Schedule | CosineAnnealing |
| Batch Size | 32 |
| Augmentation | Standard (no spatial for center samples) |

## Migration Path

### Phase 1: Heatmap Head (2-3 days)
1. Keep ResNet18 backbone
2. Replace regression head with decoder → heatmap
3. Generate Gaussian targets from existing (x,y) labels
4. Train with focal loss

### Phase 2: Enhanced Labels (1 week)
1. Add bounding box support to labels
2. Generate multi-scale Gaussian targets (small vs large enemies)
3. Add occlusion handling

### Phase 3: Multi-Enemy (1 week)
1. Train with multiple peak support
2. NMS on peak extraction
3. Performance optimization

## Expected Improvements

| Metric | Current (Regression) | Expected (Heatmap) |
|--------|---------------------|-------------------|
| Center Bias | 9.93× | 3-4× (natural resistance) |
| Edge Detection | Poor | Good |
| Multiple Enemies | Not supported | Supported |
| No Enemy Handling | Separate classifier | Natural |
| Training Stability | Medium | High |
| Inference Speed | Fast | Fast (similar) |

## Alternative: Object Detection (YOLO/SSD)

If heatmap seems too custom, consider standard object detection:

**Pros:**
- Mature ecosystem
- Built-in NMS, anchor optimization
- Multiple enemies natural
- Confidence scores built-in

**Cons:**
- Heavier than regression
- Anchor tuning required
- Still needs classification head for "no enemy"

**Recommendation:** Try heatmap first (simpler for single-class), switch to YOLO if you need production-ready multi-class detection.

## Quick Start Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class HeatmapEnemyDetector(nn.Module):
    def __init__(self, backbone='resnet18', heatmap_size=64):
        super().__init__()
        self.heatmap_size = heatmap_size
        
        # Backbone
        if backbone == 'resnet18':
            base = resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(base.children())[:-2])
            encoder_dim = 512
        
        # Decoder (simple upsampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # 1-channel heatmap
        )
    
    def forward(self, x):
        # x: (B, 3, 256, 256)
        features = self.encoder(x)  # (B, 512, 8, 8)
        heatmap = self.decoder(features)  # (B, 1, 64, 64)
        return torch.sigmoid(heatmap)
```

## Summary

**Recommendation: Migrate to heatmap-based prediction.**

It naturally solves:
1. No-enemy handling
2. Multiple enemy support
3. Spatial uncertainty
4. Training stability
5. Positional bias resistance

With minimal architecture change and significant long-term benefits.
"""
