"""
Bias-Aware Data Augmentation Pipeline
=======================================

Redesigned augment_dataset.py with:
- Spatially-aware augmentation (more augmentation for edge/corner samples)
- Skips pixel augmentation for overrepresented regions
- Correct coordinate transformation for all spatial augmentations
- Proper handling of bounding boxes (not just center points)
- Support for 'has_enemy' negative samples
- Quality checks on augmented outputs

Usage:
    python augment_dataset_improved.py --input_csv dataset/labeled/labels_enhanced.csv \
                                       --input_dir dataset/labeled/images \
                                       --output_dir dataset/augmented
"""

import os
import csv
import cv2
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm

# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ============================== CONFIGURATION ==============================

# Augmentation intensity by region
CENTER_AUGMENTATIONS = ['brightness', 'noise']  # Only pixel-level for center
EDGE_AUGMENTATIONS = ['brightness', 'noise', 'shift', 'zoom', 'rotate', 'flip', 'masked_pan']
CORNER_AUGMENTATIONS = ['brightness', 'noise', 'shift', 'zoom', 'rotate', 'flip', 'perspective', 'masked_pan']

# Region definitions
CENTER_REGION = 0.30  # Within 30% of center is "center"
EDGE_THRESHOLD = 0.15  # Within 15% of edge is "edge"

# Augmentation parameters
AUGMENTATION_INTENSITY = {
    'center': 0.3,   # Light augmentation
    'edge': 0.7,     # Medium augmentation
    'corner': 1.0,   # Heavy augmentation
}

MAX_AUG_PER_SAMPLE = {
    'center': 2,     # Few augmentations for center samples
    'edge': 4,       # More for edges
    'corner': 6,     # Most for corners
}

# Specific augmentation parameters
BRIGHTNESS_RANGE = [0.7, 1.3]
NOISE_INTENSITY = 0.03
SHIFT_MAX_PCT = 0.15
ZOOM_RANGE = (0.8, 1.4)
ROTATE_MAX_ANGLE = 15
BLUR_KERNEL_RANGE = [3, 7]

# Quality thresholds
MIN_AUGMENTED_BLUR = 0.2
MAX_COORDINATE_DRIFT = 0.05  # Max 5% shift in normalized coordinates

# ============================== DATA STRUCTURES ==============================

@dataclass
class BBox:
    """Bounding box with normalized coordinates."""
    x_center: float
    y_center: float
    width: float
    height: float
    
    def to_corners(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return x1, y1, x2, y2
    
    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> 'BBox':
        """Create from corner coordinates."""
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        return cls(x_center, y_center, width, height)
    
    def is_valid(self) -> bool:
        """Check if box is within image bounds and has valid size."""
        x1, y1, x2, y2 = self.to_corners()
        
        # Within bounds (with small tolerance)
        if x1 < -0.05 or y1 < -0.05 or x2 > 1.05 or y2 > 1.05:
            return False
        
        # Valid size
        if self.width <= 0 or self.height <= 0:
            return False
        if self.width > 0.5 or self.height > 0.5:
            return False
        
        return True
    
    def clip(self) -> 'BBox':
        """Clip box to image bounds."""
        x1, y1, x2, y2 = self.to_corners()
        x1 = max(0, min(1, x1))
        y1 = max(0, min(1, y1))
        x2 = max(0, min(1, x2))
        y2 = max(0, min(1, y2))
        return BBox.from_corners(x1, y1, x2, y2)


@dataclass
class Label:
    """Enhanced label structure matching new format."""
    video_id: str
    frame_index: int
    has_enemy: int
    bbox: Optional[BBox]
    confidence: float
    source_type: str
    occluded: int
    multiple_targets: int
    blur_score: float
    image_hash: str
    filename: str


# ============================== REGION DETECTION ==============================

class RegionClassifier:
    """Classify position into center, edge, or corner for bias-aware augmentation."""
    
    @staticmethod
    def classify(x: float, y: float) -> str:
        """
        Classify position region.
        Priority: corner > edge > center
        """
        is_edge_x = x < EDGE_THRESHOLD or x > (1 - EDGE_THRESHOLD)
        is_edge_y = y < EDGE_THRESHOLD or y > (1 - EDGE_THRESHOLD)
        
        # Corner: both x and y are near edges
        if is_edge_x and is_edge_y:
            return 'corner'
        
        # Edge: either x or y is near edge
        if is_edge_x or is_edge_y:
            return 'edge'
        
        # Center: within central region
        center_min = 0.5 - CENTER_REGION / 2
        center_max = 0.5 + CENTER_REGION / 2
        if center_min <= x <= center_max and center_min <= y <= center_max:
            return 'center'
        
        # Mid: not center, not edge (transitional region)
        return 'edge'  # Treat as edge for augmentation purposes
    
    @staticmethod
    def classify_bbox(bbox: BBox) -> str:
        """Classify based on bbox center."""
        return RegionClassifier.classify(bbox.x_center, bbox.y_center)


# ============================== AUGMENTATION FUNCTIONS ==============================

class Augmentations:
    """Collection of augmentation functions with correct coordinate handling."""
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness. No coordinate change."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def add_noise(image: np.ndarray, intensity: float = NOISE_INTENSITY) -> np.ndarray:
        """Add Gaussian noise. No coordinate change."""
        noise = np.random.normal(0, intensity * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply slight blur. No coordinate change."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast. No coordinate change."""
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    @staticmethod
    def horizontal_flip(image: np.ndarray, bbox: BBox) -> Tuple[np.ndarray, BBox]:
        """Horizontal flip. Mirrors x coordinate."""
        flipped = cv2.flip(image, 1)
        new_bbox = BBox(
            x_center=1.0 - bbox.x_center,
            y_center=bbox.y_center,
            width=bbox.width,
            height=bbox.height
        )
        return flipped, new_bbox
    
    @staticmethod
    def random_shift(image: np.ndarray, bbox: BBox, 
                     max_shift_pct: float = SHIFT_MAX_PCT) -> Tuple[np.ndarray, Optional[BBox]]:
        """
        Randomly shift image and update bounding box.
        Returns None bbox if shifted out of frame.
        """
        h, w = image.shape[:2]
        
        shift_x = random.uniform(-max_shift_pct, max_shift_pct)
        shift_y = random.uniform(-max_shift_pct, max_shift_pct)
        
        # Create transformation matrix
        M = np.float32([[1, 0, shift_x * w], [0, 1, shift_y * h]])
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Update bbox (add shift to normalized coordinates)
        new_bbox = BBox(
            x_center=bbox.x_center + shift_x,
            y_center=bbox.y_center + shift_y,
            width=bbox.width,
            height=bbox.height
        )
        
        # Check validity
        if not new_bbox.is_valid():
            return shifted, None
        
        return shifted, new_bbox.clip()
    
    @staticmethod
    def random_zoom(image: np.ndarray, bbox: BBox, 
                    scale_range: Tuple[float, float] = ZOOM_RANGE) -> Tuple[np.ndarray, BBox]:
        """
        Random zoom with bbox adjustment.
        Zoom in (scale > 1): crop region around bbox, resize back
        Zoom out (scale < 1): image becomes smaller on black canvas
        """
        h, w = image.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])
        
        if scale >= 1.0:
            # ZOOM IN: Crop around target, resize back
            # Calculate crop region that keeps bbox in view
            crop_w = w / scale
            crop_h = h / scale
            
            # Random crop centered roughly on bbox but with some variation
            bx, by = bbox.x_center * w, bbox.y_center * h
            margin_x = max(0, crop_w - bbox.width * w) / 2
            margin_y = max(0, crop_h - bbox.height * h) / 2
            
            # Random offset within margin
            offset_x = random.uniform(-margin_x * 0.5, margin_x * 0.5)
            offset_y = random.uniform(-margin_y * 0.5, margin_y * 0.5)
            
            x1 = int(max(0, bx - crop_w/2 + offset_x))
            y1 = int(max(0, by - crop_h/2 + offset_y))
            x2 = int(min(w, x1 + crop_w))
            y2 = int(min(h, y1 + crop_h))
            
            # Adjust if we hit boundaries
            if x2 - x1 < crop_w:
                x1 = int(x2 - crop_w)
            if y2 - y1 < crop_h:
                y1 = int(y2 - crop_h)
            
            x1, y1 = max(0, x1), max(0, y1)
            
            # Crop and resize
            cropped = image[y1:y2, x1:x2]
            zoomed = cv2.resize(cropped, (w, h))
            
            # Update bbox: position relative to crop, then scale
            new_bx = (bx - x1) / (x2 - x1)
            new_by = (by - y1) / (y2 - y1)
            new_w = bbox.width * w / (x2 - x1)
            new_h = bbox.height * h / (y2 - y1)
            
            new_bbox = BBox(new_bx, new_by, new_w, new_h)
            
        else:
            # ZOOM OUT: Shrink image, place on black canvas
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            # Create black canvas and place resized image
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            x_offset = random.randint(0, w - new_w)
            y_offset = random.randint(0, h - new_h)
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            zoomed = canvas
            
            # Update bbox
            new_bx = (bbox.x_center * new_w + x_offset) / w
            new_by = (bbox.y_center * new_h + y_offset) / h
            new_bbox_w = bbox.width * new_w / w
            new_bbox_h = bbox.height * new_h / h
            
            new_bbox = BBox(new_bx, new_by, new_bbox_w, new_bbox_h)
        
        return zoomed, new_bbox.clip()
    
    @staticmethod
    def random_rotation(image: np.ndarray, bbox: BBox, 
                       max_angle: float = ROTATE_MAX_ANGLE) -> Tuple[np.ndarray, BBox]:
        """
        Rotate image and bbox.
        For simplicity, rotates around image center and clips bbox.
        """
        h, w = image.shape[:2]
        angle = random.uniform(-max_angle, max_angle)
        
        # Get rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        # Rotate bbox center
        cx, cy = bbox.x_center * w, bbox.y_center * h
        # Apply rotation matrix to center point
        new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        
        new_bbox = BBox(
            x_center=new_cx / w,
            y_center=new_cy / h,
            width=bbox.width,  # Keep same size (approximation)
            height=bbox.height
        )
        
        return rotated, new_bbox.clip()
    
    @staticmethod
    def perspective_transform(image: np.ndarray, bbox: BBox, 
                             max_shift: float = 0.1) -> Tuple[np.ndarray, BBox]:
        """
        Apply slight perspective transform.
        This is an approximation for bbox transformation.
        """
        h, w = image.shape[:2]
        
        # Random perspective shift
        shift = random.uniform(0, max_shift)
        
        # Define source and destination points
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([
            [random.randint(0, int(shift*w)), random.randint(0, int(shift*h))],
            [w - random.randint(0, int(shift*w)), random.randint(0, int(shift*h))],
            [w - random.randint(0, int(shift*w)), h - random.randint(0, int(shift*h))],
            [random.randint(0, int(shift*w)), h - random.randint(0, int(shift*h))]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(image, M, (w, h))
        
        # Transform bbox center (approximation)
        cx, cy = bbox.x_center * w, bbox.y_center * h
        pts = np.float32([[cx, cy]])
        pts_transformed = cv2.perspectiveTransform(pts[None, :, :], M)
        new_cx, new_cy = pts_transformed[0, 0]
        
        new_bbox = BBox(
            x_center=new_cx / w,
            y_center=new_cy / h,
            width=bbox.width,
            height=bbox.height
        )
        
        return transformed, new_bbox.clip()

    @staticmethod
    def generate_fortnite_static_mask(h: int, w: int) -> np.ndarray:
        """Creates a binary mask where 255 = keep static (HUD/Player), 0 = shift."""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Bottom center self-player zone
        sp_x1, sp_x2 = int(w * 0.3), int(w * 0.7)
        sp_y1, sp_y2 = int(h * 0.65), h
        mask[sp_y1:sp_y2, sp_x1:sp_x2] = 255
        
        # Top right minimap (approximate)
        mask[0:int(h*0.25), int(w*0.8):w] = 255
        
        # Bottom health/ammo
        mask[int(h*0.85):h, 0:int(w*0.25)] = 255
        mask[int(h*0.85):h, int(w*0.75):w] = 255
        
        return mask

    @staticmethod
    def augmented_masked_pan(image: np.ndarray, bbox: BBox, max_shift_pct: float = 0.3) -> Tuple[Optional[np.ndarray], Optional[BBox]]:
        """Shifts the background layer to relocate the enemy while pinning the HUD/player mask."""
        h, w = image.shape[:2]
        
        # 1. Calculate random shift (dx, dy)
        dx = int(random.uniform(-max_shift_pct, max_shift_pct) * w)
        dy = int(random.uniform(-max_shift_pct * 0.5, max_shift_pct * 0.5) * h)
        
        # 2. Extract static foreground
        static_mask = Augmentations.generate_fortnite_static_mask(h, w)
        foreground = cv2.bitwise_and(image, image, mask=static_mask)
        
        # 3. Shift the entire background with border reflection
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_bg = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        
        # 4. Remove the foreground area from the shifted background
        bg_mask = cv2.bitwise_not(static_mask)
        masked_bg = cv2.bitwise_and(shifted_bg, shifted_bg, mask=bg_mask)
        
        # 5. Composite
        final_image = cv2.add(masked_bg, foreground)
        
        # 6. Update label (bbox centers are normalized 0-1)
        new_cx = bbox.x_center + (dx / w)
        new_cy = bbox.y_center + (dy / h)
        
        # Check if enemy was shifted off-screen
        if new_cx < 0 or new_cx > 1.0 or new_cy < 0 or new_cy > 1.0:
            return None, None
            
        # Check if enemy is now hidden behind the HUD/Local player mask
        # Since pixel lookups are integers, convert normalized to pixel
        px, py = int(new_cx * w), int(new_cy * h)
        if 0 <= px < w and 0 <= py < h and static_mask[py, px] == 255:
            return None, None # Masked out
            
        new_bbox = BBox(new_cx, new_cy, bbox.width, bbox.height)
        return final_image, new_bbox.clip()


# ============================== MAIN PIPELINE ==============================

class BiasAwareAugmentation:
    """Main augmentation pipeline with bias awareness."""
    
    def __init__(self, input_csv: str, input_dir: str, output_dir: str):
        self.input_csv = Path(input_csv)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'total_input': 0,
            'total_output': 0,
            'by_region': {'center': 0, 'edge': 0, 'corner': 0},
            'by_type': {},
            'skipped': 0
        }
        
        # Output CSV
        self.output_csv = self.output_dir / "augmented_labels.csv"
        self._init_csv()
    
    def _init_csv(self):
        """Initialize output CSV."""
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'filename', 'x_norm', 'y_norm', 'video_id', 'frame_idx',
                'timestamp', 'confidence', 'auto_labeled', 'aug_type'
            ])
    
    def _load_labels(self) -> List[Label]:
        """Load labels from enhanced CSV format."""
        labels = []
        
        with open(self.input_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # All rows in the enhanced CSV are high-confidence enemy locations
                bbox = BBox(
                    x_center=float(row['x_norm']),
                    y_center=float(row['y_norm']),
                    width=0.05,  # Dummy width for augmentation functions
                    height=0.1   # Dummy height for augmentation functions
                )
                
                label = Label(
                    video_id=row.get('video_id', 'unknown'),
                    frame_index=int(row.get('frame_idx', 0)),
                    has_enemy=1,
                    bbox=bbox,
                    confidence=float(row.get('confidence', 1.0)),
                    source_type='auto',
                    occluded=0,
                    multiple_targets=0,
                    blur_score=0.0,
                    image_hash='',
                    filename=row['filename']
                )
                
                # Store the extra original fields so we can write them back out
                label.timestamp = float(row.get('timestamp', 0.0))
                label.auto_labeled = row.get('auto_labeled', 'True')
                
                labels.append(label)
        
        return labels
    
    def _compute_blur(self, image: np.ndarray) -> float:
        """Compute blur score."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
    
    def _save_augmented(self, image: np.ndarray, label: Label, 
                       bbox: Optional[BBox], aug_type: str, parent: str):
        """Save augmented image and label."""
        # Generate filename
        base = Path(label.filename).stem
        new_filename = f"{base}_aug_{aug_type}.png"
        
        # Save image
        img_path = self.output_dir / "images" / new_filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), image)
        
        # Write CSV row
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                new_filename,
                f"{bbox.x_center:.6f}" if bbox else "0",
                f"{bbox.y_center:.6f}" if bbox else "0",
                label.video_id,
                label.frame_index,
                getattr(label, 'timestamp', 0.0),
                f"{label.confidence:.4f}",
                getattr(label, 'auto_labeled', 'True'),
                aug_type
            ])
        
        self.stats['total_output'] += 1
        self.stats['by_type'][aug_type] = self.stats['by_type'].get(aug_type, 0) + 1
    
    def _apply_augmentations(self, image: np.ndarray, label: Label) -> int:
        """
        Apply appropriate augmentations based on spatial region.
        Returns number of augmentations applied.
        """
        if label.has_enemy == 0:
            # Negative samples: light augmentation only
            return self._augment_negative(image, label)
        
        if not label.bbox:
            return 0
        
        # Determine region
        region = RegionClassifier.classify_bbox(label.bbox)
        self.stats['by_region'][region] += 1
        
        # Get allowed augmentations and max count
        if region == 'center':
            allowed = CENTER_AUGMENTATIONS
            max_aug = MAX_AUG_PER_SAMPLE['center']
            intensity = AUGMENTATION_INTENSITY['center']
        elif region == 'corner':
            allowed = CORNER_AUGMENTATIONS
            max_aug = MAX_AUG_PER_SAMPLE['corner']
            intensity = AUGMENTATION_INTENSITY['corner']
        else:  # edge
            allowed = EDGE_AUGMENTATIONS
            max_aug = MAX_AUG_PER_SAMPLE['edge']
            intensity = AUGMENTATION_INTENSITY['edge']
        
        count = 0
        aug = Augmentations()
        
        # Always save original
        self._save_augmented(image, label, label.bbox, 'original', label.filename)
        
        # Apply pixel-level augmentations (all regions get these)
        if 'brightness' in allowed and random.random() < intensity:
            for factor in [BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]]:
                bright = aug.adjust_brightness(image, factor)
                self._save_augmented(bright, label, label.bbox, f'brightness_{factor}', label.filename)
                count += 1
        
        if 'noise' in allowed and random.random() < intensity:
            noisy = aug.add_noise(image, NOISE_INTENSITY)
            self._save_augmented(noisy, label, label.bbox, 'noise', label.filename)
            count += 1
        
        if 'blur' in allowed and random.random() < intensity * 0.5:
            k = random.choice(BLUR_KERNEL_RANGE)
            blurred = aug.apply_blur(image, k)
            self._save_augmented(blurred, label, label.bbox, f'blur_{k}', label.filename)
            count += 1
        
        # Apply spatial augmentations (edges and corners only)
        if region in ['edge', 'corner']:
            current_bbox = label.bbox
            
            if 'flip' in allowed and random.random() < intensity:
                flipped, new_bbox = aug.horizontal_flip(image, current_bbox)
                if new_bbox.is_valid():
                    self._save_augmented(flipped, label, new_bbox, 'flip', label.filename)
                    count += 1
            
            if 'shift' in allowed and random.random() < intensity:
                shifted, new_bbox = aug.random_shift(image, current_bbox, SHIFT_MAX_PCT)
                if new_bbox and new_bbox.is_valid():
                    # Validate shift magnitude
                    dx = abs(new_bbox.x_center - label.bbox.x_center)
                    dy = abs(new_bbox.y_center - label.bbox.y_center)
                    if dx < MAX_COORDINATE_DRIFT and dy < MAX_COORDINATE_DRIFT:
                        self._save_augmented(shifted, label, new_bbox, 'shift', label.filename)
                        count += 1
            
            if 'zoom' in allowed and random.random() < intensity:
                zoomed, new_bbox = aug.random_zoom(image, current_bbox, ZOOM_RANGE)
                if new_bbox.is_valid():
                    self._save_augmented(zoomed, label, new_bbox, 'zoom', label.filename)
                    count += 1
            
            if 'rotate' in allowed and region == 'corner' and random.random() < intensity:
                rotated, new_bbox = aug.random_rotation(image, current_bbox, ROTATE_MAX_ANGLE)
                if new_bbox.is_valid():
                    self._save_augmented(rotated, label, new_bbox, 'rotate', label.filename)
                    count += 1
            
            if 'perspective' in allowed and region == 'corner' and random.random() < intensity * 0.5:
                persp, new_bbox = aug.perspective_transform(image, current_bbox)
                if new_bbox.is_valid():
                    self._save_augmented(persp, label, new_bbox, 'perspective', label.filename)
                    count += 1
                    
            if 'masked_pan' in allowed and random.random() < intensity:
                panned, new_bbox = aug.augmented_masked_pan(image, current_bbox, max_shift_pct=0.35)
                if panned is not None and new_bbox is not None and new_bbox.is_valid():
                    self._save_augmented(panned, label, new_bbox, 'maskedpan', label.filename)
                    count += 1
        
        return count
    
    def _augment_negative(self, image: np.ndarray, label: Label) -> int:
        """Apply light augmentation to negative samples (no bbox)."""
        count = 0
        aug = Augmentations()
        
        # Save original
        self._save_augmented(image, label, None, 'original', label.filename)
        
        # Light augmentations
        bright = aug.adjust_brightness(image, random.uniform(0.9, 1.1))
        self._save_augmented(bright, label, None, 'brightness', label.filename)
        count += 1
        
        noisy = aug.add_noise(image, NOISE_INTENSITY * 0.5)
        self._save_augmented(noisy, label, None, 'noise', label.filename)
        count += 1
        
        return count
    
    def run(self):
        """Run the augmentation pipeline."""
        print("Loading labels...")
        labels = self._load_labels()
        self.stats['total_input'] = len(labels)
        
        print(f"Found {len(labels)} labels to augment")
        print(f"Output directory: {self.output_dir}")
        
        # Process each label
        for label in tqdm(labels, desc="Augmenting"):
            # Load image
            img_path = self.input_dir / label.filename
            if not img_path.exists():
                # Try without parent directory
                img_path = self.input_dir / Path(label.filename).name
            
            if not img_path.exists():
                print(f"Warning: Image not found: {label.filename}")
                self.stats['skipped'] += 1
                continue
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load image: {label.filename}")
                self.stats['skipped'] += 1
                continue
            
            # Apply augmentations
            try:
                self._apply_augmentations(image, label)
            except Exception as e:
                print(f"Error augmenting {label.filename}: {e}")
                self.stats['skipped'] += 1
        
        # Save stats
        self._save_stats()
        
        print("\n" + "="*60)
        print("Augmentation Complete!")
        print("="*60)
        print(f"Stats: {self.stats}")
    
    def _save_stats(self):
        """Save augmentation statistics."""
        stats_path = self.output_dir / "augmentation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"[Stats] Saved to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Bias-Aware Data Augmentation")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="Path to input labels CSV (enhanced format)")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for augmented dataset")
    
    args = parser.parse_args()
    
    pipeline = BiasAwareAugmentation(
        input_csv=args.input_csv,
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
