"""
Improved Data Collection Pipeline
==================================

Redesigned process_video.py with:
- Smart frame sampling (engagement windows + random exploration)
- Confidence-based review (auto-label, confirm, manual)
- Multi-object tracking for label propagation
- Grid-based spatial balancing
- Enhanced label format (bbox, confidence, source type)
- Quality control (blur detection, jump detection, duplicates)

Usage:
    python process_video_improved.py --videos_dir src/videos --output dataset/labeled

Controls during review:
    'y' - Accept detection
    'n' - Reject detection
    '1-9' - Select detection number
    'c' + click - Manual correction
    't' - Start tracking (propagate to next frames)
    's' - Skip frame
    'p' - Skip to next video
    ESC - Quit
"""

import cv2
import os
import csv
import json
import random
import hashlib
import logging
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import time

# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Setup logging to stdout for better visibility in Electron
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Force flush all prints
import functools
print = functools.partial(print, flush=True)

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Install ultralytics: pip install ultralytics")

# ============================== CONFIGURATION ==============================

# Confidence thresholds for auto-labeling
HIGH_CONF_THRESHOLD = 0.70  # Auto-label without review
MED_CONF_THRESHOLD = 0.40   # Requires quick confirmation

# Tracking settings
TRACK_FRAMES = 20           # Number of frames to track forward
TRACK_CONF_DECAY = 0.90     # Confidence decay per frame
MAX_TRACK_JUMP_PX = 50      # Maximum allowed position jump between frames

# Frame sampling
ENGAGEMENT_WINDOW = [-30, -20, -10, 0, 10, 20]  # Frame offsets around engagements
RANDOM_SAMPLE_RATE = 0.02   # 2% random exploration frames (reduced for speed)
MIN_EXPLORATION_SAMPLES = 12  # Always inspect at least a handful of frames

# Quality control
BLUR_THRESHOLD = 100        # Laplacian variance threshold
MIN_BBOX_SIZE_PX = 20       # Minimum bounding box dimension
MAX_BBOX_SIZE_RATIO = 0.4   # Max bbox as ratio of image
DUPLICATE_HASH_THRESHOLD = 10  # Hamming distance for pHash

# Spatial balancing
GRID_SIZE = 5               # 5x5 grid for position tracking
MAX_SAMPLES_PER_CELL = 100  # Quota per grid cell
CENTER_CELL_PENALTY = 0.3   # Reduce center cell quota by 70%

# Detection filtering
MAX_DETECTIONS_PER_FRAME = 5
DETECTION_CONF_THRESHOLD = 0.15
MIN_ENEMY_AREA_PX = 100
MAX_ENEMY_AREA_RATIO = 0.25
MAX_BOTTOM_Y_RATIO = 0.7  # Don't accept detections below 70% of screen height

# Self-Player Filtering (Third-Person Games)
SELF_PLAYER_BOTTOM_REGION_PCT = 0.35  # Bottom 35% of screen
SELF_PLAYER_CENTER_ZONE_X = (0.3, 0.7)  # Horizontal range for player zone
SELF_PLAYER_CENTER_ZONE_Y = (0.6, 1.0)  # Vertical range (bottom 40%)
SELF_PLAYER_MAX_SIZE_RATIO = 0.15  # Max size (self player is large)
SELF_PLAYER_MIN_SIZE_RATIO = 0.05  # Min size to consider
SELF_PLAYER_IOU_THRESHOLD = 0.3  # IoU threshold for exclusion
SELF_PLAYER_BOTTOM_EDGE_THRESHOLD = 0.85  # Bottom edge proximity
SELF_PLAYER_TEMPORAL_FRAMES = 10  # Frames to track for temporal filtering

# ============================== DATA STRUCTURES ==============================

class LabelEntry:
    """Enhanced label format with full metadata."""
    def __init__(self, video_id: str, frame_index: int, has_enemy: int = 1,
                 x_center: float = 0, y_center: float = 0,
                 width: float = 0, height: float = 0,
                 confidence: float = 0, source_type: str = "unknown",
                 occluded: int = 0, multiple_targets: int = 0,
                 blur_score: float = 0, image_hash: str = ""):
        self.video_id = video_id
        self.frame_index = frame_index
        self.has_enemy = has_enemy
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.confidence = confidence
        self.source_type = source_type
        self.occluded = occluded
        self.multiple_targets = multiple_targets
        self.blur_score = blur_score
        self.image_hash = image_hash
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            'video_id': self.video_id,
            'frame_index': self.frame_index,
            'has_enemy': self.has_enemy,
            'x_center': f"{self.x_center:.6f}",
            'y_center': f"{self.y_center:.6f}",
            'width': f"{self.width:.6f}",
            'height': f"{self.height:.6f}",
            'confidence': f"{self.confidence:.4f}",
            'source_type': self.source_type,
            'occluded': self.occluded,
            'multiple_targets': self.multiple_targets,
            'blur_score': f"{self.blur_score:.2f}",
            'image_hash': self.image_hash,
            'timestamp': self.timestamp
        }


class SpatialBalancer:
    """Tracks sample distribution across screen regions."""
    def __init__(self, grid_size: int = GRID_SIZE, max_per_cell: int = MAX_SAMPLES_PER_CELL):
        self.grid_size = grid_size
        self.max_per_cell = max_per_cell
        self.cell_counts = np.zeros((grid_size, grid_size), dtype=int)
        self.total_samples = 0

    def get_cell(self, x_norm: float, y_norm: float) -> Tuple[int, int]:
        """Convert normalized coordinates to grid cell."""
        cell_x = min(int(x_norm * self.grid_size), self.grid_size - 1)
        cell_y = min(int(y_norm * self.grid_size), self.grid_size - 1)
        return cell_x, cell_y

    def is_cell_full(self, cell_x: int, cell_y: int) -> bool:
        """Check if cell has reached quota."""
        # Center cell (2,2) has reduced quota
        if cell_x == 2 and cell_y == 2:
            effective_max = int(self.max_per_cell * CENTER_CELL_PENALTY)
        else:
            effective_max = self.max_per_cell
        return self.cell_counts[cell_x, cell_y] >= effective_max

    def should_accept(self, x_norm: float, y_norm: float, 
                      prefer_edges: bool = True) -> Tuple[bool, float]:
        """
        Determine if sample should be accepted based on spatial balance.
        Returns: (accept, priority_score)
        """
        # Bias protection removed: Always accept all samples exactly as they are.
        # Still keep track of distribution for analysis
        cell_x, cell_y = self.get_cell(x_norm, y_norm)
        self.cell_counts[cell_x, cell_y] += 1
        self.total_samples += 1
        return True, 1.0

    def get_statistics(self) -> Dict:
        """Return current distribution statistics."""
        return {
            'grid_counts': self.cell_counts.tolist(),
            'total_samples': self.total_samples,
            'center_20_pct': np.sum(self.cell_counts[1:4, 1:4]) / max(self.total_samples, 1),
            'edge_pct': (np.sum(self.cell_counts[0, :]) + np.sum(self.cell_counts[-1, :]) +
                        np.sum(self.cell_counts[:, 0]) + np.sum(self.cell_counts[:, -1])) / max(self.total_samples, 1)
        }


class QualityControl:
    """Quality checks for labels and images."""
    
    @staticmethod
    def compute_blur_score(image: np.ndarray) -> float:
        """Compute Laplacian variance as blur metric (higher = sharper)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range (empirical scaling)
        normalized = min(laplacian_var / 500.0, 1.0)
        return normalized
    
    @staticmethod
    def compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
        """Compute perceptual hash for duplicate detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        # Convert to hex string
        hash_int = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        return format(hash_int, f'0{hash_size * hash_size // 4}x')
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two perceptual hashes."""
        x = int(hash1, 16) ^ int(hash2, 16)
        return bin(x).count('1')
    
    @staticmethod
    def validate_bbox(x: float, y: float, w: float, h: float) -> bool:
        """Check if bounding box is valid."""
        # All values in 0-1 range
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            return False
        # Box not too small
        if w < 0.01 or h < 0.01:
            return False
        # Box not too large
        if w > MAX_BBOX_SIZE_RATIO or h > MAX_BBOX_SIZE_RATIO:
            return False
        # Box within image bounds
        if x - w/2 < 0 or x + w/2 > 1 or y - h/2 < 0 or y + h/2 > 1:
            return False
        return True
    
    @staticmethod
    def detect_position_jump(prev_pos: Tuple[float, float], 
                            curr_pos: Tuple[float, float],
                            img_width: int, img_height: int) -> bool:
        """Detect unnatural position jump between frames."""
        dx_px = abs(prev_pos[0] - curr_pos[0]) * img_width
        dy_px = abs(prev_pos[1] - curr_pos[1]) * img_height
        distance_px = np.sqrt(dx_px**2 + dy_px**2)
        return distance_px > MAX_TRACK_JUMP_PX


class SelfPlayerFilter:
    """
    Filters out the local player character from detections.
    Critical for third-person games where player avatar is visible.
    """
    
    def __init__(self):
        # Track detections in potential self-player zone for temporal consistency
        self.temporal_detections = []  # List of (frame_idx, bbox, confidence)
        self.self_player_template = None  # Template bbox for persistent self-player
        self.frames_since_seen = 0
    
    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float], 
                    box2: Tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union for two boxes (x1, y1, x2, y2 format)."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_self_player_region(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Define the self-player exclusion region (typically bottom-center of screen).
        Returns: (x1, y1, x2, y2) in pixel coordinates.
        """
        # Bottom region (default 35% of screen)
        y_start = int(img_height * (1 - SELF_PLAYER_BOTTOM_REGION_PCT))
        y_end = img_height
        
        # Center horizontally (default 30-70% of width)
        x_start = int(img_width * SELF_PLAYER_CENTER_ZONE_X[0])
        x_end = int(img_width * SELF_PLAYER_CENTER_ZONE_X[1])
        
        return (x_start, y_start, x_end, y_end)
    
    def is_in_self_player_zone(self, bbox: Tuple[float, float, float, float], 
                               img_width: int, img_height: int) -> Tuple[bool, float]:
        """
        Check if detection is in self-player zone using IoU.
        Returns: (is_in_zone, iou_value)
        """
        sp_x1, sp_y1, sp_x2, sp_y2 = self.get_self_player_region(img_width, img_height)
        
        # Compute IoU with self-player region
        iou = self.compute_iou(
            bbox,
            (sp_x1, sp_y1, sp_x2, sp_y2)
        )
        
        return iou > SELF_PLAYER_IOU_THRESHOLD, iou
    
    def check_geometric_rules(self, bbox: Tuple[float, float, float, float],
                              img_width: int, img_height: int) -> Tuple[bool, str]:
        """
        Apply geometric filtering rules for self-player detection.
        Returns: (is_likely_self_player, reason)
        """
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        area = box_w * box_h
        
        # Normalize dimensions
        norm_w = box_w / img_width
        norm_h = box_h / img_height
        norm_area = area / (img_width * img_height)
        
        # Check 1: Unusually large box (self player is typically large)
        if norm_area > SELF_PLAYER_MAX_SIZE_RATIO:
            return True, f"large_size ({norm_area:.2%})"
        
        # Check 2: Bottom edge proximity (self player often at bottom)
        bottom_edge_proximity = y2 / img_height
        if bottom_edge_proximity > SELF_PLAYER_BOTTOM_EDGE_THRESHOLD:
            # Combined with center position
            center_x = (x1 + x2) / 2 / img_width
            if SELF_PLAYER_CENTER_ZONE_X[0] < center_x < SELF_PLAYER_CENTER_ZONE_X[1]:
                return True, f"bottom_edge ({bottom_edge_proximity:.2%})"
        
        # Check 3: Position in typical player-avatar zone (lower-center)
        center_x = (x1 + x2) / 2 / img_width
        center_y = (y1 + y2) / 2 / img_height
        
        if (SELF_PLAYER_CENTER_ZONE_X[0] < center_x < SELF_PLAYER_CENTER_ZONE_X[1] and
            SELF_PLAYER_CENTER_ZONE_Y[0] < center_y < SELF_PLAYER_CENTER_ZONE_Y[1]):
            # And within size range
            if SELF_PLAYER_MIN_SIZE_RATIO < norm_area < SELF_PLAYER_MAX_SIZE_RATIO:
                return True, f"player_zone ({center_x:.2f}, {center_y:.2f})"
        
        return False, ""
    
    def update_temporal_tracking(self, frame_idx: int, detections: List[Dict],
                                  img_width: int, img_height: int) -> List[Dict]:
        """
        Track detections over time to identify persistent self-player.
        Returns filtered detections.
        """
        # Add current detections to history
        current_dets = []
        for det in detections:
            bbox = det['bbox']
            is_sp, reason = self.check_geometric_rules(bbox, img_width, img_height)
            if is_sp:
                current_dets.append({
                    'frame': frame_idx,
                    'bbox': bbox,
                    'conf': det['confidence'],
                    'reason': reason
                })
        
        self.temporal_detections.extend(current_dets)
        
        # Keep only recent history
        cutoff_frame = frame_idx - SELF_PLAYER_TEMPORAL_FRAMES
        self.temporal_detections = [
            d for d in self.temporal_detections if d['frame'] >= cutoff_frame
        ]
        
        # Find persistent detections (appeared in multiple frames)
        persistent_bboxes = self._find_persistent_detections()
        
        # Filter current detections
        filtered = []
        for det in detections:
            bbox = det['bbox']
            
            # Check if matches persistent self-player
            is_self_player = False
            
            # 1. Check geometric rules
            is_sp_geo, reason_geo = self.check_geometric_rules(bbox, img_width, img_height)
            if is_sp_geo:
                is_self_player = True
                logger.debug(f"  [SelfPlayerFilter] Geometric match: {reason_geo}")
            
            # 2. Check IoU with self-player region
            is_sp_iou, iou_val = self.is_in_self_player_zone(bbox, img_width, img_height)
            if is_sp_iou:
                is_self_player = True
                logger.debug(f"  [SelfPlayerFilter] IoU match: {iou_val:.2f}")
            
            # 3. Check against persistent self-player template
            if persistent_bboxes:
                for pbbox in persistent_bboxes:
                    if self.compute_iou(bbox, pbbox) > 0.5:
                        is_self_player = True
                        logger.debug(f"  [SelfPlayerFilter] Temporal match (persistent)")
                        break
            
            if not is_self_player:
                filtered.append(det)
            else:
                logger.info(f"  [SelfPlayerFilter] Filtered detection (conf: {det['confidence']:.2f})")
        
        return filtered
    
    def _find_persistent_detections(self) -> List[Tuple[float, float, float, float]]:
        """Find detections that persist across multiple frames (likely self-player)."""
        if len(self.temporal_detections) < 3:
            return []
        
        # Group by similar position
        groups = []
        for det in self.temporal_detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Find matching group
            found_group = False
            for group in groups:
                g_cx = (group['center'][0] + group['center'][2]) / 2
                g_cy = (group['center'][1] + group['center'][3]) / 2
                
                # Within 50px distance
                if abs(cx - g_cx) < 50 and abs(cy - g_cy) < 50:
                    group['bboxes'].append(bbox)
                    group['frames'].append(det['frame'])
                    found_group = True
                    break
            
            if not found_group:
                groups.append({
                    'center': bbox,
                    'bboxes': [bbox],
                    'frames': [det['frame']]
                })
        
        # Return groups with 3+ detections (persistent)
        persistent = []
        for group in groups:
            if len(group['bboxes']) >= 3:
                # Average bbox
                avg_bbox = (
                    sum(b[0] for b in group['bboxes']) / len(group['bboxes']),
                    sum(b[1] for b in group['bboxes']) / len(group['bboxes']),
                    sum(b[2] for b in group['bboxes']) / len(group['bboxes']),
                    sum(b[3] for b in group['bboxes']) / len(group['bboxes']),
                )
                persistent.append(avg_bbox)
        
        return persistent
    
    def draw_exclusion_zone(self, frame: np.ndarray, alpha: float = 0.3):
        """Visualize the self-player exclusion zone on frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.get_self_player_region(w, h)
        
        # Create overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add label
        cv2.putText(frame, "SELF-PLAYER EXCLUSION ZONE", (x1 + 10, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame


class ObjectTracker:
    """Simple tracker for label propagation with drift detection."""
    
    def __init__(self, yolo_model):
        self.yolo = yolo_model
        self.last_bbox = None
        self.last_confidence = 1.0
        self.frame_count = 0
        
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initialize tracker with first detection."""
        self.last_bbox = bbox
        self.last_confidence = 1.0
        self.frame_count = 0
        # Could use OpenCV trackers here (KCF, CSRT) for better results
        # For simplicity, using detection-based tracking
        
    def update(self, frame: np.ndarray) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """
        Update tracker on new frame.
        Returns: (bbox, confidence) or None if lost
        """
        self.frame_count += 1
        
        # Decay confidence over time
        self.last_confidence *= TRACK_CONF_DECAY
        
        if self.last_confidence < 0.3:
            return None  # Lost tracking
        
        # Re-detect in region of interest around last position
        if self.last_bbox:
            x, y, w, h = self.last_bbox
            margin = int(max(w, h) * 0.5)
            
            h_img, w_img = frame.shape[:2]
            roi_x1 = max(0, x - margin)
            roi_y1 = max(0, y - margin)
            roi_x2 = min(w_img, x + w + margin)
            roi_y2 = min(h_img, y + h + margin)
            
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi.size > 0:
                results = self.yolo(roi, verbose=False)[0]
                
                if len(results.boxes) > 0:
                    # Get highest confidence detection
                    best_box = None
                    best_conf = 0
                    
                    for box in results.boxes:
                        conf = float(box.conf)
                        if conf > best_conf:
                            best_conf = conf
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            best_box = (int(x1 + roi_x1), int(y1 + roi_y1),
                                       int(x2 - x1), int(y2 - y1))
                    
                    if best_box:
                        # Validate not too far from last position
                        old_center = (self.last_bbox[0] + self.last_bbox[2]/2,
                                   self.last_bbox[1] + self.last_bbox[3]/2)
                        new_center = (best_box[0] + best_box[2]/2,
                                    best_box[1] + best_box[3]/2)
                        
                        distance = np.sqrt((old_center[0] - new_center[0])**2 +
                                         (old_center[1] - new_center[1])**2)
                        
                        if distance < MAX_TRACK_JUMP_PX * 2:
                            self.last_bbox = best_box
                            self.last_confidence = best_conf
                            return best_box, best_conf
        
        return None


# ============================== FRAME SAMPLING ==============================

class FrameSampler:
    """Smart frame sampling for diverse dataset coverage."""
    
    def __init__(self, total_frames: int, engagement_frames: List[int] = None):
        self.total_frames = total_frames
        self.engagement_frames = engagement_frames or []
        self.selected_frames = set()
        self.exploration_frames = set()
        
    def generate_engagement_samples(self) -> List[int]:
        """Generate frames around engagement timestamps."""
        samples = []
        for eng_frame in self.engagement_frames:
            for offset in ENGAGEMENT_WINDOW:
                frame_idx = eng_frame + offset
                if 0 <= frame_idx < self.total_frames:
                    samples.append(frame_idx)
        return samples
    
    def generate_exploration_samples(self, n_samples: int) -> List[int]:
        """Generate random exploration samples avoiding duplicates."""
        samples = []
        attempts = 0
        max_attempts = n_samples * 3
        
        while len(samples) < n_samples and attempts < max_attempts:
            frame_idx = random.randint(0, self.total_frames - 1)
            if frame_idx not in self.selected_frames:
                # Check not too close to existing samples
                too_close = any(abs(frame_idx - s) < 5 for s in self.selected_frames)
                if not too_close:
                    samples.append(frame_idx)
                    self.selected_frames.add(frame_idx)
            attempts += 1
        
        return samples
    
    def generate_edge_corner_samples(self, yolo_model, video_path: str, n_samples: int = 50) -> List[int]:
        """
        Pre-scan video to find off-center enemies.
        Returns frames with detections in edge/corner regions.
        """
        logger.info(f"  [Pre-scan] Scanning for edge/corner frames...")
        cap = cv2.VideoCapture(video_path)
        edge_frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // (n_samples * 2))
        frame_idx = 0
        
        logger.info(f"  [Pre-scan] Total frames: {total_frames}, sampling every {sample_interval} frames")
        
        with tqdm(total=n_samples, desc="  Pre-scanning", unit="found", leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    h, w = frame.shape[:2]
                    results = yolo_model(frame, verbose=False)[0]
                    
                    found_edge_or_corner = False
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                        
                        # Check if in edge or corner
                        is_edge = cx < 0.15 or cx > 0.85 or cy < 0.15 or cy > 0.85
                        is_corner = (cx < 0.2 or cx > 0.8) and (cy < 0.2 or cy > 0.8)
                        
                        if (is_corner or is_edge) and frame_idx not in self.selected_frames:
                            edge_frames.append((frame_idx, 'corner' if is_corner else 'edge', float(box.conf)))
                            self.selected_frames.add(frame_idx)
                            found_edge_or_corner = True
                            pbar.update(1)
                            logger.debug(f"    Found {('corner' if is_corner else 'edge')} detection at frame {frame_idx} (conf: {float(box.conf):.2f})")
                            break
                    
                    if len(edge_frames) >= n_samples:
                        logger.info(f"  [Pre-scan] Reached target of {n_samples} edge/corner frames")
                        break
                
                frame_idx += 1
                
        cap.release()
        logger.info(f"  [Pre-scan] Found {len(edge_frames)} edge/corner frames total")
        return [f[0] for f in edge_frames]
    
    def get_all_samples(self, n_exploration: int = 100) -> List[int]:
        """Combine all sampling strategies."""
        # Engagement samples
        engagement = self.generate_engagement_samples()
        self.selected_frames.update(engagement)
        
        # Random exploration
        exploration = self.generate_exploration_samples(n_exploration)
        
        return sorted(list(self.selected_frames))


# ============================== REVIEW INTERFACE ==============================

class ReviewInterface:
    """Interactive UI for human review of detections."""
    
    def __init__(self, screen_w: int = 1280, screen_h: int = 720):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.click_point = None
        self.drag_start = None
        self.drag_current = None
        self.drawn_box = None
        self.current_zoom = 1.0
        cv2.namedWindow("Review", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Review", self._mouse_callback)

    def reset_interaction(self):
        self.click_point = None
        self.drag_start = None
        self.drag_current = None
        self.drawn_box = None

    def _to_original_coords(self, x: int, y: int) -> Tuple[int, int]:
        zoom = self.current_zoom if self.current_zoom > 0 else 1.0
        return int(x / zoom), int(y / zoom)
        
    def _mouse_callback(self, event, x, y, flags, param):
        ox, oy = self._to_original_coords(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (ox, oy)
            self.drag_current = (ox, oy)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = (ox, oy)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x1 = min(self.drag_start[0], ox)
            y1 = min(self.drag_start[1], oy)
            x2 = max(self.drag_start[0], ox)
            y2 = max(self.drag_start[1], oy)
            if abs(x2 - x1) < 8 or abs(y2 - y1) < 8:
                self.click_point = (ox, oy)
                self.drawn_box = None
            else:
                self.drawn_box = (x1, y1, x2, y2)
                self.click_point = None
            self.drag_start = None
            self.drag_current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.reset_interaction()

    @staticmethod
    def _draw_manual_overlay(display: np.ndarray, click_point, drawn_box,
                             drag_start=None, drag_current=None):
        overlay_color = (255, 0, 255)
        if drawn_box is not None:
            x1, y1, x2, y2 = drawn_box
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), overlay_color, 2)
            cv2.putText(display, "MANUAL BOX", (int(x1), max(20, int(y1) - 6)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)
        elif click_point is not None:
            cx, cy = click_point
            cv2.circle(display, (int(cx), int(cy)), 6, overlay_color, -1)
            cv2.putText(display, "MANUAL POINT", (int(cx) + 8, max(20, int(cy) - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2)
        elif drag_start is not None and drag_current is not None:
            x1 = min(drag_start[0], drag_current[0])
            y1 = min(drag_start[1], drag_current[1])
            x2 = max(drag_start[0], drag_current[0])
            y2 = max(drag_start[1], drag_current[1])
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), overlay_color, 2)

    def label_frame(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Manual labeling loop.

        Controls:
        - Drag left mouse: draw bbox
        - Single left click: place point, press A to accept a default box around it
        - 1-9: select suggested detector bbox
        - A / Enter: accept current manual box/point or selected suggestion
        - N: mark no enemy and skip the frame
        - S: skip frame
        - R or right click: clear manual annotation
        - P: skip remaining frames in current video
        - ESC: quit collection
        """
        self.reset_interaction()

        h, w = frame.shape[:2]
        zoom_w = self.screen_w / w
        zoom_h = self.screen_h / h
        self.current_zoom = min(zoom_w, zoom_h, 1.0)

        selected_idx = 0 if detections else None

        while True:
            display = frame.copy()

            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                color = (0, 255, 255) if i == selected_idx else (0, 255, 0)
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{i+1}: {conf:.2f}"
                cv2.putText(display, label, (int(x1), max(20, int(y1) - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self._draw_manual_overlay(
                display,
                self.click_point,
                self.drawn_box,
                self.drag_start,
                self.drag_current,
            )

            instructions = "Draw box or click enemy. A=save 1-9=suggested N=no-enemy S=skip R=clear ESC=quit"
            cv2.putText(display, instructions, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.current_zoom < 1.0:
                display = cv2.resize(display, (int(w * self.current_zoom), int(h * self.current_zoom)))

            cv2.imshow("Review", display)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            if key in (ord('r'), ord('R')):
                self.reset_interaction()
                continue
            if key in (ord('a'), ord('A'), 13):
                if self.drawn_box is not None:
                    return {'action': 'save_manual_box', 'bbox': self.drawn_box}
                if self.click_point is not None:
                    return {'action': 'save_manual_point', 'point': self.click_point}
                if selected_idx is not None:
                    return {'action': 'accept_detection', 'detection_idx': selected_idx}
                continue
            if key == ord('n') or key == ord('N'):
                return {'action': 'no_enemy'}
            if key == ord('s') or key == ord('S'):
                return {'action': 'skip'}
            if key == ord('p') or key == ord('P'):
                return {'action': 'next_video'}
            if key == 27:
                return {'action': 'quit'}
            if ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(detections):
                    selected_idx = idx

        return {'action': 'skip'}
    
    def display(self, frame: np.ndarray, detections: List[Dict], 
                mode: str = "confirm", show_self_player_zone: bool = False) -> str:
        """
        Display frame with detections and wait for user input.
        
        Modes:
        - 'auto': Brief display, auto-accept
        - 'confirm': Show with prompt, wait for y/n
        - 'select': Multiple detections, select with number keys
        - 'manual': No detections or low conf, wait for click or skip
        
        Returns: Action string ('accept', 'reject', 'skip', '1', '2', etc., or 'click:x,y')
        """
        h, w = frame.shape[:2]
        
        # Calculate zoom factor to fit screen
        zoom_w = self.screen_w / w
        zoom_h = self.screen_h / h
        zoom = min(zoom_w, zoom_h, 1.0)
        
        # Create display copy
        display = frame.copy()
        
        # Draw self-player exclusion zone if requested
        if show_self_player_zone:
            sp_x1 = int(w * SELF_PLAYER_CENTER_ZONE_X[0])
            sp_x2 = int(w * SELF_PLAYER_CENTER_ZONE_X[1])
            sp_y1 = int(h * (1 - SELF_PLAYER_BOTTOM_REGION_PCT))
            sp_y2 = h
            
            overlay = display.copy()
            cv2.rectangle(overlay, (sp_x1, sp_y1), (sp_x2, sp_y2), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.rectangle(display, (sp_x1, sp_y1), (sp_x2, sp_y2), (0, 0, 255), 2)
            cv2.putText(display, "SELF-PLAYER ZONE", (sp_x1 + 5, sp_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw detections
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            conf = det['confidence']
            
            # Color by confidence
            if conf > HIGH_CONF_THRESHOLD:
                color = (0, 255, 0)  # Green
            elif conf > MED_CONF_THRESHOLD:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw center point
            cv2.circle(display, (int(cx), int(cy)), 5, color, -1)
            
            # Label
            label = f"{i+1}: {conf:.2f}"
            cv2.putText(display, label, (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw crosshair
        cv2.line(display, (w//2-15, h//2), (w//2+15, h//2), (255, 0, 0), 2)
        cv2.line(display, (w//2, h//2-15), (w//2, h//2+15), (255, 0, 0), 2)
        
        # Draw mode-specific instructions
        if mode == 'auto':
            msg = "AUTO-LABEL (press 's' to skip, any key to accept)"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if detections:
                logger.info(f"  [UI] Auto-label: {len(detections)} detection(s), press any key to accept or 's' to skip")
        elif mode == 'confirm':
            msg = "CONFIRM: 'y'=accept, 'n'=reject, 's'=skip"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if detections:
                logger.info(f"  [UI] Confirm detection (conf: {detections[0]['confidence']:.2f}) - waiting for input...")
        elif mode == 'select':
            msg = "SELECT: Press 1-9, 'c'+click=manual, 's'=skip"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            logger.info(f"  [UI] Select from {len(detections)} detections (1-{min(9, len(detections))}) or 's' to skip")
        elif mode == 'manual':
            msg = "MANUAL: Click enemy, 'n'=no enemy, 's'=skip"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            logger.info(f"  [UI] Manual mode: Click on enemy or press 's' to skip")
        
        # Grid overlay for balancing visualization
        grid_color = (100, 100, 100)
        for i in range(1, 5):
            x_line = int(w * i / 5)
            y_line = int(h * i / 5)
            cv2.line(display, (x_line, 0), (x_line, h), grid_color, 1)
            cv2.line(display, (0, y_line), (w, y_line), grid_color, 1)
        
        # Resize for display
        if zoom < 1.0:
            display = cv2.resize(display, (int(w*zoom), int(h*zoom)))
        
        cv2.imshow("Review", display)
        
        # Wait for key
        if mode == 'auto':
            key = cv2.waitKey(100) & 0xFF  # Brief pause for auto
        else:
            key = cv2.waitKey(0) & 0xFF
        
        # Reset click point
        clicked = self.click_point
        self.click_point = None
        
        # Map key to action
        if key == ord('y'):
            return 'accept'
        elif key == ord('n'):
            return 'reject'
        elif key == ord('s'):
            return 'skip'
        elif key == ord('p'):
            return 'next_video'
        elif key == 27:  # ESC
            return 'quit'
        elif key >= ord('1') and key <= ord('9'):
            idx = key - ord('1')
            if idx < len(detections):
                return f'select:{idx}'
        elif key == ord('t'):
            return 'track'
        elif clicked and mode in ['select', 'manual']:
            # Scale click back to original coordinates
            orig_x = int(clicked[0] / zoom)
            orig_y = int(clicked[1] / zoom)
            return f'click:{orig_x},{orig_y}'
        
        return 'skip'


# ============================== MAIN PIPELINE ==============================

class ImprovedDataPipeline:
    """Main data collection pipeline integrating all components."""
    
    def __init__(self, videos_dir: str, output_dir: str, yolo_model_path: str = "yolov8n.pt",
                 engagement_file: str = None, auto_skip: bool = False, review_uncertain: bool = False,
                 video_file: str = None):
        self.engagement_file = engagement_file
        self.videos_dir = Path(videos_dir)
        self.video_file = Path(video_file) if video_file else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store flags
        self.auto_skip = auto_skip
        self.review_uncertain = review_uncertain
        self.uncertain_frames = []  # Store frames that were auto-skipped
        self.high_confidence_labels = []  # Auto-accepted frames
        self.confirmed_frames = []  # User-confirmed uncertain frames
        self.stop_requested = False
        self.skip_current_video = False
        
        # Initialize components
        logger.info(f"Loading YOLO model from {yolo_model_path}...")
        self.yolo = YOLO(yolo_model_path)
        self.balancer = SpatialBalancer()
        self.qc = QualityControl()
        self.self_player_filter = SelfPlayerFilter()
        
        # Storage
        self.labels = []
        self.seen_hashes = {}  # hash -> frame info
        
        # Only initialize UI if not in automated mode
        if not self.auto_skip:
            self.ui = ReviewInterface()
            # Live Preview Heatmap
            self.live_preview = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.namedWindow("Live Detection Map", cv2.WINDOW_NORMAL)
        else:
            self.ui = None
            self.live_preview = None
        
        # CSV setup
        self.csv_path = self.output_dir / "labels_enhanced.csv"
        self._init_csv()
        
        # Stats
        self.stats = {
            'processed': 0,
            'auto_labeled': 0,
            'weak_auto_saved': 0,
            'confirmed': 0,
            'manual': 0,
            'tracked': 0,
            'rejected': 0,
            'skipped': 0,
            'self_player_filtered': 0,
            'uncertain': 0
        }
    
    def _init_csv(self):
        """Initialize CSV with headers if not exists."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'filename', 'class_id', 'class_name', 'has_enemy', 'x_center',
                    'y_center', 'width', 'height', 'video_id', 'frame_idx',
                    'timestamp', 'confidence', 'auto_labeled', 'bbox_source', 'aug_type'
                ])
                writer.writeheader()
    
    def _save_label(self, entry: LabelEntry, image: np.ndarray):
        """Save label entry and image."""
        # Save image
        img_name = f"{entry.video_id}_frame_{entry.frame_index:06d}.png"
        img_path = self.output_dir / "images" / img_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), image)
        
        # Save label
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.to_dict().keys())
            writer.writerow(entry.to_dict())
        
        self.labels.append(entry)
        logger.info(f"    [Saved] {img_name} ({entry.source_type}, conf: {entry.confidence:.2f})")

    @staticmethod
    def _normalize_detection(detection: Dict, frame_shape: tuple) -> Dict:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = detection['bbox']
        cx, cy = detection['center']
        return {
            'x_center': cx / w,
            'y_center': cy / h,
            'width': (x2 - x1) / w,
            'height': (y2 - y1) / h,
        }

    def _build_label_row(self, image_name: str, detection: Dict, frame_shape: tuple,
                         video_id: str, frame_idx: int, timestamp: float,
                         auto_labeled: bool, bbox_source: str = 'measured') -> Dict:
        normalized = self._normalize_detection(detection, frame_shape)
        return {
            'filename': image_name,
            'class_id': 0,
            'class_name': 'enemy',
            'has_enemy': 1,
            'x_center': f"{normalized['x_center']:.6f}",
            'y_center': f"{normalized['y_center']:.6f}",
            'width': f"{normalized['width']:.6f}",
            'height': f"{normalized['height']:.6f}",
            'video_id': video_id,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'confidence': detection['confidence'],
            'auto_labeled': auto_labeled,
            'bbox_source': bbox_source,
            'aug_type': '',
        }

    def _write_yolo_label(self, image_name: str, detection: Dict, frame_shape: tuple):
        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        normalized = self._normalize_detection(detection, frame_shape)
        label_path.write_text(
            f"0 {normalized['x_center']:.6f} {normalized['y_center']:.6f} "
            f"{normalized['width']:.6f} {normalized['height']:.6f}\n",
            encoding='utf-8'
        )

    @staticmethod
    def _bbox_to_detection(bbox: Tuple[int, int, int, int], confidence: float = 1.0) -> Dict:
        x1, y1, x2, y2 = bbox
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        return {
            'bbox': (float(x1), float(y1), float(x2), float(y2)),
            'center': (float(x1) + width / 2, float(y1) + height / 2),
            'width': width,
            'height': height,
            'confidence': float(confidence),
            'class': 0,
        }

    @staticmethod
    def _point_to_bbox(point: Tuple[int, int], frame_shape: tuple) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        default_w = max(24, int(w * 0.08))
        default_h = max(36, int(h * 0.18))
        cx, cy = point
        x1 = max(0, int(cx - default_w / 2))
        y1 = max(0, int(cy - default_h / 2))
        x2 = min(w - 1, int(cx + default_w / 2))
        y2 = min(h - 1, int(cy + default_h / 2))
        return x1, y1, x2, y2

    def _run_detection(self, frame: np.ndarray, frame_idx: int = 0) -> List[Dict]:
        """Run YOLO detection and return filtered results (excluding self-player)."""
        h, w = frame.shape[:2]
        results = self.yolo(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Filter by confidence
            if conf < DETECTION_CONF_THRESHOLD:
                continue
            
            # Filter by size
            box_w = x2 - x1
            box_h = y2 - y1
            area = box_w * box_h
            
            if area < MIN_ENEMY_AREA_PX:
                continue
            if area > MAX_ENEMY_AREA_RATIO * w * h:
                continue
            
            # Filter by position (don't accept detections too low on screen)
            if (y1 + y2) / 2 > MAX_BOTTOM_Y_RATIO * h:
                continue
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'width': box_w,
                'height': box_h,
                'confidence': conf,
                'class': int(box.cls) if hasattr(box, 'cls') else 0
            })
        
        # Apply self-player filtering
        initial_count = len(detections)
        detections = self.self_player_filter.update_temporal_tracking(
            frame_idx, detections, w, h
        )
        filtered_count = initial_count - len(detections)
        self.stats['self_player_filtered'] += filtered_count
        
        if filtered_count > 0:
            logger.debug(f"  Filtered {filtered_count} self-player detection(s)")
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit detections per frame
        return detections[:MAX_DETECTIONS_PER_FRAME]
    
    def _evaluate_confidence(self, detections: List[Dict], frame_shape: tuple) -> Dict:
        """Evaluate confidence tier for detections."""
        if not detections:
            return {'tier': 'UNCERTAIN', 'reason': 'no_detections'}
        
        # Get best detection
        best_det = max(detections, key=lambda x: x['confidence'])
        conf = best_det['confidence']
        
        # Check for self-player risk
        sp_risk = self._check_self_player_risk(best_det, frame_shape)
        
        # Size/position heuristics
        size_score = self._evaluate_size_position(best_det, frame_shape)
        
        # Combined confidence score
        combined_score = conf * size_score * (1 - sp_risk)
        
        if combined_score >= 0.7 and sp_risk < 0.3:
            return {
                'tier': 'HIGH',
                'detection': best_det,
                'score': combined_score,
                'self_player_risk': sp_risk
            }
        elif combined_score >= 0.3:
            return {
                'tier': 'UNCERTAIN',
                'detection': best_det,
                'score': combined_score,
                'self_player_risk': sp_risk,
                'reason': f'low_confidence_{sp_risk:.2f}'
            }
        else:
            return {'tier': 'INVALID', 'reason': 'very_low_confidence'}
    
    def _check_self_player_risk(self, detection: Dict, frame_shape: tuple) -> float:
        """Check if detection might be self-player."""
        h, w = frame_shape[:2]
        cx, cy = detection['center']
        
        # Risk factors
        risk = 0.0
        
        # Center zone risk
        if SELF_PLAYER_CENTER_ZONE_X[0] < cx/w < SELF_PLAYER_CENTER_ZONE_X[1]:
            risk += 0.3
        
        # Bottom region risk
        if cy/h > (1 - SELF_PLAYER_BOTTOM_REGION_PCT):
            risk += 0.4
        
        # Size risk (self-player often larger)
        area = detection['width'] * detection['height']
        if area > MAX_ENEMY_AREA_RATIO * w * h * 0.8:
            risk += 0.3
        
        return min(risk, 1.0)
    
    def _evaluate_size_position(self, detection: Dict, frame_shape: tuple) -> float:
        """Evaluate detection based on size and position heuristics."""
        h, w = frame_shape[:2]
        cx, cy = detection['center']
        area = detection['width'] * detection['height']
        
        score = 1.0
        
        # Penalize too small
        if area < MIN_ENEMY_AREA_PX * 2:
            score *= 0.7
        
        # Penalize too low on screen
        if cy/h > MAX_BOTTOM_Y_RATIO:
            score *= 0.6
        
        # Reward good center position (not too centered to avoid self-player)
        center_dist = abs(cx/w - 0.5)
        if 0.2 < center_dist < 0.4:
            score *= 1.1
        
        return min(score, 1.0)
    
    def _queue_uncertain_frame(self, frame: np.ndarray, confidence_data: Dict, 
                              video_id: str, frame_idx: int, timestamp: float):
        """Save uncertain frame to review queue."""
        # Create preview with overlays
        preview = self._create_preview_image(frame, confidence_data)
        
        # Save preview
        preview_path = self.output_dir / "previews" / f"uncertain_{video_id}_{frame_idx:06d}.png"
        preview_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(preview_path), preview)
        
        # Add to queue
        queue_item = {
            'video_id': video_id,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'detections': confidence_data.get('detections', []),
            'best_detection': confidence_data.get('detection', {}),
            'confidence_score': confidence_data.get('score', 0.0),
            'uncertainty_reason': confidence_data.get('reason', 'unknown'),
            'preview_path': str(preview_path),
            'frame_shape': frame.shape[:2],
            'self_player_risk': confidence_data.get('self_player_risk', 0.0)
        }
        
        self.uncertain_frames.append(queue_item)
        logger.debug(f"  [Queue] Uncertain frame queued: {video_id}_{frame_idx:06d} ({confidence_data.get('reason', 'unknown')})")
    
    def _create_preview_image(self, frame: np.ndarray, confidence_data: Dict) -> np.ndarray:
        """Create preview image with detection overlays."""
        preview = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw detections
        detections = confidence_data.get('detections', [])
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Color by confidence
            if conf > HIGH_CONF_THRESHOLD:
                color = (0, 255, 0)  # Green
            elif conf > MED_CONF_THRESHOLD:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(preview, f"{i+1}: {conf:.2f}", (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add uncertainty reason
        reason = confidence_data.get('reason', 'Unknown')
        cv2.putText(preview, f"Reason: {reason}", (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return preview
    
    def _save_high_confidence(self, frame: np.ndarray, confidence_data: Dict,
                             video_id: str, frame_idx: int, timestamp: float,
                             auto_labeled: bool = True, bbox_source: str = 'measured',
                             stats_key: Optional[str] = None):
        """Save high-confidence frame immediately."""
        detection = confidence_data['detection']
        cx, cy = detection['center']
        
        # Save image
        image_name = f"{video_id}_{frame_idx:06d}.png"
        img_path = self.output_dir / "images" / image_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), frame)
        self._write_yolo_label(image_name, detection, frame.shape)
        
        # Add to labels
        label = self._build_label_row(
            image_name=image_name,
            detection=detection,
            frame_shape=frame.shape,
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp=timestamp,
            auto_labeled=auto_labeled,
            bbox_source=bbox_source,
        )
        
        self.high_confidence_labels.append(label)
        resolved_stats_key = stats_key
        if resolved_stats_key is None:
            resolved_stats_key = 'auto_labeled' if auto_labeled else 'weak_auto_saved'
        if resolved_stats_key:
            self.stats[resolved_stats_key] = self.stats.get(resolved_stats_key, 0) + 1
        
        # Append to CSV dynamically
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(label.keys()))
            writer.writerow(label)
            
        # Update live preview window mask if enabled
        if self.live_preview is not None:
            vx = int((cx / frame.shape[1]) * 1280)
            vy = int((cy / frame.shape[0]) * 720)
            cv2.circle(self.live_preview, (vx, vy), 3, (0, 0, 255), -1)
            cv2.imshow("Live Detection Map", self.live_preview)
            cv2.waitKey(1)
        
        logger.info(f"  [Auto] Saved high confidence frame: {img_path.name} (conf: {detection['confidence']:.2f})")
    
    def _prompt_final_decision(self) -> str:
        """Ask user whether to review uncertain frames."""
        print("\n" + "="*60)
        print("Pipeline Processing Complete")
        print("="*60)
        print(f"High-confidence frames processed: {len(self.high_confidence_labels)}")
        print(f"Low-confidence frames queued: {len(self.uncertain_frames)}")
        print("\nOptions:")
        print("1) Review low-confidence frames (y)")
        print("2) Keep only high-confidence dataset and exit (n)")
        print("\nYour choice: ", end="")
        
        while True:
            choice = input().lower().strip()
            if choice in ['y', 'yes', '1']:
                return 'REVIEW'
            elif choice in ['n', 'no', '2']:
                return 'SKIP'
            else:
                print("Please enter 'y' to review or 'n' to skip: ", end="")
    
    def _review_uncertain_frames(self):
        """Interactive review of queued uncertain frames."""
        logger.info(f"Starting review of {len(self.uncertain_frames)} uncertain frames...")
        
        for i, frame_data in enumerate(self.uncertain_frames):
            # Load preview image
            preview = cv2.imread(frame_data['preview_path'])
            if preview is None:
                logger.warning(f"Could not load preview: {frame_data['preview_path']}")
                continue
            
            # Show with uncertainty info
            action = self.ui.display_uncertain_frame(preview, frame_data, i+1, len(self.uncertain_frames))
            
            if action == 'accept':
                self.confirmed_frames.append(frame_data)
                logger.info(f"  [Review] Accepted uncertain frame: {frame_data['video_id']}_{frame_data['frame_idx']:06d}")
            elif action == 'correct':
                corrected = self.ui.manual_correction(preview, frame_data)
                if corrected:
                    self.confirmed_frames.append(corrected)
                    logger.info(f"  [Review] Corrected frame: {frame_data['video_id']}_{frame_data['frame_idx']:06d}")
            # 'reject' does nothing
        
        logger.info(f"Review complete. Confirmed {len(self.confirmed_frames)} frames.")
    
    def _process_confirmed_frames(self):
        """Process user-confirmed uncertain frames."""
        for frame_data in self.confirmed_frames:
            # Load original frame (need to reconstruct path)
            video_path = self.videos_dir / f"{frame_data['video_id']}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_data['frame_idx'])
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                continue
            
            # Save with corrected label
            detection = frame_data.get('best_detection', {})
            if detection:
                self._save_high_confidence(
                    frame,
                    {'detection': detection, 'score': 1.0},
                    frame_data['video_id'],
                    frame_data['frame_idx'],
                    frame_data['timestamp'],
                    auto_labeled=False,
                    bbox_source='manual_review',
                    stats_key='confirmed',
                )
    
    def _process_high_confidence_batch(self):
        """Process all high-confidence frames through augmentation."""
        if not self.high_confidence_labels:
            logger.info("No high-confidence frames to process")
            return
        
        logger.info(f"Processing {len(self.high_confidence_labels)} high-confidence frames...")
        
        # Save labels CSV
        labels_path = self.output_dir / "labels_enhanced.csv"
        with open(labels_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filename', 'class_id', 'class_name', 'has_enemy', 'x_center',
                'y_center', 'width', 'height', 'video_id', 'frame_idx',
                'timestamp', 'confidence', 'auto_labeled', 'bbox_source', 'aug_type'
            ])
            writer.writeheader()
            writer.writerows(self.high_confidence_labels)
        
        # Run augmentation if requested
        logger.info("Running augmentation on high-confidence dataset...")
        # TODO: Integrate with augment_dataset_improved.py
    
    def _export_dataset(self):
        """Final dataset export and cleanup."""
        logger.info("Exporting final dataset...")
        
        # Confirmed frames are persisted during _process_confirmed_frames,
        # so the in-memory list already contains the complete dataset.
        all_labels = self.high_confidence_labels.copy()
        
        # Save final labels
        labels_path = self.output_dir / "labels_enhanced.csv"
        with open(labels_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filename', 'class_id', 'class_name', 'has_enemy', 'x_center',
                'y_center', 'width', 'height', 'video_id', 'frame_idx',
                'timestamp', 'confidence', 'auto_labeled', 'bbox_source', 'aug_type'
            ])
            writer.writeheader()
            writer.writerows(all_labels)
        
        logger.info(f"Dataset exported: {len(all_labels)} total labels")
    
    def _cleanup(self):
        """Clean up temporary files."""
        # Remove preview directory
        preview_dir = self.output_dir / "previews"
        if preview_dir.exists():
            import shutil
            shutil.rmtree(preview_dir)
            logger.debug("Cleaned up preview directory")

    def _process_frame(self, frame: np.ndarray, video_id: str, frame_idx: int, timestamp: float) -> str:
        """Manual-first frame review for bbox collection."""
        detections = self._run_detection(frame, frame_idx)
        action = self.ui.label_frame(frame, detections)

        if action['action'] == 'accept_detection':
            det_idx = action['detection_idx']
            detection = detections[det_idx]
            self._save_high_confidence(
                frame,
                {'detection': detection, 'score': detection['confidence']},
                video_id,
                frame_idx,
                timestamp,
                auto_labeled=False,
                bbox_source='manual_pick',
                stats_key='manual',
            )
            return 'saved'

        if action['action'] == 'save_manual_box':
            detection = self._bbox_to_detection(action['bbox'])
            self._save_high_confidence(
                frame,
                {'detection': detection, 'score': 1.0},
                video_id,
                frame_idx,
                timestamp,
                auto_labeled=False,
                bbox_source='manual_box',
                stats_key='manual',
            )
            return 'saved'

        if action['action'] == 'save_manual_point':
            bbox = self._point_to_bbox(action['point'], frame.shape)
            detection = self._bbox_to_detection(bbox)
            self._save_high_confidence(
                frame,
                {'detection': detection, 'score': 1.0},
                video_id,
                frame_idx,
                timestamp,
                auto_labeled=False,
                bbox_source='manual_point',
                stats_key='manual',
            )
            return 'saved'

        if action['action'] == 'no_enemy':
            self.stats['skipped'] += 1
            logger.info(f"  [Manual] Marked no enemy for frame {video_id}_{frame_idx:06d}")
            return 'skip'

        if action['action'] == 'skip':
            self.stats['skipped'] += 1
            return 'skip'

        if action['action'] == 'next_video':
            self.skip_current_video = True
            return 'next_video'

        if action['action'] == 'quit':
            self.stop_requested = True
            return 'quit'

        self.stats['skipped'] += 1
        return 'skip'
    
    def _process_frame_automated(self, frame: np.ndarray, video_id: str, frame_idx: int,
                                timestamp: float) -> bool:
        """Process frame in automated mode without user interaction."""
        # Run detection and filtering
        detections = self._run_detection(frame, frame_idx)
        
        # Determine confidence tier
        confidence_data = self._evaluate_confidence(detections, frame.shape)
        
        if confidence_data['tier'] == 'HIGH':
            # Auto-accept and save immediately
            self._save_high_confidence(frame, confidence_data, video_id, frame_idx, timestamp)
            return True
        elif confidence_data['tier'] == 'UNCERTAIN':
            detection = confidence_data.get('detection')
            if detection:
                # In unattended mode, keep medium-confidence proposals instead of
                # ending with an empty dataset. They stay marked as weak auto labels.
                self._save_high_confidence(
                    frame,
                    confidence_data,
                    video_id,
                    frame_idx,
                    timestamp,
                    auto_labeled=False,
                    bbox_source='weak_auto',
                    stats_key='weak_auto_saved',
                )
                self.stats['uncertain'] += 1
                return True
            # Queue only if something upstream classified it as uncertain without a box.
            self._queue_uncertain_frame(frame, confidence_data, video_id, frame_idx, timestamp)
            self.stats['uncertain'] += 1
            return False
        # 'INVALID' frames are silently discarded
        self.stats['rejected'] += 1
        return False
    
    def process_video(self, video_path: Path) -> Dict:
        """Process entire video in automated mode."""
        video_id = video_path.stem
        logger.info(f"Processing video: {video_id}")
        logger.info("=" * 60)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video info: {total_frames} frames @ {fps:.1f} fps")
        
        # Generate sample frames
        engagement_frames = []
        if self.engagement_file and Path(self.engagement_file).exists():
            with open(self.engagement_file, 'r') as f:
                data = json.load(f)
                engagement_frames = data.get(video_id, [])
                
        if self.auto_skip:
            sampler = FrameSampler(total_frames, engagement_frames)
            n_exploration = min(
                total_frames,
                max(MIN_EXPLORATION_SAMPLES, int(total_frames * RANDOM_SAMPLE_RATE))
            ) if total_frames > 0 else 0
            sample_frames = sampler.get_all_samples(n_exploration=n_exploration)
        else:
            sample_frames = list(range(total_frames))
        
        logger.info(f"Total frames to process: {len(sample_frames)}")
        
        # Create output directory
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Process frames
        saved_count = 0
        sample_set = set(sample_frames)
        max_frame = max(sample_set) if sample_set else 0
        self.skip_current_video = False
        
        frame_idx = 0
        pbar = tqdm(total=len(sample_set), desc=f"{video_id}")
        
        while cap.isOpened() and frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in sample_set:
                timestamp = frame_idx / fps
                
                if self.auto_skip:
                    # Automated mode - no user interaction
                    if self._process_frame_automated(frame, video_id, frame_idx, timestamp):
                        saved_count += 1
                else:
                    outcome = self._process_frame(frame, video_id, frame_idx, timestamp)
                    if outcome == 'saved':
                        saved_count += 1
                    elif outcome == 'next_video':
                        break
                    elif outcome == 'quit':
                        break
                        
                pbar.update(1)
                
            frame_idx += 1

            if self.stop_requested or self.skip_current_video:
                break
            
        pbar.close()
        
        cap.release()
        
        stats = {
            'video_id': video_id,
            'total_frames': total_frames,
            'processed_frames': len(sample_frames),
            'saved_frames': saved_count,
            'stats': dict(self.stats)
        }
        
        logger.info(f"Completed {video_id}: {saved_count}/{len(sample_frames)} frames saved")
        return stats
    
    def run(self):
        """Run the full pipeline on all videos."""
        start_time = time.time()
        
        # Find video files
        if self.video_file:
            video_files = [self.video_file]
            if not self.video_file.exists():
                logger.error(f"Video file not found: {self.video_file}")
                return
        else:
            video_files = sorted(self.videos_dir.glob("*.mp4"))
            if not video_files:
                logger.error(f"No video files found in {self.videos_dir}")
                return
        
        logger.info(f"Processing {len(video_files)} video(s)")
        
        # Process all videos
        all_stats = []
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n[Video {i}/{len(video_files)}] Starting {video_path.name}")
            stats = self.process_video(video_path)
            all_stats.append(stats)
            if self.stop_requested:
                break
        
        # Final processing
        if self.auto_skip:
            self._process_high_confidence_batch()
            
            if self.review_uncertain and self.uncertain_frames:
                choice = self._prompt_final_decision()
                
                if choice == 'REVIEW':
                    self._review_uncertain_frames()
                    self._process_confirmed_frames()
            
            self._export_dataset()
            self._cleanup()
        
        # Print summary
        elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total processing time: {elapsed/60:.1f} minutes")
        logger.info(f"Videos processed: {len(video_files)}")
        logger.info(f"High-confidence frames: {len(self.high_confidence_labels)}")
        logger.info(f"Uncertain frames: {len(self.uncertain_frames)}")
        logger.info(f"Confirmed frames: {len(self.confirmed_frames)}")
        
        if self.stats:
            logger.info("\nFrame statistics:")
            for key, val in self.stats.items():
                logger.info(f"  {key}: {val}")
        
        logger.info("="*60)


# ============================== COMMAND LINE ==============================

def main():
    print("Initializing Data Collection Script...", flush=True)
    parser = argparse.ArgumentParser(description="Improved Data Collection Pipeline")
    parser.add_argument("--videos_dir", type=str, required=True,
                       help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for labeled dataset")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                       help="Path to YOLO model")
    parser.add_argument("--engagement_file", type=str, default=None,
                       help="JSON file with engagement timestamps per video")
    parser.add_argument("--video_file", type=str, default=None,
                       help="Specific video file to process (overrides --videos_dir)")
    parser.add_argument("--auto_skip", action="store_true",
                       help="Auto-skip low/medium confidence frames, only save high confidence")
    parser.add_argument("--review_uncertain", action="store_true",
                       help="At end, review frames that were auto-skipped for confirmation")
    
    args = parser.parse_args()
    
    pipeline = ImprovedDataPipeline(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        yolo_model_path=args.yolo_model,
        engagement_file=args.engagement_file,
        auto_skip=args.auto_skip,
        review_uncertain=args.review_uncertain,
        video_file=args.video_file
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
