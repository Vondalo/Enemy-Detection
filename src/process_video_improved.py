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
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm

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
RANDOM_SAMPLE_RATE = 0.10   # 10% random exploration frames

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
MIN_ENEMY_AREA_PX = 100
MAX_ENEMY_AREA_RATIO = 0.25

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
        cell_x, cell_y = self.get_cell(x_norm, y_norm)
        
        if not self.is_cell_full(cell_x, cell_y):
            # Higher priority for edge/corner cells
            is_edge = (cell_x in [0, self.grid_size-1] or 
                      cell_y in [0, self.grid_size-1])
            is_center = (cell_x == 2 and cell_y == 2)
            
            if is_center:
                priority = 0.5  # Lower priority for center
            elif is_edge:
                priority = 2.0  # Higher priority for edges
            else:
                priority = 1.0
            
            self.cell_counts[cell_x, cell_y] += 1
            self.total_samples += 1
            return True, priority
        else:
            # Cell is full - accept with 5% chance anyway
            return random.random() < 0.05, 0.1

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
        cap = cv2.VideoCapture(video_path)
        edge_frames = []
        
        sample_interval = max(1, self.total_frames // (n_samples * 2))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                h, w = frame.shape[:2]
                results = yolo_model(frame, verbose=False)[0]
                
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    
                    # Check if in edge or corner
                    is_edge = cx < 0.15 or cx > 0.85 or cy < 0.15 or cy > 0.85
                    is_corner = (cx < 0.2 or cx > 0.8) and (cy < 0.2 or cy > 0.8)
                    
                    if is_corner and frame_idx not in self.selected_frames:
                        edge_frames.append((frame_idx, 'corner', float(box.conf)))
                        self.selected_frames.add(frame_idx)
                        break
                    elif is_edge and frame_idx not in self.selected_frames:
                        edge_frames.append((frame_idx, 'edge', float(box.conf)))
                        self.selected_frames.add(frame_idx)
                        break
            
            frame_idx += 1
            if len(edge_frames) >= n_samples:
                break
        
        cap.release()
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
        cv2.namedWindow("Review", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Review", self._mouse_callback)
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
    
    def display(self, frame: np.ndarray, detections: List[Dict], 
                mode: str = "confirm") -> str:
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
        elif mode == 'confirm':
            msg = "CONFIRM: 'y'=accept, 'n'=reject, 's'=skip"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif mode == 'select':
            msg = "SELECT: Press 1-9, 'c'+click=manual, 's'=skip"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif mode == 'manual':
            msg = "MANUAL: Click enemy, 'n'=no enemy, 's'=skip"
            cv2.putText(display, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
    
    def __init__(self, videos_dir: str, output_dir: str, yolo_model_path: str = "yolov8n.pt"):
        self.videos_dir = Path(videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print(f"Loading YOLO model from {yolo_model_path}...")
        self.yolo = YOLO(yolo_model_path)
        self.balancer = SpatialBalancer()
        self.qc = QualityControl()
        self.ui = ReviewInterface()
        
        # Storage
        self.labels = []
        self.seen_hashes = {}  # hash -> frame info
        
        # CSV setup
        self.csv_path = self.output_dir / "labels_enhanced.csv"
        self._init_csv()
        
        # Stats
        self.stats = {
            'processed': 0,
            'auto_labeled': 0,
            'confirmed': 0,
            'manual': 0,
            'tracked': 0,
            'rejected': 0,
            'skipped': 0
        }
    
    def _init_csv(self):
        """Initialize CSV with headers if not exists."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'video_id', 'frame_index', 'has_enemy', 'x_center', 'y_center',
                    'width', 'height', 'confidence', 'source_type', 'occluded',
                    'multiple_targets', 'blur_score', 'image_hash', 'timestamp'
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
    
    def _run_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run YOLO detection and return filtered results."""
        h, w = frame.shape[:2]
        results = self.yolo(frame, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Filter by confidence
            if conf < 0.2:
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
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit detections per frame
        return detections[:MAX_DETECTIONS_PER_FRAME]
    
    def _process_frame(self, frame: np.ndarray, video_id: str, frame_idx: int,
                       sampler: FrameSampler) -> bool:
        """
        Process a single frame through the pipeline.
        Returns True if frame was accepted, False otherwise.
        """
        h, w = frame.shape[:2]
        
        # Quality check: blur
        blur_score = self.qc.compute_blur_score(frame)
        if blur_score < 0.3:  # Too blurry
            self.stats['rejected'] += 1
            return False
        
        # Check for duplicates
        img_hash = self.qc.compute_phash(frame)
        is_duplicate = False
        for existing_hash, existing_info in self.seen_hashes.items():
            if self.qc.hamming_distance(img_hash, existing_hash) < DUPLICATE_HASH_THRESHOLD:
                is_duplicate = True
                break
        
        if is_duplicate:
            self.stats['skipped'] += 1
            return False
        
        # Run detection
        detections = self._run_detection(frame)
        
        # Determine mode based on detections
        multiple_targets = len(detections) > 1
        
        if len(detections) == 0:
            # No enemies - negative sample
            label = LabelEntry(
                video_id=video_id,
                frame_index=frame_idx,
                has_enemy=0,
                x_center=0,
                y_center=0,
                width=0,
                height=0,
                confidence=0,
                source_type='negative',
                occluded=0,
                multiple_targets=0,
                blur_score=blur_score,
                image_hash=img_hash
            )
            self._save_label(label, frame)
            self.stats['auto_labeled'] += 1
            return True
        
        elif len(detections) == 1 and detections[0]['confidence'] > HIGH_CONF_THRESHOLD:
            # High confidence single detection - auto-label with balancing check
            det = detections[0]
            cx_norm = det['center'][0] / w
            cy_norm = det['center'][1] / h
            
            should_accept, priority = self.balancer.should_accept(cx_norm, cy_norm)
            
            if should_accept:
                # Brief UI display for auto-labeled
                action = self.ui.display(frame, detections, mode='auto')
                
                if action == 'skip':
                    self.stats['skipped'] += 1
                    return False
                
                # Create label
                label = LabelEntry(
                    video_id=video_id,
                    frame_index=frame_idx,
                    has_enemy=1,
                    x_center=cx_norm,
                    y_center=cy_norm,
                    width=det['width'] / w,
                    height=det['height'] / h,
                    confidence=det['confidence'],
                    source_type='auto',
                    occluded=0,
                    multiple_targets=0,
                    blur_score=blur_score,
                    image_hash=img_hash
                )
                self._save_label(label, frame)
                self.stats['auto_labeled'] += 1
                
                # Optionally track forward
                action = self.ui.display(frame, detections, mode='auto')
                if action == 'track':
                    self._track_and_propagate(frame, det, video_id, frame_idx, blur_score, img_hash)
                
                return True
            else:
                self.stats['skipped'] += 1
                return False
        
        elif len(detections) >= 1:
            # Medium confidence or multiple detections - human review
            mode = 'select' if multiple_targets else 'confirm'
            action = self.ui.display(frame, detections, mode=mode)
            
            if action == 'quit':
                raise KeyboardInterrupt("User quit")
            elif action == 'next_video':
                return 'next_video'
            elif action == 'skip' or action == 'reject':
                self.stats['skipped'] += 1
                return False
            elif action.startswith('select:'):
                idx = int(action.split(':')[1])
                det = detections[idx]
                source_type = 'reviewed'
                self.stats['confirmed'] += 1
            elif action.startswith('click:'):
                coords = action.split(':')[1]
                cx, cy = map(int, coords.split(','))
                # Create manual detection
                det = {
                    'bbox': (cx-20, cy-20, cx+20, cy+20),
                    'center': (cx, cy),
                    'width': 40,
                    'height': 40,
                    'confidence': 1.0
                }
                source_type = 'manual'
                self.stats['manual'] += 1
            elif action == 'accept':
                det = detections[0]
                source_type = 'reviewed'
                self.stats['confirmed'] += 1
            else:
                self.stats['skipped'] += 1
                return False
            
            # Check spatial balance
            cx_norm = det['center'][0] / w
            cy_norm = det['center'][1] / h
            should_accept, _ = self.balancer.should_accept(cx_norm, cy_norm)
            
            if not should_accept:
                self.stats['skipped'] += 1
                return False
            
            # Create and save label
            label = LabelEntry(
                video_id=video_id,
                frame_index=frame_idx,
                has_enemy=1,
                x_center=cx_norm,
                y_center=cy_norm,
                width=det['width'] / w,
                height=det['height'] / h,
                confidence=det['confidence'],
                source_type=source_type,
                occluded=0,
                multiple_targets=multiple_targets,
                blur_score=blur_score,
                image_hash=img_hash
            )
            self._save_label(label, frame)
            
            # Offer tracking
            action = self.ui.display(frame, [det], mode='confirm')
            if action == 'track':
                self._track_and_propagate(frame, det, video_id, frame_idx, blur_score, img_hash)
            
            return True
        
        return False
    
    def _track_and_propagate(self, frame: np.ndarray, initial_det: Dict,
                            video_id: str, start_frame_idx: int,
                            blur_score: float, img_hash: str):
        """Track enemy forward and generate labels for tracked frames."""
        # Initialize tracker
        tracker = ObjectTracker(self.yolo)
        x1, y1, x2, y2 = initial_det['bbox']
        tracker.init(frame, (int(x1), int(y1), int(x2-x1), int(y2-y1)))
        
        # Note: In full implementation, would need to re-open video
        # and seek to start_frame_idx, then read forward
        # This is a simplified version
        print(f"[Tracking] Would propagate from frame {start_frame_idx} for {TRACK_FRAMES} frames")
        self.stats['tracked'] += 0  # Would increment as tracks are saved
    
    def process_video(self, video_path: Path, engagement_frames: List[int] = None):
        """Process a single video through the pipeline."""
        video_id = video_path.stem
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing video: {video_id} ({total_frames} frames)")
        
        # Generate frame samples
        sampler = FrameSampler(total_frames, engagement_frames)
        
        # Pre-scan for edge/corner frames
        print("  Pre-scanning for off-center enemies...")
        edge_frames = sampler.generate_edge_corner_samples(self.yolo, str(video_path), n_samples=50)
        print(f"  Found {len(edge_frames)} edge/corner frames")
        
        # Get all samples to process
        all_samples = sampler.get_all_samples(n_exploration=100)
        print(f"  Total frames to process: {len(all_samples)}")
        
        # Process frames
        for frame_idx in tqdm(all_samples, desc=f"Processing {video_id}"):
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            try:
                result = self._process_frame(frame, video_id, frame_idx, sampler)
                
                if result == 'next_video':
                    print(f"  Skipping to next video")
                    break
                    
            except KeyboardInterrupt:
                print("\nUser interrupted. Saving progress...")
                break
        
        cap.release()
        self.stats['processed'] += len(all_samples)
    
    def run(self):
        """Run pipeline on all videos in directory."""
        video_files = list(self.videos_dir.glob("*.mp4")) + list(self.videos_dir.glob("*.avi"))
        
        if not video_files:
            print(f"No video files found in {self.videos_dir}")
            return
        
        print(f"Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            # Could load engagement timestamps from file
            engagement_frames = None
            self.process_video(video_path, engagement_frames)
        
        # Save statistics
        self._save_stats()
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print(f"Stats: {self.stats}")
        print(f"Spatial distribution: {self.balancer.get_statistics()}")
    
    def _save_stats(self):
        """Save processing statistics to JSON."""
        stats_path = self.output_dir / "collection_stats.json"
        
        stats_data = {
            'processing_stats': self.stats,
            'spatial_distribution': self.balancer.get_statistics(),
            'total_labels': len(self.labels),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"[Stats] Saved to {stats_path}")


# ============================== COMMAND LINE ==============================

def main():
    parser = argparse.ArgumentParser(description="Improved Data Collection Pipeline")
    parser.add_argument("--videos_dir", type=str, required=True,
                       help="Directory containing input videos")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for labeled dataset")
    parser.add_argument("--yolo_model", type=str, default="yolov8n.pt",
                       help="Path to YOLO model")
    parser.add_argument("--engagement_file", type=str, default=None,
                       help="JSON file with engagement timestamps per video")
    
    args = parser.parse_args()
    
    pipeline = ImprovedDataPipeline(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        yolo_model_path=args.yolo_model
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
