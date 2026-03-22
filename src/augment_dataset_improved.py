"""
Bias-aware data augmentation for YOLO-style bounding-box datasets.

This version operates at the image level instead of the CSV-row level so that
multi-box frames stay consistent across:
- augmented images
- YOLO label files
- CSV exports used by the rest of the pipeline
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    DEFAULT_CLASS_ID,
    DEFAULT_CLASS_NAME,
    PLAYER_CLASS_ID,
    DetectionAnnotation,
    find_image_path,
    group_annotations_by_filename,
    load_annotations,
    write_annotations_csv,
    write_yolo_label_file,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
print = functools.partial(print, flush=True)


# ============================== CONFIGURATION ==============================

CENTER_AUGMENTATIONS = ["brightness", "noise"]
EDGE_AUGMENTATIONS = ["brightness", "noise", "shift", "zoom", "rotate", "flip", "masked_pan"]
CORNER_AUGMENTATIONS = ["brightness", "noise", "shift", "zoom", "rotate", "flip", "perspective", "masked_pan"]

CENTER_REGION = 0.30
EDGE_THRESHOLD = 0.15

AUGMENTATION_INTENSITY = {
    "center": 0.3,
    "edge": 0.7,
    "corner": 1.0,
}

MAX_AUG_PER_SAMPLE = {
    "center": 2,
    "edge": 4,
    "corner": 6,
}

BRIGHTNESS_RANGE = [0.7, 1.3]
NOISE_INTENSITY = 0.03
SHIFT_MAX_PCT = 0.15
ZOOM_RANGE = (0.8, 1.4)
ROTATE_MAX_ANGLE = 15
BLUR_KERNEL_RANGE = [3, 7]
MIN_BOX_SIZE = 0.01


# ============================== DATA STRUCTURES ==============================


@dataclass
class BBox:
    x_center: float
    y_center: float
    width: float
    height: float

    def to_corners(self) -> Tuple[float, float, float, float]:
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return x1, y1, x2, y2

    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> "BBox":
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        return cls(x_center, y_center, width, height)

    @classmethod
    def from_annotation(cls, annotation: DetectionAnnotation) -> "BBox":
        return cls(
            x_center=annotation.x_center,
            y_center=annotation.y_center,
            width=annotation.width,
            height=annotation.height,
        )

    def clipped(self) -> Optional["BBox"]:
        x1, y1, x2, y2 = self.to_corners()
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        clipped = BBox.from_corners(x1, y1, x2, y2)
        if clipped.width < MIN_BOX_SIZE or clipped.height < MIN_BOX_SIZE:
            return None
        return clipped

    def is_valid(self) -> bool:
        x1, y1, x2, y2 = self.to_corners()
        if self.width <= 0 or self.height <= 0:
            return False
        if self.width > 1 or self.height > 1:
            return False
        return x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1


# ============================== REGION DETECTION ==============================


class RegionClassifier:
    @staticmethod
    def classify(x: float, y: float) -> str:
        is_edge_x = x < EDGE_THRESHOLD or x > (1 - EDGE_THRESHOLD)
        is_edge_y = y < EDGE_THRESHOLD or y > (1 - EDGE_THRESHOLD)

        if is_edge_x and is_edge_y:
            return "corner"
        if is_edge_x or is_edge_y:
            return "edge"

        center_min = 0.5 - CENTER_REGION / 2
        center_max = 0.5 + CENTER_REGION / 2
        if center_min <= x <= center_max and center_min <= y <= center_max:
            return "center"

        return "edge"

    @staticmethod
    def classify_bbox(bbox: BBox) -> str:
        return RegionClassifier.classify(bbox.x_center, bbox.y_center)


# ============================== BOX HELPERS ==============================


def choose_primary_annotation(annotations: Sequence[DetectionAnnotation]) -> Optional[DetectionAnnotation]:
    positive = [ann for ann in annotations if ann.has_enemy]
    if not positive:
        return None

    enemy_boxes = [ann for ann in positive if ann.class_id != PLAYER_CLASS_ID]
    pool = enemy_boxes or positive
    return max(pool, key=lambda ann: ann.width * ann.height)


def clone_with_box(
    annotation: DetectionAnnotation,
    bbox: Optional[BBox],
    *,
    filename: Optional[str] = None,
    aug_type: Optional[str] = None,
    has_enemy: Optional[int] = None,
) -> DetectionAnnotation:
    if bbox is None:
        x_center = 0.5
        y_center = 0.5
        width = 0.0
        height = 0.0
    else:
        clipped = bbox.clipped()
        if clipped is None:
            clipped = bbox
        x_center = clipped.x_center
        y_center = clipped.y_center
        width = clipped.width
        height = clipped.height

    return DetectionAnnotation(
        filename=filename or annotation.filename,
        class_id=annotation.class_id,
        class_name=annotation.class_name,
        has_enemy=annotation.has_enemy if has_enemy is None else has_enemy,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        video_id=annotation.video_id,
        frame_idx=annotation.frame_idx,
        timestamp=annotation.timestamp,
        confidence=annotation.confidence,
        auto_labeled=annotation.auto_labeled,
        bbox_source=annotation.bbox_source,
        aug_type=annotation.aug_type if aug_type is None else aug_type,
    )


def make_negative_annotation(
    source_annotations: Sequence[DetectionAnnotation],
    *,
    filename: str,
    aug_type: str,
) -> DetectionAnnotation:
    source = source_annotations[0]
    return DetectionAnnotation(
        filename=filename,
        class_id=DEFAULT_CLASS_ID,
        class_name=DEFAULT_CLASS_NAME,
        has_enemy=0,
        x_center=0.5,
        y_center=0.5,
        width=0.0,
        height=0.0,
        video_id=source.video_id,
        frame_idx=source.frame_idx,
        timestamp=source.timestamp,
        confidence=0.0,
        auto_labeled=source.auto_labeled,
        bbox_source="negative_augmented",
        aug_type=aug_type,
    )


# ============================== AUGMENTATION FUNCTIONS ==============================


class Augmentations:
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def add_noise(image: np.ndarray, intensity: float = NOISE_INTENSITY) -> np.ndarray:
        noise = np.random.normal(0, intensity * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def horizontal_flip(
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
    ) -> Tuple[np.ndarray, List[Optional[BBox]]]:
        flipped = cv2.flip(image, 1)
        boxes = []
        for annotation in annotations:
            if not annotation.has_enemy:
                boxes.append(None)
                continue
            bbox = BBox.from_annotation(annotation)
            boxes.append(
                BBox(
                    x_center=1.0 - bbox.x_center,
                    y_center=bbox.y_center,
                    width=bbox.width,
                    height=bbox.height,
                ).clipped()
            )
        return flipped, boxes

    @staticmethod
    def random_shift(
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
        max_shift_pct: float = SHIFT_MAX_PCT,
    ) -> Tuple[np.ndarray, List[Optional[BBox]]]:
        h, w = image.shape[:2]
        shift_x = random.uniform(-max_shift_pct, max_shift_pct)
        shift_y = random.uniform(-max_shift_pct, max_shift_pct)

        matrix = np.float32([[1, 0, shift_x * w], [0, 1, shift_y * h]])
        shifted = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        boxes: List[Optional[BBox]] = []
        for annotation in annotations:
            if not annotation.has_enemy:
                boxes.append(None)
                continue
            bbox = BBox.from_annotation(annotation)
            boxes.append(
                BBox(
                    x_center=bbox.x_center + shift_x,
                    y_center=bbox.y_center + shift_y,
                    width=bbox.width,
                    height=bbox.height,
                ).clipped()
            )

        return shifted, boxes

    @staticmethod
    def random_zoom(
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
        focus_annotation: DetectionAnnotation,
        scale_range: Tuple[float, float] = ZOOM_RANGE,
    ) -> Tuple[np.ndarray, List[Optional[BBox]]]:
        h, w = image.shape[:2]
        focus_bbox = BBox.from_annotation(focus_annotation)
        scale = random.uniform(scale_range[0], scale_range[1])

        transformed_boxes: List[Optional[BBox]] = []

        if scale >= 1.0:
            crop_w = w / scale
            crop_h = h / scale

            bx = focus_bbox.x_center * w
            by = focus_bbox.y_center * h
            margin_x = max(0.0, crop_w - focus_bbox.width * w) / 2
            margin_y = max(0.0, crop_h - focus_bbox.height * h) / 2

            offset_x = random.uniform(-margin_x * 0.5, margin_x * 0.5)
            offset_y = random.uniform(-margin_y * 0.5, margin_y * 0.5)

            x1 = int(max(0, bx - crop_w / 2 + offset_x))
            y1 = int(max(0, by - crop_h / 2 + offset_y))
            x2 = int(min(w, x1 + crop_w))
            y2 = int(min(h, y1 + crop_h))

            if x2 - x1 < crop_w:
                x1 = int(max(0, x2 - crop_w))
            if y2 - y1 < crop_h:
                y1 = int(max(0, y2 - crop_h))

            cropped = image[y1:y2, x1:x2]
            zoomed = cv2.resize(cropped, (w, h))
            crop_width = max(1, x2 - x1)
            crop_height = max(1, y2 - y1)

            for annotation in annotations:
                if not annotation.has_enemy:
                    transformed_boxes.append(None)
                    continue
                bbox = BBox.from_annotation(annotation)
                x_a1, y_a1, x_a2, y_a2 = bbox.to_corners()
                x_a1 = (x_a1 * w - x1) / crop_width
                y_a1 = (y_a1 * h - y1) / crop_height
                x_a2 = (x_a2 * w - x1) / crop_width
                y_a2 = (y_a2 * h - y1) / crop_height
                transformed_boxes.append(BBox.from_corners(x_a1, y_a1, x_a2, y_a2).clipped())
        else:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            x_offset = random.randint(0, w - new_w)
            y_offset = random.randint(0, h - new_h)
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            zoomed = canvas

            for annotation in annotations:
                if not annotation.has_enemy:
                    transformed_boxes.append(None)
                    continue
                bbox = BBox.from_annotation(annotation)
                x1_norm, y1_norm, x2_norm, y2_norm = bbox.to_corners()
                px1 = x1_norm * new_w + x_offset
                py1 = y1_norm * new_h + y_offset
                px2 = x2_norm * new_w + x_offset
                py2 = y2_norm * new_h + y_offset
                transformed_boxes.append(BBox.from_corners(px1 / w, py1 / h, px2 / w, py2 / h).clipped())

        return zoomed, transformed_boxes

    @staticmethod
    def random_rotation(
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
        max_angle: float = ROTATE_MAX_ANGLE,
    ) -> Tuple[np.ndarray, List[Optional[BBox]]]:
        h, w = image.shape[:2]
        angle = random.uniform(-max_angle, max_angle)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        boxes: List[Optional[BBox]] = []
        for annotation in annotations:
            if not annotation.has_enemy:
                boxes.append(None)
                continue
            bbox = BBox.from_annotation(annotation)
            cx = bbox.x_center * w
            cy = bbox.y_center * h
            new_cx = matrix[0, 0] * cx + matrix[0, 1] * cy + matrix[0, 2]
            new_cy = matrix[1, 0] * cx + matrix[1, 1] * cy + matrix[1, 2]
            boxes.append(
                BBox(
                    x_center=new_cx / w,
                    y_center=new_cy / h,
                    width=bbox.width,
                    height=bbox.height,
                ).clipped()
            )

        return rotated, boxes

    @staticmethod
    def perspective_transform(
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
        max_shift: float = 0.1,
    ) -> Tuple[np.ndarray, List[Optional[BBox]]]:
        h, w = image.shape[:2]
        shift = random.uniform(0, max_shift)
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        pts2 = np.float32([
            [random.randint(0, int(shift * w)), random.randint(0, int(shift * h))],
            [w - random.randint(0, int(shift * w)), random.randint(0, int(shift * h))],
            [w - random.randint(0, int(shift * w)), h - random.randint(0, int(shift * h))],
            [random.randint(0, int(shift * w)), h - random.randint(0, int(shift * h))],
        ])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(image, matrix, (w, h))

        boxes: List[Optional[BBox]] = []
        for annotation in annotations:
            if not annotation.has_enemy:
                boxes.append(None)
                continue
            bbox = BBox.from_annotation(annotation)
            pts = np.float32([[[bbox.x_center * w, bbox.y_center * h]]])
            pts_transformed = cv2.perspectiveTransform(pts, matrix)
            new_cx, new_cy = pts_transformed[0, 0]
            boxes.append(
                BBox(
                    x_center=new_cx / w,
                    y_center=new_cy / h,
                    width=bbox.width,
                    height=bbox.height,
                ).clipped()
            )

        return transformed, boxes

    @staticmethod
    def generate_fortnite_static_mask(h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.65):h, int(w * 0.3):int(w * 0.7)] = 255
        mask[0:int(h * 0.25), int(w * 0.8):w] = 255
        mask[int(h * 0.85):h, 0:int(w * 0.25)] = 255
        mask[int(h * 0.85):h, int(w * 0.75):w] = 255
        return mask

    @staticmethod
    def augmented_masked_pan(
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
        max_shift_pct: float = 0.3,
    ) -> Tuple[Optional[np.ndarray], Optional[List[Optional[BBox]]]]:
        h, w = image.shape[:2]
        dx = int(random.uniform(-max_shift_pct, max_shift_pct) * w)
        dy = int(random.uniform(-max_shift_pct * 0.5, max_shift_pct * 0.5) * h)

        static_mask = Augmentations.generate_fortnite_static_mask(h, w)
        foreground = cv2.bitwise_and(image, image, mask=static_mask)

        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_bg = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        bg_mask = cv2.bitwise_not(static_mask)
        masked_bg = cv2.bitwise_and(shifted_bg, shifted_bg, mask=bg_mask)
        final_image = cv2.add(masked_bg, foreground)

        boxes: List[Optional[BBox]] = []
        for annotation in annotations:
            if not annotation.has_enemy:
                boxes.append(None)
                continue

            bbox = BBox.from_annotation(annotation)
            if annotation.class_id == PLAYER_CLASS_ID:
                boxes.append(bbox.clipped())
                continue

            new_cx = bbox.x_center + (dx / w)
            new_cy = bbox.y_center + (dy / h)
            if not (0.0 <= new_cx <= 1.0 and 0.0 <= new_cy <= 1.0):
                boxes.append(None)
                continue

            px = int(new_cx * w)
            py = int(new_cy * h)
            if 0 <= px < w and 0 <= py < h and static_mask[py, px] == 255:
                boxes.append(None)
                continue

            boxes.append(BBox(new_cx, new_cy, bbox.width, bbox.height).clipped())

        return final_image, boxes


# ============================== MAIN PIPELINE ==============================


class BiasAwareAugmentation:
    def __init__(self, input_csv: str, input_dir: str, output_dir: str):
        self.input_csv = Path(input_csv)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.output_csv = self.output_dir / "augmented_labels.csv"
        self.output_annotations: List[DetectionAnnotation] = []

        self.stats = {
            "total_input": 0,
            "total_input_annotations": 0,
            "total_output": 0,
            "total_output_annotations": 0,
            "by_region": {"center": 0, "edge": 0, "corner": 0, "negative": 0},
            "by_type": {},
            "skipped": 0,
        }

    def _make_augmented_filename(self, source_filename: str, aug_type: str) -> str:
        source_path = Path(source_filename)
        digest = hashlib.sha1(source_filename.encode("utf-8")).hexdigest()[:8]
        return f"{source_path.stem}_{digest}_aug_{aug_type}.png"

    def _normalize_annotations_for_output(
        self,
        source_annotations: Sequence[DetectionAnnotation],
        new_filename: str,
        aug_type: str,
        transformed_boxes: Sequence[Optional[BBox]],
    ) -> List[DetectionAnnotation]:
        normalized: List[DetectionAnnotation] = []
        for annotation, bbox in zip(source_annotations, transformed_boxes):
            if not annotation.has_enemy:
                normalized.append(
                    clone_with_box(
                        annotation,
                        None,
                        filename=new_filename,
                        aug_type=aug_type,
                        has_enemy=0,
                    )
                )
                continue

            if bbox is None:
                continue

            clipped = bbox.clipped()
            if clipped is None:
                continue

            normalized.append(
                clone_with_box(
                    annotation,
                    clipped,
                    filename=new_filename,
                    aug_type=aug_type,
                )
            )

        positives = [annotation for annotation in normalized if annotation.has_enemy]
        if positives:
            return positives

        return [make_negative_annotation(source_annotations, filename=new_filename, aug_type=aug_type)]

    def _save_augmented(
        self,
        image: np.ndarray,
        source_annotations: Sequence[DetectionAnnotation],
        transformed_boxes: Sequence[Optional[BBox]],
        aug_type: str,
    ) -> None:
        new_filename = self._make_augmented_filename(source_annotations[0].filename, aug_type)

        img_path = self.output_dir / "images" / new_filename
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), image)

        normalized_annotations = self._normalize_annotations_for_output(
            source_annotations,
            new_filename,
            aug_type,
            transformed_boxes,
        )

        label_path = self.output_dir / "labels" / f"{Path(new_filename).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        write_yolo_label_file(label_path, normalized_annotations)

        self.output_annotations.extend(normalized_annotations)
        self.stats["total_output"] += 1
        self.stats["total_output_annotations"] += len(normalized_annotations)
        self.stats["by_type"][aug_type] = self.stats["by_type"].get(aug_type, 0) + 1

    def _apply_augmentations(
        self,
        image: np.ndarray,
        annotations: Sequence[DetectionAnnotation],
    ) -> None:
        primary = choose_primary_annotation(annotations)
        region = "negative"
        if primary is not None:
            region = RegionClassifier.classify_bbox(BBox.from_annotation(primary))
        self.stats["by_region"][region] = self.stats["by_region"].get(region, 0) + 1

        aug = Augmentations()
        identity_boxes = [
            BBox.from_annotation(annotation) if annotation.has_enemy else None
            for annotation in annotations
        ]
        self._save_augmented(image, annotations, identity_boxes, "original")

        if primary is None:
            bright = aug.adjust_brightness(image, random.uniform(0.9, 1.1))
            self._save_augmented(bright, annotations, identity_boxes, "brightness")

            noisy = aug.add_noise(image, NOISE_INTENSITY * 0.5)
            self._save_augmented(noisy, annotations, identity_boxes, "noise")
            return

        allowed = {
            "center": CENTER_AUGMENTATIONS,
            "edge": EDGE_AUGMENTATIONS,
            "corner": CORNER_AUGMENTATIONS,
        }.get(region, CENTER_AUGMENTATIONS)
        intensity = AUGMENTATION_INTENSITY.get(region, AUGMENTATION_INTENSITY["center"])
        max_aug = MAX_AUG_PER_SAMPLE.get(region, MAX_AUG_PER_SAMPLE["center"])
        applied = 0

        if "brightness" in allowed and random.random() < intensity and applied < max_aug:
            for factor in BRIGHTNESS_RANGE:
                self._save_augmented(
                    aug.adjust_brightness(image, factor),
                    annotations,
                    identity_boxes,
                    f"brightness_{factor}",
                )
                applied += 1
                if applied >= max_aug:
                    break

        if "noise" in allowed and random.random() < intensity and applied < max_aug:
            self._save_augmented(aug.add_noise(image, NOISE_INTENSITY), annotations, identity_boxes, "noise")
            applied += 1

        if "blur" in allowed and random.random() < intensity * 0.5 and applied < max_aug:
            kernel = random.choice(BLUR_KERNEL_RANGE)
            self._save_augmented(aug.apply_blur(image, kernel), annotations, identity_boxes, f"blur_{kernel}")
            applied += 1

        if region in {"edge", "corner"} and applied < max_aug:
            if "flip" in allowed and random.random() < intensity and applied < max_aug:
                flipped, new_boxes = aug.horizontal_flip(image, annotations)
                self._save_augmented(flipped, annotations, new_boxes, "flip")
                applied += 1

            if "shift" in allowed and random.random() < intensity and applied < max_aug:
                shifted, new_boxes = aug.random_shift(image, annotations, SHIFT_MAX_PCT)
                self._save_augmented(shifted, annotations, new_boxes, "shift")
                applied += 1

            if "zoom" in allowed and random.random() < intensity and applied < max_aug:
                zoomed, new_boxes = aug.random_zoom(image, annotations, primary, ZOOM_RANGE)
                self._save_augmented(zoomed, annotations, new_boxes, "zoom")
                applied += 1

            if "rotate" in allowed and region == "corner" and random.random() < intensity and applied < max_aug:
                rotated, new_boxes = aug.random_rotation(image, annotations, ROTATE_MAX_ANGLE)
                self._save_augmented(rotated, annotations, new_boxes, "rotate")
                applied += 1

            if "perspective" in allowed and region == "corner" and random.random() < intensity * 0.5 and applied < max_aug:
                persp, new_boxes = aug.perspective_transform(image, annotations)
                self._save_augmented(persp, annotations, new_boxes, "perspective")
                applied += 1

            if "masked_pan" in allowed and random.random() < intensity and applied < max_aug:
                panned, new_boxes = aug.augmented_masked_pan(image, annotations, max_shift_pct=0.35)
                if panned is not None and new_boxes is not None:
                    self._save_augmented(panned, annotations, new_boxes, "maskedpan")

    def run(self) -> None:
        print("Loading labels...")
        annotations = load_annotations(self.input_csv)
        grouped = group_annotations_by_filename(annotations)

        self.stats["total_input"] = len(grouped)
        self.stats["total_input_annotations"] = len(annotations)

        print(f"Found {len(grouped)} images and {len(annotations)} annotations to augment")
        print(f"Output directory: {self.output_dir}")

        for filename, image_annotations in tqdm(grouped.items(), desc="Augmenting"):
            img_path = find_image_path(self.input_dir, filename)
            if img_path is None:
                print(f"Warning: Image not found: {filename}")
                self.stats["skipped"] += 1
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load image: {filename}")
                self.stats["skipped"] += 1
                continue

            try:
                self._apply_augmentations(image, image_annotations)
            except Exception as exc:
                print(f"Error augmenting {filename}: {exc}")
                self.stats["skipped"] += 1

        write_annotations_csv(self.output_csv, self.output_annotations)
        self._save_stats()

        print("\n" + "=" * 60)
        print("Augmentation Complete!")
        print("=" * 60)
        print(f"Stats: {self.stats}")

    def _save_stats(self) -> None:
        stats_path = self.output_dir / "augmentation_stats.json"
        with open(stats_path, "w", encoding="utf-8") as handle:
            json.dump(self.stats, handle, indent=2)
        print(f"[Stats] Saved to {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias-aware YOLO dataset augmentation")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the bbox annotation CSV")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for augmented dataset")
    args = parser.parse_args()

    pipeline = BiasAwareAugmentation(
        input_csv=args.input_csv,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
