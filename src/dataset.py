import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_CLASS_ID = 0
DEFAULT_CLASS_NAME = "enemy"
PLAYER_CLASS_ID = 1
PLAYER_CLASS_NAME = "player"
KNOWN_CLASS_NAMES = {
    DEFAULT_CLASS_ID: DEFAULT_CLASS_NAME,
    PLAYER_CLASS_ID: PLAYER_CLASS_NAME,
}
DEFAULT_BOX_WIDTH = 0.08
DEFAULT_BOX_HEIGHT = 0.18

ANNOTATION_FIELDNAMES = [
    "filename",
    "class_id",
    "class_name",
    "has_enemy",
    "x_center",
    "y_center",
    "width",
    "height",
    "video_id",
    "frame_idx",
    "timestamp",
    "confidence",
    "auto_labeled",
    "bbox_source",
    "aug_type",
]


def _parse_float(value, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _parse_int(value, default: int) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _parse_bool(value, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class DetectionAnnotation:
    filename: str
    class_id: int = DEFAULT_CLASS_ID
    class_name: str = DEFAULT_CLASS_NAME
    has_enemy: int = 1
    x_center: float = 0.5
    y_center: float = 0.5
    width: float = DEFAULT_BOX_WIDTH
    height: float = DEFAULT_BOX_HEIGHT
    video_id: str = "unknown"
    frame_idx: int = 0
    timestamp: float = 0.0
    confidence: float = 1.0
    auto_labeled: bool = True
    bbox_source: str = "measured"
    aug_type: str = ""

    def __post_init__(self):
        self.class_id = int(self.class_id)
        self.has_enemy = int(self.has_enemy)
        self.x_center = float(self.x_center)
        self.y_center = float(self.y_center)
        self.width = float(self.width)
        self.height = float(self.height)
        self.frame_idx = int(self.frame_idx)
        self.timestamp = float(self.timestamp)
        self.confidence = float(self.confidence)
        self.auto_labeled = bool(self.auto_labeled)

    @property
    def x1(self) -> float:
        return self.x_center - self.width / 2

    @property
    def y1(self) -> float:
        return self.y_center - self.height / 2

    @property
    def x2(self) -> float:
        return self.x_center + self.width / 2

    @property
    def y2(self) -> float:
        return self.y_center + self.height / 2

    def clipped(self) -> "DetectionAnnotation":
        if self.has_enemy == 0:
            return self

        x1 = _clamp(self.x1)
        y1 = _clamp(self.y1)
        x2 = _clamp(self.x2)
        y2 = _clamp(self.y2)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        return DetectionAnnotation(
            filename=self.filename,
            class_id=self.class_id,
            class_name=self.class_name,
            has_enemy=self.has_enemy,
            x_center=x1 + width / 2,
            y_center=y1 + height / 2,
            width=width,
            height=height,
            video_id=self.video_id,
            frame_idx=self.frame_idx,
            timestamp=self.timestamp,
            confidence=self.confidence,
            auto_labeled=self.auto_labeled,
            bbox_source=self.bbox_source,
            aug_type=self.aug_type,
        )

    def is_valid(self) -> bool:
        if self.has_enemy == 0:
            return True
        if self.width <= 0 or self.height <= 0:
            return False
        if self.width > 1 or self.height > 1:
            return False
        return self.x1 >= 0 and self.y1 >= 0 and self.x2 <= 1 and self.y2 <= 1

    def to_yolo_line(self) -> str:
        clipped = self.clipped()
        return (
            f"{clipped.class_id} {clipped.x_center:.6f} {clipped.y_center:.6f} "
            f"{clipped.width:.6f} {clipped.height:.6f}"
        )

    def to_csv_row(self) -> Dict[str, str]:
        clipped = self.clipped()
        return {
            "filename": clipped.filename,
            "class_id": str(clipped.class_id),
            "class_name": clipped.class_name,
            "has_enemy": str(clipped.has_enemy),
            "x_center": f"{clipped.x_center:.6f}",
            "y_center": f"{clipped.y_center:.6f}",
            "width": f"{clipped.width:.6f}",
            "height": f"{clipped.height:.6f}",
            "video_id": clipped.video_id,
            "frame_idx": str(clipped.frame_idx),
            "timestamp": f"{clipped.timestamp:.6f}",
            "confidence": f"{clipped.confidence:.4f}",
            "auto_labeled": "True" if clipped.auto_labeled else "False",
            "bbox_source": clipped.bbox_source,
            "aug_type": clipped.aug_type,
        }

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "DetectionAnnotation":
        x_center = row.get("x_center", row.get("x_norm", 0.5))
        y_center = row.get("y_center", row.get("y_norm", 0.5))
        width_value = row.get("width", row.get("bbox_width"))
        height_value = row.get("height", row.get("bbox_height"))
        bbox_source = row.get("bbox_source", "measured")

        if width_value in (None, "") or height_value in (None, ""):
            width_value = DEFAULT_BOX_WIDTH
            height_value = DEFAULT_BOX_HEIGHT
            if row.get("x_norm") not in (None, "") or row.get("y_norm") not in (None, ""):
                bbox_source = "synthetic_from_point"

        annotation = cls(
            filename=row["filename"],
            class_id=_parse_int(row.get("class_id"), DEFAULT_CLASS_ID),
            class_name=row.get("class_name", DEFAULT_CLASS_NAME) or DEFAULT_CLASS_NAME,
            has_enemy=_parse_int(row.get("has_enemy"), 1),
            x_center=_parse_float(x_center, 0.5),
            y_center=_parse_float(y_center, 0.5),
            width=_parse_float(width_value, DEFAULT_BOX_WIDTH),
            height=_parse_float(height_value, DEFAULT_BOX_HEIGHT),
            video_id=row.get("video_id", "unknown") or "unknown",
            frame_idx=_parse_int(row.get("frame_idx"), 0),
            timestamp=_parse_float(row.get("timestamp"), 0.0),
            confidence=_parse_float(row.get("confidence"), 1.0),
            auto_labeled=_parse_bool(row.get("auto_labeled"), True),
            bbox_source=bbox_source,
            aug_type=row.get("aug_type", ""),
        )
        return annotation.clipped()


def annotation_fieldnames() -> List[str]:
    return list(ANNOTATION_FIELDNAMES)


def infer_class_names(annotations: Iterable[DetectionAnnotation]) -> List[str]:
    class_names: Dict[int, str] = dict(KNOWN_CLASS_NAMES)
    max_class_id = DEFAULT_CLASS_ID

    for annotation in annotations:
        class_id = int(annotation.class_id)
        class_name = (annotation.class_name or KNOWN_CLASS_NAMES.get(class_id, f"class_{class_id}")).strip()
        class_names[class_id] = class_name
        max_class_id = max(max_class_id, class_id)

    return [class_names.get(class_id, f"class_{class_id}") for class_id in range(max_class_id + 1)]


def load_annotations(csv_path: str | Path) -> List[DetectionAnnotation]:
    annotations: List[DetectionAnnotation] = []
    with open(csv_path, "r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("filename"):
                continue
            annotation = DetectionAnnotation.from_row(row)
            if annotation.has_enemy == 0 or annotation.is_valid():
                annotations.append(annotation)
    return annotations


def write_annotations_csv(csv_path: str | Path, annotations: Iterable[DetectionAnnotation]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fieldnames())
        writer.writeheader()
        for annotation in annotations:
            writer.writerow(annotation.to_csv_row())


def group_annotations_by_filename(
    annotations: Iterable[DetectionAnnotation],
) -> Dict[str, List[DetectionAnnotation]]:
    grouped: Dict[str, List[DetectionAnnotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.filename, []).append(annotation)
    return grouped


def group_annotations_by_video(
    annotations: Iterable[DetectionAnnotation],
) -> Dict[str, List[DetectionAnnotation]]:
    grouped: Dict[str, List[DetectionAnnotation]] = {}
    for annotation in annotations:
        key = annotation.video_id or "unknown"
        grouped.setdefault(key, []).append(annotation)
    return grouped


def center_ratio(annotations: Iterable[DetectionAnnotation]) -> float:
    items = list(annotations)
    if not items:
        return 0.0
    center_hits = 0
    for annotation in items:
        if 0.4 <= annotation.x_center <= 0.6 and 0.4 <= annotation.y_center <= 0.6:
            center_hits += 1
    return center_hits / len(items)


def split_annotations_by_video(
    annotations: List[DetectionAnnotation],
    val_ratio: float = 0.2,
    seed: int = 42,
    stratified: bool = False,
) -> Tuple[List[DetectionAnnotation], List[DetectionAnnotation], List[str], List[str]]:
    video_groups = group_annotations_by_video(annotations)
    video_ids = list(video_groups.keys())
    rng = random.Random(seed)

    if len(video_ids) <= 1:
        shuffled = list(annotations)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            return shuffled, [], ["mixed"], ["mixed"]
        split_idx = min(len(shuffled) - 1, max(1, int(len(shuffled) * (1 - val_ratio))))
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]
        return train, val, ["mixed"], ["mixed"]

    if stratified:
        video_ids.sort(key=lambda video_id: center_ratio(video_groups[video_id]))
        low_bias = video_ids[::2]
        high_bias = video_ids[1::2]
        rng.shuffle(low_bias)
        rng.shuffle(high_bias)
        ordered_ids = []
        while low_bias or high_bias:
            if low_bias:
                ordered_ids.append(low_bias.pop())
            if high_bias:
                ordered_ids.append(high_bias.pop())
        video_ids = ordered_ids
    else:
        rng.shuffle(video_ids)

    n_val = max(1, int(round(len(video_ids) * val_ratio)))
    if n_val >= len(video_ids):
        n_val = len(video_ids) - 1

    val_ids = set(video_ids[:n_val])
    train_ids = [video_id for video_id in video_ids if video_id not in val_ids]
    val_keys = [video_id for video_id in video_ids if video_id in val_ids]

    train_annotations = [ann for ann in annotations if ann.video_id in train_ids]
    val_annotations = [ann for ann in annotations if ann.video_id in val_ids]

    return train_annotations, val_annotations, train_ids, val_keys


def find_image_path(img_dir: str | Path, filename: str) -> Path | None:
    img_dir = Path(img_dir)
    direct = img_dir / filename
    if direct.exists():
        return direct

    basename = Path(filename).name
    nested = img_dir / basename
    if nested.exists():
        return nested

    matches = list(img_dir.rglob(basename))
    if matches:
        return matches[0]

    return None


def write_yolo_label_file(
    label_path: str | Path,
    annotations: Iterable[DetectionAnnotation],
) -> None:
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [ann.to_yolo_line() for ann in annotations if ann.has_enemy]
    content = "\n".join(lines)
    label_path.write_text(content, encoding="utf-8")


def export_split_dataset(
    annotations: List[DetectionAnnotation],
    src_img_dir: str | Path,
    split_dir: str | Path,
) -> Dict[str, int]:
    split_dir = Path(split_dir)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_annotations_by_filename(annotations)
    copied_images = 0
    exported_annotations: List[DetectionAnnotation] = []
    for filename, image_annotations in grouped.items():
        src_path = find_image_path(src_img_dir, filename)
        if src_path is None:
            continue

        destination = images_dir / Path(filename).name
        shutil.copy2(src_path, destination)
        copied_images += 1

        label_path = labels_dir / f"{destination.stem}.txt"
        normalized = []
        for annotation in image_annotations:
            adjusted = DetectionAnnotation(
                filename=destination.name,
                class_id=annotation.class_id,
                class_name=annotation.class_name,
                has_enemy=annotation.has_enemy,
                x_center=annotation.x_center,
                y_center=annotation.y_center,
                width=annotation.width,
                height=annotation.height,
                video_id=annotation.video_id,
                frame_idx=annotation.frame_idx,
                timestamp=annotation.timestamp,
                confidence=annotation.confidence,
                auto_labeled=annotation.auto_labeled,
                bbox_source=annotation.bbox_source,
                aug_type=annotation.aug_type,
            )
            normalized.append(adjusted)
            exported_annotations.append(adjusted)
        write_yolo_label_file(label_path, normalized)

    write_annotations_csv(split_dir / "labels.csv", exported_annotations)
    return {"images": copied_images, "annotations": len(exported_annotations)}


def write_data_yaml(
    dataset_root: str | Path,
    train_images: str = "train/images",
    val_images: str = "val/images",
    class_names: List[str] | None = None,
) -> Path:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    class_names = class_names or [DEFAULT_CLASS_NAME]
    yaml_path = dataset_root / "data.yaml"
    names = ", ".join(f"'{name}'" for name in class_names)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                f"train: {train_images}",
                f"val: {val_images}",
                f"nc: {len(class_names)}",
                f"names: [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path
