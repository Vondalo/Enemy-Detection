import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.dataset import (
    export_split_dataset,
    group_annotations_by_filename,
    infer_class_names,
    load_annotations,
    split_annotations_by_video,
    write_data_yaml,
)
from src.model import DEFAULT_MODEL_CHOICE, format_model_choices, resolve_model_source

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REVIEW_IOU_THRESHOLD = 0.5
REVIEW_CONFIDENCE_THRESHOLD = 0.25
REVIEW_MAX_DETECTIONS = 100


def _count_box_sources(annotations):
    summary = {}
    for annotation in annotations:
        summary[annotation.bbox_source] = summary.get(annotation.bbox_source, 0) + 1
    return summary


def _normalize_class_name(raw_name: str | int) -> tuple[str, str]:
    text = str(raw_name).strip()
    normalized = text.lower()

    if normalized in {"enemy", "enemies", "opponent", "opponents", "target"}:
        return "enemy", "Enemy"
    if normalized in {"player", "players", "friendly", "friend", "teammate", "ally"}:
        return "player", "Player"

    if not text:
        return "unknown", "Unknown"

    return normalized, text.title()


def _resolve_result_names(raw_names) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(raw_names)}
    return {}


def _box_confidence(box) -> float:
    confidence = getattr(box, "conf", None)
    if confidence is None:
        return 0.0
    if hasattr(confidence, "numel") and confidence.numel() > 0:
        return float(confidence.flatten()[0].item())
    return float(confidence)


def _box_class_id(box) -> int:
    class_id = getattr(box, "cls", None)
    if class_id is None:
        return 0
    if hasattr(class_id, "numel") and class_id.numel() > 0:
        return int(class_id.flatten()[0].item())
    return int(class_id)


def _as_float(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return value


def _clamp_ratio(value: float, name: str) -> float:
    if value <= 0 or value >= 1:
        raise ValueError(f"{name} must be between 0 and 1 (exclusive), received {value}.")
    return value


def _resolve_test_split(args) -> float:
    test_split = _clamp_ratio(args.test_split, "--test_split")
    if args.val_split is None:
        return test_split

    legacy_split = _clamp_ratio(args.val_split, "--val_split")
    if abs(legacy_split - test_split) > 1e-9:
        raise ValueError("--test_split and --val_split were both provided with different values.")
    return legacy_split


def _annotation_to_detection(annotation, image_width: int, image_height: int) -> dict:
    x1 = max(0.0, min(float(image_width), annotation.x1 * image_width))
    y1 = max(0.0, min(float(image_height), annotation.y1 * image_height))
    x2 = max(0.0, min(float(image_width), annotation.x2 * image_width))
    y2 = max(0.0, min(float(image_height), annotation.y2 * image_height))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    class_key, class_name = _normalize_class_name(annotation.class_name)
    return {
        "class_id": int(annotation.class_id),
        "class_key": class_key,
        "class_name": class_name,
        "confidence": float(annotation.confidence),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xyxy_normalized": [
            x1 / image_width if image_width else 0.0,
            y1 / image_height if image_height else 0.0,
            x2 / image_width if image_width else 0.0,
            y2 / image_height if image_height else 0.0,
        ],
        "x_center": float(annotation.x_center),
        "y_center": float(annotation.y_center),
        "width": float(annotation.width),
        "height": float(annotation.height),
        "source": "ground_truth",
    }


def _prediction_to_detection(box, names: dict[int, str], image_width: int, image_height: int) -> dict:
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    class_id = _box_class_id(box)
    raw_class_name = names.get(class_id, str(class_id))
    class_key, class_name = _normalize_class_name(raw_class_name)
    return {
        "class_id": class_id,
        "class_key": class_key,
        "class_name": class_name,
        "model_class_name": str(raw_class_name),
        "confidence": _box_confidence(box),
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_xyxy_normalized": [
            x1 / image_width if image_width else 0.0,
            y1 / image_height if image_height else 0.0,
            x2 / image_width if image_width else 0.0,
            y2 / image_height if image_height else 0.0,
        ],
        "x_center": (x1 + width / 2) / image_width if image_width else 0.0,
        "y_center": (y1 + height / 2) / image_height if image_height else 0.0,
        "width": width / image_width if image_width else 0.0,
        "height": height / image_height if image_height else 0.0,
        "source": "prediction",
    }


def _compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _match_detections(ground_truth: list[dict], predictions: list[dict], iou_threshold: float) -> tuple[list[dict], list[dict], list[dict]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt_item in enumerate(ground_truth):
        for pred_index, pred_item in enumerate(predictions):
            if int(gt_item["class_id"]) != int(pred_item["class_id"]):
                continue
            iou = _compute_iou(gt_item["bbox_xyxy"], pred_item["bbox_xyxy"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gt = set()
    used_pred = set()
    matches: list[dict] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append({
            "ground_truth_index": gt_index,
            "prediction_index": pred_index,
            "iou": iou,
        })

    missed = [ground_truth[index] for index in range(len(ground_truth)) if index not in used_gt]
    false_positives = [predictions[index] for index in range(len(predictions)) if index not in used_pred]
    return matches, missed, false_positives


def _render_review_image(image_path: Path, review_path: Path, ground_truth: list[dict], predictions: list[dict]) -> str | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    def draw_items(items: list[dict], color: tuple[int, int, int], prefix: str, include_conf: bool) -> None:
        for item in items:
            x1, y1, x2, y2 = [int(round(value)) for value in item["bbox_xyxy"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{prefix} {item['class_name']}"
            if include_conf:
                label = f"{label} {item['confidence']:.2f}"
            text_origin = (x1, max(18, y1 - 8))
            cv2.putText(
                image,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    draw_items(ground_truth, (0, 220, 120), "GT", False)
    draw_items(predictions, (255, 80, 80), "Pred", True)

    review_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_path), image)
    return str(review_path.resolve())


def _write_review_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "status",
        "image_path",
        "review_image_path",
        "ground_truth_count",
        "prediction_count",
        "matched_count",
        "missed_ground_truth_count",
        "false_positive_count",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_eval_metrics(results) -> dict:
    metrics = {
        "precision": None,
        "recall": None,
        "map50": None,
        "map75": None,
        "map50_95": None,
        "fitness": None,
        "per_class": [],
        "results_dict": {},
    }

    box_metrics = getattr(results, "box", None)
    if box_metrics is not None:
        metrics["precision"] = _as_float(getattr(box_metrics, "mp", None))
        metrics["recall"] = _as_float(getattr(box_metrics, "mr", None))
        metrics["map50"] = _as_float(getattr(box_metrics, "map50", None))
        metrics["map75"] = _as_float(getattr(box_metrics, "map75", None))
        metrics["map50_95"] = _as_float(getattr(box_metrics, "map", None))

    metrics["fitness"] = _as_float(getattr(results, "fitness", None))
    raw_results_dict = getattr(results, "results_dict", {}) or {}
    metrics["results_dict"] = {
        str(key): _as_float(value) if _as_float(value) is not None else value
        for key, value in raw_results_dict.items()
    }

    try:
        metrics["per_class"] = results.summary(normalize=True, decimals=6)
    except Exception:
        metrics["per_class"] = []

    return metrics


def _format_metric_for_log(value) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.4f}"


def _build_review_status(ground_truth_count: int, prediction_count: int, missed_count: int, false_positive_count: int) -> str:
    if ground_truth_count == 0 and prediction_count == 0:
        return "negative_correct"
    if ground_truth_count == 0 and prediction_count > 0:
        return "false_positive"
    if missed_count == 0 and false_positive_count == 0:
        return "match"
    if missed_count > 0 and false_positive_count == 0:
        return "missed_ground_truth"
    if missed_count == 0 and false_positive_count > 0:
        return "extra_prediction"
    return "mixed"


def _resolve_holdout_dataset(dataset_stats: dict) -> tuple[Path | None, Path | None]:
    images_dir = dataset_stats.get("test_images_dir")
    labels_csv = dataset_stats.get("test_labels_csv")
    if not images_dir or not labels_csv:
        return None, None
    return Path(images_dir), Path(labels_csv)


def _evaluate_holdout_split(
    model_path: Path,
    data_yaml: Path,
    dataset_stats: dict,
    output_dir: Path,
    imgsz: int,
    batch_size: int,
    workers: int,
    device: str,
) -> dict:
    detector = YOLO(str(model_path))
    val_results = detector.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch_size,
        workers=workers,
        device=device,
        plots=False,
        verbose=False,
    )
    aggregate_metrics = _extract_eval_metrics(val_results)

    test_images_dir, test_labels_csv = _resolve_holdout_dataset(dataset_stats)
    if test_images_dir is None or test_labels_csv is None or not test_images_dir.exists() or not test_labels_csv.exists():
        return {
            "user_split_name": "test",
            "internal_split_name": "val",
            "metrics": aggregate_metrics,
            "review_available": False,
            "review_error": "Holdout labels.csv or images directory is not available for per-image review export.",
        }

    annotations = load_annotations(test_labels_csv)
    grouped_annotations = group_annotations_by_filename(annotations)
    filenames = sorted(grouped_annotations.keys())

    evaluation_dir = output_dir / "test_evaluation"
    review_images_dir = evaluation_dir / "test_review_images"
    metrics_path = evaluation_dir / "test_metrics.json"
    manifest_path = evaluation_dir / "test_review_manifest.json"
    review_csv_path = evaluation_dir / "test_predictions.csv"

    status_counts: dict[str, int] = {}
    entries: list[dict] = []
    review_rows: list[dict] = []
    total_ground_truth = 0
    total_predictions = 0
    total_matches = 0
    total_missed = 0
    total_false_positives = 0

    for index, filename in enumerate(filenames, start=1):
        image_path = test_images_dir / Path(filename).name
        if not image_path.exists():
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image_height, image_width = image.shape[:2]
        ground_truth = [
            _annotation_to_detection(annotation, image_width, image_height)
            for annotation in grouped_annotations.get(filename, [])
            if annotation.has_enemy
        ]

        result = detector.predict(
            source=str(image_path),
            conf=REVIEW_CONFIDENCE_THRESHOLD,
            max_det=REVIEW_MAX_DETECTIONS,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        names = _resolve_result_names(getattr(result, "names", None))
        predictions = [
            _prediction_to_detection(box, names, image_width, image_height)
            for box in getattr(result, "boxes", []) or []
        ]
        predictions.sort(key=lambda item: item["confidence"], reverse=True)

        matches, missed_ground_truth, false_positives = _match_detections(
            ground_truth,
            predictions,
            REVIEW_IOU_THRESHOLD,
        )
        status = _build_review_status(
            len(ground_truth),
            len(predictions),
            len(missed_ground_truth),
            len(false_positives),
        )
        status_counts[status] = status_counts.get(status, 0) + 1

        review_image_path = _render_review_image(
            image_path,
            review_images_dir / Path(filename).name,
            ground_truth,
            predictions,
        )

        entry = {
            "filename": Path(filename).name,
            "image_path": str(image_path.resolve()),
            "review_image_path": review_image_path,
            "status": status,
            "image_size": {"width": image_width, "height": image_height},
            "ground_truth": ground_truth,
            "predictions": predictions,
            "matches": matches,
            "missed_ground_truth": missed_ground_truth,
            "false_positives": false_positives,
            "counts": {
                "ground_truth": len(ground_truth),
                "predictions": len(predictions),
                "matched": len(matches),
                "missed_ground_truth": len(missed_ground_truth),
                "false_positives": len(false_positives),
            },
        }
        entries.append(entry)
        review_rows.append({
            "filename": entry["filename"],
            "status": status,
            "image_path": entry["image_path"],
            "review_image_path": review_image_path or "",
            "ground_truth_count": len(ground_truth),
            "prediction_count": len(predictions),
            "matched_count": len(matches),
            "missed_ground_truth_count": len(missed_ground_truth),
            "false_positive_count": len(false_positives),
        })

        total_ground_truth += len(ground_truth)
        total_predictions += len(predictions)
        total_matches += len(matches)
        total_missed += len(missed_ground_truth)
        total_false_positives += len(false_positives)

        if index == 1 or index % 10 == 0 or index == len(filenames):
            print(f"[Eval] Reviewed {index}/{len(filenames)} holdout image(s)...")

    review_summary = {
        "images": len(entries),
        "ground_truth_boxes": total_ground_truth,
        "predicted_boxes": total_predictions,
        "matched_boxes": total_matches,
        "missed_ground_truth_boxes": total_missed,
        "false_positive_boxes": total_false_positives,
        "status_counts": status_counts,
    }

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = _to_jsonable(
        {
            "aggregate_metrics": aggregate_metrics,
            "review_summary": review_summary,
            "iou_match_threshold": REVIEW_IOU_THRESHOLD,
            "confidence_threshold": REVIEW_CONFIDENCE_THRESHOLD,
        }
    )
    manifest_payload = _to_jsonable(
        {
            "entries": entries,
            "summary": review_summary,
            "aggregate_metrics": aggregate_metrics,
            "iou_match_threshold": REVIEW_IOU_THRESHOLD,
            "confidence_threshold": REVIEW_CONFIDENCE_THRESHOLD,
        }
    )
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    _write_review_csv(review_csv_path, review_rows)

    return {
        "user_split_name": "test",
        "internal_split_name": "val",
        "metrics": aggregate_metrics,
        "review_available": True,
        "review_summary": review_summary,
        "iou_match_threshold": REVIEW_IOU_THRESHOLD,
        "confidence_threshold": REVIEW_CONFIDENCE_THRESHOLD,
        "metrics_path": str(metrics_path),
        "review_manifest_path": str(manifest_path),
        "review_csv_path": str(review_csv_path),
        "review_images_dir": str(review_images_dir),
    }


def _prepare_dataset(args) -> tuple[Path, dict]:
    output_dir = Path(args.output_dir)
    prepared_dir = output_dir / "prepared_dataset"

    if args.force_rebuild and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"No data.yaml found in dataset_dir: {dataset_dir}")
        test_images_dir = dataset_dir / "val" / "images"
        test_labels_csv = dataset_dir / "val" / "labels.csv"
        stats = {
            "dataset_mode": "existing_yolo_dataset",
            "data_yaml": str(data_yaml),
            "test_images_dir": str(test_images_dir) if test_images_dir.exists() else None,
            "test_labels_csv": str(test_labels_csv) if test_labels_csv.exists() else None,
            "internal_test_dir_name": "val",
        }
        return data_yaml, stats

    train_csv = Path(args.train_csv or args.csv)
    train_dir = Path(args.train_dir or args.img_dir)
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Training image directory not found: {train_dir}")

    train_annotations = load_annotations(train_csv)
    if not train_annotations:
        raise RuntimeError(f"No valid annotations found in {train_csv}")

    test_split = _resolve_test_split(args)

    if args.val_csv and args.val_dir:
        val_csv = Path(args.val_csv)
        val_dir = Path(args.val_dir)
        if not val_csv.exists():
            raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
        if not val_dir.exists():
            raise FileNotFoundError(f"Validation image directory not found: {val_dir}")
        val_annotations = load_annotations(val_csv)
        train_video_ids = sorted({ann.video_id for ann in train_annotations})
        val_video_ids = sorted({ann.video_id for ann in val_annotations})
    else:
        train_annotations, val_annotations, train_video_ids, val_video_ids = split_annotations_by_video(
            train_annotations,
            val_ratio=test_split,
            seed=args.seed,
            stratified=args.stratified_split,
        )
        val_dir = train_dir

    if not val_annotations:
        raise RuntimeError("Validation split is empty. Provide more labeled data or a smaller val split.")

    prepared_dir.mkdir(parents=True, exist_ok=True)
    train_stats = export_split_dataset(train_annotations, train_dir, prepared_dir / "train")
    val_stats = export_split_dataset(val_annotations, val_dir, prepared_dir / "val")
    class_names = infer_class_names(train_annotations + val_annotations)
    data_yaml = write_data_yaml(prepared_dir, class_names=class_names)

    dataset_stats = {
        "dataset_mode": "generated_from_csv",
        "data_yaml": str(data_yaml),
        "class_names": class_names,
        "train_annotations": len(train_annotations),
        "test_annotations": len(val_annotations),
        "val_annotations": len(val_annotations),
        "train_images": train_stats["images"],
        "test_images": val_stats["images"],
        "val_images": val_stats["images"],
        "train_video_ids": train_video_ids,
        "test_video_ids": val_video_ids,
        "val_video_ids": val_video_ids,
        "bbox_sources": _count_box_sources(train_annotations + val_annotations),
        "test_split": test_split,
        "test_split_effective_images": val_stats["images"] / max(1, train_stats["images"] + val_stats["images"]),
        "test_images_dir": str((prepared_dir / "val" / "images").resolve()),
        "test_labels_csv": str((prepared_dir / "val" / "labels.csv").resolve()),
        "internal_test_dir_name": "val",
    }
    return data_yaml, dataset_stats


def _find_best_weights(runs_dir: Path) -> Path | None:
    candidates = list(runs_dir.rglob("best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_best_weights(output_dir: Path, runs_dir: Path, detector: YOLO) -> Path | None:
    search_roots: list[Path] = []

    trainer = getattr(detector, "trainer", None)
    trainer_save_dir = getattr(trainer, "save_dir", None)
    if trainer_save_dir:
        search_roots.append(Path(trainer_save_dir))

    search_roots.extend([
        runs_dir,
        output_dir,
        PROJECT_ROOT / "runs",
    ])

    seen = set()
    for root in search_roots:
        resolved = Path(root)
        key = str(resolved.resolve()) if resolved.exists() else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        best = _find_best_weights(resolved)
        if best is not None:
            return best

    return None


def _resolve_training_device(requested_mode: str | None, explicit_device: str | None) -> tuple[str, str, bool]:
    if explicit_device:
        actual_device = explicit_device
        use_cuda = explicit_device != "cpu" and torch.cuda.is_available()
        return "manual", actual_device, use_cuda

    mode = (requested_mode or "auto").strip().lower()
    cuda_available = torch.cuda.is_available()

    if mode == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA mode was requested, but PyTorch does not report an available NVIDIA CUDA device. "
                "Install a CUDA-enabled PyTorch build or switch the training device to Auto/CPU."
            )
        return mode, "0", True

    if mode == "cpu":
        return mode, "cpu", False

    return "auto", ("0" if cuda_available else "cpu"), cuda_available


def _configure_cuda_runtime(use_cuda: bool) -> dict:
    runtime_flags = {
        "amp": bool(use_cuda),
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
    }
    if not use_cuda:
        return runtime_flags

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
        runtime_flags["tf32_matmul"] = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        runtime_flags["cudnn_benchmark"] = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
            runtime_flags["tf32_cudnn"] = True

    return runtime_flags


def _resolve_worker_count(requested_workers: int, use_cuda: bool) -> int:
    if requested_workers > 0:
        return requested_workers
    if not use_cuda:
        return 0
    cpu_count = os.cpu_count() or 4
    return max(2, min(4, cpu_count - 1))


def _print_runtime_summary(requested_mode: str, actual_device: str, use_cuda: bool, runtime_flags: dict, workers: int) -> None:
    print(f"[Runtime] Torch: {torch.__version__}")
    print(f"[Runtime] Requested device mode: {requested_mode}")
    print(f"[Runtime] CUDA available: {torch.cuda.is_available()} | CUDA build: {torch.version.cuda or 'not available'}")
    print(f"[Runtime] Actual training device: {actual_device}")
    print(f"[Runtime] Workers: {workers} | AMP: {runtime_flags['amp']} | TF32 matmul: {runtime_flags['tf32_matmul']} | TF32 cuDNN: {runtime_flags['tf32_cudnn']} | cuDNN benchmark: {runtime_flags['cudnn_benchmark']}")

    if not use_cuda:
        print("[Runtime] Training will run on CPU.")
        return

    device_index = 0
    if actual_device.startswith("cuda:"):
        device_index = int(actual_device.split(":", 1)[1])
    elif actual_device.isdigit():
        device_index = int(actual_device)

    props = torch.cuda.get_device_properties(device_index)
    total_vram_gb = props.total_memory / (1024 ** 3)
    cudnn_version = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else "unknown"
    print(f"[Runtime] NVIDIA GPU: {torch.cuda.get_device_name(device_index)}")
    print(
        f"[Runtime] GPU count: {torch.cuda.device_count()} | Capability: {props.major}.{props.minor} | "
        f"VRAM: {total_vram_gb:.1f} GB | cuDNN: {cudnn_version}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train a multi-class Fortnite character detector with YOLO/Ultralytics")
    parser.add_argument("--dataset_dir", type=str, help="Existing YOLO dataset directory containing data.yaml")
    parser.add_argument("--csv", type=str, default="dataset/cleaned/labels_cleaned.csv",
                        help="Path to a bbox annotation CSV")
    parser.add_argument("--img_dir", type=str, default="dataset/cleaned/images",
                        help="Directory containing images referenced by --csv")
    parser.add_argument("--train_csv", type=str, help="Training annotation CSV")
    parser.add_argument("--train_dir", type=str, help="Training image directory")
    parser.add_argument("--val_csv", type=str, help="Validation annotation CSV")
    parser.add_argument("--val_dir", type=str, help="Validation image directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_CHOICE,
                        help="Detector backbone key or weights path")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--device", type=str, default=None, help="Training device, e.g. 0 or cpu")
    parser.add_argument("--device_mode", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="High-level device preference used when --device is not set")
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save outputs")
    parser.add_argument("--run_name", type=str, default="enemy_detector", help="Ultralytics run name")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Held-out test split ratio when only one CSV is given")
    parser.add_argument("--val_split", type=float,
                        help="Deprecated alias for --test_split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--stratified_split", action="store_true",
                        help="Balance the auto-generated split by screen position bias")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Rebuild the prepared YOLO dataset cache before training")
    parser.add_argument("--print_model_choices", action="store_true",
                        help="Print supported detector choices and exit")
    args = parser.parse_args()

    if args.print_model_choices:
        print(format_model_choices())
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = resolve_model_source(args.model, PROJECT_ROOT)
    requested_device_mode, device, use_cuda = _resolve_training_device(args.device_mode, args.device)
    runtime_flags = _configure_cuda_runtime(use_cuda)
    workers = _resolve_worker_count(args.workers, use_cuda)
    data_yaml, dataset_stats = _prepare_dataset(args)

    print(f"[Config] Device: {device}")
    print(f"[Config] Model:  {model_source}")
    print(f"[Config] Data:   {data_yaml}")
    print(f"[Config] Epochs: {args.epochs} | Batch: {args.batch_size} | ImgSz: {args.imgsz}")
    _print_runtime_summary(requested_device_mode, device, use_cuda, runtime_flags, workers)

    detector = YOLO(model_source)
    runs_dir = output_dir / "runs"
    detector.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        workers=workers,
        device=device,
        patience=args.patience,
        project=str(runs_dir),
        name=args.run_name,
        exist_ok=True,
        amp=runtime_flags["amp"],
        verbose=True,
    )

    best_weights = _resolve_best_weights(output_dir, runs_dir, detector)
    if best_weights is None:
        raise RuntimeError(
            "Training completed but no best.pt was found under the expected run directories. "
            f"Checked {runs_dir}, {output_dir}, and {PROJECT_ROOT / 'runs'}."
        )

    stable_best = output_dir / "best_model.pt"
    shutil.copy2(best_weights, stable_best)
    evaluation_summary = _evaluate_holdout_split(
        model_path=stable_best,
        data_yaml=data_yaml,
        dataset_stats=dataset_stats,
        output_dir=output_dir,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        workers=workers,
        device=device,
    )

    summary = {
        "chosen_model": model_source,
        "best_weights": str(best_weights),
        "stable_best_model": str(stable_best),
        "dataset": dataset_stats,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "requested_device_mode": requested_device_mode,
        "device": device,
        "use_cuda": use_cuda,
        "workers": workers,
        "runtime_flags": runtime_flags,
        "evaluation": evaluation_summary,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Best model copied to: {stable_best}")
    if evaluation_summary.get("review_available"):
        metrics = evaluation_summary.get("metrics", {})
        print(
            "[Eval] Holdout metrics:"
            f" P={_format_metric_for_log(metrics.get('precision'))}"
            f" R={_format_metric_for_log(metrics.get('recall'))}"
            f" mAP50={_format_metric_for_log(metrics.get('map50'))}"
            f" mAP50-95={_format_metric_for_log(metrics.get('map50_95'))}"
        )
        print(f"[Eval] Review manifest:  {evaluation_summary.get('review_manifest_path')}")
    else:
        print(f"[Eval] Review export unavailable: {evaluation_summary.get('review_error')}")
    print(f"Training summary:     {summary_path}")


if __name__ == "__main__":
    main()
