import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import torch

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.model import resolve_inference_model_source


def _emit(event_type: str, **payload) -> None:
    print(json.dumps({"type": event_type, **payload}), flush=True)


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


def _resolve_names(result) -> dict[int, str]:
    if isinstance(result.names, dict):
        return result.names
    if isinstance(result.names, (list, tuple)):
        return {idx: name for idx, name in enumerate(result.names)}
    return {}


def _predict_frame(detector: YOLO, frame, conf: float, max_det: int, device: str | int):
    result = detector.predict(
        source=frame,
        conf=conf,
        max_det=max_det,
        verbose=False,
        device=device,
    )[0]

    detections = []
    class_summary = {"enemy": 0, "player": 0}
    image_width, image_height = result.orig_shape[1], result.orig_shape[0]
    names = _resolve_names(result)

    for box in result.boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        class_id = _box_class_id(box)
        raw_class_name = names.get(class_id, str(class_id))
        class_key, class_name = _normalize_class_name(raw_class_name)
        if class_key in class_summary:
            class_summary[class_key] += 1
        detections.append({
            "class_id": class_id,
            "class_key": class_key,
            "class_name": class_name,
            "model_class_name": str(raw_class_name),
            "confidence": _box_confidence(box),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xyxy_normalized": [
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            ],
            "x_center": x_center / image_width,
            "y_center": y_center / image_height,
            "width": width / image_width,
            "height": height / image_height,
        })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections, class_summary, image_width, image_height


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detections over a video and stream structured JSON events.")
    parser.add_argument("video_path", help="Path to the input video.")
    parser.add_argument("--mode", choices=["precompute", "stream"], default="precompute",
                        help="Inference mode requested by the UI.")
    parser.add_argument("--model", default=None,
                        help="Path to trained detector weights. Defaults to the active model, then models/best_model.pt.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max_det", type=int, default=10, help="Maximum detections per frame.")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    model_path = Path(resolve_inference_model_source(args.model, PROJECT_ROOT))

    if not video_path.exists():
        _emit("error", message=f"Video not found: {video_path}")
        sys.exit(1)

    if not model_path.exists():
        _emit("error", message=f"Model not found at {model_path}. Please train the detector first.")
        sys.exit(1)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        _emit("error", message=f"Failed to open video: {video_path}")
        sys.exit(1)

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if frame_count > 0 and fps > 0 else 0.0
        device = 0 if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        detector = YOLO(str(model_path))

        _emit(
            "started",
            mode=args.mode,
            video_path=str(video_path.resolve()),
            model_path=str(model_path.resolve()),
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            frame_width=frame_width,
            frame_height=frame_height,
            device=f"cuda:{device}" if device != "cpu" else "cpu",
            device_name=device_name,
        )

        processed_frames = 0
        frame_index = 0
        progress_interval = max(1, min(30, int(round(fps))))

        while True:
            success, frame = capture.read()
            if not success:
                break

            detections, class_summary, image_width, image_height = _predict_frame(
                detector=detector,
                frame=frame,
                conf=args.conf,
                max_det=args.max_det,
                device=device,
            )

            _emit(
                "frame",
                frame_index=frame_index,
                time_s=(frame_index / fps) if fps > 0 else 0.0,
                detections=detections,
                class_summary=class_summary,
                image_size={"width": image_width, "height": image_height},
            )

            processed_frames += 1
            if processed_frames == 1 or processed_frames % progress_interval == 0 or (frame_count > 0 and processed_frames >= frame_count):
                total_frames = frame_count if frame_count > 0 else processed_frames
                percent = (processed_frames / total_frames) * 100 if total_frames > 0 else 0.0
                _emit(
                    "progress",
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                    percent=percent,
                )

            frame_index += 1

        total_frames = frame_count if frame_count > 0 else processed_frames
        _emit(
            "complete",
            processed_frames=processed_frames,
            total_frames=total_frames,
            percent=100.0 if total_frames > 0 else 0.0,
        )
    except Exception as exc:
        _emit("error", message=str(exc))
        sys.exit(1)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
