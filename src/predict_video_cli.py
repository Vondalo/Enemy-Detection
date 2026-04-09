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


def _color_for_class(class_key: str) -> tuple[int, int, int]:
    if class_key == "player":
        return (36, 191, 251)
    if class_key == "enemy":
        return (94, 63, 244)
    return (148, 163, 184)


def _render_annotated_frame(frame, detections):
    rendered = frame.copy()
    image_height, image_width = rendered.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width - 1, x2))
        y2 = max(0, min(image_height - 1, y2))

        color = _color_for_class(str(detection.get("class_key", "unknown")))
        label = f'{detection.get("class_name", "Unknown")} {float(detection.get("confidence", 0.0)) * 100:.0f}%'

        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_left = max(0, min(x1, image_width - text_width - 10))
        label_top = max(0, y1 - text_height - baseline - 8)
        label_right = min(image_width - 1, label_left + text_width + 10)
        label_bottom = min(image_height - 1, label_top + text_height + baseline + 6)
        text_y = max(text_height, label_bottom - baseline - 3)

        cv2.rectangle(rendered, (label_left, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            rendered,
            label,
            (label_left + 5, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return rendered


def _normalize_frame_size(frame_size: tuple[int, int]) -> tuple[int, int]:
    width, height = [int(value) for value in frame_size]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video frame size {width}x{height}.")

    # Some OpenCV/codec combinations refuse odd dimensions for encoded output.
    if width % 2 != 0 and width > 1:
        width -= 1
    if height % 2 != 0 and height > 1:
        height -= 1

    return width, height


def _iter_video_writer_targets(save_path: Path):
    seen: set[tuple[str, str]] = set()

    def _add(candidate_path: Path, codec: str):
        key = (str(candidate_path), codec)
        if key not in seen:
            seen.add(key)
            yield candidate_path, codec

    if save_path.suffix.lower() != ".mp4":
        yield from _add(save_path.with_suffix(".mp4"), "mp4v")
        yield from _add(save_path.with_suffix(".mp4"), "avc1")
        yield from _add(save_path.with_suffix(".avi"), "XVID")
        yield from _add(save_path.with_suffix(".avi"), "MJPG")
        return

    yield from _add(save_path, "mp4v")
    yield from _add(save_path, "avc1")
    yield from _add(save_path.with_suffix(".avi"), "XVID")
    yield from _add(save_path.with_suffix(".avi"), "MJPG")


def _open_video_writer(save_path: Path, fps: float, frame_size: tuple[int, int]):
    width, height = _normalize_frame_size(frame_size)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    effective_fps = fps if fps > 0 else 30.0
    attempted_targets = []

    for candidate_path, codec in _iter_video_writer_targets(save_path):
        writer = cv2.VideoWriter(
            str(candidate_path),
            cv2.VideoWriter_fourcc(*codec),
            effective_fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, candidate_path, (width, height), codec
        attempted_targets.append(f"{candidate_path.name} ({codec})")
        writer.release()

    raise RuntimeError(
        f"Unable to open a video writer for {save_path}. Tried: {', '.join(attempted_targets)}."
    )


def _prepare_frame_for_writer(frame, target_size: tuple[int, int]):
    target_width, target_height = target_size
    current_height, current_width = frame.shape[:2]

    if current_width == target_width and current_height == target_height:
        return frame

    # Prefer dropping a trailing row/column over resampling when we only adjusted odd dimensions.
    if current_width >= target_width and current_height >= target_height:
        cropped = frame[:target_height, :target_width]
        if cropped.shape[1] == target_width and cropped.shape[0] == target_height:
            return cropped

    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _finalize_saved_video(video_writer, save_path: Path | None, save_error: str | None):
    if video_writer is not None:
        try:
            video_writer.release()
        except Exception as exc:
            if save_error is None:
                save_error = f"Failed to finalize annotated video at {save_path}: {exc}"
        video_writer = None

    if save_path:
        try:
            if not save_path.exists():
                if save_error is None:
                    save_error = f"Annotated video was not created at {save_path}."
            elif save_path.stat().st_size <= 0 and save_error is None:
                save_error = f"Annotated video at {save_path} is empty."
        except Exception as exc:
            if save_error is None:
                save_error = f"Failed to verify annotated video at {save_path}: {exc}"

    return video_writer, save_error


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
    parser.add_argument("--save_path", default=None, help="Optional path to save an annotated video file.")
    parser.add_argument("--stop_file", default=None, help="Optional sentinel file that requests a graceful stop.")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    model_path = Path(resolve_inference_model_source(args.model, PROJECT_ROOT))
    save_path = Path(args.save_path).expanduser().resolve() if args.save_path else None
    stop_file = Path(args.stop_file).expanduser().resolve() if args.stop_file else None

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
        requested_save_path = str(save_path) if save_path else None
        video_writer = None
        actual_save_path = save_path
        writer_frame_size = None
        save_error = None

        if save_path and frame_width > 0 and frame_height > 0:
            try:
                video_writer, actual_save_path, writer_frame_size, _ = _open_video_writer(
                    save_path,
                    fps,
                    (frame_width, frame_height),
                )
            except Exception as exc:
                save_error = str(exc)

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
            saved_video_path=str(actual_save_path) if actual_save_path else requested_save_path,
            save_error=save_error,
        )

        processed_frames = 0
        frame_index = 0
        progress_interval = max(1, min(30, int(round(fps))))
        stop_requested = False

        while True:
            if stop_file and stop_file.exists():
                stop_requested = True
                break

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

            if save_path and video_writer is None and save_error is None:
                try:
                    video_writer, actual_save_path, writer_frame_size, _ = _open_video_writer(
                        save_path,
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                except Exception as exc:
                    save_error = str(exc)

            if video_writer is not None:
                rendered_frame = _render_annotated_frame(frame, detections)
                video_writer.write(_prepare_frame_for_writer(rendered_frame, writer_frame_size))

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
        video_writer, save_error = _finalize_saved_video(video_writer, actual_save_path, save_error)
        if stop_requested:
            _emit(
                "stopped",
                processed_frames=processed_frames,
                total_frames=total_frames,
                percent=(processed_frames / total_frames) * 100 if total_frames > 0 else 0.0,
                saved_video_path=str(actual_save_path) if actual_save_path else requested_save_path,
                save_error=save_error,
            )
        else:
            _emit(
                "complete",
                processed_frames=processed_frames,
                total_frames=total_frames,
                percent=100.0 if total_frames > 0 else 0.0,
                saved_video_path=str(actual_save_path) if actual_save_path else requested_save_path,
                save_error=save_error,
            )
    except Exception as exc:
        _emit("error", message=str(exc))
        sys.exit(1)
    finally:
        capture.release()
        if "video_writer" in locals() and video_writer is not None:
            video_writer.release()


if __name__ == "__main__":
    main()
