import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


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


def main():
    parser = argparse.ArgumentParser(description="Run object detection on a single image.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "best_model.pt"),
                        help="Path to trained detector weights.")
    parser.add_argument("--save_path", default=None,
                        help="Optional path to save the rendered detection image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max_det", type=int, default=10, help="Maximum detections to return.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model)

    if not image_path.exists():
        print(json.dumps({"error": f"Image not found: {image_path}"}))
        sys.exit(1)

    if not model_path.exists():
        print(json.dumps({"error": f"Model not found at {model_path}. Please train the detector first."}))
        sys.exit(1)

    try:
        detector = YOLO(str(model_path))
        result = detector.predict(
            source=str(image_path),
            conf=args.conf,
            max_det=args.max_det,
            verbose=False,
        )[0]

        detections = []
        image_width, image_height = result.orig_shape[1], result.orig_shape[0]
        if isinstance(result.names, dict):
            names = result.names
        elif isinstance(result.names, (list, tuple)):
            names = {idx: name for idx, name in enumerate(result.names)}
        else:
            names = {}

        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            x_center = x1 + width / 2
            y_center = y1 + height / 2
            class_id = _box_class_id(box)
            detections.append({
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "confidence": _box_confidence(box),
                "bbox_xyxy": [x1, y1, x2, y2],
                "x_center": x_center / image_width,
                "y_center": y_center / image_height,
                "width": width / image_width,
                "height": height / image_height,
            })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        top_detection = detections[0] if detections else None

        saved_image_path = None
        if args.save_path:
            rendered = result.plot()
            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), rendered)
            saved_image_path = str(save_path.resolve())

        print(json.dumps({
            "detections": detections,
            "count": len(detections),
            "top_detection": top_detection,
            "prediction": [top_detection["x_center"], top_detection["y_center"]] if top_detection else None,
            "truth": None,
            "saved_image_path": saved_image_path,
            "model_path": str(model_path.resolve()),
        }))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
