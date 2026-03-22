import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.dataset import (
    DEFAULT_CLASS_NAME,
    export_split_dataset,
    load_annotations,
    split_annotations_by_video,
    write_data_yaml,
)
from src.model import DEFAULT_MODEL_CHOICE, format_model_choices, resolve_model_source

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _count_box_sources(annotations):
    summary = {}
    for annotation in annotations:
        summary[annotation.bbox_source] = summary.get(annotation.bbox_source, 0) + 1
    return summary


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
        stats = {
            "dataset_mode": "existing_yolo_dataset",
            "data_yaml": str(data_yaml),
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
            val_ratio=args.val_split,
            seed=args.seed,
            stratified=args.stratified_split,
        )
        val_dir = train_dir

    if not val_annotations:
        raise RuntimeError("Validation split is empty. Provide more labeled data or a smaller val split.")

    prepared_dir.mkdir(parents=True, exist_ok=True)
    train_stats = export_split_dataset(train_annotations, train_dir, prepared_dir / "train")
    val_stats = export_split_dataset(val_annotations, val_dir, prepared_dir / "val")
    data_yaml = write_data_yaml(prepared_dir, class_names=[DEFAULT_CLASS_NAME])

    dataset_stats = {
        "dataset_mode": "generated_from_csv",
        "data_yaml": str(data_yaml),
        "train_annotations": len(train_annotations),
        "val_annotations": len(val_annotations),
        "train_images": train_stats["images"],
        "val_images": val_stats["images"],
        "train_video_ids": train_video_ids,
        "val_video_ids": val_video_ids,
        "bbox_sources": _count_box_sources(train_annotations + val_annotations),
    }
    return data_yaml, dataset_stats


def _find_best_weights(runs_dir: Path) -> Path | None:
    candidates = list(runs_dir.rglob("best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Train an enemy object detector with YOLO/Ultralytics")
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
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save outputs")
    parser.add_argument("--run_name", type=str, default="enemy_detector", help="Ultralytics run name")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split when only one CSV is given")
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
    device = args.device if args.device else ("0" if torch.cuda.is_available() else "cpu")
    data_yaml, dataset_stats = _prepare_dataset(args)

    print(f"[Config] Device: {device}")
    print(f"[Config] Model:  {model_source}")
    print(f"[Config] Data:   {data_yaml}")
    print(f"[Config] Epochs: {args.epochs} | Batch: {args.batch_size} | ImgSz: {args.imgsz}")

    detector = YOLO(model_source)
    runs_dir = output_dir / "runs"
    detector.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        workers=args.workers,
        device=device,
        patience=args.patience,
        project=str(runs_dir),
        name=args.run_name,
        exist_ok=True,
        verbose=True,
    )

    best_weights = _find_best_weights(runs_dir)
    if best_weights is None:
        raise RuntimeError(f"Training completed but no best.pt was found under {runs_dir}")

    stable_best = output_dir / "best_model.pt"
    shutil.copy2(best_weights, stable_best)

    summary = {
        "chosen_model": model_source,
        "best_weights": str(best_weights),
        "stable_best_model": str(stable_best),
        "dataset": dataset_stats,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "device": device,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Best model copied to: {stable_best}")
    print(f"Training summary:     {summary_path}")


if __name__ == "__main__":
    main()
