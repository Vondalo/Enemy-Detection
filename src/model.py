import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DetectorChoice:
    key: str
    weights: str
    summary: str
    strengths: str
    tradeoffs: str
    recommended_for: str


MODEL_CHOICES: Dict[str, DetectorChoice] = {
    "yolov8n": DetectorChoice(
        key="yolov8n",
        weights="yolov8n.pt",
        summary="Smallest YOLOv8 checkpoint with the fastest iteration speed.",
        strengths="Great default for quick experiments, weaker GPUs, and frequent retraining.",
        tradeoffs="Lowest recall ceiling of the bundled options on tiny or distant enemies.",
        recommended_for="Most local finetuning runs and first-pass dataset validation.",
    ),
    "yolov8s": DetectorChoice(
        key="yolov8s",
        weights="yolov8s.pt",
        summary="A larger YOLOv8 backbone with better small-object recall.",
        strengths="Usually a stronger balance when enemies are far away or partially occluded.",
        tradeoffs="Slower training and inference than `yolov8n`.",
        recommended_for="Accuracy-focused finetuning once the bbox pipeline is stable.",
    ),
    "yolov8m": DetectorChoice(
        key="yolov8m",
        weights="yolov8m.pt",
        summary="Mid-sized YOLOv8 detector with a larger accuracy budget.",
        strengths="Higher capacity for harder scenes, clutter, and scale variation.",
        tradeoffs="Needs more VRAM and noticeably more training time.",
        recommended_for="Later-stage training once you have enough clean data.",
    ),
    "rtdetr-l": DetectorChoice(
        key="rtdetr-l",
        weights="rtdetr-l.pt",
        summary="Transformer-style detector with strong global reasoning.",
        strengths="Can do well when targets blend into busy scenes.",
        tradeoffs="Heavier and less convenient for quick iteration than YOLOv8.",
        recommended_for="A stronger baseline to compare against YOLO after the dataset matures.",
    ),
}

DEFAULT_MODEL_CHOICE = "yolov8n"
ACTIVE_MODEL_FILENAME = "active_model.json"
DEFAULT_INFERENCE_MODEL_PATH = Path("models") / "best_model.pt"


def list_model_choices() -> List[DetectorChoice]:
    return list(MODEL_CHOICES.values())


def get_model_choice(key: str) -> DetectorChoice:
    normalized = (key or DEFAULT_MODEL_CHOICE).strip().lower()
    if normalized not in MODEL_CHOICES:
        available = ", ".join(sorted(MODEL_CHOICES))
        raise ValueError(f"Unknown model choice '{key}'. Available choices: {available}")
    return MODEL_CHOICES[normalized]


def resolve_model_source(model_name: str | None, project_root: str | Path | None = None) -> str:
    target = (model_name or DEFAULT_MODEL_CHOICE).strip()
    candidate = Path(target)
    if candidate.exists():
        return str(candidate)

    root = Path(project_root) if project_root else None
    if root is not None:
        rooted_candidate = root / target
        if rooted_candidate.exists():
            return str(rooted_candidate)

    normalized = target.lower()
    if normalized in MODEL_CHOICES:
        choice = MODEL_CHOICES[normalized]
        if root is not None:
            bundled = root / choice.weights
            if bundled.exists():
                return str(bundled)
        return choice.weights

    return target


def _resolve_project_root(project_root: str | Path | None = None) -> Path:
    if project_root is not None:
        return Path(project_root)
    return Path(__file__).resolve().parents[1]


def _resolve_existing_project_path(project_root: Path, candidate_path: str | Path | None) -> Path | None:
    if not candidate_path:
        return None

    candidate = Path(candidate_path)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    rooted = project_root / candidate
    return rooted if rooted.exists() else None


def read_active_model_selection(project_root: str | Path | None = None) -> dict | None:
    root = _resolve_project_root(project_root)
    active_model_path = root / "models" / ACTIVE_MODEL_FILENAME
    if not active_model_path.exists():
        return None

    try:
        payload = json.loads(active_model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def resolve_active_model_weights(project_root: str | Path | None = None) -> str | None:
    root = _resolve_project_root(project_root)
    active_selection = read_active_model_selection(root)
    if not active_selection:
        return None

    for key in ("weights_path", "model_path", "stable_best_model"):
        resolved = _resolve_existing_project_path(root, active_selection.get(key))
        if resolved is not None:
            return str(resolved)

    return None


def resolve_inference_model_source(model_name: str | None, project_root: str | Path | None = None) -> str:
    root = _resolve_project_root(project_root)
    if model_name:
        return resolve_model_source(model_name, root)

    active_weights = resolve_active_model_weights(root)
    if active_weights:
        return active_weights

    return str(root / DEFAULT_INFERENCE_MODEL_PATH)


def format_model_choices() -> str:
    lines = []
    for choice in list_model_choices():
        lines.append(f"- {choice.key}: {choice.summary}")
        lines.append(f"  strengths: {choice.strengths}")
        lines.append(f"  tradeoffs: {choice.tradeoffs}")
        lines.append(f"  best for: {choice.recommended_for}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Available detector backbones:")
    print(format_model_choices())
