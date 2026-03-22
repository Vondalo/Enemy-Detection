from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import PLAYER_CLASS_ID, load_annotations

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def plot_spatial_distribution(csv_path: str, output_path: str) -> None:
    annotations = load_annotations(csv_path)

    enemy_x = []
    enemy_y = []
    player_x = []
    player_y = []

    for annotation in annotations:
        if not annotation.has_enemy:
            continue
        if annotation.class_id == PLAYER_CLASS_ID:
            player_x.append(annotation.x_center)
            player_y.append(annotation.y_center)
        else:
            enemy_x.append(annotation.x_center)
            enemy_y.append(annotation.y_center)

    plt.figure(figsize=(12, 8), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    if enemy_x:
        heatmap, xedges, yedges = np.histogram2d(enemy_x, enemy_y, bins=30, range=[[0, 1], [0, 1]])
        ax.imshow(
            heatmap.T,
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="Reds",
            alpha=0.45,
            aspect="auto",
        )
        ax.scatter(enemy_x, enemy_y, color="#dc2626", s=24, alpha=0.65, edgecolors="none", label=f"enemy ({len(enemy_x)})")

    if player_x:
        ax.scatter(player_x, player_y, color="#0ea5e9", s=20, alpha=0.5, edgecolors="none", label=f"player ({len(player_x)})")

    if not enemy_x and not player_x:
        ax.text(0.5, 0.5, "No valid annotations found", ha="center", va="center", fontsize=14, color="#64748b")

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    ax.set_title(f"Spatial Annotation Distribution\nSource: {Path(csv_path).name}", color="black", pad=20)
    ax.set_xlabel("Normalized X", color="black")
    ax.set_ylabel("Normalized Y", color="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()

    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_visible(True)

    if enemy_x or player_x:
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cbd5e1")

    plt.grid(True, linestyle=":", alpha=0.3, color="gray")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white")
    plt.close()
    print(f"Heatmap saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize spatial annotation bias for YOLO datasets")
    parser.add_argument("--csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output", type=str, default="dataset/center_bias_heatmap.png", help="Output PNG path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_spatial_distribution(args.csv, args.output)
