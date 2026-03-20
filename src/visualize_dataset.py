import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def plot_spatial_distribution(csv_path, output_path):
    """Generate a 2D heatmap of enemy locations."""
    x_coords = []
    y_coords = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both formats (enhanced and original)
            x = row.get('x_norm', row.get('x_center'))
            y = row.get('y_norm', row.get('y_center'))
            
            if x and y:
                x_coords.append(float(x))
                y_coords.append(float(y))
    
    if not x_coords:
        print("No valid coordinates found in CSV.")
        return

    plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Scatter Plot (Red Dots)
    plt.scatter(x_coords, y_coords, color='red', s=20, alpha=0.6, edgecolors='none')
    
    # Add center crosshair lines
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add labels
    plt.title(f"Enemy Position Distribution\nSource: {Path(csv_path).name}", color='black', pad=20)
    plt.xlabel("Normalized X", color='black')
    plt.ylabel("Normalized Y", color='black')
    
    # Set axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Invert Y to match screen coords (0,0 is top-left)
    plt.gca().invert_yaxis()
    
    # Styling
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_visible(True)
    
    plt.grid(True, linestyle=':', alpha=0.3, color='gray')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white')
    plt.close()
    print(f"Heatmap saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Dataset Bias")
    parser.add_argument("--csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output", type=str, default="dataset/center_bias_heatmap.png", help="Output PNG path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_spatial_distribution(args.csv, args.output)
