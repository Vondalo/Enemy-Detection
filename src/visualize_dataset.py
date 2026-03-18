import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

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

    plt.figure(figsize=(10, 8), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    
    # 2D Histogram (Heatmap)
    # Origin at top-left to match image coordinates
    plt.hist2d(x_coords, y_coords, bins=40, range=[[0, 1], [0, 1]], cmap='magma', cmin=1)
    
    # Add center crosshair lines
    plt.axvline(x=0.5, color='white', linestyle='--', alpha=0.3)
    plt.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
    
    # Add labels
    plt.title(f"Enemy Spatial Distribution\nSource: {Path(csv_path).name}", color='white', pad=20)
    plt.xlabel("Normalized X", color='white')
    plt.ylabel("Normalized Y", color='white')
    plt.colorbar(label='Sample Count')
    
    # Invert Y to match screen coords (0,0 is top-left)
    plt.gca().invert_yaxis()
    
    # Styling
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, facecolor='#121212')
    plt.close()
    print(f"Heatmap saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Dataset Bias")
    parser.add_argument("--csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--output", type=str, default="dataset/center_bias_heatmap.png", help="Output PNG path")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_spatial_distribution(args.csv, args.output)
