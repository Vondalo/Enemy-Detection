"""
Dataset Positional Bias Analysis Tool
====================================
Analyzes enemy position distribution to detect if the dataset suffers from
center-screen bias which could lead to the model learning positional shortcuts.

Usage: python analyze_positional_bias.py --csv src/dataset/uncleaned/labels.csv
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os
from scipy.stats import gaussian_kde


def load_coordinates(csv_path):
    """Load normalized coordinates from CSV."""
    x_coords = []
    y_coords = []
    filenames = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_coords.append(float(row['x_norm']))
            y_coords.append(float(row['y_norm']))
            filenames.append(row['filename'])

    return np.array(x_coords), np.array(y_coords), filenames


def create_screen_canvas():
    """Create a blank 2D canvas representing the screen (0-1 normalized)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('X (normalized 0-1)', fontsize=12)
    ax.set_ylabel('Y (normalized 0-1)', fontsize=12)
    ax.set_title('Enemy Position Distribution on Screen', fontsize=14, fontweight='bold')
    return fig, ax


def plot_scatter(x, y, save_path=None):
    """Create scatter plot of all enemy positions."""
    fig, ax = create_screen_canvas()

    # Plot points with slight transparency to show density
    ax.scatter(x, y, alpha=0.4, s=20, c='blue', edgecolors='none')

    # Add center crosshair marker
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Add center region box (middle 20% of screen)
    center_box = patches.Rectangle((0.4, 0.4), 0.2, 0.2,
                                   linewidth=2, edgecolor='red',
                                   facecolor='red', alpha=0.1, linestyle='--')
    ax.add_patch(center_box)
    ax.text(0.5, 0.42, 'Center 20%\n(crosshair region)', ha='center', va='bottom',
            fontsize=9, color='red', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Scatter plot saved to {save_path}")
    return fig


def plot_white_background(x, y, save_path=None):
    """Create a blank white image with only the position points shown."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Set limits (normalized 0-1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove all axes, labels, and spines
    ax.axis('off')

    # Plot points in red for visibility
    ax.scatter(x, y, alpha=0.5, s=15, c='red', edgecolors='none')

    # Add subtle center marker (very faint)
    ax.axhline(y=0.5, color='lightgray', linestyle='--', alpha=0.2, linewidth=0.5)
    ax.axvline(x=0.5, color='lightgray', linestyle='--', alpha=0.2, linewidth=0.5)

    plt.tight_layout(pad=0)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
        print(f"[Plot] White background plot saved to {save_path}")
    return fig


def plot_heatmap(x, y, save_path=None, bins=50):
    """Create 2D heatmap/density map of enemy positions."""
    fig, ax = create_screen_canvas()

    # Create 2D histogram/heatmap
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]])

    # Smooth with gaussian filter for better visualization
    from scipy.ndimage import gaussian_filter
    heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)

    # Create custom colormap (dark blue to yellow/red)
    colors = ['#000033', '#000055', '#0000ff', '#0055ff', '#00ffff', '#55ff00', '#ffff00', '#ff5500', '#ff0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(heatmap_smooth.T, extent=extent, origin='lower', cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Enemy Density', rotation=270, labelpad=20)

    # Add center markers
    ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Heatmap saved to {save_path}")
    return fig


def plot_density_comparison(x, y, save_path=None):
    """Create side-by-side comparison: scatter vs contour density."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Scatter with density coloring
    ax1 = axes[0]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (normalized 0-1)', fontsize=11)
    ax1.set_ylabel('Y (normalized 0-1)', fontsize=11)
    ax1.set_title('Scatter Plot (colored by density)', fontsize=12, fontweight='bold')

    # Calculate point density for coloring
    xy = np.vstack([x, y])
    try:
        density = gaussian_kde(xy)(xy)
        scatter = ax1.scatter(x, y, c=density, s=25, cmap='YlOrRd', alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04, label='Density')
    except:
        ax1.scatter(x, y, alpha=0.4, s=20, c='blue', edgecolors='none')

    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)

    # Right: Contour density plot
    ax2 = axes[1]
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (normalized 0-1)', fontsize=11)
    ax2.set_ylabel('Y (normalized 0-1)', fontsize=11)
    ax2.set_title('Density Contours', fontsize=12, fontweight='bold')

    # Create density contours
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[0:1:100j, 0:1:100j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        levels = np.linspace(zi.min(), zi.max(), 10)
        ax2.contourf(xi, yi, zi, levels=levels, cmap='YlOrRd', alpha=0.7)
        ax2.contour(xi, yi, zi, levels=levels[::2], colors='black', alpha=0.3, linewidths=0.5)
    except:
        ax2.hist2d(x, y, bins=30, cmap='YlOrRd')

    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Density comparison saved to {save_path}")
    return fig


def analyze_distribution(x, y):
    """Quantify the distribution of points across screen regions."""
    results = {}
    n = len(x)

    # Define regions
    # Center 20% (crosshair region)
    center_mask = (x >= 0.4) & (x <= 0.6) & (y >= 0.4) & (y <= 0.6)
    results['center_20pct'] = {
        'count': np.sum(center_mask),
        'percentage': np.sum(center_mask) / n * 100
    }

    # Center 30% (expanded crosshair region)
    center_30_mask = (x >= 0.35) & (x <= 0.65) & (y >= 0.35) & (y <= 0.65)
    results['center_30pct'] = {
        'count': np.sum(center_30_mask),
        'percentage': np.sum(center_30_mask) / n * 100
    }

    # Screen quadrants
    top_left = (x <= 0.5) & (y >= 0.5)
    top_right = (x >= 0.5) & (y >= 0.5)
    bottom_left = (x <= 0.5) & (y <= 0.5)
    bottom_right = (x >= 0.5) & (y <= 0.5)

    results['quadrants'] = {
        'top_left': np.sum(top_left) / n * 100,
        'top_right': np.sum(top_right) / n * 100,
        'bottom_left': np.sum(bottom_left) / n * 100,
        'bottom_right': np.sum(bottom_right) / n * 100
    }

    # Edges (within 10% of any edge)
    edge_margin = 0.1
    edges_mask = (x <= edge_margin) | (x >= 1 - edge_margin) | (y <= edge_margin) | (y >= 1 - edge_margin)
    results['edges_10pct'] = {
        'count': np.sum(edges_mask),
        'percentage': np.sum(edges_mask) / n * 100
    }

    # Corners (within 15% of any corner)
    corner_margin = 0.15
    corners_mask = ((x <= corner_margin) | (x >= 1 - corner_margin)) & ((y <= corner_margin) | (y >= 1 - corner_margin))
    results['corners_15pct'] = {
        'count': np.sum(corners_mask),
        'percentage': np.sum(corners_mask) / n * 100
    }

    # Mean and std
    results['stats'] = {
        'x_mean': np.mean(x),
        'x_std': np.std(x),
        'y_mean': np.mean(y),
        'y_std': np.std(y),
        'total_samples': n
    }

    return results


def print_analysis_report(results):
    """Print detailed analysis report."""
    print("\n" + "=" * 60)
    print("  DATASET POSITIONAL BIAS ANALYSIS REPORT")
    print("=" * 60)

    stats = results['stats']
    print(f"\n📊 DATASET OVERVIEW")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   X distribution: mean={stats['x_mean']:.4f}, std={stats['x_std']:.4f}")
    print(f"   Y distribution: mean={stats['y_mean']:.4f}, std={stats['y_std']:.4f}")

    print(f"\n🎯 CENTER REGION ANALYSIS")
    print(f"   Center 20% (crosshair): {results['center_20pct']['count']} samples ({results['center_20pct']['percentage']:.1f}%)")
    print(f"   Center 30% (expanded):  {results['center_30pct']['count']} samples ({results['center_30pct']['percentage']:.1f}%)")

    print(f"\n📐 QUADRANT DISTRIBUTION")
    q = results['quadrants']
    print(f"   Top-Left:     {q['top_left']:.1f}%")
    print(f"   Top-Right:    {q['top_right']:.1f}%")
    print(f"   Bottom-Left:  {q['bottom_left']:.1f}%")
    print(f"   Bottom-Right: {q['bottom_right']:.1f}%")

    print(f"\n🔲 EDGE & CORNER COVERAGE")
    print(f"   Edges (within 10%):   {results['edges_10pct']['count']} samples ({results['edges_10pct']['percentage']:.1f}%)")
    print(f"   Corners (within 15%): {results['corners_15pct']['count']} samples ({results['corners_15pct']['percentage']:.1f}%)")

    print("\n" + "=" * 60)
    print("  BIAS ASSESSMENT")
    print("=" * 60)

    # Determine bias level
    center_20_pct = results['center_20pct']['percentage']
    center_30_pct = results['center_30pct']['percentage']
    edge_pct = results['edges_10pct']['percentage']
    corner_pct = results['corners_15pct']['percentage']

    # Expected values for uniform distribution
    expected_center_20 = 4.0  # 20% * 20% = 4% of area
    expected_center_30 = 9.0  # 30% * 30% = 9% of area
    expected_edges = 36.0     # ~36% of perimeter area
    expected_corners = 9.0    # ~9% of corner area

    print(f"\n📈 COMPARISON TO UNIFORM DISTRIBUTION")
    print(f"   Center 20%: {center_20_pct:.1f}% actual vs {expected_center_20:.1f}% expected")
    print(f"   Center 30%: {center_30_pct:.1f}% actual vs {expected_center_30:.1f}% expected")
    print(f"   Edges:      {edge_pct:.1f}% actual vs {expected_edges:.1f}% expected")
    print(f"   Corners:    {corner_pct:.1f}% actual vs {expected_corners:.1f}% expected")

    # Bias calculation
    center_bias_ratio = center_20_pct / expected_center_20

    print(f"\n🔍 BIAS METRIC")
    print(f"   Center bias ratio: {center_bias_ratio:.2f}x")
    print(f"   (Values > 2.0 indicate strong center bias)")

    print("\n" + "=" * 60)
    print("  CONCLUSION")
    print("=" * 60)

    if center_bias_ratio > 4.0:
        bias_level = "🔴 SEVERE BIAS"
        conclusion = """The dataset shows SEVERE center-screen bias. The model will likely learn
to predict enemies near the center regardless of visual features.
This is a critical issue that will harm generalization significantly."""
    elif center_bias_ratio > 2.5:
        bias_level = "🟠 MODERATE BIAS"
        conclusion = """The dataset shows MODERATE center-screen bias. The model may develop
a tendency to predict enemies near the center, potentially affecting
accuracy for edge/corner enemies."""
    elif center_bias_ratio > 1.5:
        bias_level = "🟡 MILD BIAS"
        conclusion = """The dataset shows MILD center-screen bias. While not severe, the model
may still benefit from data augmentation or additional edge/corner samples."""
    else:
        bias_level = "🟢 WELL BALANCED"
        conclusion = """The dataset appears well-balanced with good coverage across the screen.
No significant positional bias detected."""

    print(f"\n{bias_level}")
    print(f"\n{conclusion}")

    print("\n" + "=" * 60)


def save_report(results, save_path):
    """Save analysis report to text file."""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  DATASET POSITIONAL BIAS ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        stats = results['stats']
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"X distribution: mean={stats['x_mean']:.4f}, std={stats['x_std']:.4f}\n")
        f.write(f"Y distribution: mean={stats['y_mean']:.4f}, std={stats['y_std']:.4f}\n\n")

        f.write(f"Center 20% (crosshair): {results['center_20pct']['count']} samples ({results['center_20pct']['percentage']:.1f}%)\n")
        f.write(f"Center 30% (expanded):  {results['center_30pct']['count']} samples ({results['center_30pct']['percentage']:.1f}%)\n\n")

        q = results['quadrants']
        f.write(f"Quadrant distribution:\n")
        f.write(f"  Top-Left:     {q['top_left']:.1f}%\n")
        f.write(f"  Top-Right:    {q['top_right']:.1f}%\n")
        f.write(f"  Bottom-Left:  {q['bottom_left']:.1f}%\n")
        f.write(f"  Bottom-Right: {q['bottom_right']:.1f}%\n\n")

        f.write(f"Edges (within 10%):   {results['edges_10pct']['count']} samples ({results['edges_10pct']['percentage']:.1f}%)\n")
        f.write(f"Corners (within 15%): {results['corners_15pct']['count']} samples ({results['corners_15pct']['percentage']:.1f}%)\n\n")

        center_bias_ratio = results['center_20pct']['percentage'] / 4.0
        f.write(f"Center bias ratio: {center_bias_ratio:.2f}x\n")

        if center_bias_ratio > 4.0:
            f.write("\nConclusion: SEVERE BIAS - Dataset needs rebalancing\n")
        elif center_bias_ratio > 2.5:
            f.write("\nConclusion: MODERATE BIAS - Consider data augmentation\n")
        elif center_bias_ratio > 1.5:
            f.write("\nConclusion: MILD BIAS - Minor improvements recommended\n")
        else:
            f.write("\nConclusion: WELL BALANCED - No significant issues\n")

    print(f"[Report] Analysis saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze positional bias in enemy detection dataset")
    parser.add_argument("--csv", type=str, default="src/dataset/uncleaned/labels.csv",
                        help="Path to labels CSV file")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                        help="Output directory for visualizations and report")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"[Loading] Reading coordinates from {args.csv}...")
    x, y, filenames = load_coordinates(args.csv)
    print(f"[Loaded] {len(x)} samples")

    # Generate visualizations
    print("[Generating] Creating visualizations...")

    plot_scatter(x, y, save_path=os.path.join(args.output_dir, "scatter_plot.png"))
    plot_heatmap(x, y, save_path=os.path.join(args.output_dir, "heatmap.png"))
    plot_density_comparison(x, y, save_path=os.path.join(args.output_dir, "density_comparison.png"))
    plot_white_background(x, y, save_path=os.path.join(args.output_dir, "white_background.png"))

    # Analyze distribution
    print("[Analyzing] Computing distribution statistics...")
    results = analyze_distribution(x, y)

    # Print and save report
    print_analysis_report(results)
    save_report(results, os.path.join(args.output_dir, "bias_analysis_report.txt"))

    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
