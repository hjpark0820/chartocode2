"""
Synthetic Chart Generator v2 - Improved Realism
Generates charts that better match the visual style of patent/scientific paper figures.

Key improvements over v1:
- Grayscale/low-contrast styling matching patent figures
- Varied marker sizes (smaller, matching real chart proportions)
- Thicker marker edges for open symbols
- Background noise and scan-like artifacts
- Font styling matching patent figures
- Variable DPI and image resolution
- Varied line styles connecting data points
"""

import os
import argparse
from pathlib import Path
import json
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageEnhance

# Define your output directory
output_dir = r"C:\Users\ziola\OneDrive\documents\projects\chart-data-extractor\src\data\synthetic_v2\train\images"

# This will create 'images' and any parent folders (train, synthetic_v2, etc.) if they don't exist
os.makedirs(output_dir, exist_ok=True)

@dataclass
class SymbolDef:
    name: str
    matplotlib_marker: str
    filled: bool
    class_id: int


SYMBOL_DEFS = [
    SymbolDef("filled_circle",   "o", True,  0),
    SymbolDef("open_circle",     "o", False, 1),
    SymbolDef("filled_square",   "s", True,  2),
    SymbolDef("open_square",     "s", False, 3),
    SymbolDef("filled_triangle", "^", True,  4),
    SymbolDef("open_triangle",   "^", False, 5),
    SymbolDef("filled_diamond",  "D", True,  6),
    SymbolDef("open_diamond",    "D", False, 7),
]


@dataclass
class DetectionAnnotation:
    bbox: List[float]
    category_id: int
    center_px: Tuple[float, float]
    real_x: float
    real_y: float
    series_id: int


def generate_series_data(n_points, x_range, curve_type, noise_std, log_x):
    """Generate data points for one series."""
    if log_x:
        x = np.logspace(np.log10(max(x_range[0], 1e-15)), np.log10(x_range[1]), n_points)
    else:
        x = np.linspace(x_range[0], x_range[1], n_points)

    if curve_type == "sigmoid":
        mid = random.uniform(0.3, 0.7) * (x_range[1] - x_range[0]) + x_range[0]
        if log_x:
            mid = 10 ** (random.uniform(0.3, 0.7) * (np.log10(x_range[1]) - np.log10(max(x_range[0], 1e-15))) + np.log10(max(x_range[0], 1e-15)))
            t = (np.log10(x) - np.log10(mid)) / (0.15 * (np.log10(x_range[1]) - np.log10(max(x_range[0], 1e-15))))
        else:
            t = (x - mid) / (0.15 * (x_range[1] - x_range[0]))
        y_base = random.uniform(0.08, 0.15)
        y_top = random.uniform(0.28, 0.38)
        y = y_base + (y_top - y_base) / (1 + np.exp(-t))
    elif curve_type == "linear":
        slope = random.uniform(-0.15, 0.25)
        intercept = random.uniform(0.08, 0.20)
        y = slope * (x - x_range[0]) / (x_range[1] - x_range[0]) + intercept
    elif curve_type == "exponential":
        rate = random.uniform(2, 8)
        y = 0.1 + 0.25 * (1 - np.exp(-rate * (x - x_range[0]) / (x_range[1] - x_range[0])))
    else:
        y = np.random.uniform(0.08, 0.35, n_points)

    y = y + np.random.normal(0, noise_std, len(y))
    y = np.clip(y, 0.05, 0.40)
    return x, y


def apply_scan_artifacts(img_path, intensity="medium"):
    """Apply scan-like artifacts to make the image look like a scanned patent figure."""
    img = Image.open(img_path)

    # Convert to grayscale
    img_gray = img.convert('L')

    # Adjust contrast (patent figures tend to be lower contrast)
    contrast_factor = random.uniform(0.7, 0.95) if intensity != "low" else random.uniform(0.9, 1.0)
    enhancer = ImageEnhance.Contrast(img_gray)
    img_gray = enhancer.enhance(contrast_factor)

    # Add slight Gaussian noise (like scan noise)
    arr = np.array(img_gray).astype(np.float32)
    noise_level = {"low": 2, "medium": 4, "high": 8}.get(intensity, 4)
    noise = np.random.normal(0, noise_level, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    # Slight blur (like print/scan degradation)
    img_out = Image.fromarray(arr)
    if random.random() < 0.3:
        img_out = img_out.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Slight brightness variation
    enhancer = ImageEnhance.Brightness(img_out)
    img_out = enhancer.enhance(random.uniform(0.95, 1.05))

    # Convert back to RGB for consistency
    img_out = img_out.convert('RGB')
    img_out.save(img_path)
    return img_out


def generate_chart_v2(
    chart_id: int,
    output_dir: str,
    n_series: int = 4,
    n_points_range: Tuple[int, int] = (8, 16),
    img_size: Tuple[int, int] = (800, 600),
    dpi: int = 100,
    log_x: bool = True,
    symbol_size: float = 7.0,
    line_width: float = 1.0,
    overlap_intensity: str = "medium",
    style: str = "patent",
) -> Tuple[str, List[DetectionAnnotation], int, int]:
    """Generate one chart with improved realism."""
    fig_w = img_size[0] / dpi
    fig_h = img_size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Configure axes
    if log_x:
        ax.set_xscale('log')
        x_range = (1e-15, 1e-7)
    else:
        x_range = (0, 10)

    ax.set_ylim(0.05, 0.40)
    ax.set_xlim(x_range)

    # Select symbols for each series
    if n_series <= len(SYMBOL_DEFS):
        series_symbols = random.sample(SYMBOL_DEFS, n_series)
    else:
        series_symbols = [random.choice(SYMBOL_DEFS) for _ in range(n_series)]

    curve_types = ["sigmoid", "linear", "exponential", "flat", "random"]
    noise_map = {"low": 0.005, "medium": 0.015, "high": 0.03}
    noise_std = noise_map.get(overlap_intensity, 0.015)

    annotations: List[DetectionAnnotation] = []
    all_series_data = []

    # Style parameters based on mode
    if style == "patent":
        # Patent-like styling
        marker_edge_width = random.uniform(1.0, 2.0)
        line_color = random.choice(['black', '#333333', '#444444'])
        bg_color = random.choice(['white', '#FAFAFA', '#F5F5F5'])
        font_family = random.choice(['serif', 'Times New Roman', 'DejaVu Serif'])
        font_size_label = random.randint(9, 12)
        font_size_tick = random.randint(7, 10)
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)
    else:
        marker_edge_width = 1.2
        line_color = 'black'
        font_family = 'sans-serif'
        font_size_label = 10
        font_size_tick = 8

    # Line style variation
    line_styles = ['-', '-', '-', '--', '-.']

    for s_idx in range(n_series):
        sym = series_symbols[s_idx]
        n_pts = random.randint(*n_points_range)
        curve_type = random.choice(curve_types[:3])

        x_data, y_data = generate_series_data(
            n_pts, x_range, curve_type, noise_std, log_x
        )
        all_series_data.append((x_data, y_data, sym, s_idx))

        # Marker style
        facecolor = 'black' if sym.filled else 'white'
        edgecolor = line_color

        # Open symbols need thicker edges to be visible
        mew = marker_edge_width * (1.3 if not sym.filled else 1.0)

        ls = random.choice(line_styles)

        ax.plot(
            x_data, y_data,
            marker=sym.matplotlib_marker,
            markersize=symbol_size,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markeredgewidth=mew,
            linestyle=ls,
            linewidth=line_width,
            color=line_color,
            zorder=2 + s_idx,
        )

    # Axis styling
    ax.set_xlabel(random.choice([
        "Concentration, M", "Antibody Concentration, M",
        "Dose (mg/kg)", "Time (hours)", "X Axis",
        "Concentration (nM)", "Log Concentration",
    ]), fontsize=font_size_label, fontfamily=font_family)

    ax.set_ylabel(random.choice([
        "OD 570/630 nm", "Absorbance", "Response",
        "% Inhibition", "Binding (AU)", "Y Axis",
    ]), fontsize=font_size_label, fontfamily=font_family)

    ax.tick_params(labelsize=font_size_tick)

    # Grid: sometimes present, sometimes not
    if random.random() < 0.3:
        ax.grid(True, alpha=0.2, linewidth=0.5)
    else:
        ax.grid(False)

    # Legend with varied placement
    legend_labels = [f"Series {i}" for i in range(n_series)]
    legend_loc = random.choice(['best', 'upper left', 'lower right', 'upper right'])
    if random.random() < 0.7:
        ax.legend(legend_labels, fontsize=max(6, font_size_tick - 1),
                  loc=legend_loc, framealpha=0.8)

    # Title (sometimes)
    if random.random() < 0.3:
        fig_num = random.randint(1, 15)
        ax.set_title(f"FIG. {fig_num}", fontsize=font_size_label + 2,
                     fontfamily=font_family, fontweight='bold')

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(random.uniform(0.8, 1.5))

    fig.tight_layout()

    # Extract pixel coordinates
    fig.canvas.draw()
    marker_radius_px = symbol_size * dpi / 72 / 2

    for x_data, y_data, sym, s_idx in all_series_data:
        for i in range(len(x_data)):
            display_coords = ax.transData.transform((x_data[i], y_data[i]))
            px_x = display_coords[0]
            px_y = img_size[1] - display_coords[1]

            bbox_size = marker_radius_px * 2 + 2
            bbox = [
                px_x - bbox_size / 2,
                px_y - bbox_size / 2,
                bbox_size,
                bbox_size,
            ]

            annotations.append(DetectionAnnotation(
                bbox=bbox,
                category_id=sym.class_id,
                center_px=(px_x, px_y),
                real_x=float(x_data[i]),
                real_y=float(y_data[i]),
                series_id=s_idx,
            ))

    # Save image
    img_path = os.path.join(output_dir, "images", f"chart_{chart_id:05d}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    fig.savefig(img_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    # Apply scan artifacts for realism
    if style == "patent":
        apply_scan_artifacts(img_path, intensity=random.choice(["low", "medium"]))

    # Get actual size
    saved_img = Image.open(img_path)
    actual_w, actual_h = saved_img.size
    saved_img.close()

    return img_path, annotations, actual_w, actual_h


def generate_dataset_v2(
    output_dir: str,
    n_charts: int = 100,
    seed: int = 123,
) -> dict:
    """Generate dataset with improved realism."""
    random.seed(seed)
    np.random.seed(seed)
    
    images_path = os.path.join(output_dir, "images")
    os.makedirs(images_path, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": s.class_id, "name": s.name}
            for s in SYMBOL_DEFS
        ],
    }

    ann_id = 0

    for chart_id in tqdm(range(n_charts), desc="Generating v2 charts"):
        n_series = random.choice([2, 3, 4, 4, 4, 5, 6])
        n_points_range = (random.randint(6, 10), random.randint(11, 18))
        log_x = random.choice([True, True, True, False])  # bias toward log (patent style)

        # More varied symbol sizes (real charts have smaller markers)
        symbol_size = random.uniform(4.0, 8.0)
        line_width = random.uniform(0.6, 1.5)
        overlap = random.choice(["low", "medium", "medium", "high"])
        dpi = random.choice([100, 100, 120, 150])
        style = random.choice(["patent", "patent", "patent", "default"])

        img_path, anns, img_w, img_h = generate_chart_v2(
            chart_id=chart_id,
            output_dir=output_dir,
            n_series=n_series,
            n_points_range=n_points_range,
            img_size=(800, 600),
            dpi=dpi,
            log_x=log_x,
            symbol_size=symbol_size,
            line_width=line_width,
            overlap_intensity=overlap,
            style=style,
        )

        coco["images"].append({
            "id": chart_id,
            "file_name": os.path.basename(img_path),
            "width": img_w,
            "height": img_h,
        })

        for ann in anns:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": chart_id,
                "category_id": ann.category_id,
                "bbox": [round(v, 2) for v in ann.bbox],
                "area": round(ann.bbox[2] * ann.bbox[3], 2),
                "center_px": [round(ann.center_px[0], 2), round(ann.center_px[1], 2)],
                "real_x": ann.real_x,
                "real_y": ann.real_y,
                "series_id": ann.series_id,
                "iscrowd": 0,
            })
            ann_id += 1

    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"\nv2 Dataset generated:")
    print(f"  Charts: {n_charts}")
    print(f"  Total annotations: {ann_id}")
    print(f"  Images: {os.path.join(output_dir, 'images')}")
    print(f"  Annotations: {ann_path}")

    return coco


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="C:\\Users\\ziola\\OneDrive\\documents\\projects\\chart-data-extractor\\src\\data\\synthetic_v2\\train")	
    parser.add_argument("--n-charts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    generate_dataset_v2(args.output, args.n_charts, args.seed)
