"""
Synthetic Scientific Chart Generator
Generates scatter/line plots with ground truth annotations for training
symbol detectors and classifiers.

Output:
  - Chart images (PNG)
  - COCO-format annotations (JSON) for object detection
  - Data tables (CSV) with real-value coordinates
"""

import os
import json
import random
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.markers as mmarkers
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm


# ─── Symbol Definitions ────────────────────────────────────────────────

@dataclass
class SymbolDef:
    """A marker symbol definition."""
    name: str
    matplotlib_marker: str
    filled: bool  # True = filled, False = open/hollow
    class_id: int


SYMBOL_DEFS = [
    SymbolDef("filled_circle",  "o", True,  0),
    SymbolDef("open_circle",    "o", False, 1),
    SymbolDef("filled_square",  "s", True,  2),
    SymbolDef("open_square",    "s", False, 3),
    SymbolDef("filled_triangle","^", True,  4),
    SymbolDef("open_triangle",  "^", False, 5),
    SymbolDef("filled_diamond", "D", True,  6),
    SymbolDef("open_diamond",   "D", False, 7),
]

NUM_SYMBOL_CLASSES = len(SYMBOL_DEFS)


# ─── Data Series Generation ────────────────────────────────────────────

def generate_series_data(
    n_points: int,
    x_range: Tuple[float, float],
    curve_type: str = "sigmoid",
    noise_std: float = 0.02,
    log_x: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (x, y) data for one series."""
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
        y_top = random.uniform(0.25, 0.38)
        y = y_base + (y_top - y_base) / (1 + np.exp(-t))
    elif curve_type == "linear":
        slope = random.uniform(-0.3, 0.3)
        intercept = random.uniform(0.1, 0.3)
        y = slope * (x - x_range[0]) / (x_range[1] - x_range[0]) + intercept
    elif curve_type == "exponential":
        rate = random.uniform(0.5, 3.0)
        y = 0.1 + 0.25 * (1 - np.exp(-rate * (x - x_range[0]) / (x_range[1] - x_range[0])))
    elif curve_type == "flat":
        y = np.full_like(x, random.uniform(0.08, 0.35))
    else:
        y = np.random.uniform(0.08, 0.35, size=n_points)

    # Add noise
    y += np.random.normal(0, noise_std, size=n_points)
    y = np.clip(y, 0.05, 0.40)

    return x, y


# ─── Chart Generation ───────────────────────────────────────────────────

@dataclass
class DetectionAnnotation:
    """COCO-format annotation for one detected symbol."""
    bbox: List[float]       # [x, y, w, h] in pixels
    category_id: int
    center_px: Tuple[float, float]  # pixel center
    real_x: float           # real data value
    real_y: float
    series_id: int


def generate_chart(
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
) -> Tuple[str, List[DetectionAnnotation]]:
    """
    Generate one synthetic chart with ground truth annotations.

    Returns:
        (image_path, list_of_annotations)
    """
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
    annotations: List[DetectionAnnotation] = []

    # Control overlap by adjusting noise and spacing
    noise_map = {"low": 0.005, "medium": 0.015, "high": 0.03}
    noise_std = noise_map.get(overlap_intensity, 0.015)

    all_series_data = []

    for s_idx in range(n_series):
        sym = series_symbols[s_idx]
        n_pts = random.randint(*n_points_range)
        curve_type = random.choice(curve_types[:3])  # sigmoid, linear, exp

        x_data, y_data = generate_series_data(
            n_pts, x_range, curve_type, noise_std, log_x
        )
        all_series_data.append((x_data, y_data, sym, s_idx))

        # Marker style
        facecolor = 'black' if sym.filled else 'white'
        edgecolor = 'black'

        ax.plot(
            x_data, y_data,
            marker=sym.matplotlib_marker,
            markersize=symbol_size,
            markerfacecolor=facecolor,
            markeredgecolor=edgecolor,
            markeredgewidth=1.2,
            linestyle='-',
            linewidth=line_width,
            color='black',
            zorder=2 + s_idx,
        )

    # Style the plot
    ax.set_xlabel("X Axis", fontsize=10)
    ax.set_ylabel("Y Axis", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(False)

    # Add legend
    legend_labels = [f"Series {i} ({s.name})" for i, (_, _, s, _) in enumerate(all_series_data)]
    ax.legend(legend_labels, fontsize=7, loc='best')

    # Tight layout
    fig.tight_layout()

    # Now extract pixel coordinates for each data point
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Compute the marker radius in pixels (approximate)
    # markersize is in points; 1 point = dpi/72 pixels
    marker_radius_px = symbol_size * dpi / 72 / 2

    for x_data, y_data, sym, s_idx in all_series_data:
        for i in range(len(x_data)):
            # Transform data coordinates to pixel (display) coordinates
            display_coords = ax.transData.transform((x_data[i], y_data[i]))
            px_x = display_coords[0]
            px_y = img_size[1] - display_coords[1]  # flip y for image coords

            # Bounding box: [x, y, w, h] centered on the marker
            bbox_size = marker_radius_px * 2 + 2  # small padding
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

    # Re-read saved image to get actual size (tight_layout may change it)
    from PIL import Image
    saved_img = Image.open(img_path)
    actual_w, actual_h = saved_img.size
    saved_img.close()

    # Adjust annotations for actual image size vs intended size
    # The fig.savefig with bbox_inches='tight' may crop differently
    # We need to re-transform. Let's regenerate with fixed size instead.
    # For simplicity, store the actual image dimensions with annotations.

    return img_path, annotations, actual_w, actual_h


def generate_dataset(
    output_dir: str,
    n_charts: int = 100,
    seed: int = 42,
) -> dict:
    """
    Generate a full dataset of synthetic charts with COCO annotations.

    Returns:
        COCO-format annotation dict
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # COCO annotation structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": s.class_id, "name": s.name}
            for s in SYMBOL_DEFS
        ],
    }

    ann_id = 0

    for chart_id in tqdm(range(n_charts), desc="Generating charts"):
        # Randomize chart parameters
        n_series = random.choice([2, 3, 4, 4, 4, 5, 6])
        n_points_range = (random.randint(6, 10), random.randint(11, 18))
        log_x = random.choice([True, True, False])  # bias toward log
        symbol_size = random.uniform(5.0, 9.0)
        overlap = random.choice(["low", "medium", "medium", "high"])
        dpi = random.choice([100, 100, 150])

        img_path, anns, img_w, img_h = generate_chart(
            chart_id=chart_id,
            output_dir=output_dir,
            n_series=n_series,
            n_points_range=n_points_range,
            img_size=(800, 600),
            dpi=dpi,
            log_x=log_x,
            symbol_size=symbol_size,
            overlap_intensity=overlap,
        )

        # Add image info
        coco["images"].append({
            "id": chart_id,
            "file_name": os.path.basename(img_path),
            "width": img_w,
            "height": img_h,
        })

        # Add annotations
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

    # Save annotations
    ann_path = os.path.join(output_dir, "annotations.json")
    with open(ann_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"\nDataset generated:")
    print(f"  Charts: {n_charts}")
    print(f"  Total annotations: {ann_id}")
    print(f"  Images: {os.path.join(output_dir, 'images')}")
    print(f"  Annotations: {ann_path}")

    return coco


# ─── Patch Extraction for Classification ────────────────────────────────

def extract_classification_patches(
    output_dir: str,
    coco_annotations: dict,
    images_dir: str,
    patch_size: int = 32,
    n_negative_per_image: int = 10,
    n_offset_per_symbol: int = 2,
) -> dict:
    """
    Extract patches for symbol classification training.

    Three types:
    1. Positive: centered on each symbol (< 2px offset)
    2. Offset-negative: 3-5px off-center (labeled as background)
    3. Random background: random crops without symbols
    """
    from PIL import Image

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    # Create class directories
    for sym in SYMBOL_DEFS:
        os.makedirs(os.path.join(patches_dir, sym.name), exist_ok=True)
    os.makedirs(os.path.join(patches_dir, "background"), exist_ok=True)

    patch_metadata = []
    patch_id = 0
    half = patch_size // 2

    # Group annotations by image
    img_anns = {}
    for ann in coco_annotations["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    for img_info in tqdm(coco_annotations["images"], desc="Extracting patches"):
        img_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        img = Image.open(img_path).convert('L')  # grayscale
        img_arr = np.array(img)
        h, w = img_arr.shape

        anns = img_anns.get(img_id, [])
        symbol_centers = []

        for ann in anns:
            cx, cy = ann["center_px"]
            cx, cy = int(round(cx)), int(round(cy))
            symbol_centers.append((cx, cy))

            # 1. Positive patch
            if half <= cx < w - half and half <= cy < h - half:
                patch = img_arr[cy - half:cy + half, cy - half:cy + half]
                # Actually should be img_arr[cy-half:cy+half, cx-half:cx+half]
                patch = img_arr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    cat_name = SYMBOL_DEFS[ann["category_id"]].name
                    fname = f"patch_{patch_id:06d}.png"
                    Image.fromarray(patch).save(
                        os.path.join(patches_dir, cat_name, fname)
                    )
                    patch_metadata.append({
                        "id": patch_id,
                        "file": f"{cat_name}/{fname}",
                        "class": cat_name,
                        "class_id": ann["category_id"],
                        "source_image": img_info["file_name"],
                    })
                    patch_id += 1

            # 2. Offset-negative patches
            for _ in range(n_offset_per_symbol):
                off_x = cx + random.choice([-1, 1]) * random.randint(3, 5)
                off_y = cy + random.choice([-1, 1]) * random.randint(3, 5)
                if half <= off_x < w - half and half <= off_y < h - half:
                    patch = img_arr[off_y - half:off_y + half, off_x - half:off_x + half]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        fname = f"patch_{patch_id:06d}.png"
                        Image.fromarray(patch).save(
                            os.path.join(patches_dir, "background", fname)
                        )
                        patch_metadata.append({
                            "id": patch_id,
                            "file": f"background/{fname}",
                            "class": "background",
                            "class_id": NUM_SYMBOL_CLASSES,
                            "source_image": img_info["file_name"],
                        })
                        patch_id += 1

        # 3. Random background patches
        for _ in range(n_negative_per_image):
            rx = random.randint(half, w - half - 1)
            ry = random.randint(half, h - half - 1)
            # Check not too close to any symbol
            too_close = any(
                abs(rx - sx) < patch_size and abs(ry - sy) < patch_size
                for sx, sy in symbol_centers
            )
            if not too_close:
                patch = img_arr[ry - half:ry + half, rx - half:rx + half]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    fname = f"patch_{patch_id:06d}.png"
                    Image.fromarray(patch).save(
                        os.path.join(patches_dir, "background", fname)
                    )
                    patch_metadata.append({
                        "id": patch_id,
                        "file": f"background/{fname}",
                        "class": "background",
                        "class_id": NUM_SYMBOL_CLASSES,
                        "source_image": img_info["file_name"],
                    })
                    patch_id += 1

    # Save metadata
    meta_path = os.path.join(output_dir, "patch_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(patch_metadata, f, indent=2)

    # Print stats
    class_counts = {}
    for p in patch_metadata:
        c = p["class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    print(f"\nPatches extracted: {patch_id}")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    return patch_metadata


# ─── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic chart dataset")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    parser.add_argument("--n-charts", type=int, default=100, help="Number of charts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--extract-patches", action="store_true", help="Also extract classification patches")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size for classification")
    args = parser.parse_args()

    coco = generate_dataset(args.output, args.n_charts, args.seed)

    if args.extract_patches:
        extract_classification_patches(
            args.output, coco,
            os.path.join(args.output, "images"),
            patch_size=args.patch_size,
        )
