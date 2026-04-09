"""
Coordinate Extraction Pipeline
Extracts real-value coordinates from detected symbol pixel positions.

Includes:
- Plotting area detection
- Axis tick label OCR
- Pixel-to-value coordinate mapping (linear and log scales)
- Post-processing with RANSAC and curve fitting
"""

import os
import json
import re
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class AxisInfo:
    """Information about a chart axis."""
    orientation: str       # "x" or "y"
    is_log: bool          # True if logarithmic scale
    tick_positions_px: List[float]  # pixel positions of ticks
    tick_values: List[float]        # numeric values at ticks
    axis_start_px: float  # pixel position of axis start
    axis_end_px: float    # pixel position of axis end
    label: str = ""


@dataclass
class PlottingArea:
    """Detected plotting area bounds."""
    x_min_px: float
    x_max_px: float
    y_min_px: float  # top (image coords)
    y_max_px: float  # bottom (image coords)


def detect_plotting_area(image_path: str) -> PlottingArea:
    """
    Detect the plotting area boundaries using edge detection.
    Returns pixel coordinates of the plot area.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w, *_ = img.shape

    # Use Hough line detection to find axis lines
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=w // 4, maxLineGap=10)

    if lines is None:
        # Fallback: assume standard margins
        return PlottingArea(
            x_min_px=w * 0.12,
            x_max_px=w * 0.88,
            y_min_px=h * 0.08,
            y_max_px=h * 0.82,
        )

    # Find horizontal and vertical lines
    horizontal = []
    vertical = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if angle < 0.15:  # near horizontal
            horizontal.append((y1, x1, x2, length))
        elif angle > 1.4:  # near vertical
            vertical.append((x1, y1, y2, length))

    # X-axis: lowest horizontal line in bottom half
    bottom_horiz = [l for l in horizontal if l[0] > h * 0.5]
    if bottom_horiz:
        x_axis = max(bottom_horiz, key=lambda l: l[0])
        y_max_px = x_axis[0]
        x_min_from_axis = min(x_axis[1], x_axis[2])
        x_max_from_axis = max(x_axis[1], x_axis[2])
    else:
        y_max_px = h * 0.82
        x_min_from_axis = w * 0.12
        x_max_from_axis = w * 0.88

    # Y-axis: leftmost vertical line in left half
    left_vert = [l for l in vertical if l[0] < w * 0.3]
    if left_vert:
        y_axis = min(left_vert, key=lambda l: l[0])
        x_min_px = y_axis[0]
        y_min_px = min(y_axis[1], y_axis[2])
    else:
        x_min_px = w * 0.12
        y_min_px = h * 0.08

    return PlottingArea(
        x_min_px=float(x_min_px),
        x_max_px=float(x_max_from_axis),
        y_min_px=float(y_min_px),
        y_max_px=float(y_max_px),
    )


def parse_tick_labels_manual(
    x_ticks: List[Tuple[float, str]] = None,
    y_ticks: List[Tuple[float, str]] = None,
) -> Tuple[AxisInfo, AxisInfo]:
    """
    Create axis info from manually specified tick labels.
    This is used when OCR is not available or for testing.

    Format: list of (pixel_position, value_string) tuples
    """
    def parse_value(s: str) -> float:
        s = s.strip()
        # Handle scientific notation like "10^-15" or "1e-15"
        if 'e' in s.lower():
            return float(s)
        if '^' in s:
            base, exp = s.split('^')
            return float(base) ** float(exp)
        if '×' in s or 'x' in s:
            parts = re.split(r'[×x]', s)
            return float(parts[0]) * float(parts[1].replace('10', '10'))
        return float(s)

    x_info = None
    y_info = None

    if x_ticks:
        positions = [t[0] for t in x_ticks]
        values = [parse_value(t[1]) for t in x_ticks]
        # Detect log scale
        if len(values) >= 2:
            ratios = [values[i+1] / values[i] if values[i] != 0 else 0
                      for i in range(len(values)-1)]
            is_log = all(abs(r - ratios[0]) < ratios[0] * 0.1
                        for r in ratios) and ratios[0] > 1.5
        else:
            is_log = False

        x_info = AxisInfo(
            orientation="x",
            is_log=is_log,
            tick_positions_px=positions,
            tick_values=values,
            axis_start_px=positions[0],
            axis_end_px=positions[-1],
        )

    if y_ticks:
        positions = [t[0] for t in y_ticks]
        values = [parse_value(t[1]) for t in y_ticks]
        is_log = False  # Y-axis is typically linear for these charts

        y_info = AxisInfo(
            orientation="y",
            is_log=is_log,
            tick_positions_px=positions,
            tick_values=values,
            axis_start_px=positions[0],
            axis_end_px=positions[-1],
        )

    return x_info, y_info


def _interp_sorted(pixel_pos: float, positions: np.ndarray, values: np.ndarray) -> float:
    """np.interp requires sorted xp. Sort positions and values together."""
    sort_idx = np.argsort(positions)
    return float(np.interp(pixel_pos, positions[sort_idx], values[sort_idx]))


def pixel_to_value(
    pixel_pos: float,
    axis: AxisInfo,
) -> float:
    """Convert a pixel coordinate to a real value using the axis mapping."""
    positions = np.array(axis.tick_positions_px)
    values = np.array(axis.tick_values)

    if axis.is_log:
        # For log scale: interpolate in log space
        log_values = np.log10(np.maximum(values, 1e-30))
        value_log = _interp_sorted(pixel_pos, positions, log_values)
        return 10 ** value_log
    else:
        return _interp_sorted(pixel_pos, positions, values)


def pixel_to_value_y(
    pixel_pos: float,
    axis: AxisInfo,
) -> float:
    """Convert Y pixel coordinate (inverted) to value."""
    # Y-axis is inverted in image coords: higher pixel = lower value
    # np.interp requires monotonically increasing xp, so sort first
    positions = np.array(axis.tick_positions_px)
    values = np.array(axis.tick_values)

    if axis.is_log:
        log_values = np.log10(np.maximum(values, 1e-30))
        value_log = _interp_sorted(pixel_pos, positions, log_values)
        return 10 ** value_log
    else:
        return _interp_sorted(pixel_pos, positions, values)


def extract_coordinates(
    detections: List[dict],
    x_axis: AxisInfo,
    y_axis: AxisInfo,
    plot_area: PlottingArea,
) -> List[dict]:
    """
    Convert pixel detections to real-value coordinates.

    Returns detections augmented with 'real_x' and 'real_y' fields.
    """
    results = []

    for det in detections:
        cx, cy = det["center_px"]

        # Check if detection is within plotting area
        if (cx < plot_area.x_min_px - 5 or cx > plot_area.x_max_px + 5 or
            cy < plot_area.y_min_px - 5 or cy > plot_area.y_max_px + 5):
            continue

        # Convert pixel to real value
        real_x = pixel_to_value(cx, x_axis)
        real_y = pixel_to_value_y(cy, y_axis)

        result = dict(det)
        result["real_x"] = real_x
        result["real_y"] = real_y
        results.append(result)

    return results


def group_by_series(
    detections: List[dict],
) -> Dict[str, List[dict]]:
    """Group detections by symbol class (proxy for data series)."""
    series = {}
    for det in detections:
        cls = det["final_class"]
        if cls not in series:
            series[cls] = []
        series[cls].append(det)

    # Sort each series by x coordinate
    for cls in series:
        series[cls].sort(key=lambda d: d["real_x"])

    return series


def ransac_filter(
    series_data: List[dict],
    residual_threshold: float = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Use RANSAC to filter outliers from a data series.
    Returns (inliers, outliers).

    Adaptive threshold: defaults to 10% of the Y range of the series.
    """
    if len(series_data) < 5:
        return series_data, []

    X = np.array([d["real_x"] for d in series_data]).reshape(-1, 1)
    y = np.array([d["real_y"] for d in series_data])

    # For log-scale X, work in log space
    if np.min(X) > 0 and np.max(X) / np.min(X) > 100:
        X_fit = np.log10(X)
    else:
        X_fit = X

    # Adaptive threshold: 10% of Y range or minimum 0.02
    if residual_threshold is None:
        y_range = np.max(y) - np.min(y)
        residual_threshold = max(0.02, y_range * 0.15)

    try:
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=100,
            random_state=42,
        )
        ransac.fit(X_fit, y)
        inlier_mask = ransac.inlier_mask_

        inliers = [d for d, m in zip(series_data, inlier_mask) if m]
        outliers = [d for d, m in zip(series_data, inlier_mask) if not m]
        return inliers, outliers
    except Exception:
        return series_data, []


def fit_smooth_curve(
    series_data: List[dict],
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a smooth curve through data points using polynomial or spline.
    Returns (x_smooth, y_smooth) for visualization.
    """
    if len(series_data) < 3:
        x = np.array([d["real_x"] for d in series_data])
        y = np.array([d["real_y"] for d in series_data])
        return x, y

    X = np.array([d["real_x"] for d in series_data])
    Y = np.array([d["real_y"] for d in series_data])

    # Sort by X
    sort_idx = np.argsort(X)
    X = X[sort_idx]
    Y = Y[sort_idx]

    # Work in log space for X if needed
    if np.min(X) > 0 and np.max(X) / np.min(X) > 100:
        X_fit = np.log10(X)
    else:
        X_fit = X

    # Fit polynomial (degree 3-5 depending on number of points)
    degree = min(5, len(X) - 1)
    try:
        coeffs = np.polyfit(X_fit, Y, degree)
        x_smooth_fit = np.linspace(X_fit.min(), X_fit.max(), n_points)
        y_smooth = np.polyval(coeffs, x_smooth_fit)

        if np.min(X) > 0 and np.max(X) / np.min(X) > 100:
            x_smooth = 10 ** x_smooth_fit
        else:
            x_smooth = x_smooth_fit

        return x_smooth, y_smooth
    except Exception:
        return X, Y


def reconstruct_and_compare(
    series_dict: Dict[str, List[dict]],
    x_axis: AxisInfo,
    output_path: str,
):
    """
    Reconstruct the chart from extracted data and save comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Marker styles
    style_map = {
        "filled_circle": ("o", "black", "black"),
        "open_circle": ("o", "white", "black"),
        "filled_square": ("s", "black", "black"),
        "open_square": ("s", "white", "black"),
        "filled_triangle": ("^", "black", "black"),
        "open_triangle": ("^", "white", "black"),
        "filled_diamond": ("D", "black", "black"),
        "open_diamond": ("D", "white", "black"),
    }

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan"]

    for i, (cls, points) in enumerate(series_dict.items()):
        if not points:
            continue

        x_vals = [p["real_x"] for p in points]
        y_vals = [p["real_y"] for p in points]

        marker, facecolor, edgecolor = style_map.get(cls, ("o", "black", "black"))
        color = colors[i % len(colors)]

        ax.plot(x_vals, y_vals,
                marker=marker,
                markersize=8,
                markerfacecolor=facecolor if facecolor == "black" else "white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                linestyle='-',
                linewidth=1,
                color=color,
                label=cls,
                zorder=2 + i)

    if x_axis.is_log:
        ax.set_xscale('log')

    ax.legend(fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Reconstructed Chart from Extracted Data")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Reconstructed chart saved to: {output_path}")


def run_full_extraction(
    image_path: str,
    detections: List[dict],
    x_ticks: List[Tuple[float, str]],
    y_ticks: List[Tuple[float, str]],
    output_dir: str,
) -> Dict[str, List[dict]]:
    """
    Run the complete coordinate extraction pipeline.

    Args:
        image_path: path to chart image
        detections: list of detected symbols from YOLO+CNN pipeline
        x_ticks: manually specified X axis ticks [(pixel_pos, value_string), ...]
        y_ticks: manually specified Y axis ticks [(pixel_pos, value_string), ...]
        output_dir: directory for output files
    """
    os.makedirs(output_dir, exist_ok=True)
    from pathlib import Path
    fname = Path(image_path).stem

    # 1. Detect plotting area
    print("1. Detecting plotting area...")
    plot_area = detect_plotting_area(image_path)
    print(f"   Plot area: x=[{plot_area.x_min_px:.0f}, {plot_area.x_max_px:.0f}], "
          f"y=[{plot_area.y_min_px:.0f}, {plot_area.y_max_px:.0f}]")

    # 2. Parse axis information
    print("2. Parsing axis information...")
    x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)
    print(f"   X-axis: {'log' if x_axis.is_log else 'linear'}, "
          f"range [{x_axis.tick_values[0]}, {x_axis.tick_values[-1]}]")
    print(f"   Y-axis: {'log' if y_axis.is_log else 'linear'}, "
          f"range [{y_axis.tick_values[0]}, {y_axis.tick_values[-1]}]")

    # 3. Extract coordinates
    print("3. Extracting coordinates...")
    coords = extract_coordinates(detections, x_axis, y_axis, plot_area)
    print(f"   {len(coords)} points with valid coordinates (out of {len(detections)} detections)")

    # 4. Group by series
    print("4. Grouping by data series...")
    series = group_by_series(coords)
    for cls, points in series.items():
        print(f"   {cls}: {len(points)} points")

    # 5. RANSAC filtering
    print("5. RANSAC outlier filtering...")
    filtered_series = {}
    all_outliers = []
    for cls, points in series.items():
        inliers, outliers = ransac_filter(points)
        filtered_series[cls] = inliers
        all_outliers.extend(outliers)
        if outliers:
            print(f"   {cls}: removed {len(outliers)} outliers")

    # 6. Reconstruct chart
    print("6. Reconstructing chart...")
    recon_path = os.path.join(output_dir, f"{fname}_reconstructed.png")
    reconstruct_and_compare(filtered_series, x_axis, recon_path)

    # 7. Save extracted data as CSV
    csv_path = os.path.join(output_dir, f"{fname}_extracted_data.csv")
    with open(csv_path, 'w') as f:
        f.write("series,symbol_type,real_x,real_y,pixel_x,pixel_y,det_conf\n")
        for cls, points in filtered_series.items():
            for p in points:
                f.write(f"{cls},{p['final_class']},{p['real_x']:.6e},"
                        f"{p['real_y']:.6f},{p['center_px'][0]:.1f},"
                        f"{p['center_px'][1]:.1f},{p['det_conf']:.3f}\n")

    print(f"   Data saved to: {csv_path}")

    # 8. Summary
    print(f"\n{'='*50}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*50}")
    total_points = sum(len(pts) for pts in filtered_series.values())
    print(f"Total data series: {len(filtered_series)}")
    print(f"Total data points: {total_points}")
    print(f"Outliers removed: {len(all_outliers)}")
    for cls, points in filtered_series.items():
        if points:
            x_range = (min(p["real_x"] for p in points), max(p["real_x"] for p in points))
            y_range = (min(p["real_y"] for p in points), max(p["real_y"] for p in points))
            print(f"\n  {cls} ({len(points)} points):")
            print(f"    X range: [{x_range[0]:.3e}, {x_range[1]:.3e}]")
            print(f"    Y range: [{y_range[0]:.4f}, {y_range[1]:.4f}]")

    return filtered_series


# ─── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Coordinate extractor module. Use run_full_extraction() for end-to-end extraction.")
    print("See run_full_pipeline.py for the complete pipeline.")
