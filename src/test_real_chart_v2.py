"""
Test v2 models on real FIG.3 with improved post-processing.

Key improvements over test_real_chart.py:
- Tighter plotting area bounds (exclude legend, axis labels)
- Higher confidence threshold for cleaner detections
- Proximity deduplication
- Minimum series size filtering
- Better RANSAC with polynomial fit support
"""

import os
import sys
import json
import gc
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from run_detection_pipeline import (
    load_models, detect_symbols, detect_symbols_sahi,
    visualize_detections,
)
from coordinate_extractor import (
    parse_tick_labels_manual, extract_coordinates,
    group_by_series, ransac_filter, reconstruct_and_compare,
    PlottingArea, AxisInfo,
)

PROJECT_DIR = "/Users/jason/MyProject/RA/Hongju"


def get_fig3_plot_area_manual():
    """
    Precisely determined plotting area for FIG.3 from axis line detection.

    Image: 1430x1009 pixels
    Y-axis line at x=441
    X-axis line at y=713
    X-axis ticks at: x = 440, 514, 588, 662, 735, 806, 880, 953, 1027 (spacing ~73-74px)
    Y-axis ticks at: y = 617(0.10), 524(0.15), 434(0.20), 342(0.25), 249(0.30), ~157(0.35)

    Plot area (data region, excluding axes and legend):
    """
    return PlottingArea(
        x_min_px=445,    # Just right of Y-axis line (x=441)
        x_max_px=1025,   # Just left of last X tick (x=1027)
        y_min_px=155,    # Just below 0.35 tick
        y_max_px=620,    # Just above 0.10 tick / X-axis
    )


def get_fig3_axis_info():
    """
    Get precise axis tick positions for FIG.3 from detected tick marks.

    FIG.3 axes (verified by pixel scanning):
    - X: log scale, 10^-15 to 10^-7
      Tick positions: x = 440, 514, 588, 662, 735, 806, 880, 953, 1027
    - Y: linear, 0.10 to 0.35
      Tick positions: y = 617, 524, 434, 342, 249, 157 (extrapolated)
    """
    plot_area = get_fig3_plot_area_manual()

    # X-axis ticks: precisely detected positions
    x_tick_positions = [440, 514, 588, 662, 735, 806, 880, 953, 1027]
    x_tick_values = [-15, -14, -13, -12, -11, -10, -9, -8, -7]
    x_ticks = [(px, f"1e{exp}") for px, exp in zip(x_tick_positions, x_tick_values)]

    # Y-axis ticks: detected + extrapolated 0.35
    y_ticks = [
        (617, "0.10"),
        (524, "0.15"),
        (434, "0.20"),
        (342, "0.25"),
        (249, "0.30"),
        (157, "0.35"),
    ]

    return x_ticks, y_ticks, plot_area


def merge_and_dedup(std_dets, sahi_dets, dedup_radius=10, conf_boost=0.05):
    """
    Merge standard and SAHI detections with deduplication.
    Standard detections get a confidence boost (higher precision).
    """
    merged = []
    for det in std_dets:
        d = dict(det)
        d["det_conf"] = min(1.0, d["det_conf"] + conf_boost)
        d["source"] = "std"
        merged.append(d)

    for sahi_det in sahi_dets:
        cx, cy = sahi_det["center_px"]
        is_duplicate = False
        for existing in merged:
            ex, ey = existing["center_px"]
            if ((cx - ex)**2 + (cy - ey)**2) ** 0.5 < dedup_radius:
                is_duplicate = True
                break
        if not is_duplicate:
            d = dict(sahi_det)
            d["source"] = "sahi"
            merged.append(d)

    return merged


def filter_by_area(detections, plot_area, margin=5):
    """Filter detections to only those within the plotting area (with small margin)."""
    filtered = []
    for det in detections:
        cx, cy = det["center_px"]
        if (plot_area.x_min_px - margin <= cx <= plot_area.x_max_px + margin and
            plot_area.y_min_px - margin <= cy <= plot_area.y_max_px + margin):
            filtered.append(det)
    return filtered


def filter_near_axes(detections, plot_area, axis_margin=8):
    """Remove detections that are very close to axis lines (likely tick marks)."""
    filtered = []
    for det in detections:
        cx, cy = det["center_px"]
        # Too close to X-axis (bottom)
        if cy > plot_area.y_max_px - axis_margin:
            continue
        # Too close to Y-axis (left)
        if cx < plot_area.x_min_px + axis_margin:
            continue
        filtered.append(det)
    return filtered


def filter_by_confidence(detections, min_conf=0.25):
    """Filter by minimum detection confidence."""
    return [d for d in detections if d["det_conf"] >= min_conf]


def filter_small_series(series_dict, min_points=4):
    """Remove series with too few points (likely false positives)."""
    filtered = {}
    for cls, points in series_dict.items():
        if len(points) >= min_points:
            filtered[cls] = points
        else:
            print(f"    Removed {cls}: only {len(points)} points (min={min_points})")
    return filtered


def ransac_filter_improved(series_data, residual_threshold=None):
    """
    Improved RANSAC with better handling of sigmoid-like data.
    Uses polynomial fit in log-x space for better inlier detection.
    """
    from sklearn.linear_model import RANSACRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    if len(series_data) < 5:
        return series_data, []

    X = np.array([d["real_x"] for d in series_data]).reshape(-1, 1)
    y = np.array([d["real_y"] for d in series_data])

    # Work in log-x space
    if np.min(X) > 0 and np.max(X) / np.min(X) > 100:
        X_fit = np.log10(X)
    else:
        X_fit = X

    # Adaptive threshold
    if residual_threshold is None:
        y_range = np.max(y) - np.min(y)
        residual_threshold = max(0.02, y_range * 0.15)

    try:
        # Try polynomial RANSAC (degree 2) for better sigmoid fitting
        poly_model = make_pipeline(
            PolynomialFeatures(degree=min(2, len(series_data) - 2)),
            RANSACRegressor(
                residual_threshold=residual_threshold,
                max_trials=200,
                random_state=42,
            )
        )
        poly_model.fit(X_fit, y)
        inlier_mask = poly_model.named_steps['ransacregressor'].inlier_mask_

        inliers = [d for d, m in zip(series_data, inlier_mask) if m]
        outliers = [d for d, m in zip(series_data, inlier_mask) if not m]
        return inliers, outliers
    except Exception:
        # Fallback to simple RANSAC
        return ransac_filter(series_data, residual_threshold)


def run_real_chart_test_v2():
    """Run improved pipeline on FIG.3 with v2 models."""
    image_path = os.path.join(PROJECT_DIR, "data/raw/original_plot_fig3.png")
    output_dir = os.path.join(PROJECT_DIR, "experiments/real_chart_test_v2_improved")
    os.makedirs(output_dir, exist_ok=True)

    yolo_path = os.path.join(PROJECT_DIR, "models/yolo_detector_v2/train/weights/best.pt")
    classifier_path = os.path.join(PROJECT_DIR, "models/classifier_v2/best_model.pth")

    print("=" * 60)
    print("REAL CHART TEST v2 (Improved Post-Processing)")
    print("=" * 60)

    # Load models
    print("\n1. Loading v2 models...")
    detector, classifier = load_models(yolo_path, classifier_path, device="cpu")

    # Get axis info with tighter plot area
    x_ticks, y_ticks, plot_area = get_fig3_axis_info()
    x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)
    print(f"   Plot area: x=[{plot_area.x_min_px}, {plot_area.x_max_px}], "
          f"y=[{plot_area.y_min_px}, {plot_area.y_max_px}]")

    # Standard detection (higher threshold for precision)
    print("\n2. Standard detection (conf=0.30)...")
    std_dets = detect_symbols(
        image_path, detector, classifier,
        conf_threshold=0.30, device="cpu",
    )
    print(f"   {len(std_dets)} raw detections")

    # SAHI detection (lower threshold for recall)
    print("\n3. SAHI detection (conf=0.25)...")
    gc.collect()
    sahi_dets = detect_symbols_sahi(
        image_path, yolo_path, classifier,
        conf_threshold=0.25, slice_size=320,
        overlap_ratio=0.2, device="cpu",
    )
    print(f"   {len(sahi_dets)} raw detections")

    # Merge and deduplicate
    print("\n4. Merging and deduplicating...")
    merged = merge_and_dedup(std_dets, sahi_dets, dedup_radius=10)
    print(f"   {len(merged)} merged detections")

    # Filter by plotting area
    print("\n5. Filtering by plotting area...")
    in_area = filter_by_area(merged, plot_area, margin=3)
    print(f"   {len(in_area)} detections in plot area (removed {len(merged) - len(in_area)})")

    # Filter near axes
    in_area = filter_near_axes(in_area, plot_area, axis_margin=6)
    print(f"   {len(in_area)} after axis proximity filter")

    # Filter by confidence
    in_area = filter_by_confidence(in_area, min_conf=0.20)
    print(f"   {len(in_area)} after confidence filter")

    # Show class distribution
    cls_counts = {}
    for d in in_area:
        cls = d["final_class"]
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
    for cls, cnt in sorted(cls_counts.items()):
        print(f"     {cls}: {cnt}")

    # Visualize filtered detections
    vis_path = os.path.join(output_dir, "fig3_filtered_detections.png")
    visualize_detections(image_path, in_area, vis_path)

    # Extract coordinates
    print("\n6. Extracting coordinates...")
    coords = extract_coordinates(in_area, x_axis, y_axis, plot_area)
    print(f"   {len(coords)} points with coordinates")

    # Group by series
    print("\n7. Grouping by series...")
    series = group_by_series(coords)
    for cls, pts in series.items():
        print(f"   {cls}: {len(pts)} points")

    # RANSAC filtering (improved)
    print("\n8. RANSAC filtering (improved)...")
    filtered_series = {}
    for cls, points in series.items():
        inliers, outliers = ransac_filter_improved(points)
        filtered_series[cls] = inliers
        print(f"   {cls}: {len(points)} → {len(inliers)} (removed {len(outliers)})")

    # Remove small series (keep open symbols even with fewer points)
    print("\n9. Filtering small series (min=2 for expected types, min=4 for others)...")
    expected_types = {"filled_circle", "open_circle", "filled_square", "open_square"}
    final_series = {}
    for cls, points in filtered_series.items():
        min_pts = 2 if cls in expected_types else 4
        if len(points) >= min_pts:
            final_series[cls] = points
        else:
            print(f"    Removed {cls}: only {len(points)} points (min={min_pts})")
    filtered_series = final_series

    # Reconstruct chart
    print("\n10. Reconstructing chart...")
    recon_path = os.path.join(output_dir, "fig3_reconstructed_v2_improved.png")
    reconstruct_and_compare(filtered_series, x_axis, recon_path)

    # Save data
    csv_path = os.path.join(output_dir, "fig3_extracted_data_v2_improved.csv")
    with open(csv_path, 'w') as f:
        f.write("series,real_x,real_y,pixel_x,pixel_y,det_conf\n")
        for cls, points in filtered_series.items():
            for p in points:
                f.write(f"{cls},{p['real_x']:.6e},{p['real_y']:.6f},"
                        f"{p['center_px'][0]:.1f},{p['center_px'][1]:.1f},"
                        f"{p['det_conf']:.3f}\n")
    print(f"\n   Data saved to: {csv_path}")

    # Final summary
    expected_mapping = {
        "filled_circle": "D2E7-IgG4",
        "open_circle": "2SD4-IgG1",
        "filled_square": "MAK195-IgG1",
        "open_square": "MAK195 F(ab')2",
    }

    print(f"\n{'='*60}")
    print("FINAL EXTRACTION SUMMARY")
    print(f"{'='*60}")
    total = sum(len(pts) for pts in filtered_series.values())
    print(f"Total series: {len(filtered_series)}")
    print(f"Total valid points: {total}")

    for cls, points in sorted(filtered_series.items()):
        if not points:
            continue
        name = expected_mapping.get(cls, f"[unknown: {cls}]")
        x_vals = [p["real_x"] for p in points]
        y_vals = [p["real_y"] for p in points]
        confs = [p["det_conf"] for p in points]
        print(f"\n  {cls} → {name}")
        print(f"    Points: {len(points)}")
        print(f"    X range: [{min(x_vals):.2e}, {max(x_vals):.2e}] M")
        print(f"    Y range: [{min(y_vals):.4f}, {max(y_vals):.4f}] OD")
        print(f"    Avg confidence: {np.mean(confs):.3f}")
        print(f"    Data points:")
        for p in points:
            print(f"      X={p['real_x']:.3e} M, Y={p['real_y']:.4f} OD "
                  f"(conf={p['det_conf']:.2f})")

    return filtered_series


if __name__ == "__main__":
    run_real_chart_test_v2()
