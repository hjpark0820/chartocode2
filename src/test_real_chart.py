"""
Test the full pipeline on the real FIG.3 patent chart.
Manually specified axis tick positions based on visual inspection.

FIG.3: Antibody Concentration vs OD 570/630 nm
- X-axis: log scale, 10^-15 to 10^-7 M
- Y-axis: linear scale, 0.10 to 0.35
- 4 series: D2E7-IgG4 (filled circle), 2SD4-IgG1 (open circle),
            MAK195-IgG1 (filled square), MAK195 F(ab')2 (open square)
"""

import os
import sys
import json
import gc

sys.path.insert(0, os.path.dirname(__file__))

from run_detection_pipeline import (
    load_models, detect_symbols, detect_symbols_sahi,
    detect_symbols_best, visualize_detections,
)
from coordinate_extractor import (
    detect_plotting_area, parse_tick_labels_manual,
    extract_coordinates, group_by_series, ransac_filter,
    reconstruct_and_compare, PlottingArea,
)

PROJECT_DIR = "/Users/jason/MyProject/RA/Hongju"


def get_fig3_axis_info(image_path):
    """
    Get axis tick positions for FIG.3 by analyzing the image.
    We use plotting area detection + known axis values to derive tick positions.

    FIG.3 axes:
    - X: log scale, 10^-15 to 10^-7 (9 decades, ticks at each decade)
    - Y: linear, 0.10 to 0.35 (ticks at 0.05 intervals)
    """
    # First detect the plotting area to get boundaries
    plot_area = detect_plotting_area(image_path)
    print(f"  Detected plot area: x=[{plot_area.x_min_px:.0f}, {plot_area.x_max_px:.0f}], "
          f"y=[{plot_area.y_min_px:.0f}, {plot_area.y_max_px:.0f}]")

    # X-axis ticks: 9 decades from 10^-15 to 10^-7, evenly spaced in log
    x_start = plot_area.x_min_px
    x_end = plot_area.x_max_px
    n_decades = 8  # 10^-15 to 10^-7 = 8 decades
    x_ticks = []
    for i, exp in enumerate(range(-15, -6)):  # -15 to -7 inclusive = 9 ticks
        px = x_start + (x_end - x_start) * i / n_decades
        x_ticks.append((px, f"1e{exp}"))

    # Y-axis ticks: linear from 0.10 to 0.35
    y_bottom = plot_area.y_max_px  # bottom of plot (high pixel = low value)
    y_top = plot_area.y_min_px     # top of plot (low pixel = high value)
    y_range_val = 0.35 - 0.10
    y_ticks = []
    for val in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        frac = (val - 0.10) / y_range_val
        px = y_bottom - frac * (y_bottom - y_top)
        y_ticks.append((px, f"{val:.2f}"))

    return x_ticks, y_ticks, plot_area


def run_real_chart_test():
    """Run the full pipeline on FIG.3."""
    image_path = os.path.join(PROJECT_DIR, "data/raw/original_plot_fig3.png")
    output_dir = os.path.join(PROJECT_DIR, "experiments/real_chart_test")
    os.makedirs(output_dir, exist_ok=True)

    yolo_path = os.path.join(PROJECT_DIR, "models/yolo_detector_v1/train/weights/best.pt")
    classifier_path = os.path.join(PROJECT_DIR, "models/classifier_v1/best_model.pth")

    # === Detection Phase ===
    print("=" * 60)
    print("REAL CHART TEST: FIG.3")
    print("=" * 60)

    print("\n1. Loading models...")
    detector, classifier = load_models(yolo_path, classifier_path, device="cpu")

    # Standard detection
    print("\n2a. Standard detection...")
    std_dets = detect_symbols(
        image_path, detector, classifier,
        conf_threshold=0.25, device="cpu",
    )
    print(f"   {len(std_dets)} detections")
    std_classes = {}
    for d in std_dets:
        cls = d["final_class"]
        std_classes[cls] = std_classes.get(cls, 0) + 1
    for cls, cnt in sorted(std_classes.items()):
        print(f"   {cls}: {cnt}")

    # SAHI detection
    print("\n2b. SAHI detection...")
    gc.collect()
    sahi_dets = detect_symbols_sahi(
        image_path, yolo_path, classifier,
        conf_threshold=0.2, slice_size=256,
        overlap_ratio=0.25, device="cpu",
    )
    print(f"   {len(sahi_dets)} detections")
    sahi_classes = {}
    for d in sahi_dets:
        cls = d["final_class"]
        sahi_classes[cls] = sahi_classes.get(cls, 0) + 1
    for cls, cnt in sorted(sahi_classes.items()):
        print(f"   {cls}: {cnt}")

    # Combined detection
    print("\n2c. Combined (Standard + SAHI dedup)...")
    gc.collect()
    combined_dets = detect_symbols_best(
        image_path, detector, yolo_path, classifier,
        conf_threshold=0.2, device="cpu",
    )
    print(f"   {len(combined_dets)} detections")
    comb_classes = {}
    for d in combined_dets:
        cls = d["final_class"]
        comb_classes[cls] = comb_classes.get(cls, 0) + 1
    for cls, cnt in sorted(comb_classes.items()):
        print(f"   {cls}: {cnt}")

    # Visualize all three
    for name, dets in [("std", std_dets), ("sahi", sahi_dets), ("combined", combined_dets)]:
        vis_path = os.path.join(output_dir, f"fig3_{name}_detections.png")
        visualize_detections(image_path, dets, vis_path)

    # === Coordinate Extraction Phase ===
    print("\n3. Extracting axis information...")
    x_ticks, y_ticks, plot_area = get_fig3_axis_info(image_path)
    x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)
    print(f"   X-axis: {'log' if x_axis.is_log else 'linear'}, "
          f"range [{x_axis.tick_values[0]:.1e}, {x_axis.tick_values[-1]:.1e}]")
    print(f"   Y-axis: range [{y_axis.tick_values[0]:.2f}, {y_axis.tick_values[-1]:.2f}]")

    # Use combined detections for coordinate extraction
    print("\n4. Extracting coordinates (combined detections)...")
    coords = extract_coordinates(combined_dets, x_axis, y_axis, plot_area)
    print(f"   {len(coords)} points with valid coordinates")

    print("\n5. Grouping by series...")
    series = group_by_series(coords)
    for cls, points in series.items():
        print(f"   {cls}: {len(points)} points")

    print("\n6. RANSAC filtering...")
    filtered_series = {}
    for cls, points in series.items():
        inliers, outliers = ransac_filter(points)
        filtered_series[cls] = inliers
        if outliers:
            print(f"   {cls}: removed {len(outliers)} outliers")

    print("\n7. Reconstructing chart...")
    recon_path = os.path.join(output_dir, "fig3_reconstructed.png")
    reconstruct_and_compare(filtered_series, x_axis, recon_path)

    # Save extracted data
    csv_path = os.path.join(output_dir, "fig3_extracted_data.csv")
    with open(csv_path, 'w') as f:
        f.write("series,real_x,real_y,pixel_x,pixel_y,det_conf\n")
        for cls, points in filtered_series.items():
            for p in points:
                f.write(f"{cls},{p['real_x']:.6e},{p['real_y']:.6f},"
                        f"{p['center_px'][0]:.1f},{p['center_px'][1]:.1f},"
                        f"{p['det_conf']:.3f}\n")
    print(f"   Data saved to: {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY FOR FIG.3")
    print(f"{'='*60}")
    total = sum(len(pts) for pts in filtered_series.values())
    print(f"Total series: {len(filtered_series)}")
    print(f"Total points: {total}")

    # Expected: ~10-12 points per series, 4 series
    # D2E7-IgG4 (filled_circle), 2SD4-IgG1 (open_circle),
    # MAK195-IgG1 (filled_square), MAK195 F(ab')2 (open_square)
    expected_mapping = {
        "filled_circle": "D2E7-IgG4",
        "open_circle": "2SD4-IgG1",
        "filled_square": "MAK195-IgG1",
        "open_square": "MAK195 F(ab')2",
    }

    for cls, points in filtered_series.items():
        if not points:
            continue
        name = expected_mapping.get(cls, cls)
        x_vals = [p["real_x"] for p in points]
        y_vals = [p["real_y"] for p in points]
        print(f"\n  {cls} → {name} ({len(points)} points):")
        print(f"    X range: [{min(x_vals):.2e}, {max(x_vals):.2e}] M")
        print(f"    Y range: [{min(y_vals):.4f}, {max(y_vals):.4f}] OD")
        for p in points:
            print(f"      ({p['real_x']:.2e}, {p['real_y']:.4f})")

    return filtered_series


if __name__ == "__main__":
    run_real_chart_test()
