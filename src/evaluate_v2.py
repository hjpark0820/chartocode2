"""
Evaluate v2 models (YOLO v2 + CNN v2) on:
1. Synthetic v2 validation data (10 images)
2. Real FIG.3 patent chart

Compare with v1 results to measure improvement.
"""

import os
import sys
import json
import gc
import numpy as np
from pathlib import Path


# Setup dynamic paths
# SRC_DIR is the 'src' folder where this script lives
SRC_DIR = Path(__file__).parent.absolute()

# PROJECT_ROOT is the main folder containing 'src', 'models', and 'experiments'
PROJECT_ROOT = SRC_DIR.parent

# Ensure local imports work
sys.path.insert(0, str(SRC_DIR))

from run_detection_pipeline import load_models, detect_symbols, detect_symbols_sahi, detect_symbols_best, visualize_detections
from test_coordinate_extraction import derive_tick_positions_from_gt
from coordinate_extractor import (
    detect_plotting_area, parse_tick_labels_manual,
    extract_coordinates, group_by_series, ransac_filter,
    reconstruct_and_compare,
)
from evaluate_pipeline import evaluate_detection, evaluate_coordinates


def eval_synthetic_v2(n_images=10):
    """Evaluate v2 models on v2 synthetic validation data."""
    print(f"\n{'='*60}")
    print("PART 1: v2 Models on v2 Synthetic Validation Data")
    print(f"{'='*60}")

    # Point to the synthetic_v2 data you just generated
    ann_path = SRC_DIR / "data" / "synthetic_v2" / "val" / "annotations.json"
    
    if not ann_path.exists():
        print(f"Error: Could not find annotations at {ann_path}")
        return None

    with open(ann_path) as f:
        data = json.load(f)

    # Update model paths to match your Windows structure
    yolo_path = PROJECT_ROOT / "models" / "yolo_detector_v2" / "train" / "weights" / "best.pt"
    classifier_path = SRC_DIR / "models" / "classifier_v2" / "best_model.pth"
    
    print(f"Loading YOLO from: {yolo_path}")
    detector, classifier = load_models(str(yolo_path), str(classifier_path), device="cpu")

    all_det_metrics = []
    all_x_errors = []
    all_y_errors = []

    for img_id in range(n_images):
        img_name = f"chart_{img_id:05d}.png"
        img_path = SRC_DIR / "data" / "synthetic_v2" / "val" / "images" / img_name
        
        if not img_path.exists():
            print(f"Skipping {img_name}: File not found.")
            continue

        gt_anns = [a for a in data["annotations"] if a["image_id"] == img_id]

        detections = detect_symbols(
            str(img_path), detector, classifier,
            conf_threshold=0.3, device="cpu",
        )

        det_metrics = evaluate_detection(detections, gt_anns)
        all_det_metrics.append(det_metrics)

        # Coordinate extraction
        x_ticks, y_ticks = derive_tick_positions_from_gt(data["annotations"], img_id)
        if x_ticks and y_ticks:
            plot_area = detect_plotting_area(str(img_path))
            x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)
            coords = extract_coordinates(detections, x_axis, y_axis, plot_area)
            x_errs, y_errs = evaluate_coordinates(coords, gt_anns)
            all_x_errors.extend(x_errs)
            all_y_errors.extend(y_errs)

        print(f"  Image {img_id}: det={len(detections)}, gt={len(gt_anns)}, "
              f"P={det_metrics['precision']:.3f}, R={det_metrics['recall']:.3f}, "
              f"F1={det_metrics['f1']:.3f}")
        gc.collect()

    if not all_det_metrics:
        return None

    avg_p = np.mean([m["precision"] for m in all_det_metrics])
    avg_r = np.mean([m["recall"] for m in all_det_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_det_metrics])
    total_tp = sum(m["tp"] for m in all_det_metrics)
    total_fp = sum(m["fp"] for m in all_det_metrics)
    total_fn = sum(m["fn"] for m in all_det_metrics)

    print(f"\n  DETECTION SUMMARY (v2 models, v2 data):")
    print(f"    Avg P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f1:.3f}")
    print(f"    Total TP={total_tp}, FP={total_fp}, FN={total_fn}")
    if all_x_errors:
        print(f"    X error: mean={np.mean(all_x_errors):.4f} decades")
        print(f"    Y error: mean={np.mean(all_y_errors):.6f}")

    return {
        "avg_precision": avg_p,
        "avg_recall": avg_r,
        "avg_f1": avg_f1,
        "x_error_mean": np.mean(all_x_errors) if all_x_errors else None,
        "y_error_mean": np.mean(all_y_errors) if all_y_errors else None,
    }


def eval_real_fig3():
    """Evaluate v2 models on real FIG.3 patent chart."""
    print(f"\n{'='*60}")
    print("PART 2: v2 Models on Real FIG.3")
    print(f"{'='*60}")
    
	
    image_path = SRC_DIR / "data" / "raw" / "original_plot_fig3.png"
    output_dir = PROJECT_ROOT / "experiments" / "real_chart_test_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_path = PROJECT_ROOT / "models" / "yolo_detector_v2" / "train" / "weights" / "best.pt"
    classifier_path = SRC_DIR/ "models" / "classifier_v2" / "best_model.pth"

    print("\n  Loading v2 models...")
    detector, classifier = load_models(str(yolo_path), str(classifier_path), device="cpu")

    # Standard detection
    print("\n  Standard detection...")
    std_dets = detect_symbols(
        str(image_path), detector, classifier,
        conf_threshold=0.25, device="cpu",
    )
    print(f"    {len(std_dets)} detections")
    
    # ... classification summary ...
    std_classes = {}
    for d in std_dets:
        cls = d["final_class"]
        std_classes[cls] = std_classes.get(cls, 0) + 1
    for cls, cnt in sorted(std_classes.items()):
        print(f"      {cls}: {cnt}")

    # Combined detection
    print("\n  Combined (Standard + SAHI) detection...")
    gc.collect()
    combined_dets = detect_symbols_best(
        str(image_path), detector, str(yolo_path), classifier,
        conf_threshold=0.2, device="cpu",
    )
    print(f"    {len(combined_dets)} detections")

    # Visualize results
    for name, dets in [("std_v2", std_dets), ("combined_v2", combined_dets)]:
        vis_path = output_dir / f"fig3_{name}_detections.png"
        visualize_detections(str(image_path), dets, str(vis_path))

    # Coordinate extraction
    print("\n  Extracting coordinates...")
    from test_real_chart import get_fig3_axis_info
    x_ticks, y_ticks, plot_area = get_fig3_axis_info(str(image_path))
    x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)

    coords = extract_coordinates(combined_dets, x_axis, y_axis, plot_area)
    
    series = group_by_series(coords)
    filtered_series = {}
    for cls, points in series.items():
        inliers, outliers = ransac_filter(points)
        filtered_series[cls] = inliers
        print(f"    {cls}: {len(points)} → {len(inliers)} (RANSAC removed {len(outliers)})")

    recon_path = output_dir / "fig3_reconstructed_v2.png"
    reconstruct_and_compare(filtered_series, x_axis, str(recon_path))

    # Save CSV
    csv_path = output_dir / "fig3_extracted_data_v2.csv"
    with open(csv_path, 'w') as f:
        f.write("series,real_x,real_y,pixel_x,pixel_y,det_conf\n")
        for cls, points in filtered_series.items():
            for p in points:
                f.write(f"{cls},{p['real_x']:.6e},{p['real_y']:.6f},"
                        f"{p['center_px'][0]:.1f},{p['center_px'][1]:.1f},"
                        f"{p['det_conf']:.3f}\n")
    print(f"    Data saved to: {csv_path}")

    # Summary classes for final print
    comb_classes = {}
    for d in combined_dets:
        cls = d["final_class"]
        comb_classes[cls] = comb_classes.get(cls, 0) + 1

    return filtered_series, comb_classes


if __name__ == "__main__":
    # Part 1: Synthetic evaluation
    synth_results = eval_synthetic_v2(n_images=10)

    gc.collect()

    # Part 2: Real chart evaluation
    if synth_results:
        real_results, real_classes = eval_real_fig3()

        # Comparison with v1 (from experiment log)
        print(f"\n{'='*60}")
        print("COMPARISON: v1 vs v2")
        print(f"{'='*60}")
        print("  v1 (synthetic val): P=0.993, R=0.858, F1=0.929")
        print(f"  v2 (synthetic val): P={synth_results['avg_precision']:.3f}, "
              f"R={synth_results['avg_recall']:.3f}, "
              f"F1={synth_results['avg_f1']:.3f}")
        print(f"\n  v1 (real FIG.3): open_circle=1, open_square=0")
        print(f"  v2 (real FIG.3): ", end="")
        for cls in ["filled_circle", "open_circle", "filled_square", "open_square"]:
            print(f"{cls}={real_classes.get(cls, 0)}", end=" ")
        print()
    else:
        print("Synthetic evaluation failed. Please check paths.")
