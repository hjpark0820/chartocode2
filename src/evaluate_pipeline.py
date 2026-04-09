"""
Multi-image evaluation of the full pipeline.
Compares standard YOLO vs SAHI detection and evaluates coordinate extraction accuracy.
"""

import os
import sys
import json
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from run_detection_pipeline import load_models, detect_symbols, detect_symbols_sahi
from test_coordinate_extraction import derive_tick_positions_from_gt
from coordinate_extractor import (
    detect_plotting_area,
    parse_tick_labels_manual,
    extract_coordinates,
    group_by_series,
    ransac_filter,
    reconstruct_and_compare,
)

PROJECT_DIR = "/Users/jason/MyProject/RA/Hongju"


def evaluate_detection(detections, gt_anns, match_radius=10):
    """Evaluate detection quality against ground truth."""
    matched_gt = set()
    matched_det = set()

    for i, det in enumerate(detections):
        cx, cy = det["center_px"]
        best_dist = float("inf")
        best_j = -1
        for j, ann in enumerate(gt_anns):
            if j in matched_gt:
                continue
            gx, gy = ann["center_px"]
            dist = ((cx - gx)**2 + (cy - gy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_dist < match_radius and best_j >= 0:
            matched_gt.add(best_j)
            matched_det.add(i)

    tp = len(matched_gt)
    fp = len(detections) - tp
    fn = len(gt_anns) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "matched_det_indices": matched_det,
        "matched_gt_indices": matched_gt,
    }


def evaluate_coordinates(detections, gt_anns, match_radius=10):
    """Evaluate coordinate extraction accuracy for matched detections."""
    x_errors = []
    y_errors = []

    for det in detections:
        if "real_x" not in det:
            continue
        cx, cy = det["center_px"]
        # Find closest GT
        best_dist = float("inf")
        best_ann = None
        for ann in gt_anns:
            gx, gy = ann["center_px"]
            dist = ((cx - gx)**2 + (cy - gy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_ann = ann

        if best_dist < match_radius and best_ann:
            gt_x, gt_y = best_ann["real_x"], best_ann["real_y"]
            pred_x, pred_y = det["real_x"], det["real_y"]

            if gt_x > 0 and pred_x > 0:
                x_errors.append(abs(np.log10(pred_x) - np.log10(gt_x)))
            y_errors.append(abs(pred_y - gt_y))

    return x_errors, y_errors


def run_evaluation(n_images=10, use_sahi=False):
    """Run evaluation on multiple synthetic validation images."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {'SAHI' if use_sahi else 'Standard'} detection on {n_images} images")
    print(f"{'='*60}")

    # Load annotations
    ann_path = os.path.join(PROJECT_DIR, "data/synthetic/val/annotations.json")
    with open(ann_path) as f:
        data = json.load(f)

    cat_names = {c["id"]: c["name"] for c in data["categories"]}

    # Load models
    print("Loading models...")
    yolo_path = os.path.join(PROJECT_DIR, "models/yolo_detector_v1/train/weights/best.pt")
    classifier_path = os.path.join(PROJECT_DIR, "models/classifier_v1/best_model.pth")
    detector, classifier = load_models(yolo_path, classifier_path, device="cpu")

    all_det_metrics = []
    all_x_errors = []
    all_y_errors = []
    all_coord_points = 0

    for img_id in range(n_images):
        img_path = os.path.join(PROJECT_DIR, f"data/synthetic/val/images/chart_{img_id:05d}.png")
        if not os.path.exists(img_path):
            continue

        gt_anns = [a for a in data["annotations"] if a["image_id"] == img_id]

        # Run detection
        if use_sahi:
            detections = detect_symbols_sahi(
                img_path, yolo_path, classifier,
                conf_threshold=0.25, slice_size=320,
                overlap_ratio=0.2, device="cpu",
            )
        else:
            detections = detect_symbols(
                img_path, detector, classifier,
                conf_threshold=0.3, device="cpu",
            )

        # Evaluate detection
        det_metrics = evaluate_detection(detections, gt_anns)
        all_det_metrics.append(det_metrics)

        # Coordinate extraction
        x_ticks, y_ticks = derive_tick_positions_from_gt(data["annotations"], img_id)
        if x_ticks and y_ticks:
            plot_area = detect_plotting_area(img_path)
            x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)
            coords = extract_coordinates(detections, x_axis, y_axis, plot_area)

            x_errs, y_errs = evaluate_coordinates(coords, gt_anns)
            all_x_errors.extend(x_errs)
            all_y_errors.extend(y_errs)
            all_coord_points += len(coords)

        print(f"  Image {img_id}: det={len(detections)}, gt={len(gt_anns)}, "
              f"P={det_metrics['precision']:.3f}, R={det_metrics['recall']:.3f}, "
              f"F1={det_metrics['f1']:.3f}")

        # Free memory periodically
        gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    avg_p = np.mean([m["precision"] for m in all_det_metrics])
    avg_r = np.mean([m["recall"] for m in all_det_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_det_metrics])
    total_tp = sum(m["tp"] for m in all_det_metrics)
    total_fp = sum(m["fp"] for m in all_det_metrics)
    total_fn = sum(m["fn"] for m in all_det_metrics)

    print(f"  Avg Precision: {avg_p:.3f}")
    print(f"  Avg Recall:    {avg_r:.3f}")
    print(f"  Avg F1:        {avg_f1:.3f}")
    print(f"  Total TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  Micro Precision: {total_tp/(total_tp+total_fp):.3f}")
    print(f"  Micro Recall:    {total_tp/(total_tp+total_fn):.3f}")

    if all_x_errors:
        print(f"\nCOORDINATE ACCURACY ({all_coord_points} points)")
        print(f"  X error (log10 decades): "
              f"mean={np.mean(all_x_errors):.4f}, "
              f"median={np.median(all_x_errors):.4f}, "
              f"95th={np.percentile(all_x_errors, 95):.4f}")
        print(f"  Y error (absolute): "
              f"mean={np.mean(all_y_errors):.6f}, "
              f"median={np.median(all_y_errors):.6f}, "
              f"95th={np.percentile(all_y_errors, 95):.6f}")

    return all_det_metrics, all_x_errors, all_y_errors


if __name__ == "__main__":
    # Evaluate standard detection
    std_metrics, std_x, std_y = run_evaluation(n_images=10, use_sahi=False)

    # Clean up memory before SAHI evaluation
    gc.collect()

    # Evaluate SAHI detection
    sahi_metrics, sahi_x, sahi_y = run_evaluation(n_images=10, use_sahi=True)

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Standard vs SAHI")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Standard':>12} {'SAHI':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Avg Recall':<20} {np.mean([m['recall'] for m in std_metrics]):>12.3f} "
          f"{np.mean([m['recall'] for m in sahi_metrics]):>12.3f}")
    print(f"  {'Avg Precision':<20} {np.mean([m['precision'] for m in std_metrics]):>12.3f} "
          f"{np.mean([m['precision'] for m in sahi_metrics]):>12.3f}")
    print(f"  {'Avg F1':<20} {np.mean([m['f1'] for m in std_metrics]):>12.3f} "
          f"{np.mean([m['f1'] for m in sahi_metrics]):>12.3f}")
    if std_x and sahi_x:
        print(f"  {'X err (decades)':<20} {np.mean(std_x):>12.4f} {np.mean(sahi_x):>12.4f}")
        print(f"  {'Y err (abs)':<20} {np.mean(std_y):>12.6f} {np.mean(sahi_y):>12.6f}")
