"""
Test the coordinate extraction pipeline on synthetic validation data.

Tests:
1. Pixel-to-value mapping accuracy (using ground truth pixel+value pairs)
2. Full pipeline: detection → classification → coordinate extraction
3. Reports error statistics per series
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from coordinate_extractor import (
    detect_plotting_area,
    parse_tick_labels_manual,
    pixel_to_value,
    pixel_to_value_y,
    extract_coordinates,
    group_by_series,
    ransac_filter,
    reconstruct_and_compare,
    run_full_extraction,
    AxisInfo,
    PlottingArea,
)

PROJECT_DIR = "/Users/jason/MyProject/RA/Hongju"


def derive_tick_positions_from_gt(annotations, image_id=0):
    """
    Derive axis tick pixel positions from ground truth annotations.
    Uses known (pixel, value) pairs to establish the mapping.
    Automatically detects log vs linear X-axis.
    """
    anns = [a for a in annotations if a["image_id"] == image_id]
    if not anns:
        return None, None

    # Collect all (px, val) pairs
    x_pairs = [(a["center_px"][0], a["real_x"]) for a in anns]
    y_pairs = [(a["center_px"][1], a["real_y"]) for a in anns]

    # Sort by pixel position
    x_pairs.sort(key=lambda p: p[0])
    y_pairs.sort(key=lambda p: p[0])

    x_px = np.array([p[0] for p in x_pairs])
    x_val = np.array([p[1] for p in x_pairs])

    # Detect log vs linear X-axis
    x_min, x_max = np.min(x_val), np.max(x_val)
    is_log_x = x_min > 0 and (x_max / max(x_min, 1e-30)) > 1000

    x_ticks = []
    if is_log_x:
        # Log scale: fit in (px, log10(val)) space
        x_log = np.log10(np.maximum(x_val, 1e-30))
        a_x, b_x = np.polyfit(x_px, x_log, 1)

        # Derive tick pixel positions for each decade
        min_exp = int(np.floor(np.log10(max(x_min, 1e-30))))
        max_exp = int(np.ceil(np.log10(x_max)))
        for exp in range(min_exp, max_exp + 1):
            px = (exp - b_x) / a_x
            x_ticks.append((px, f"1e{exp}"))
    else:
        # Linear scale: fit in (px, val) space
        a_x, b_x = np.polyfit(x_px, x_val, 1)

        # Derive tick positions for nice round values
        step = max(1, round((x_max - x_min) / 10))
        tick_start = int(np.floor(x_min / step) * step)
        tick_end = int(np.ceil(x_max / step) * step)
        for val in range(tick_start, tick_end + 1, step):
            px = (val - b_x) / a_x
            x_ticks.append((px, f"{val}"))

    # For Y-axis (linear scale): fit linear
    y_px = np.array([p[0] for p in y_pairs])
    y_val = np.array([p[1] for p in y_pairs])

    # Fit linear: val = a * px + b  (note: inverted, higher px = lower val)
    a_y, b_y = np.polyfit(y_px, y_val, 1)

    # Derive tick positions for Y values: 0.05, 0.10, 0.15, ..., 0.40
    y_ticks = []
    for val in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        px = (val - b_y) / a_y
        y_ticks.append((px, f"{val:.2f}"))

    return x_ticks, y_ticks


def test_pixel_to_value_accuracy():
    """
    Test 1: Accuracy of pixel-to-value mapping using ground truth.
    Uses GT annotations' pixel positions and checks if extracted values match.
    """
    print("=" * 60)
    print("TEST 1: Pixel-to-value mapping accuracy")
    print("=" * 60)

    # Load ground truth
    ann_path = os.path.join(PROJECT_DIR, "data/synthetic/val/annotations.json")
    with open(ann_path) as f:
        data = json.load(f)

    # Test on first 5 images
    all_x_errors = []
    all_y_errors = []

    for img_id in range(5):
        anns = [a for a in data["annotations"] if a["image_id"] == img_id]
        if not anns:
            continue

        # Derive tick positions from GT
        x_ticks, y_ticks = derive_tick_positions_from_gt(data["annotations"], img_id)
        if x_ticks is None:
            continue

        # Create axis info
        x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)

        # Test each annotation
        x_errors = []
        y_errors = []
        for ann in anns:
            px_x, px_y = ann["center_px"]
            gt_x, gt_y = ann["real_x"], ann["real_y"]

            # Extract coordinate
            pred_x = pixel_to_value(px_x, x_axis)
            pred_y = pixel_to_value_y(px_y, y_axis)

            # Log-scale relative error for X
            if gt_x > 0 and pred_x > 0:
                x_err = abs(np.log10(pred_x) - np.log10(gt_x))
                x_errors.append(x_err)

            # Absolute error for Y
            y_err = abs(pred_y - gt_y)
            y_errors.append(y_err)

        all_x_errors.extend(x_errors)
        all_y_errors.extend(y_errors)

        print(f"\n  Image {img_id}: {len(anns)} annotations")
        print(f"    X error (log10 decades): mean={np.mean(x_errors):.4f}, "
              f"max={np.max(x_errors):.4f}")
        print(f"    Y error (absolute): mean={np.mean(y_errors):.6f}, "
              f"max={np.max(y_errors):.6f}")

    print(f"\n  OVERALL ({len(all_x_errors)} points):")
    print(f"    X error (log10 decades): mean={np.mean(all_x_errors):.6f}, "
          f"median={np.median(all_x_errors):.6f}, max={np.max(all_x_errors):.6f}")
    print(f"    Y error (absolute): mean={np.mean(all_y_errors):.6f}, "
          f"median={np.median(all_y_errors):.6f}, max={np.max(all_y_errors):.6f}")

    x_pass = np.mean(all_x_errors) < 0.1  # Less than 0.1 decade error
    y_pass = np.mean(all_y_errors) < 0.01  # Less than 0.01 absolute error
    print(f"\n  RESULT: X mapping {'PASS' if x_pass else 'FAIL'} | "
          f"Y mapping {'PASS' if y_pass else 'FAIL'}")
    return x_pass and y_pass


def test_plotting_area_detection():
    """
    Test 2: Plotting area detection on synthetic images.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Plotting area detection")
    print("=" * 60)

    img_dir = os.path.join(PROJECT_DIR, "data/synthetic/val/images")

    for img_id in range(3):
        img_path = os.path.join(img_dir, f"chart_{img_id:05d}.png")
        if not os.path.exists(img_path):
            continue

        area = detect_plotting_area(img_path)
        print(f"\n  Image {img_id}: x=[{area.x_min_px:.0f}, {area.x_max_px:.0f}], "
              f"y=[{area.y_min_px:.0f}, {area.y_max_px:.0f}]")

        # Load GT annotations to check if detected area contains all data points
        ann_path = os.path.join(PROJECT_DIR, "data/synthetic/val/annotations.json")
        with open(ann_path) as f:
            data = json.load(f)
        anns = [a for a in data["annotations"] if a["image_id"] == img_id]

        inside = 0
        for ann in anns:
            cx, cy = ann["center_px"]
            if (area.x_min_px - 10 <= cx <= area.x_max_px + 10 and
                area.y_min_px - 10 <= cy <= area.y_max_px + 10):
                inside += 1

        pct = inside / len(anns) * 100 if anns else 0
        print(f"    {inside}/{len(anns)} GT points inside detected area ({pct:.1f}%)")


def test_full_pipeline_synthetic():
    """
    Test 3: Full pipeline on a synthetic image.
    Detection → Classification → Coordinate Extraction → Comparison with GT.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Full pipeline on synthetic image")
    print("=" * 60)

    # Use existing detection results if available
    det_json = os.path.join(PROJECT_DIR, "experiments/detection_results/chart_00000_detections.json")
    img_path = os.path.join(PROJECT_DIR, "data/synthetic/val/images/chart_00000.png")

    if os.path.exists(det_json):
        print(f"\n  Loading existing detections from: {det_json}")
        with open(det_json) as f:
            detections = json.load(f)
        print(f"  {len(detections)} detections loaded")
    else:
        print("\n  Running detection pipeline...")
        from run_detection_pipeline import load_models, detect_symbols
        detector, classifier = load_models(device="cpu")
        detections = detect_symbols(img_path, detector, classifier, conf_threshold=0.3, device="cpu")
        print(f"  {len(detections)} symbols detected")

    # Load GT annotations
    ann_path = os.path.join(PROJECT_DIR, "data/synthetic/val/annotations.json")
    with open(ann_path) as f:
        data = json.load(f)

    gt_anns = [a for a in data["annotations"] if a["image_id"] == 0]
    print(f"  Ground truth: {len(gt_anns)} annotations")

    # Derive tick positions from GT
    x_ticks, y_ticks = derive_tick_positions_from_gt(data["annotations"], 0)

    # Run coordinate extraction
    output_dir = os.path.join(PROJECT_DIR, "experiments/coord_extraction_test")
    os.makedirs(output_dir, exist_ok=True)

    filtered_series = run_full_extraction(
        image_path=img_path,
        detections=detections,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        output_dir=output_dir,
    )

    # Compare with ground truth
    print("\n  --- Comparison with Ground Truth ---")
    total_extracted = sum(len(pts) for pts in filtered_series.values())
    print(f"  Extracted: {total_extracted} data points")
    print(f"  Ground truth: {len(gt_anns)} annotations")

    # Build GT lookup by approximate pixel position
    gt_by_pos = {}
    for ann in gt_anns:
        key = (round(ann["center_px"][0], 0), round(ann["center_px"][1], 0))
        gt_by_pos[key] = ann

    # Match extracted points to GT
    matched = 0
    x_errors_matched = []
    y_errors_matched = []

    for cls, points in filtered_series.items():
        for pt in points:
            cx, cy = pt["center_px"]
            # Find closest GT point
            best_dist = float("inf")
            best_gt = None
            for ann in gt_anns:
                gx, gy = ann["center_px"]
                dist = ((cx - gx)**2 + (cy - gy)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_gt = ann

            if best_dist < 15:  # Within 15 pixels
                matched += 1
                gt_x, gt_y = best_gt["real_x"], best_gt["real_y"]
                pred_x, pred_y = pt["real_x"], pt["real_y"]

                if gt_x > 0 and pred_x > 0:
                    x_err = abs(np.log10(pred_x) - np.log10(gt_x))
                    x_errors_matched.append(x_err)

                y_err = abs(pred_y - gt_y)
                y_errors_matched.append(y_err)

    print(f"\n  Matched to GT: {matched}/{total_extracted} extracted points "
          f"({matched/total_extracted*100:.1f}%)")

    if x_errors_matched:
        print(f"  X coordinate error (log10 decades):")
        print(f"    mean={np.mean(x_errors_matched):.4f}, "
              f"median={np.median(x_errors_matched):.4f}, "
              f"max={np.max(x_errors_matched):.4f}")
    if y_errors_matched:
        print(f"  Y coordinate error (absolute):")
        print(f"    mean={np.mean(y_errors_matched):.6f}, "
              f"median={np.median(y_errors_matched):.6f}, "
              f"max={np.max(y_errors_matched):.6f}")

    # Detection recall
    print(f"\n  Detection recall: {matched}/{len(gt_anns)} GT points found "
          f"({matched/len(gt_anns)*100:.1f}%)")

    return filtered_series


def test_gt_only_extraction():
    """
    Test 4: Coordinate extraction using GT annotations as detections.
    This isolates the coordinate extraction accuracy from detection errors.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Coordinate extraction with GT annotations (no detection)")
    print("=" * 60)

    ann_path = os.path.join(PROJECT_DIR, "data/synthetic/val/annotations.json")
    with open(ann_path) as f:
        data = json.load(f)

    # Category mapping
    cat_names = {c["id"]: c["name"] for c in data["categories"]}

    # Test on image 0
    img_id = 0
    img_path = os.path.join(PROJECT_DIR, f"data/synthetic/val/images/chart_{img_id:05d}.png")
    gt_anns = [a for a in data["annotations"] if a["image_id"] == img_id]

    # Convert GT annotations to detection format
    fake_detections = []
    for ann in gt_anns:
        fake_detections.append({
            "bbox": ann["bbox"],
            "center_px": tuple(ann["center_px"]),
            "det_class": cat_names[ann["category_id"]],
            "det_conf": 1.0,
            "cls_class": cat_names[ann["category_id"]],
            "cls_conf": 1.0,
            "final_class": cat_names[ann["category_id"]],
            "gt_real_x": ann["real_x"],
            "gt_real_y": ann["real_y"],
            "gt_series_id": ann["series_id"],
        })

    print(f"  Using {len(fake_detections)} GT annotations as detections")

    # Derive tick positions from GT
    x_ticks, y_ticks = derive_tick_positions_from_gt(data["annotations"], img_id)

    # Run extraction
    output_dir = os.path.join(PROJECT_DIR, "experiments/coord_extraction_gt_test")
    filtered_series = run_full_extraction(
        image_path=img_path,
        detections=fake_detections,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        output_dir=output_dir,
    )

    # Calculate errors
    print("\n  --- Accuracy Analysis ---")
    total_x_err = []
    total_y_err = []

    for cls, points in filtered_series.items():
        x_errs = []
        y_errs = []
        for pt in points:
            gt_x = pt.get("gt_real_x")
            gt_y = pt.get("gt_real_y")
            if gt_x is None:
                continue

            pred_x = pt["real_x"]
            pred_y = pt["real_y"]

            if gt_x > 0 and pred_x > 0:
                x_err = abs(np.log10(pred_x) - np.log10(gt_x))
                x_errs.append(x_err)
                total_x_err.append(x_err)

            y_err = abs(pred_y - gt_y)
            y_errs.append(y_err)
            total_y_err.append(y_err)

        if x_errs:
            print(f"  {cls}: X err={np.mean(x_errs):.4f} decades, "
                  f"Y err={np.mean(y_errs):.6f}")

    print(f"\n  OVERALL: X err={np.mean(total_x_err):.6f} decades (log10), "
          f"Y err={np.mean(total_y_err):.6f} (absolute)")
    print(f"  X: 95th percentile = {np.percentile(total_x_err, 95):.6f} decades")
    print(f"  Y: 95th percentile = {np.percentile(total_y_err, 95):.6f}")

    return total_x_err, total_y_err


if __name__ == "__main__":
    print("=" * 60)
    print("COORDINATE EXTRACTION PIPELINE TESTS")
    print("=" * 60)

    # Test 1: Pixel-to-value mapping
    test_pixel_to_value_accuracy()

    # Test 2: Plotting area detection
    test_plotting_area_detection()

    # Test 4: GT-only extraction (isolates coordinate accuracy)
    test_gt_only_extraction()

    # Test 3: Full pipeline (requires detection models)
    try:
        test_full_pipeline_synthetic()
    except Exception as e:
        print(f"\n  Test 3 skipped: {e}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
