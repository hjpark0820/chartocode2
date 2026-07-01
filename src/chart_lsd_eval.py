"""
chart_lsd_eval.py
=================
Evaluate two OpenCV LSD variants on synthetic chart plots:

  • LSD_REFINE_NONE  — standard LSD (no post-processing refinement)
  • LSD_REFINE_ADV   — LSD with advanced refinement (gradient-based endpoint polish)

Validation is performed against the ground-truth segments stored in the
GT JSON files produced by chart_marker_detector_v3.py (or migrated by
chart_segment_detector.py).

Outputs (saved to  data/lsd_eval/):
  per_plot_metrics.csv   — one row per plot × variant
  summary_metrics.csv    — mean ± std and micro-averaged P/R/F1 per variant
  images/<variant>/plot_NNNNN.png
      3-panel PNG: original crop | GT (green) + predicted (red) | overlay

Usage
-----
  # Evaluate all plots
  python chart_lsd_eval.py

  # First 100 plots only
  python chart_lsd_eval.py --n-plots 100

  # Skip saving images
  python chart_lsd_eval.py --no-images

  # Change match distance/angle thresholds
  python chart_lsd_eval.py --match-dist 15 --match-angle 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths  (same layout as chart_marker_detector_v3.py)
# ---------------------------------------------------------------------------
_SRC_DIR    = Path(__file__).resolve().parent          # …/chartocode/src/
SYNTH_DIR   = _SRC_DIR / ".." / "data" / "synthetic_plots"  # …/chartocode/data/synthetic_plots/
EVAL_DIR    = _SRC_DIR / ".." / "data" / "lsd_eval"          # …/chartocode/data/lsd_eval/

# ---------------------------------------------------------------------------
# Matching thresholds
# ---------------------------------------------------------------------------
MATCH_DIST_DEFAULT  = 10   # px — midpoint distance for TP
MATCH_ANGLE_DEFAULT = 15   # deg — angular difference for TP

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
def _bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100.0 * done / max(total, 1)
    eta    = ""
    return f"[{bar}] {done:>{len(str(total))}}/{total}  {pct:5.1f}%  {eta}"


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------
MIN_SEG_LEN = 5.0  # px — minimum segment length to include in GT


def _segments_from_points(points: List[Dict]) -> List[Dict]:
    """
    Compute GT segments from marker points.

    Works with both chart_marker_detector_v3.py GT files (which store only
    {cx, cy, class_idx, class_name}) and chart_segment_detector.py GT files
    (which also store series_idx and point_idx).

    In v3 files each series uses a unique symbol class, so
    class_idx == series_idx.  point_idx is inferred from order of appearance.
    """
    from collections import defaultdict
    by_series: Dict = defaultdict(list)
    for pt in points:
        si = pt.get("series_idx", pt.get("class_idx"))
        if si is None:
            continue
        by_series[si].append(pt)

    segments: List[Dict] = []
    for si, pts_s in by_series.items():
        # Assign _pi (point index) from stored value or order of appearance
        for order, pt in enumerate(pts_s):
            pt["_pi"] = pt.get("point_idx", order)
        pts_s.sort(key=lambda p: p["_pi"])
        for i in range(len(pts_s) - 1):
            p0, p1 = pts_s[i], pts_s[i + 1]
            if p1["_pi"] != p0["_pi"] + 1:
                continue
            x1, y1 = p0["cx"], p0["cy"]
            x2, y2 = p1["cx"], p1["cy"]
            length = math.hypot(x2 - x1, y2 - y1)
            if length >= MIN_SEG_LEN:
                segments.append({
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "series_idx": si,
                    "length": round(length, 2),
                })
    return segments


def _segments_from_series_pixels(series_pixels: List[List[Dict]]) -> List[Dict]:
    """
    Derive GT segments from series_pixels (the ordered pixel sequences saved
    by chart_marker_detector_v3.py).  ALL consecutive pairs are included;
    the MIN_SEP filter only governs ViT subimage patch selection and must
    not gate segment GT coverage.
    """
    segments: List[Dict] = []
    for si, sp in enumerate(series_pixels):
        for i in range(len(sp) - 1):
            p0, p1 = sp[i], sp[i + 1]
            length = math.hypot(p1["cx"] - p0["cx"], p1["cy"] - p0["cy"])
            if length >= MIN_SEG_LEN:
                segments.append({
                    "x1": p0["cx"], "y1": p0["cy"],
                    "x2": p1["cx"], "y2": p1["cy"],
                    "series_idx": si,
                    "length": round(length, 2),
                })
    return segments


def _load_gt(gt_path: Path) -> Dict:
    """
    Load a GT JSON file.  Segments are derived in priority order:
      1. Use 'segments' key if present and non-empty (new-format GT files
         generated by the updated chart_marker_detector_v3.py).
      2. Use 'series_pixels' + 'points' to derive segments correctly
         (GT files that have series_pixels but no segments key).
      3. Fall back to reconstructing from 'points' using class_idx as
         series_idx (old-format GT files without series_pixels).
    Derived segments are written back to the file for future use.
    Returns the full data dict.
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("segments"):
        if data.get("series_pixels"):
            # Priority 2: use series_pixels (correct ordering, all pairs)
            segs = _segments_from_series_pixels(data["series_pixels"])
        elif data.get("points"):
            # Priority 3: fallback reconstruction from class_idx
            segs = _segments_from_points(data["points"])
        else:
            segs = []
        data["segments"] = segs
        try:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except PermissionError:
            pass  # read-only filesystem — use in-memory segments only

    return data


def _load_gt_segments(gt_path: Path) -> List[Dict]:
    """Return the list of GT segment dicts from a GT JSON file."""
    return _load_gt(gt_path).get("segments") or []


def _get_plot_area(gt_path: Path) -> Dict:
    data = _load_gt(gt_path)
    return data.get("pa") or data.get("plot_area") or {}


# ---------------------------------------------------------------------------
# Segment matching
# ---------------------------------------------------------------------------
def _seg_midpoint(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _seg_angle_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Undirected angle in [0, 90) degrees."""
    dx, dy = x2 - x1, y2 - y1
    a = math.degrees(math.atan2(abs(dy), abs(dx)))
    return a  # already in [0, 90]


def _angle_diff_deg(a1: float, a2: float) -> float:
    """Smallest difference between two undirected angles (both in [0,90])."""
    d = abs(a1 - a2)
    if d > 90:
        d = 180 - d
    return d


def match_segments(
    pred_segs: List[Tuple[float, float, float, float]],
    gt_segs:   List[Tuple[float, float, float, float]],
    match_dist:  float,
    match_angle: float,
) -> Tuple[int, int, int]:
    """
    Greedy one-to-one matching between predicted and GT segments.

    Returns (TP, FP, FN).
    A predicted segment is a TP if:
      - its midpoint is within `match_dist` px of an unmatched GT midpoint, AND
      - the angular difference is within `match_angle` degrees.
    """
    matched_gt = set()
    tp = 0

    for px1, py1, px2, py2 in pred_segs:
        pmx, pmy = _seg_midpoint(px1, py1, px2, py2)
        pa = _seg_angle_deg(px1, py1, px2, py2)
        best_dist = math.inf
        best_idx  = -1
        for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_segs):
            if gi in matched_gt:
                continue
            gmx, gmy = _seg_midpoint(gx1, gy1, gx2, gy2)
            dist = math.hypot(pmx - gmx, pmy - gmy)
            if dist > match_dist:
                continue
            ga = _seg_angle_deg(gx1, gy1, gx2, gy2)
            if _angle_diff_deg(pa, ga) > match_angle:
                continue
            if dist < best_dist:
                best_dist = dist
                best_idx  = gi
        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(pred_segs) - tp
    fn = len(gt_segs)   - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# LSD runners
# ---------------------------------------------------------------------------
def _run_lsd(gray: np.ndarray, refine: int) -> List[Tuple[float, float, float, float]]:
    """
    Run OpenCV LSD on a grayscale uint8 image.

    refine: cv2.LSD_REFINE_NONE or cv2.LSD_REFINE_ADV
    Returns list of (x1, y1, x2, y2).
    """
    lsd   = cv2.createLineSegmentDetector(refine)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return []
    result = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        result.append((float(x1), float(y1), float(x2), float(y2)))
    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _draw_segments(
    img_bgr: np.ndarray,
    segments: List[Tuple[float, float, float, float]],
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    out = img_bgr.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(out,
                 (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))),
                 color, thickness, cv2.LINE_AA)
    return out


def _save_vis(
    save_path: Path,
    crop_gray: np.ndarray,
    gt_segs:   List[Tuple[float, float, float, float]],
    pred_segs: List[Tuple[float, float, float, float]],
) -> None:
    """Save a 3-panel PNG: original | GT (green) | GT+pred overlay."""
    h, w = crop_gray.shape
    bgr   = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)

    panel_gt   = _draw_segments(bgr, gt_segs,   (0, 200, 0),   thickness=1)
    panel_pred = _draw_segments(bgr, gt_segs,   (0, 200, 0),   thickness=1)
    panel_pred = _draw_segments(panel_pred, pred_segs, (0, 0, 220), thickness=1)

    # Labels
    for panel, label in [(panel_gt, "GT"), (panel_pred, "Predicted (red) + GT (green)")]:
        cv2.putText(panel, label, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    sheet = np.concatenate([bgr, panel_gt, panel_pred], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), sheet)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
VARIANTS = [
    ("LSD_NONE", cv2.LSD_REFINE_NONE),
    ("LSD_ADV",  cv2.LSD_REFINE_ADV),
]


def evaluate(
    n_plots:     int   = 0,
    match_dist:  float = MATCH_DIST_DEFAULT,
    match_angle: float = MATCH_ANGLE_DEFAULT,
    save_images: bool  = True,
) -> None:
    # ------------------------------------------------------------------
    # Collect GT files
    # ------------------------------------------------------------------
    gt_files = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not gt_files:
        print(f"[ERROR] No GT files found in {SYNTH_DIR}")
        print("  Generate plots first with chart_marker_detector_v3.py --mode train")
        sys.exit(1)

    if n_plots > 0:
        gt_files = gt_files[:n_plots]

    total = len(gt_files)
    print(f"\n  Evaluating {total} plots  "
          f"(match_dist={match_dist} px, match_angle={match_angle}°) …\n")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-plot metrics accumulator
    # ------------------------------------------------------------------
    # rows: {plot_id, variant, tp, fp, fn, precision, recall, f1,
    #        n_pred, n_gt, img_w, img_h}
    rows: List[Dict] = []

    # Accumulators for micro-average
    accum = {name: {"tp": 0, "fp": 0, "fn": 0} for name, _ in VARIANTS}

    # ------------------------------------------------------------------
    # Process each plot
    # ------------------------------------------------------------------
    for idx, gt_path in enumerate(gt_files, 1):
        # Progress
        sys.stdout.write(f"\r  {_bar(idx, total)}")
        sys.stdout.flush()

        plot_id = gt_path.stem.replace("gt_", "")
        img_path = SYNTH_DIR / f"plot_{plot_id}.png"
        if not img_path.exists():
            continue

        # Load image and crop to plotting area
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        pa = _get_plot_area(gt_path)
        if not pa:
            continue
        x0, y0 = int(pa["x0"]), int(pa["y0"])
        x1, y1 = int(pa["x1"]), int(pa["y1"])
        crop_bgr  = img_bgr[y0:y1, x0:x1]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        ch, cw    = crop_gray.shape

        # Load GT segments and translate to crop coordinates
        raw_segs = _load_gt_segments(gt_path)
        gt_segs: List[Tuple[float, float, float, float]] = []
        for s in raw_segs:
            gx1 = float(s["x1"]) - x0
            gy1 = float(s["y1"]) - y0
            gx2 = float(s["x2"]) - x0
            gy2 = float(s["y2"]) - y0
            # Keep only segments whose midpoint is inside the crop
            mx, my = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            if 0 <= mx < cw and 0 <= my < ch:
                gt_segs.append((gx1, gy1, gx2, gy2))

        # Run both LSD variants
        for var_name, refine in VARIANTS:
            pred_segs = _run_lsd(crop_gray, refine)
            tp, fp, fn = match_segments(pred_segs, gt_segs, match_dist, match_angle)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            accum[var_name]["tp"] += tp
            accum[var_name]["fp"] += fp
            accum[var_name]["fn"] += fn

            rows.append({
                "plot_id":   plot_id,
                "variant":   var_name,
                "n_gt":      len(gt_segs),
                "n_pred":    len(pred_segs),
                "tp":        tp,
                "fp":        fp,
                "fn":        fn,
                "precision": round(prec, 4),
                "recall":    round(rec,  4),
                "f1":        round(f1,   4),
                "img_w":     cw,
                "img_h":     ch,
            })

            # Save visualisation (only for the first variant to avoid duplication;
            # save both variants' predictions side-by-side)
            if save_images and var_name == "LSD_ADV":
                # Save ADV variant image
                vis_path = EVAL_DIR / "images" / var_name / f"plot_{plot_id}.png"
                _save_vis(vis_path, crop_gray, gt_segs, pred_segs)

        # Also save NONE variant image
        if save_images:
            pred_none = _run_lsd(crop_gray, cv2.LSD_REFINE_NONE)
            vis_path  = EVAL_DIR / "images" / "LSD_NONE" / f"plot_{plot_id}.png"
            _save_vis(vis_path, crop_gray, gt_segs, pred_none)

    print()  # newline after progress bar

    # ------------------------------------------------------------------
    # Write per-plot CSV
    # ------------------------------------------------------------------
    per_plot_csv = EVAL_DIR / "per_plot_metrics.csv"
    fieldnames = ["plot_id", "variant", "n_gt", "n_pred",
                  "tp", "fp", "fn", "precision", "recall", "f1",
                  "img_w", "img_h"]
    with open(per_plot_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------------
    # Compute summary statistics per variant
    # ------------------------------------------------------------------
    summary_rows = []
    for var_name, _ in VARIANTS:
        var_rows = [r for r in rows if r["variant"] == var_name]
        precs = [r["precision"] for r in var_rows]
        recs  = [r["recall"]    for r in var_rows]
        f1s   = [r["f1"]        for r in var_rows]

        micro_tp = accum[var_name]["tp"]
        micro_fp = accum[var_name]["fp"]
        micro_fn = accum[var_name]["fn"]
        micro_p  = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
        micro_r  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

        summary_rows.append({
            "variant":          var_name,
            "n_plots":          len(var_rows),
            "mean_precision":   round(float(np.mean(precs)), 4),
            "std_precision":    round(float(np.std(precs)),  4),
            "mean_recall":      round(float(np.mean(recs)),  4),
            "std_recall":       round(float(np.std(recs)),   4),
            "mean_f1":          round(float(np.mean(f1s)),   4),
            "std_f1":           round(float(np.std(f1s)),    4),
            "micro_precision":  round(micro_p,  4),
            "micro_recall":     round(micro_r,  4),
            "micro_f1":         round(micro_f1, 4),
            "total_tp":         micro_tp,
            "total_fp":         micro_fp,
            "total_fn":         micro_fn,
        })

    summary_csv = EVAL_DIR / "summary_metrics.csv"
    sum_fields  = ["variant", "n_plots",
                   "mean_precision", "std_precision",
                   "mean_recall",    "std_recall",
                   "mean_f1",        "std_f1",
                   "micro_precision", "micro_recall", "micro_f1",
                   "total_tp", "total_fp", "total_fn"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n  {'Variant':<12}  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}  "
          f"{'Mean-F1':>8}  {'±Std':>6}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
    print("  " + "-" * 80)
    for r in summary_rows:
        print(f"  {r['variant']:<12}  "
              f"{r['micro_precision']:8.4f}  "
              f"{r['micro_recall']:8.4f}  "
              f"{r['micro_f1']:9.4f}  "
              f"{r['mean_f1']:8.4f}  "
              f"±{r['std_f1']:5.4f}  "
              f"{r['total_tp']:6d}  "
              f"{r['total_fp']:6d}  "
              f"{r['total_fn']:6d}")

    print(f"\n  Results saved to: {EVAL_DIR}")
    print(f"    per_plot_metrics.csv  ({len(rows)} rows)")
    print(f"    summary_metrics.csv   ({len(summary_rows)} rows)")
    if save_images:
        print(f"    images/LSD_NONE/      (annotated crops)")
        print(f"    images/LSD_ADV/       (annotated crops)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate standard LSD and LSD+ADV on synthetic chart plots"
    )
    parser.add_argument("--n-plots",    type=int,   default=0,
                        help="Number of plots to evaluate (0 = all)")
    parser.add_argument("--match-dist", type=float, default=MATCH_DIST_DEFAULT,
                        help="Midpoint distance threshold for TP matching (px)")
    parser.add_argument("--match-angle",type=float, default=MATCH_ANGLE_DEFAULT,
                        help="Angular difference threshold for TP matching (deg)")
    parser.add_argument("--no-images",  action="store_true",
                        help="Skip saving annotated images")
    args = parser.parse_args()

    evaluate(
        n_plots     = args.n_plots,
        match_dist  = args.match_dist,
        match_angle = args.match_angle,
        save_images = not args.no_images,
    )


if __name__ == "__main__":
    main()
