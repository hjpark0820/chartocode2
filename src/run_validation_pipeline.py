"""
run_validation_pipeline.py
==========================
End-to-end validation script for the chart-digitisation pipeline.

Usage
-----
  # From the src/ directory:
  python run_validation_pipeline.py                          # generate 10 plots, run pipeline
  python run_validation_pipeline.py --n 20                  # use 20 plots
  python run_validation_pipeline.py --plots-dir /path/to/existing/plots  # skip generation
  python run_validation_pipeline.py --no-figures            # skip visualisation output

Pipeline stages
---------------
  Stage 1a  1_point_detection_v3.py           -- ViT sliding-window detector
  Stage 1b  2_point_detection_adaptive_nms.py -- adaptive x-column NMS
                                                 (mode_xs returned as the x-grid)
  Stage 2   3_segment_detection.py            -- directional-probe segment detector
  Stage 3   4_segment_refinement.py           -- mode_xs grid segment refinement
  Stage 4   5_correction.py                  -- greedy SSIM correction

Outputs (under ../experiments/validation_results/)
---------------------------------------------------
  plots/                  -- generated synthetic plot PNGs + GT JSON files
  results/
    validation_metrics.csv  -- per-class TP/FP/FN/Precision/Recall/F1 table
    seg_refined_overview.jpg -- refined segment visualisation (6 plots)
    seg_comparison.jpg       -- raw vs refined side-by-side (first 6 plots)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Resolve paths relative to src/ ───────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "chart_marker_net_v3.pth"
DETECTOR_PY  = SRC_DIR / "chart_marker_detector_v3.py"

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Validate the chart digitisation pipeline.")
parser.add_argument("--n",          type=int,  default=10,
                    help="Number of synthetic plots to generate (default 10).")
parser.add_argument("--plots-dir",  type=str,  default=None,
                    help="Use existing plots directory instead of generating new ones.")
parser.add_argument("--out-dir",    type=str,  default=None,
                    help="Output directory (default: ../experiments/validation_results).")
parser.add_argument("--no-figures", action="store_true",
                    help="Skip saving visualisation figures.")
parser.add_argument("--conf",       type=float, default=0.65,
                    help="Detector confidence threshold (default 0.65).")
parser.add_argument("--stride",     type=int,   default=8,
                    help="Sliding-window stride in pixels (default 8).")
args = parser.parse_args()

N_VAL      = args.n
CONF_THRESH = args.conf
STRIDE      = args.stride
SAVE_FIGS   = not args.no_figures

OUT_DIR  = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "experiments" / "validation_results"
PLOT_DIR = Path(args.plots_dir) if args.plots_dir else OUT_DIR / "plots"
RES_DIR  = OUT_DIR / "results"
for d in [PLOT_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load modules ──────────────────────────────────────────────────────────────
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

print("Loading pipeline modules …")
gen_mod = _load("chart_marker_detector_v3",       DETECTOR_PY)
mod2    = _load("adaptive_nms",                   SRC_DIR / "2_point_detection_adaptive_nms.py")
mod3    = _load("segment_detection",              SRC_DIR / "3_segment_detection.py")
mod4    = _load("segment_refinement",             SRC_DIR / "4_segment_refinement.py")
mod5    = _load("correction",                     SRC_DIR / "5_correction.py")

CLASS_NAMES = gen_mod.CLASS_NAMES
N_CLASSES   = len(CLASS_NAMES)
MIN_SEP     = int(round(gen_mod.P * 1.5))   # match-radius for TP counting

# ── Stage 0: Generate synthetic plots ─────────────────────────────────────────
if args.plots_dir is None:
    print(f"\n=== Generating {N_VAL} synthetic plots → {PLOT_DIR} ===")
    seeds = [random.randint(0, 2**31) for _ in range(N_VAL)]
    for i, seed in enumerate(seeds):
        img_p, _ = gen_mod.generate_one_plot((i, str(PLOT_DIR), seed))
        print(f"  [{i+1}/{N_VAL}] {Path(img_p).name}")
else:
    print(f"\n=== Using existing plots from {PLOT_DIR} ===")

plot_paths = sorted(PLOT_DIR.glob("*.png"))[:N_VAL]
# GT files are named gt_NNNNN.json alongside plot_NNNNN.png
gt_paths   = [p.parent / p.name.replace("plot_", "gt_").replace(".png", ".json") for p in plot_paths]

if not plot_paths:
    sys.exit(f"ERROR: No PNG files found in {PLOT_DIR}")

# ── Run pipeline ──────────────────────────────────────────────────────────────
print(f"\n=== Running pipeline on {len(plot_paths)} plots ===")
records = []

for i, (img_path, gt_path) in enumerate(zip(plot_paths, gt_paths)):
    t0 = time.time()
    print(f"  [{i+1}/{len(plot_paths)}] {img_path.name} …", end=" ", flush=True)

    # Load GT
    gt_pts = []
    if gt_path.exists():
        with open(gt_path) as f:
            gt = json.load(f)
        gt_pts = gt.get("points", [])

    # Stage 1a + 1b: detection + adaptive NMS
    result2 = mod2.detect_with_adaptive_nms(
        img_path         = str(img_path),
        model_path       = str(MODEL_PATH),
        known_classes    = CLASS_NAMES[:-1],
        detector_py_path = str(DETECTOR_PY),
        conf_thresh      = CONF_THRESH,
        stride           = STRIDE,
    )
    kept     = result2["kept"]
    mode_xs  = result2["mode_xs"]
    d_est    = result2["d_est"]
    img_bgr  = result2["img_bgr"]

    # Use KDE mode_xs directly as the x-grid (the key fix)
    grid_xs = sorted(float(x) for x in mode_xs)

    # Stage 2: raw segment detection
    segs_raw = mod3.detect(img_bgr)

    # Stage 3: segment refinement with mode_xs grid
    segs_ref, ref_log = mod4.refine(segs_raw, grid_xs)

    # Stage 4: greedy SSIM correction (optional — comment out to skip for speed)
    # segs_final = mod5.run_correction(
    #     img_path         = str(img_path),
    #     model_path       = str(MODEL_PATH),
    #     detector_py_path = str(DETECTOR_PY),
    #     known_classes    = CLASS_NAMES[:-1],
    #     out_dir          = None,
    #     mode_xs          = mode_xs,
    # )["history"]
    segs_final = segs_ref  # use refined segments directly when Stage 4 is skipped

    elapsed = time.time() - t0
    print(f"kept={len(kept)} pts  grid={len(grid_xs)} cols  "
          f"raw={len(segs_raw)}→refined={len(segs_ref)} segs  ({elapsed:.1f}s)")

    records.append({
        "name":     img_path.name,
        "img_bgr":  img_bgr,
        "kept":     kept,
        "gt_pts":   gt_pts,
        "grid_xs":  grid_xs,
        "d_est":    d_est,
        "segs_raw": segs_raw,
        "segs_ref": segs_ref,
        "ref_log":  ref_log,
    })

# ── Compute point-detection metrics ──────────────────────────────────────────
print("\n=== Computing point-detection metrics ===")
agg = {c: {"TP": 0, "FP": 0, "FN": 0} for c in range(N_CLASSES)}

for r in records:
    gt_by_cls  = {c: [] for c in range(N_CLASSES)}
    det_by_cls = {c: [] for c in range(N_CLASSES)}

    for pt in r["gt_pts"]:
        ci = pt.get("class_idx", -1)
        if 0 <= ci < N_CLASSES:
            gt_by_cls[ci].append((pt["cx"], pt["cy"]))

    for d in r["kept"]:
        cn = d.get("class_name", "")
        ci = CLASS_NAMES.index(cn) if cn in CLASS_NAMES else -1
        if 0 <= ci < N_CLASSES:
            det_by_cls[ci].append((d["cx"], d["cy"]))

    for c in range(N_CLASSES):
        gts = list(gt_by_cls[c])
        dts = list(det_by_cls[c])
        tp = 0
        for dx, dy in dts:
            best_dist, best_gi = float("inf"), -1
            for gi, (gx, gy) in enumerate(gts):
                dist = math.hypot(dx - gx, dy - gy)
                if dist < best_dist and dist < MIN_SEP:
                    best_dist, best_gi = dist, gi
            if best_gi >= 0:
                tp += 1
                gts.pop(best_gi)
        fp = len(dts) - tp
        fn = len(gt_by_cls[c]) - tp
        agg[c]["TP"] += tp
        agg[c]["FP"] += fp
        agg[c]["FN"] += fn

# Print and save metrics table
rows = []
for c in range(N_CLASSES):
    tp, fp, fn = agg[c]["TP"], agg[c]["FP"], agg[c]["FN"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else float("nan")
    rows.append({"class": CLASS_NAMES[c], "TP": tp, "FP": fp, "FN": fn,
                 "Precision": prec, "Recall": rec, "F1": f1})

print(f"\n{'Class':<22} {'TP':>5} {'FP':>5} {'FN':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
print("-" * 64)
for r in rows:
    p_s = f"{r['Precision']:.3f}" if not math.isnan(r['Precision']) else "  N/A"
    r_s = f"{r['Recall']:.3f}"    if not math.isnan(r['Recall'])    else "  N/A"
    f_s = f"{r['F1']:.3f}"        if not math.isnan(r['F1'])        else "  N/A"
    print(f"{r['class']:<22} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5}  {p_s:>6}  {r_s:>6}  {f_s:>6}")

csv_path = RES_DIR / "validation_metrics.csv"
with open(csv_path, "w") as f:
    f.write("class,TP,FP,FN,Precision,Recall,F1\n")
    for r in rows:
        p_s = f"{r['Precision']:.4f}" if not math.isnan(r['Precision']) else "nan"
        r_s = f"{r['Recall']:.4f}"    if not math.isnan(r['Recall'])    else "nan"
        f_s = f"{r['F1']:.4f}"        if not math.isnan(r['F1'])        else "nan"
        f.write(f"{r['class']},{r['TP']},{r['FP']},{r['FN']},{p_s},{r_s},{f_s}\n")
print(f"\nMetrics saved → {csv_path}")

# ── Visualisation figures ─────────────────────────────────────────────────────
if SAVE_FIGS and records:
    print("\n=== Saving visualisation figures ===")

    def _draw_segs(img_bgr, segs, color_bgr=(40, 40, 220), thickness=2):
        out = img_bgr.copy()
        for x1, y1, x2, y2 in segs:
            cv2.line(out, (int(round(x1)), int(round(y1))),
                     (int(round(x2)), int(round(y2))), color_bgr, thickness)
        return out

    def _draw_grid(img_bgr, grid_xs, color_bgr=(0, 180, 0), thickness=1):
        out = img_bgr.copy()
        H = img_bgr.shape[0]
        for gx in grid_xs:
            cv2.line(out, (int(round(gx)), 0), (int(round(gx)), H), color_bgr, thickness)
        return out

    n_show = min(6, len(records))
    show   = records[:n_show]

    # Figure 1: refined segments overview
    cols = min(3, n_show)
    rows_fig = math.ceil(n_show / cols)
    fig1, axes1 = plt.subplots(rows_fig, cols, figsize=(cols * 6, rows_fig * 4))
    axes1 = np.array(axes1).flatten()
    for idx, rec in enumerate(show):
        img = _draw_segs(_draw_grid(rec["img_bgr"], rec["grid_xs"]), rec["segs_ref"])
        axes1[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes1[idx].axis("off")
        log = rec["ref_log"]
        axes1[idx].set_title(
            f"{rec['name']}\n"
            f"grid={len(rec['grid_xs'])}cols  "
            f"refined={len(rec['segs_ref'])}  "
            f"(pruned={log.get('n_pruned_short',0)}, "
            f"cuts={log.get('n_cuts',0)}, "
            f"snapped={log.get('n_endpoints_snapped',0)})",
            fontsize=7)
    for ax in axes1[n_show:]:
        ax.axis("off")
    seg_p = mpatches.Patch(color=(40/255, 40/255, 220/255), label="Refined segment (blue)")
    grd_p = mpatches.Patch(color=(0, 180/255, 0),           label="Grid — KDE mode_xs (green)")
    fig1.legend(handles=[seg_p, grd_p], loc="lower center", ncol=2, fontsize=9)
    fig1.suptitle("Stage 3 — Refined Segments with KDE mode_xs Grid", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    p1 = RES_DIR / "seg_refined_overview.jpg"
    plt.savefig(str(p1), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p1}")

    # Figure 2: raw vs refined comparison
    fig2, axes2 = plt.subplots(n_show, 2, figsize=(14, n_show * 3.5))
    if n_show == 1:
        axes2 = axes2[np.newaxis, :]
    for idx, rec in enumerate(show):
        raw_rgb = cv2.cvtColor(
            _draw_segs(rec["img_bgr"], rec["segs_raw"], color_bgr=(220, 40, 40)),
            cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(
            _draw_segs(_draw_grid(rec["img_bgr"], rec["grid_xs"]), rec["segs_ref"]),
            cv2.COLOR_BGR2RGB)
        axes2[idx, 0].imshow(raw_rgb)
        axes2[idx, 0].axis("off")
        axes2[idx, 0].set_title(f"Raw — {len(rec['segs_raw'])} segs", fontsize=8)
        axes2[idx, 1].imshow(ref_rgb)
        axes2[idx, 1].axis("off")
        log = rec["ref_log"]
        axes2[idx, 1].set_title(
            f"Refined — {len(rec['segs_ref'])} segs  "
            f"(pruned={log.get('n_pruned_short',0)}, "
            f"cuts={log.get('n_cuts',0)}, "
            f"snapped={log.get('n_endpoints_snapped',0)})  "
            f"grid={len(rec['grid_xs'])}cols",
            fontsize=8)
        axes2[idx, 0].set_ylabel(rec["name"], fontsize=7, rotation=0, labelpad=80, va="center")
    fig2.suptitle("Stage 2 vs Stage 3 — Raw vs Refined Segments  [KDE mode_xs grid]",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = RES_DIR / "seg_comparison.jpg"
    plt.savefig(str(p2), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p2}")

print("\nValidation complete.")
print(f"Results directory: {RES_DIR}")
