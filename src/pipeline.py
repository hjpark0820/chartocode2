"""
pipeline.py
===========
Unified chart-digitisation pipeline — Stages 1 through 5.

Stages
------
  Stage 1a  1_point_detection_v3.py           -- ViT sliding-window detector
             (also contains the synthetic-plot generator used for validation)
  Stage 1b  2_point_detection_adaptive_nms.py -- adaptive x-column NMS
                                                 (produces KDE mode_xs grid)
  Stage 2   3_segment_detection.py            -- directional-probe segment detector
  Stage 3   4_segment_refinement.py           -- mode_xs grid segment refinement
  Stage 4   5_correction.py                  -- greedy SSIM correction

Usage — run on a single image
------------------------------
  python pipeline.py image path/to/chart.png

Usage — run validation on N synthetic plots
--------------------------------------------
  python pipeline.py validate                         # 10 plots (default)
  python pipeline.py validate --n 20                  # 20 plots
  python pipeline.py validate --plots-dir path/       # use existing plots
  python pipeline.py validate --skip-correction       # Stages 1-3 only (fast)

All outputs are written to:
  ../experiments/pipeline_out/<plot_name>/   for single-image mode
  ../experiments/validation_results/         for validation mode

Output files (per plot)
-----------------------
  iter01.jpg … iterNN.jpg          -- per-iteration correction panels (Stage 4)
  convergence.jpg                  -- 1-SSIM convergence curve (Stage 4)
  all_iterations_combined.jpg      -- all iterations stacked (Stage 4)
  seg_refined_overview.jpg         -- refined segments + grid (validation only)
  seg_comparison.jpg               -- raw vs refined (validation only)
  validation_metrics.csv           -- per-class TP/FP/FN/Prec/Rec/F1 (validation only)
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
DETECTOR_PY  = SRC_DIR / "1_point_detection_v3.py"

# ── Module loader ─────────────────────────────────────────────────────────────
def _load(name: str, path: Path):
    """Dynamically load a module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def _load_all_modules():
    """Load all 5 pipeline modules and return them as a namespace dict."""
    print("Loading pipeline modules …")
    # Stage 1a: detector / generator
    gen_mod = _load("chart_marker_detector_v3", DETECTOR_PY)
    # Stage 1b: adaptive NMS
    mod2 = _load("adaptive_nms",        SRC_DIR / "2_point_detection_adaptive_nms.py")
    # Stage 2: segment detection
    # Stage 5 imports this as 'segment_detector' — register under both names
    mod3 = _load("segment_detector",    SRC_DIR / "3_segment_detection.py")
    _load("segment_detection",          SRC_DIR / "3_segment_detection.py")
    # Stage 3: segment refinement
    # Stage 5 imports this as 'refine_segments' — register under both names
    mod4 = _load("refine_segments",     SRC_DIR / "4_segment_refinement.py")
    _load("segment_refinement",         SRC_DIR / "4_segment_refinement.py")
    # Stage 4: greedy SSIM correction
    mod5 = _load("correction",          SRC_DIR / "5_correction.py")
    return gen_mod, mod2, mod3, mod4, mod5


# ── Single-image pipeline ─────────────────────────────────────────────────────
def run_single(img_path: str, out_dir: str = None,
               conf_thresh: float = 0.65, stride: int = 8,
               skip_correction: bool = False):
    """
    Run the full pipeline (Stages 1–4) on a single chart image.

    Parameters
    ----------
    img_path        : path to the input PNG/JPG
    out_dir         : output directory (default: ../experiments/pipeline_out/<name>)
    conf_thresh     : detector confidence threshold
    stride          : sliding-window stride (px)
    skip_correction : if True, stop after Stage 3 (no SSIM correction)

    Returns
    -------
    dict with keys:
        kept        -- final active detections (list of dicts)
        grid_xs     -- KDE mode x-positions (list of float)
        segs_raw    -- raw segments from Stage 2
        segs_ref    -- refined segments from Stage 3
        history     -- correction convergence history (Stage 4), or []
        out_dir     -- path to the output directory
    """
    gen_mod, mod2, mod3, mod4, mod5 = _load_all_modules()

    img_path = str(img_path)
    plot_name = Path(img_path).stem
    if out_dir is None:
        out_dir = str(PROJECT_ROOT / "experiments" / "pipeline_out" / plot_name)
    os.makedirs(out_dir, exist_ok=True)

    CLASS_NAMES = gen_mod.CLASS_NAMES

    print(f"\n{'='*65}")
    print(f"  Image : {img_path}")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}")

    # ── Stage 1a + 1b: detection + adaptive NMS ───────────────────────────
    print("\n[Stage 1] Point detection + adaptive NMS …")
    t0 = time.time()
    result2 = mod2.detect_with_adaptive_nms(
        img_path         = img_path,
        model_path       = str(MODEL_PATH),
        known_classes    = CLASS_NAMES[:-1],
        detector_py_path = str(DETECTOR_PY),
        conf_thresh      = conf_thresh,
        stride           = stride,
        out_dir          = os.path.join(out_dir, "stage1_nms"),
    )
    kept    = result2["kept"]
    mode_xs = result2["mode_xs"]
    d_est   = result2["d_est"]
    img_bgr = result2["img_bgr"]
    grid_xs = sorted(float(x) for x in mode_xs)
    print(f"  kept={len(kept)} pts  |  grid={len(grid_xs)} cols  "
          f"|  d_est={d_est:.1f}px  |  ({time.time()-t0:.1f}s)")

    # ── Stage 2: raw segment detection ────────────────────────────────────
    print("\n[Stage 2] Segment detection …")
    t0 = time.time()
    segs_raw = mod3.detect(img_bgr)
    print(f"  raw segments={len(segs_raw)}  ({time.time()-t0:.1f}s)")

    # ── Stage 3: segment refinement ───────────────────────────────────────
    print("\n[Stage 3] Segment refinement (mode_xs grid) …")
    t0 = time.time()
    segs_ref, ref_log = mod4.refine(segs_raw, grid_xs)
    print(f"  refined={len(segs_ref)}  "
          f"(pruned={ref_log.get('n_pruned_short',0)}, "
          f"cuts={ref_log.get('n_cuts',0)}, "
          f"snapped={ref_log.get('n_endpoints_snapped',0)})  "
          f"({time.time()-t0:.1f}s)")

    # Save Stage 1–3 visualisation
    _save_stages123_figure(img_bgr, segs_raw, segs_ref, grid_xs,
                           os.path.join(out_dir, "stages1_3_overview.jpg"),
                           plot_name)

    # ── Stage 4: greedy SSIM correction ───────────────────────────────────
    history = []
    if not skip_correction:
        print("\n[Stage 4] Greedy SSIM correction …")
        t0 = time.time()
        corr_result = mod5.run_correction(
            img_path         = img_path,
            model_path       = str(MODEL_PATH),
            detector_py_path = str(DETECTOR_PY),
            known_classes    = CLASS_NAMES[:-1],
            out_dir          = out_dir,
            mode_xs          = mode_xs,
        )
        history = corr_result["history"]
        print(f"  Correction done  ({time.time()-t0:.1f}s)")
        print(f"\n  Output files in: {out_dir}")
    else:
        print("\n[Stage 4] Skipped (--skip-correction).")
        print(f"\n  Output files in: {out_dir}")

    return {
        "kept":     kept,
        "grid_xs":  grid_xs,
        "segs_raw": segs_raw,
        "segs_ref": segs_ref,
        "history":  history,
        "out_dir":  out_dir,
    }


# ── Validation pipeline ───────────────────────────────────────────────────────
def run_validation(n: int = 10,
                   plots_dir: str = None,
                   out_dir: str = None,
                   conf_thresh: float = 0.65,
                   stride: int = 8,
                   skip_correction: bool = False):
    """
    Generate N synthetic plots and run the full pipeline on each.

    Parameters
    ----------
    n               : number of synthetic plots to generate
    plots_dir       : if given, use existing plots instead of generating new ones
    out_dir         : root output directory
    conf_thresh     : detector confidence threshold
    stride          : sliding-window stride (px)
    skip_correction : if True, stop after Stage 3
    """
    gen_mod, mod2, mod3, mod4, mod5 = _load_all_modules()

    CLASS_NAMES = gen_mod.CLASS_NAMES
    N_CLASSES   = len(CLASS_NAMES)
    MIN_SEP     = int(round(gen_mod.P * 1.5))

    if out_dir is None:
        out_dir = str(PROJECT_ROOT / "experiments" / "validation_results")
    PLOT_DIR = Path(plots_dir) if plots_dir else Path(out_dir) / "plots"
    RES_DIR  = Path(out_dir) / "results"
    for d in [PLOT_DIR, RES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Generate plots ────────────────────────────────────────────────────
    if plots_dir is None:
        print(f"\n=== Generating {n} synthetic plots → {PLOT_DIR} ===")
        seeds = [random.randint(0, 2**31) for _ in range(n)]
        for i, seed in enumerate(seeds):
            img_p, _ = gen_mod.generate_one_plot((i, str(PLOT_DIR), seed))
            print(f"  [{i+1}/{n}] {Path(img_p).name}")
    else:
        print(f"\n=== Using existing plots from {PLOT_DIR} ===")

    plot_paths = sorted(PLOT_DIR.glob("*.png"))[:n]
    gt_paths   = [p.parent / p.name.replace("plot_", "gt_").replace(".png", ".json")
                  for p in plot_paths]

    if not plot_paths:
        sys.exit(f"ERROR: No PNG files found in {PLOT_DIR}")

    # ── Run pipeline on each plot ─────────────────────────────────────────
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

        # Stage 1a + 1b
        result2 = mod2.detect_with_adaptive_nms(
            img_path         = str(img_path),
            model_path       = str(MODEL_PATH),
            known_classes    = CLASS_NAMES[:-1],
            detector_py_path = str(DETECTOR_PY),
            conf_thresh      = conf_thresh,
            stride           = stride,
        )
        kept    = result2["kept"]
        mode_xs = result2["mode_xs"]
        d_est   = result2["d_est"]
        img_bgr = result2["img_bgr"]
        grid_xs = sorted(float(x) for x in mode_xs)

        # Stage 2
        segs_raw = mod3.detect(img_bgr)

        # Stage 3
        segs_ref, ref_log = mod4.refine(segs_raw, grid_xs)

        # Stage 4 (optional)
        history = []
        if not skip_correction:
            corr_out = str(RES_DIR / "correction" / img_path.stem)
            corr_res = mod5.run_correction(
                img_path         = str(img_path),
                model_path       = str(MODEL_PATH),
                detector_py_path = str(DETECTOR_PY),
                known_classes    = CLASS_NAMES[:-1],
                out_dir          = corr_out,
                mode_xs          = mode_xs,
            )
            history = corr_res["history"]

        elapsed = time.time() - t0
        print(f"kept={len(kept)}  grid={len(grid_xs)}cols  "
              f"raw={len(segs_raw)}→ref={len(segs_ref)}  "
              f"corr_iters={len(history)-1 if history else 'skip'}  "
              f"({elapsed:.1f}s)")

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
            "history":  history,
        })

    # ── Point-detection metrics ───────────────────────────────────────────
    print("\n=== Point-detection metrics ===")
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
            agg[c]["TP"] += tp
            agg[c]["FP"] += len(dts) - tp
            agg[c]["FN"] += len(gt_by_cls[c]) - tp

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

    # ── Visualisation figures ─────────────────────────────────────────────
    print("\n=== Saving visualisation figures ===")
    n_show = min(6, len(records))
    show   = records[:n_show]

    # Figure 1: refined segments overview (first 6 plots)
    cols     = min(3, n_show)
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
            f"grid={len(rec['grid_xs'])}cols  refined={len(rec['segs_ref'])}  "
            f"(pruned={log.get('n_pruned_short',0)}, "
            f"cuts={log.get('n_cuts',0)}, "
            f"snapped={log.get('n_endpoints_snapped',0)})",
            fontsize=7)
    for ax in axes1[n_show:]:
        ax.axis("off")
    seg_p = mpatches.Patch(color=(220/255, 40/255, 40/255), label="Refined segment (red)")
    grd_p = mpatches.Patch(color=(0, 180/255, 0),           label="Grid — KDE mode_xs (green)")
    fig1.legend(handles=[seg_p, grd_p], loc="lower center", ncol=2, fontsize=9)
    fig1.suptitle("Stages 1–3 — Refined Segments with KDE mode_xs Grid",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    p1 = RES_DIR / "seg_refined_overview.jpg"
    plt.savefig(str(p1), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p1}")

    # Figure 2: raw vs refined comparison (first 6 plots)
    fig2, axes2 = plt.subplots(n_show, 2, figsize=(14, n_show * 3.5))
    if n_show == 1:
        axes2 = axes2[np.newaxis, :]
    for idx, rec in enumerate(show):
        raw_rgb = cv2.cvtColor(
            _draw_segs(rec["img_bgr"], rec["segs_raw"], color_bgr=(40, 40, 220)),
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
        axes2[idx, 0].set_ylabel(rec["name"], fontsize=7, rotation=0,
                                  labelpad=80, va="center")
    fig2.suptitle("Stage 2 vs Stage 3 — Raw vs Refined Segments  [KDE mode_xs grid]",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = RES_DIR / "seg_comparison.jpg"
    plt.savefig(str(p2), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p2}")

    print(f"\nValidation complete.  Results → {RES_DIR}")
    return records


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _draw_segs(img_bgr, segs, color_bgr=(220, 40, 40), thickness=2):
    out = img_bgr.copy()
    for x1, y1, x2, y2 in segs:
        cv2.line(out, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), color_bgr, thickness)
    return out


def _draw_grid(img_bgr, grid_xs, color_bgr=(0, 180, 0), thickness=1):
    out = img_bgr.copy()
    H   = img_bgr.shape[0]
    for gx in grid_xs:
        cv2.line(out, (int(round(gx)), 0), (int(round(gx)), H), color_bgr, thickness)
    return out


def _save_stages123_figure(img_bgr, segs_raw, segs_ref, grid_xs, out_path, plot_name):
    """Save a 3-panel figure: original | raw segs | refined segs + grid."""
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raw_rgb  = cv2.cvtColor(_draw_segs(img_bgr, segs_raw, (40, 40, 220)), cv2.COLOR_BGR2RGB)
    ref_rgb  = cv2.cvtColor(_draw_segs(_draw_grid(img_bgr, grid_xs), segs_ref), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, img, title in zip(axes,
                               [orig_rgb, raw_rgb, ref_rgb],
                               ["Original",
                                f"Stage 2 — Raw segments ({len(segs_raw)})",
                                f"Stage 3 — Refined segments ({len(segs_ref)})  "
                                f"grid={len(grid_xs)} cols"]):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"{plot_name} — Stages 1–3 overview", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Stages 1–3 figure → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _build_parser():
    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Chart digitisation pipeline — Stages 1–5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Run on a single image (all 5 stages):
  python pipeline.py image path/to/chart.png

  # Run on a single image, skip Stage 4 correction (faster):
  python pipeline.py image path/to/chart.png --skip-correction

  # Validation: generate 10 plots and run all stages:
  python pipeline.py validate

  # Validation: 20 plots, skip Stage 4:
  python pipeline.py validate --n 20 --skip-correction

  # Validation: use existing plots:
  python pipeline.py validate --plots-dir ../data/synthetic_plots
""")

    sub = parser.add_subparsers(dest="mode", required=True)

    # ── image sub-command ────────────────────────────────────────────────
    p_img = sub.add_parser("image", help="Run pipeline on a single chart image.")
    p_img.add_argument("img_path", help="Path to the input chart image (PNG/JPG).")
    p_img.add_argument("--out-dir", default=None,
                       help="Output directory  "
                            "(default: ../experiments/pipeline_out/<name>).")
    p_img.add_argument("--skip-correction", action="store_true",
                       help="Stop after Stage 3 (skip SSIM correction).")
    p_img.add_argument("--conf",   type=float, default=0.65,
                       help="Detector confidence threshold (default 0.65).")
    p_img.add_argument("--stride", type=int,   default=8,
                       help="Sliding-window stride in pixels (default 8).")

    # ── validate sub-command ─────────────────────────────────────────────
    p_val = sub.add_parser("validate",
                           help="Generate synthetic plots and run the pipeline.")
    p_val.add_argument("--n",          type=int, default=10,
                       help="Number of synthetic plots to generate (default 10).")
    p_val.add_argument("--plots-dir",  default=None,
                       help="Use existing plots directory instead of generating.")
    p_val.add_argument("--out-dir",    default=None,
                       help="Root output directory  "
                            "(default: ../experiments/validation_results).")
    p_val.add_argument("--skip-correction", action="store_true",
                       help="Stop after Stage 3 (skip SSIM correction, faster).")
    p_val.add_argument("--conf",   type=float, default=0.65,
                       help="Detector confidence threshold (default 0.65).")
    p_val.add_argument("--stride", type=int,   default=8,
                       help="Sliding-window stride in pixels (default 8).")

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    if args.mode == "image":
        run_single(
            img_path        = args.img_path,
            out_dir         = args.out_dir,
            conf_thresh     = args.conf,
            stride          = args.stride,
            skip_correction = args.skip_correction,
        )

    elif args.mode == "validate":
        run_validation(
            n               = args.n,
            plots_dir       = args.plots_dir,
            out_dir         = args.out_dir,
            conf_thresh     = args.conf,
            stride          = args.stride,
            skip_correction = args.skip_correction,
        )
