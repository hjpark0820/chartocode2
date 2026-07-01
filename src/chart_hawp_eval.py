"""
chart_hawp_eval.py
==================
Working directory : <project>/src/
ONNX model        : <project>/weights/hawp.onnx

Evaluates HAWP line detection on synthetic chart plots using ONNX Runtime.
No Git clone, no C++ compiler, no pip install hawp needed — just onnxruntime.

SETUP (one-time)
----------------
1. Install onnxruntime:
     pip install onnxruntime-gpu     # NVIDIA GPU
     pip install onnxruntime         # CPU only

2. Download the ONNX model (run once):
     python chart_hawp_eval.py --download-weights

3. Run:
     python chart_hawp_eval.py --n-plots 50

OUTPUT
------
  data/hawp_eval/
    per_plot_metrics.csv   - one row per plot (TP/FP/FN/P/R/F1)
    summary_metrics.csv    - mean +/- std across all plots
    images/                - side-by-side: original | annotated (GT green, pred red)

USAGE
-----
  python chart_hawp_eval.py                    # all plots
  python chart_hawp_eval.py --n-plots 100      # first 100 plots
  python chart_hawp_eval.py --threshold 0.3    # lower = more lines detected
  python chart_hawp_eval.py --image crop.png   # single image, no GT needed
  python chart_hawp_eval.py --download-weights # download model and exit

REQUIREMENTS
------------
  pip install onnxruntime-gpu   (or onnxruntime for CPU-only)
  pip install opencv-python numpy
"""

from __future__ import annotations
import argparse, csv, json, math, sys, time, urllib.request, warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ==============================================================================
#  PATHS
# ==============================================================================
_SRC_DIR     = Path(__file__).parent
SYNTH_DIR    = _SRC_DIR / ".." / "data" / "synthetic_plots"
WEIGHTS_DIR  = _SRC_DIR / ".." / "weights"
EVAL_OUT_DIR = _SRC_DIR / ".." / "data" / "hawp_eval"

ONNX_FILENAME = "hawp.onnx"
ONNX_URL = (
    "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/"
    "108_HAWP/hawp_480x640/hawp_480x640.onnx"
)

# HAWP was exported at this fixed resolution
MODEL_H = 480
MODEL_W = 640

# ==============================================================================
#  CONSTANTS  (identical to chart_deeplsd_eval.py)
# ==============================================================================
MATCH_DIST      = 10.0
MATCH_ANGLE_DEG = 15.0
MIN_SEG_LEN     = 5.0
DEFAULT_THRESH  = 0.5   # line score threshold; lower = more lines


# ==============================================================================
#  PROGRESS BAR  (identical to chart_deeplsd_eval.py)
# ==============================================================================
def _pbar(done: int, total: int, t0: float, width: int = 40,
          prefix: str = "") -> None:
    frac    = done / total if total else 1.0
    filled  = int(width * frac)
    bar     = "\u2588" * filled + "\u2591" * (width - filled)
    elapsed = time.time() - t0
    if frac > 0:
        eta_s = int(elapsed / frac * (1 - frac))
        mm, ss = divmod(eta_s, 60)
        hh, mm = divmod(mm, 60)
        eta = (f"{hh}h{mm:02d}m{ss:02d}s" if hh
               else f"{mm}m{ss:02d}s" if mm
               else f"{ss}s")
    else:
        eta = "--"
    line = (f"\r  {prefix}[{bar}] {done:>{len(str(total))}}/{total}"
            f"  {frac*100:5.1f}%  ETA {eta}")
    print(line, end="", flush=True)
    if done == total:
        print()


# ==============================================================================
#  WEIGHT DOWNLOAD
# ==============================================================================
def download_weights() -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = WEIGHTS_DIR / ONNX_FILENAME
    if dst.exists():
        print(f"  Already exists: {dst}")
        return dst
    print(f"  Downloading {ONNX_FILENAME} ...")
    print(f"  Source: {ONNX_URL}")
    try:
        urllib.request.urlretrieve(ONNX_URL, str(dst))
        print(f"  Saved -> {dst}")
        return dst
    except Exception as e:
        print(f"\n  [ERROR] Automatic download failed: {e}")
        print("\n  Please download manually:")
        print("  1. Visit: https://github.com/PINTO0309/PINTO_model_zoo/tree/main/108_HAWP")
        print("  2. Download any hawp_*.onnx file")
        print(f"  3. Rename it to '{ONNX_FILENAME}' and place in:")
        print(f"     {WEIGHTS_DIR}/")
        sys.exit(1)


# ==============================================================================
#  MODEL LOADER
# ==============================================================================
def load_model(onnx_path: Path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n[ERROR] onnxruntime not found.")
        print("Install:  pip install onnxruntime-gpu")
        print("      or: pip install onnxruntime")
        sys.exit(1)

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        use         = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        device_name = "cuda"
    else:
        use         = ["CPUExecutionProvider"]
        device_name = "cpu"

    session = ort.InferenceSession(str(onnx_path), providers=use)
    print(f"  Loaded HAWP ONNX: {onnx_path.name}  (device={device_name})")

    inp  = session.get_inputs()[0]
    outs = session.get_outputs()
    print(f"  Input : {inp.name}  shape={inp.shape}")
    for o in outs:
        print(f"  Output: {o.name}  shape={o.shape}")

    return session


# ==============================================================================
#  INFERENCE
# ==============================================================================
def run_hawp(session, img_bgr: np.ndarray,
             threshold: float = DEFAULT_THRESH) -> np.ndarray:
    """
    Run HAWP ONNX on a BGR uint8 crop.
    Returns (N, 2, 2) float32: lines[i] = [[x1,y1],[x2,y2]] in crop coords.
    """
    oh, ow = img_bgr.shape[:2]

    # Pre-process: resize to model input size, ImageNet normalise
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resz = cv2.resize(img_rgb, (MODEL_W, MODEL_H),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp_t = ((img_resz - mean) / std).transpose(2, 0, 1)[None]  # (1,3,H,W)

    # Inference
    input_name = session.get_inputs()[0].name
    raw        = session.run(None, {input_name: inp_t})

    # Identify lines (N,4) and scores (N,) from output list
    lines_np = scores_np = None
    for arr in raw:
        a = np.array(arr).squeeze()
        if a.ndim == 2 and a.shape[-1] == 4:
            lines_np = a.astype(np.float32)
        elif a.ndim == 1:
            scores_np = a.astype(np.float32)

    if lines_np is None or len(lines_np) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)
    if scores_np is None:
        scores_np = np.ones(len(lines_np), dtype=np.float32)

    # Scale from model coords back to original crop size
    sx = ow / MODEL_W
    sy = oh / MODEL_H

    filtered = []
    for i, line in enumerate(lines_np):
        if float(scores_np[i]) < threshold:
            continue
        x1, y1 = float(line[0]) * sx, float(line[1]) * sy
        x2, y2 = float(line[2]) * sx, float(line[3]) * sy
        if math.hypot(x2 - x1, y2 - y1) < MIN_SEG_LEN:
            continue
        filtered.append([[x1, y1], [x2, y2]])

    if not filtered:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.array(filtered, dtype=np.float32)


# ==============================================================================
#  GT SEGMENT LOADING  (identical to chart_deeplsd_eval.py)
# ==============================================================================
def load_gt_segments(gt_path: Path, crop_x0: int, crop_y0: int) -> list[dict]:
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    segs = []
    for s in gt.get("segments", []):
        x1 = s["x1"] - crop_x0
        y1 = s["y1"] - crop_y0
        x2 = s["x2"] - crop_x0
        y2 = s["y2"] - crop_y0
        length = math.hypot(x2 - x1, y2 - y1)
        if length < MIN_SEG_LEN:
            continue
        segs.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "length": length,
            "angle_rad": math.atan2(y2 - y1, x2 - x1) % math.pi,
        })
    return segs


# ==============================================================================
#  SEGMENT MATCHING  (identical to chart_deeplsd_eval.py)
# ==============================================================================
def _midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _angle_diff_deg(a1: float, a2: float) -> float:
    diff = abs(a1 - a2) % math.pi
    if diff > math.pi / 2:
        diff = math.pi - diff
    return math.degrees(diff)


def match_segments(pred_lines: np.ndarray, gt_segs: list[dict]) -> dict:
    if len(gt_segs) == 0 and len(pred_lines) == 0:
        return {"TP": 0, "FP": 0, "FN": 0,
                "precision": 1.0, "recall": 1.0, "F1": 1.0}

    gt_mids    = [_midpoint(s["x1"], s["y1"], s["x2"], s["y2"]) for s in gt_segs]
    gt_angles  = [s["angle_rad"] for s in gt_segs]
    gt_matched = [False] * len(gt_segs)
    TP = FP = 0

    for line in pred_lines:
        px1, py1 = float(line[0][0]), float(line[0][1])
        px2, py2 = float(line[1][0]), float(line[1][1])
        pmid     = _midpoint(px1, py1, px2, py2)
        pang     = math.atan2(py2 - py1, px2 - px1) % math.pi
        best_i, best_d = -1, float("inf")
        for i, (gm, ga) in enumerate(zip(gt_mids, gt_angles)):
            if gt_matched[i]:
                continue
            d = math.hypot(pmid[0] - gm[0], pmid[1] - gm[1])
            if d > MATCH_DIST or _angle_diff_deg(pang, ga) > MATCH_ANGLE_DEG:
                continue
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            gt_matched[best_i] = True
            TP += 1
        else:
            FP += 1

    FN        = sum(1 for m in gt_matched if not m)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"TP": TP, "FP": FP, "FN": FN,
            "precision": precision, "recall": recall, "F1": F1}


# ==============================================================================
#  VISUALISATION  (identical to chart_deeplsd_eval.py)
# ==============================================================================
def draw_segments(img_bgr, gt_segs, pred_lines, metrics):
    vis = img_bgr.copy()
    for s in gt_segs:
        cv2.line(vis,
                 (int(round(s["x1"])), int(round(s["y1"]))),
                 (int(round(s["x2"])), int(round(s["y2"]))),
                 (0, 200, 0), 1, cv2.LINE_AA)
    for line in pred_lines:
        cv2.line(vis,
                 (int(round(float(line[0][0]))), int(round(float(line[0][1])))),
                 (int(round(float(line[1][0]))), int(round(float(line[1][1])))),
                 (0, 0, 220), 1, cv2.LINE_AA)
    txt = (f"TP={metrics['TP']} FP={metrics['FP']} FN={metrics['FN']} "
           f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
           f"F1={metrics['F1']:.2f}")
    cv2.putText(vis, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ==============================================================================
#  MAIN EVALUATION LOOP
# ==============================================================================
def evaluate(n_plots=None, onnx_path=None,
             threshold=DEFAULT_THRESH, save_images=True):

    if onnx_path is None:
        onnx_path = WEIGHTS_DIR / ONNX_FILENAME
    if not onnx_path.exists():
        print(f"\n[ERROR] ONNX model not found: {onnx_path}")
        print("Run:  python chart_hawp_eval.py --download-weights")
        sys.exit(1)

    gt_files = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not gt_files:
        print(f"\n[ERROR] No GT files found in {SYNTH_DIR}")
        sys.exit(1)
    if n_plots is not None:
        gt_files = gt_files[:n_plots]

    print(f"\n  Evaluating {len(gt_files)} plots ...")
    session = load_model(onnx_path)

    img_out_dir = EVAL_OUT_DIR / "images"
    if save_images:
        img_out_dir.mkdir(parents=True, exist_ok=True)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_plot_csv = EVAL_OUT_DIR / "per_plot_metrics.csv"
    summary_csv  = EVAL_OUT_DIR / "summary_metrics.csv"
    csv_fields   = ["plot_id", "n_gt_segs", "n_pred_segs",
                    "TP", "FP", "FN", "precision", "recall", "F1"]
    all_metrics  = []
    t0 = time.time()

    with open(per_plot_csv, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        writer.writeheader()

        for i, gt_path in enumerate(gt_files):
            _pbar(i + 1, len(gt_files), t0, prefix="  Processing: ")

            img_path = (gt_path.parent /
                        gt_path.name.replace("gt_", "plot_").replace(".json", ".png"))
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
            pa = gt["pa"]
            x0, y0, x1, y1 = pa["x0"], pa["y0"], pa["x1"], pa["y1"]
            crop_bgr = img_bgr[y0:y1, x0:x1]
            if crop_bgr.size == 0:
                continue

            pred_lines = run_hawp(session, crop_bgr, threshold)
            gt_segs    = load_gt_segments(gt_path, x0, y0)
            metrics    = match_segments(pred_lines, gt_segs)
            metrics["n_gt_segs"]   = len(gt_segs)
            metrics["n_pred_segs"] = len(pred_lines)
            metrics["plot_id"]     = gt_path.stem.replace("gt_", "")
            all_metrics.append(metrics)

            try:
                writer.writerow({k: metrics[k] for k in csv_fields})
            except PermissionError:
                pass

            if save_images:
                vis   = draw_segments(crop_bgr, gt_segs, pred_lines, metrics)
                panel = np.hstack([crop_bgr, vis])
                try:
                    cv2.imwrite(str(img_out_dir / f"plot_{metrics['plot_id']}.png"), panel)
                except Exception:
                    pass

    if not all_metrics:
        print("\n  No results to summarise.")
        return

    # Summary stats
    keys = ["n_gt_segs", "n_pred_segs", "TP", "FP", "FN", "precision", "recall", "F1"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[k] = {
            "mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min":  float(np.min(vals)),  "max": float(np.max(vals)),
            "total": float(np.sum(vals)) if k in (
                "TP", "FP", "FN", "n_gt_segs", "n_pred_segs") else None,
        }

    total_tp = sum(m["TP"] for m in all_metrics)
    total_fp = sum(m["FP"] for m in all_metrics)
    total_fn = sum(m["FN"] for m in all_metrics)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) > 0 else 0.0)

    try:
        with open(summary_csv, "w", newline="", encoding="utf-8") as csvf:
            w = csv.writer(csvf)
            w.writerow(["metric", "mean", "std", "min", "max", "total"])
            for k in keys:
                s = summary[k]
                w.writerow([k, f"{s['mean']:.4f}", f"{s['std']:.4f}",
                             f"{s['min']:.4f}", f"{s['max']:.4f}",
                             f"{s['total']:.0f}" if s["total"] is not None else ""])
            w.writerow(["micro_precision", f"{micro_p:.4f}", "", "", "", ""])
            w.writerow(["micro_recall",    f"{micro_r:.4f}", "", "", "", ""])
            w.writerow(["micro_F1",        f"{micro_f1:.4f}", "", "", "", ""])
    except PermissionError:
        pass

    elapsed = time.time() - t0
    print(f"\n  Evaluation complete in {elapsed:.1f}s  ({len(all_metrics)} plots)")
    print(f"\n  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  {'Total':>8}")
    print("  " + "-" * 52)
    for k in keys:
        s   = summary[k]
        tot = f"{s['total']:.0f}" if s["total"] is not None else ""
        print(f"  {k:<22}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {tot:>8}")
    print(f"\n  Micro precision : {micro_p:.4f}")
    print(f"  Micro recall    : {micro_r:.4f}")
    print(f"  Micro F1        : {micro_f1:.4f}")
    print(f"\n  Per-plot CSV -> {per_plot_csv}")
    print(f"  Summary CSV  -> {summary_csv}")
    if save_images:
        print(f"  Images       -> {img_out_dir}")


# ==============================================================================
#  SINGLE-IMAGE MODE
# ==============================================================================
def detect_image(image_path, onnx_path, threshold=DEFAULT_THRESH, output_path=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read: {image_path}")
        sys.exit(1)
    session    = load_model(onnx_path)
    pred_lines = run_hawp(session, img_bgr, threshold)
    vis = img_bgr.copy()
    for line in pred_lines:
        cv2.line(vis,
                 (int(round(float(line[0][0]))), int(round(float(line[0][1])))),
                 (int(round(float(line[1][0]))), int(round(float(line[1][1])))),
                 (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(vis, f"{len(pred_lines)} segments", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{len(pred_lines)} segments", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"  Saved -> {output_path}")
    else:
        cv2.imshow("HAWP", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==============================================================================
#  ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate HAWP (ONNX runtime) on synthetic chart segments.")
    parser.add_argument("--download-weights", action="store_true",
                        help="Download ONNX model to weights/ and exit.")
    parser.add_argument("--n-plots", type=int, default=None,
                        help="Number of plots to evaluate (default: all).")
    parser.add_argument("--onnx", type=str, default=None,
                        help=f"Path to ONNX file (default: weights/{ONNX_FILENAME}).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESH,
                        help=f"Score threshold 0-1 (default {DEFAULT_THRESH}). "
                             "Lower = more lines.")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving annotated images.")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path (no GT needed).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for --image result.")
    args = parser.parse_args()

    if args.download_weights:
        download_weights()
        sys.exit(0)

    onnx_path = Path(args.onnx) if args.onnx else WEIGHTS_DIR / ONNX_FILENAME

    if args.image:
        detect_image(args.image, onnx_path, args.threshold, args.output)
    else:
        evaluate(n_plots=args.n_plots, onnx_path=onnx_path,
                 threshold=args.threshold, save_images=not args.no_images)
