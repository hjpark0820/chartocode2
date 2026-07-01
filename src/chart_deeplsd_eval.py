"""
chart_deeplsd_eval.py
=====================
Working directory : <project>/src/
Weights directory : <project>/weights/   (deeplsd_md.tar or deeplsd_wireframe.tar)

Evaluates the PRETRAINED DeepLSD model on the synthetic dose-response charts
produced by chart_marker_detector_v3.py (or chart_segment_detector.py).

This script is FULLY SELF-CONTAINED: it bundles the VGGUNet backbone and
DeepLSD heads directly, so NO DeepLSD package installation is required.
The only dependencies are torch, torchvision, opencv-python, numpy, matplotlib.

The pretrained weights (deeplsd_md.tar / deeplsd_wireframe.tar) are loaded
directly from the .tar checkpoint files.

Post-processing uses OpenCV's built-in LSD (cv2.createLineSegmentDetector)
guided by the predicted distance and angle fields, replacing the pytlsd step.

GT segment info is used ONLY for computing validation metrics (P/R/F1).

OUTPUT
------
  data/deeplsd_eval/
    per_plot_metrics.csv        — one row per plot
    summary_metrics.csv         — overall mean ± std across all plots
    images/plot_NNNNN.png       — annotated image: GT (green) vs predicted (red)

USAGE
-----
  # Download weights first (run once):
  python chart_deeplsd_eval.py --download-weights

  # Run evaluation on all available plots:
  python chart_deeplsd_eval.py

  # Run on a specific number of plots:
  python chart_deeplsd_eval.py --n-plots 100

  # Run on a single image (already cropped to plotting area):
  python chart_deeplsd_eval.py --image path/to/crop.png

REQUIREMENTS
------------
  pip install torch torchvision opencv-python matplotlib numpy
  (No C++ compiler or DeepLSD package needed)
"""

from __future__ import annotations
import argparse, csv, json, math, os, sys, time, urllib.request, warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
_SRC_DIR      = Path(__file__).parent
SYNTH_DIR     = _SRC_DIR / ".." / "data" / "synthetic_plots"
WEIGHTS_DIR   = _SRC_DIR / ".." / "weights"
EVAL_OUT_DIR  = _SRC_DIR / ".." / "data" / "deeplsd_eval"

WEIGHT_URLS = {
    "deeplsd_md.tar":        "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar",
    "deeplsd_wireframe.tar": "https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar",
}
DEFAULT_WEIGHT = "deeplsd_md.tar"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MATCH_DIST      = 10.0   # px — midpoint distance tolerance for TP matching
MATCH_ANGLE_DEG = 15.0   # degrees — angular tolerance for TP matching
MIN_SEG_LEN     = 5.0    # px — ignore GT segments shorter than this

LINE_NEIGHBORHOOD = 5    # DeepLSD default: used in normalize_df

# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════
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
    line = f"\r  {prefix}[{bar}] {done:>{len(str(total))}}/{total}  {frac*100:5.1f}%  ETA {eta}"
    print(line, end="", flush=True)
    if done == total:
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED DEEPLSD BACKBONE
#  Replicates the VGGUNet + DF/angle heads from the DeepLSD paper.
#  Weight keys match those in the official .tar checkpoints exactly.
# ══════════════════════════════════════════════════════════════════════════════
class _VGGUNet(nn.Module):
    """
    VGG-style U-Net backbone used in DeepLSD.
    Input:  (B, 1, H, W) grayscale float32 in [0, 1]
    Output: (B, 64, H, W) feature map

    Exact channel sizes verified against the official deeplsd_md.tar checkpoint:
      Encoder:  1→64, 64→128, 128→256, 256→512
      deblock4: 512→256→256   (no skip at this level)
      deblock3: (256+256)→256→128  (skip from block3: 256ch)
      deblock2: (128+128)→128→64   (skip from block2: 128ch)
      deblock1: (64+64)→64→64      (skip from block1: 64ch)
    """
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def _enc(ci, co):
            """Encoder block: two Conv-ReLU-BN layers, same output channels."""
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co),
                nn.Conv2d(co, co, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co),
            )

        def _dec(ci, cmid, co):
            """Decoder block: Conv ci→cmid then Conv cmid→co (asymmetric)."""
            return nn.Sequential(
                nn.Conv2d(ci,   cmid, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cmid),
                nn.Conv2d(cmid, co,   3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co),
            )

        # Encoder
        self.block1 = _enc(1,   64)
        self.block2 = _enc(64,  128)
        self.block3 = _enc(128, 256)
        self.block4 = _enc(256, 512)

        # Decoder  (input channels = upsample_out + skip_out)
        self.deblock4 = _dec(512,       256, 256)   # no skip
        self.deblock3 = _dec(256 + 256, 256, 128)   # skip from block3 (256ch)
        self.deblock2 = _dec(128 + 128, 128,  64)   # skip from block2 (128ch)
        self.deblock1 = _dec( 64 +  64,  64,  64)   # skip from block1 (64ch)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(self.pool(f1))
        f3 = self.block3(self.pool(f2))
        f4 = self.block4(self.pool(f3))

        out = self.deblock4(f4)
        out = self.deblock3(torch.cat([
            F.interpolate(out, f3.shape[2:], mode="bilinear", align_corners=False), f3], dim=1))
        out = self.deblock2(torch.cat([
            F.interpolate(out, f2.shape[2:], mode="bilinear", align_corners=False), f2], dim=1))
        out = self.deblock1(torch.cat([
            F.interpolate(out, f1.shape[2:], mode="bilinear", align_corners=False), f1], dim=1))
        return out   # (B, 64, H, W)


def _head(in_ch: int, out_ch: int, final_act: nn.Module) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
        nn.Conv2d(64,    64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
        nn.Conv2d(64, out_ch, 1),
        final_act,
    )


class DeepLSDNet(nn.Module):
    """
    Self-contained DeepLSD model (backbone + DF head + angle head).
    Loads weights directly from the official .tar checkpoint.
    """
    def __init__(self):
        super().__init__()
        self.backbone   = _VGGUNet()
        self.df_head    = _head(64, 1, nn.ReLU(inplace=True))
        self.angle_head = _head(64, 1, nn.Sigmoid())

    def forward(self, x):
        """
        x: (B, 1, H, W) float32 in [0, 1]
        Returns:
          df    : (B, 1, H, W) — raw distance field (before normalisation)
          angle : (B, 1, H, W) — angle in [0, 1] (mapped to [0, π])
        """
        feat  = self.backbone(x)
        df    = self.df_head(feat)
        angle = self.angle_head(feat)
        return df, angle

    @staticmethod
    def normalize_df(df: torch.Tensor,
                     line_neighborhood: int = LINE_NEIGHBORHOOD) -> torch.Tensor:
        """Convert raw DF output to the normalised log-distance used by DeepLSD."""
        return -torch.log(df / line_neighborhood + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  WEIGHT DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_weights() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in WEIGHT_URLS.items():
        dst = WEIGHTS_DIR / fname
        if dst.exists():
            print(f"  Already exists: {dst}")
            continue
        print(f"  Downloading {fname} …")
        urllib.request.urlretrieve(url, str(dst))
        print(f"  Saved → {dst}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_model(weight_file: str = DEFAULT_WEIGHT,
               device: torch.device | None = None) -> tuple:
    """
    Load the pretrained DeepLSD weights into the self-contained DeepLSDNet.
    Returns (net, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = WEIGHTS_DIR / weight_file
    if not ckpt_path.exists():
        print(f"\n[ERROR] Weight file not found: {ckpt_path}")
        print("Run:  python chart_deeplsd_eval.py --download-weights")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    net = DeepLSDNet()

    # The official checkpoint uses key names like:
    #   backbone.block1.0.weight  → maps directly to our _VGGUNet
    #   df_head.0.weight          → maps directly to our df_head
    #   angle_head.0.weight       → maps directly to our angle_head
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    net = net.to(device).eval()
    print(f"  Loaded DeepLSD weights: {ckpt_path.name}  (device={device})")
    return net, device


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE: predict DF + angle, then extract segments with OpenCV LSD
# ══════════════════════════════════════════════════════════════════════════════
def run_deeplsd(net: DeepLSDNet,
                gray_crop: np.ndarray,
                device: torch.device,
                grad_thresh: float = 3.0) -> np.ndarray:
    """
    Run the DeepLSD model on a uint8 grayscale crop.

    Steps:
      1. Predict distance field (DF) and angle field.
      2. Build a gradient-weighted grayscale image from the DF.
      3. Run OpenCV LSD on that image to extract line segments.
      4. Filter segments using the predicted angle field.

    Returns lines as (N, 2, 2) float32 array: lines[i] = [[x1,y1],[x2,y2]]
    """
    h, w = gray_crop.shape[:2]

    # ── Step 1: predict DF and angle ─────────────────────────────────────────
    inp = torch.tensor(gray_crop, dtype=torch.float32, device=device)[None, None] / 255.0
    with torch.no_grad():
        df_raw, angle_pred = net(inp)

    df_norm  = DeepLSDNet.normalize_df(df_raw)   # (1,1,H,W) log-distance
    df_np    = df_norm[0, 0].cpu().float().numpy()
    angle_np = (angle_pred[0, 0].cpu().float().numpy() * math.pi)  # [0, π]

    # ── Step 2: build gradient-weighted image for LSD ────────────────────────
    # Use the distance field as an attention map: pixels close to a line
    # (small df_norm) get high weight.  We invert and normalise to [0, 255].
    df_inv = np.exp(-df_np)                        # high near lines
    df_u8  = np.clip(df_inv * 255, 0, 255).astype(np.uint8)

    # Blend with original image to preserve gradient information
    gray_f  = gray_crop.astype(np.float32)
    blend   = 0.5 * gray_f + 0.5 * df_inv * 255.0
    blend_u8 = np.clip(blend, 0, 255).astype(np.uint8)

    # ── Step 3: run OpenCV LSD ────────────────────────────────────────────────
    lsd = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_ADV,
        scale=0.8,
        sigma_scale=0.6,
        quant=2.0,
        ang_th=22.5,
        log_eps=0.0,
        density_th=0.7,
        n_bins=1024,
    )
    lines_cv, widths, prec, nfa = lsd.detect(blend_u8)

    if lines_cv is None or len(lines_cv) == 0:
        return np.zeros((0, 2, 2), dtype=np.float32)

    lines_cv = lines_cv.reshape(-1, 4)  # (N, 4): x1,y1,x2,y2

    # ── Step 4: filter by angle consistency with predicted angle field ────────
    filtered = []
    for seg in lines_cv:
        x1, y1, x2, y2 = seg
        length = math.hypot(x2 - x1, y2 - y1)
        if length < MIN_SEG_LEN:
            continue

        # Sample predicted angle at midpoint
        mx = int(round((x1 + x2) / 2.0))
        my = int(round((y1 + y2) / 2.0))
        mx = max(0, min(w - 1, mx))
        my = max(0, min(h - 1, my))
        pred_ang = float(angle_np[my, mx])

        # Detected segment angle
        det_ang = math.atan2(y2 - y1, x2 - x1) % math.pi

        # Angular difference (π-ambiguity)
        diff = abs(pred_ang - det_ang) % math.pi
        if diff > math.pi / 2:
            diff = math.pi - diff

        # Keep if angle is consistent OR if DF value at midpoint is small
        df_val = float(df_np[my, mx])
        if math.degrees(diff) < 30.0 or df_val < grad_thresh:
            filtered.append([[x1, y1], [x2, y2]])

    if not filtered:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.array(filtered, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  GT SEGMENT LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_gt_segments(gt_path: Path,
                     crop_x0: int, crop_y0: int) -> list[dict]:
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


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT MATCHING
# ══════════════════════════════════════════════════════════════════════════════
def _midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _angle_diff_deg(a1: float, a2: float) -> float:
    diff = abs(a1 - a2) % math.pi
    if diff > math.pi / 2:
        diff = math.pi - diff
    return math.degrees(diff)

def match_segments(pred_lines: np.ndarray,
                   gt_segs: list[dict],
                   match_dist: float = MATCH_DIST,
                   match_angle_deg: float = MATCH_ANGLE_DEG) -> dict:
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
        pmid = _midpoint(px1, py1, px2, py2)
        pang = math.atan2(py2 - py1, px2 - px1) % math.pi

        best_i, best_d = -1, float("inf")
        for i, (gm, ga) in enumerate(zip(gt_mids, gt_angles)):
            if gt_matched[i]:
                continue
            d = math.hypot(pmid[0] - gm[0], pmid[1] - gm[1])
            if d > match_dist or _angle_diff_deg(pang, ga) > match_angle_deg:
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


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
def draw_segments(img_bgr: np.ndarray,
                  gt_segs: list[dict],
                  pred_lines: np.ndarray,
                  metrics: dict) -> np.ndarray:
    vis = img_bgr.copy()
    for s in gt_segs:
        cv2.line(vis, (int(round(s["x1"])), int(round(s["y1"]))),
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(n_plots: int | None = None,
             weight_file: str = DEFAULT_WEIGHT,
             save_images: bool = True) -> None:
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not gt_files:
        print(f"\n[ERROR] No GT files found in {SYNTH_DIR}")
        sys.exit(1)
    if n_plots is not None:
        gt_files = gt_files[:n_plots]

    print(f"\n  Evaluating {len(gt_files)} plots …")
    net, device = load_model(weight_file)

    img_out_dir = EVAL_OUT_DIR / "images"
    if save_images:
        img_out_dir.mkdir(parents=True, exist_ok=True)
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_plot_csv = EVAL_OUT_DIR / "per_plot_metrics.csv"
    summary_csv  = EVAL_OUT_DIR / "summary_metrics.csv"
    csv_fields   = ["plot_id", "n_gt_segs", "n_pred_segs",
                    "TP", "FP", "FN", "precision", "recall", "F1"]

    all_metrics: list[dict] = []
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

            gray_crop  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            pred_lines = run_deeplsd(net, gray_crop, device)
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
                vis = draw_segments(crop_bgr, gt_segs, pred_lines, metrics)
                panel = np.hstack([crop_bgr, vis])
                try:
                    cv2.imwrite(str(img_out_dir / f"plot_{metrics['plot_id']}.png"), panel)
                except Exception:
                    pass

    if not all_metrics:
        print("\n  No results to summarise.")
        return

    keys = ["n_gt_segs", "n_pred_segs", "TP", "FP", "FN",
            "precision", "recall", "F1"]
    summary: dict[str, dict] = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary[k] = {
            "mean":  float(np.mean(vals)),
            "std":   float(np.std(vals)),
            "min":   float(np.min(vals)),
            "max":   float(np.max(vals)),
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
            writer = csv.writer(csvf)
            writer.writerow(["metric", "mean", "std", "min", "max", "total"])
            for k in keys:
                s = summary[k]
                writer.writerow([k,
                                  f"{s['mean']:.4f}", f"{s['std']:.4f}",
                                  f"{s['min']:.4f}",  f"{s['max']:.4f}",
                                  f"{s['total']:.0f}" if s["total"] is not None else ""])
            writer.writerow(["micro_precision", f"{micro_p:.4f}", "", "", "", ""])
            writer.writerow(["micro_recall",    f"{micro_r:.4f}", "", "", "", ""])
            writer.writerow(["micro_F1",        f"{micro_f1:.4f}", "", "", "", ""])
    except PermissionError:
        pass

    elapsed = time.time() - t0
    print(f"\n  Evaluation complete in {elapsed:.1f}s  ({len(all_metrics)} plots)")
    print(f"\n  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  {'Total':>8}")
    print("  " + "-" * 52)
    for k in keys:
        s = summary[k]
        tot_str = f"{s['total']:.0f}" if s["total"] is not None else ""
        print(f"  {k:<22}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {tot_str:>8}")
    print(f"\n  Micro precision : {micro_p:.4f}")
    print(f"  Micro recall    : {micro_r:.4f}")
    print(f"  Micro F1        : {micro_f1:.4f}")
    print(f"\n  Per-plot CSV  → {per_plot_csv}")
    print(f"  Summary CSV   → {summary_csv}")
    if save_images:
        print(f"  Images        → {img_out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-IMAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_image(image_path: str,
                 weight_file: str = DEFAULT_WEIGHT,
                 output_path: str | None = None) -> None:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    net, device = load_model(weight_file)
    pred_lines  = run_deeplsd(net, gray, device)

    vis = img_bgr.copy()
    for line in pred_lines:
        cv2.line(vis,
                 (int(round(float(line[0][0]))), int(round(float(line[0][1])))),
                 (int(round(float(line[1][0]))), int(round(float(line[1][1])))),
                 (0, 0, 220), 1, cv2.LINE_AA)
    cv2.putText(vis, f"{len(pred_lines)} segments detected",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{len(pred_lines)} segments detected",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"  Saved → {output_path}")
    else:
        cv2.imshow("DeepLSD detections", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained DeepLSD on synthetic chart segments "
                    "(no DeepLSD package required).")
    parser.add_argument("--download-weights", action="store_true",
                        help="Download pretrained DeepLSD weights and exit.")
    parser.add_argument("--n-plots", type=int, default=None,
                        help="Number of plots to evaluate (default: all).")
    parser.add_argument("--weight", type=str, default=DEFAULT_WEIGHT,
                        help=f"Weight file in weights/ dir (default: {DEFAULT_WEIGHT}).")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving annotated images.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single plotting-area image.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for single-image result.")
    args = parser.parse_args()

    if args.download_weights:
        download_weights()
        sys.exit(0)

    if args.image:
        detect_image(args.image, weight_file=args.weight,
                     output_path=args.output)
    else:
        evaluate(n_plots=args.n_plots,
                 weight_file=args.weight,
                 save_images=not args.no_images)
