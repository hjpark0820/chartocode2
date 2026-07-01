"""
chart_segment_detector.py
==========================
Working directory : <project>/src/
Model save path   : ../models/chart_segment_net.pth  (relative to src/)

Trains a Deep-LSD-style convolutional model to detect line segments in the
plotting area of synthetic dose-response charts.

Ground truth segments are the connections between consecutive markers of the
same series, restricted to pairs where BOTH endpoints survived the MIN_SEP
filter (i.e. both appear in gt["points"]).

The model predicts two dense maps over the plotting-area crop:
  • distance field  D(x,y)  — Euclidean distance to the nearest segment
  • angle field     A(x,y)  — orientation angle of the nearest segment (radians)

These are trained with L1 loss on D and a wrapped angular L1 loss on A,
masked to pixels within MAX_DIST of at least one segment.

Inference: given the two predicted maps, line segments are extracted by
tracing the gradient of D guided by A (same post-processing as Deep LSD).

USAGE
-----
  # Generate plots + train
  python chart_segment_detector.py --mode train

  # Detect segments in a plotting-area image
  python chart_segment_detector.py --mode detect --image path/to/plotting_area.png

REQUIREMENTS
------------
  pip install torch torchvision opencv-python matplotlib numpy scipy
  For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations
import argparse, json, math, os, random, time, warnings
from pathlib import Path
import multiprocessing as mp

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
mp.freeze_support()

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
_SRC_DIR        = Path(__file__).parent
MODEL_SAVE_PATH = _SRC_DIR / ".." / "models" / "chart_segment_net.pth"
# Shared synthetic plots directory (same plots used by marker detector)
SYNTH_DIR       = _SRC_DIR / ".." / "data" / "synthetic_plots"
# Per-epoch log storage
EPOCH_LOG_DIR   = _SRC_DIR / ".." / "data" / "seg_epoch_logs"
# Cache for pre-computed dataset tensors (avoids recomputing on every run)
DATASET_CACHE_DIR = _SRC_DIR / ".." / "data" / "seg_dataset_cache"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS  (must match chart_marker_detector_v3.py)
# ══════════════════════════════════════════════════════════════════════════════
N_PLOTS      = 1000
MARKER_PT    = 8
DPI          = 100
PLOT_W_IN    = 5.6
PLOT_H_IN    = 4.2
PLOT_W_PX    = int(PLOT_W_IN * DPI)   # 560
PLOT_H_PX    = int(PLOT_H_IN * DPI)   # 420
N_POINTS     = 12

EPOCHS       = 100
LR           = 3e-4
BATCH_SIZE   = 8      # safe on 6 GB+ VRAM; increase if you have more
IMG_SIZE     = 320    # all crops resized to IMG_SIZE×IMG_SIZE before training
                      # smaller = faster; 320 is a good balance for ~340×420 crops
VAL_FRAC     = 0.15
WORKERS      = min(4, mp.cpu_count())
USE_COMPILE  = False  # set True on PyTorch ≥ 2.0 for extra ~10-20% speedup

# Distance-field parameters
MAX_DIST     = 8.0    # pixels; beyond this, distance is clamped and loss masked
                      # 8px gives ~15% mask coverage on 320px crops with typical
                      # segment density — tight enough for a clear learning signal
LINE_HALF_W  = 1.5    # half-width used when rasterising segments for the mask

# Minimum segment length (pixels) — shorter segments are ignored in GT
MIN_SEG_LEN  = 5.0

# ══════════════════════════════════════════════════════════════════════════════
#  SYMBOL / SERIES CONSTANTS  (must match chart_marker_detector_v3.py)
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "filled_circle", "open_circle",
    "filled_square",  "open_square",
    "open_triangle",  "open_inv_triangle",
    "filled_triangle","filled_inv_triangle",
    "open_rhombus",   "filled_rhombus",
    "x_marker",       "plus_marker",
]
N_SYMBOLS = len(CLASS_NAMES)   # 12

_MPL_MARKERS = [
    ('o', True),  ('o', False),
    ('s', True),  ('s', False),
    ('^', False), ('v', False),
    ('^', True),  ('v', True),
    ('D', False), ('D', True),
    ('x', True),  ('+', True),
]

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
#  AUGMENTATION HELPERS  (same as marker detector)
# ══════════════════════════════════════════════════════════════════════════════
def _add_paper_texture(img: np.ndarray, strength: float, rng) -> np.ndarray:
    grain = rng.normal(0, strength * 30, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)

def _add_yellowing(img: np.ndarray, strength: float) -> np.ndarray:
    out = img.astype(np.float32)
    out[:, :, 0] -= strength * 40
    out[:, :, 1] -= strength * 10
    out[:, :, 2] += strength * 20
    out += strength * 15
    return np.clip(out, 0, 255).astype(np.uint8)

def _add_blur(img: np.ndarray, strength: float) -> np.ndarray:
    k = max(1, int(strength * 3))
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), strength * 0.8)

def _add_elastic_warp(img: np.ndarray, strength: float, rng) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    h, w = img.shape[:2]
    alpha = strength * 8
    dx = gaussian_filter(rng.random((h, w)).astype(np.float32) * 2 - 1, 20) * alpha
    dy = gaussian_filter(rng.random((h, w)).astype(np.float32) * 2 - 1, 20) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)

def _apply_scan_noise(img: np.ndarray, level: str, rng) -> np.ndarray:
    if level == 'none':   return img
    if level == 'light':
        img = _add_paper_texture(img, 0.15, rng)
        return _add_blur(img, 0.4)
    if level == 'medium':
        img = _add_yellowing(img, 0.4)
        img = _add_paper_texture(img, 0.35, rng)
        img = _add_blur(img, 0.8)
        return _add_elastic_warp(img, 0.5, rng)
    if level == 'heavy':
        img = _add_yellowing(img, 0.8)
        img = _add_paper_texture(img, 0.6, rng)
        img = _add_blur(img, 1.2)
        img = _add_elastic_warp(img, 1.0, rng)
        return img
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  HILL FUNCTION  (same as marker detector)
# ══════════════════════════════════════════════════════════════════════════════
LOG_MIN       = -3.0
LOG_MAX       =  3.0
LOG_MIN_PLOT  = -3.2
LOG_MAX_PLOT  =  3.2
Y_MIN         =  0.0
Y_MAX         =  1.0
Y_MIN_PLOT    = -0.05
Y_MAX_PLOT    =  1.05
LOG_EC50_MIN  = -2.0
LOG_EC50_MAX  =  2.0

def _hill(x, bottom, top, ec50, n):
    return bottom + (top - bottom) / (1.0 + (ec50 / np.maximum(x, 1e-12)) ** n)


# ══════════════════════════════════════════════════════════════════════════════
#  MEASURE SYMBOL DIAMETER (for MIN_SEP, same as marker detector)
# ══════════════════════════════════════════════════════════════════════════════
def _measure_p() -> int:
    fig, ax = plt.subplots(figsize=(1.0, 1.0), dpi=DPI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot(0.5, 0.5, marker='o', markersize=MARKER_PT,
            markerfacecolor='black', markeredgecolor='black', linestyle='none')
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    gray = cv2.cvtColor(buf, cv2.COLOR_RGBA2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = np.argwhere(bw > 0)
    diam = int(max(coords.max(axis=0) - coords.min(axis=0)) + 1) if len(coords) else MARKER_PT
    return int(math.ceil(diam * 1.20 * 1.20)) | 1

_P = _measure_p()
_MIN_SEP = int(round(_P * 1.5))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT GENERATION  (shared with marker detector; adds "segments" to GT JSON)
# ══════════════════════════════════════════════════════════════════════════════
def generate_one_plot(args_tuple):
    """
    Generate one synthetic dose-response plot and save:
      plot_NNNNN.png  — the rendered image
      gt_NNNNN.json   — ground truth with keys:
                          plot_w, plot_h
                          pa        : plotting area bounding box (pixels)
                          points    : MIN_SEP-filtered marker positions
                          all_points: all rendered marker positions
                          segments  : list of {x1,y1,x2,y2,series_idx}
                                      where both endpoints are in `points`
    """
    idx, out_dir, seed = args_tuple
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ec50_slot_w = (LOG_EC50_MAX - LOG_EC50_MIN) / N_SYMBOLS
    ec50_slots  = list(range(N_SYMBOLS))
    rng.shuffle(ec50_slots)

    x_sub_w   = (LOG_MAX - LOG_MIN) / N_POINTS
    x_nominal = np.array([
        10 ** (LOG_MIN + (k + 0.5) * x_sub_w) for k in range(N_POINTS)
    ])

    series_data = []
    for si in range(N_SYMBOLS):
        slot   = ec50_slots[si]
        log_ec = LOG_EC50_MIN + ec50_slot_w * slot + np_rng.uniform(0.05, 0.95) * ec50_slot_w
        ec50   = 10 ** log_ec
        bottom = np_rng.uniform(0.02, 0.20)
        top    = np_rng.uniform(0.75, 1.00)
        n_hill = np_rng.uniform(0.8, 4.0)
        jitter = np_rng.uniform(-0.05, 0.05, N_POINTS) * x_sub_w
        x_vals = np.array([
            10 ** (np.log10(x_nominal[k]) + jitter[k]) for k in range(N_POINTS)
        ])
        y_vals = _hill(x_vals, bottom, top, ec50, n_hill)
        y_vals += np_rng.normal(0, 0.005, N_POINTS)
        y_vals  = np.clip(y_vals, Y_MIN + 0.01, Y_MAX - 0.01)
        series_data.append((x_vals, y_vals))

    z_order = list(range(N_SYMBOLS))
    rng.shuffle(z_order)

    fig, ax = plt.subplots(figsize=(PLOT_W_IN, PLOT_H_IN), dpi=DPI)
    fig.patch.set_facecolor('none'); fig.patch.set_alpha(0.0)
    ax.set_facecolor('none');        ax.patch.set_alpha(0.0)

    # Draw connecting lines first (below markers)
    for si in z_order:
        x_vals, y_vals = series_data[si]
        ax.plot(x_vals, y_vals, color='black', linewidth=0.8,
                marker='none', zorder=si)

    ax.set_xscale('log')
    ax.set_xlim(10**LOG_MIN_PLOT, 10**LOG_MAX_PLOT)
    ax.set_ylim(Y_MIN_PLOT, Y_MAX_PLOT)
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    # Draw markers on top
    plot_base_ms = float(np_rng.uniform(MARKER_PT * 0.8, MARKER_PT * 1.2))
    for z, si in enumerate(z_order):
        x_vals, y_vals = series_data[si]
        mcode, filled  = _MPL_MARKERS[si]
        ms_pt = float(np_rng.uniform(plot_base_ms * 0.95, plot_base_ms * 1.05))
        lw    = float(np_rng.uniform(0.5, 1.2))
        fc    = 'black' if filled else 'white'
        ax.plot(x_vals, y_vals,
                color='black', marker=mcode, markersize=ms_pt,
                markerfacecolor=fc, markeredgecolor='black',
                markeredgewidth=lw, linestyle='none',
                zorder=N_SYMBOLS + z * 3)

    fig.canvas.draw()

    buf    = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf    = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgba_f = buf.astype(np.float32) / 255.0
    alpha  = rgba_f[:, :, 3:4]
    rgb_c  = rgba_f[:, :, :3] * alpha + np.ones_like(rgba_f[:, :, :3]) * (1 - alpha)
    img_bgr = cv2.cvtColor((rgb_c * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    noise_level = rng.choices(['none', 'light', 'medium', 'heavy'],
                              weights=[0.1, 0.3, 0.4, 0.2])[0]
    img_bgr = _apply_scan_noise(img_bgr, noise_level, np_rng)

    H_px, W_px = img_bgr.shape[:2]

    # ── Build raw_points (all rendered symbol centres) ────────────────────────
    raw_points = []
    for si in range(N_SYMBOLS):
        x_vals, y_vals = series_data[si]
        for xi, yi in zip(x_vals, y_vals):
            disp = ax.transData.transform((xi, yi))
            px = int(round(disp[0]))
            py = int(round(H_px - disp[1]))
            px = max(0, min(W_px - 1, px))
            py = max(0, min(H_px - 1, py))
            raw_points.append({"cx": px, "cy": py,
                               "class_idx": si,
                               "class_name": CLASS_NAMES[si],
                               "series_idx": si,
                               "point_idx": int(len([p for p in raw_points
                                                     if p["series_idx"] == si]))})

    # ── MIN_SEP filter ────────────────────────────────────────────────────────
    accepted_coords = []
    gt_points = []
    for pt in raw_points:
        cx, cy = pt["cx"], pt["cy"]
        too_close = any(
            (cx - ax2) ** 2 + (cy - ay2) ** 2 < _MIN_SEP ** 2
            for ax2, ay2 in accepted_coords
        )
        if not too_close:
            accepted_coords.append((cx, cy))
            gt_points.append(pt)

    # ── Build accepted set as (series_idx, point_idx) for segment filtering ──
    accepted_keys = {(p["series_idx"], p["point_idx"]) for p in gt_points}

    # ── Build segment GT ──────────────────────────────────────────────────────
    # For each series, connect consecutive raw_points[i] → raw_points[i+1]
    # only when BOTH endpoints are in the accepted (MIN_SEP-filtered) set.
    segments = []
    # Group raw_points by series
    by_series: dict[int, list] = {}
    for pt in raw_points:
        by_series.setdefault(pt["series_idx"], []).append(pt)
    # Sort each series by point_idx to ensure consecutive order
    for si, pts_s in by_series.items():
        pts_s.sort(key=lambda p: p["point_idx"])
        for i in range(len(pts_s) - 1):
            p0 = pts_s[i]
            p1 = pts_s[i + 1]
            k0 = (p0["series_idx"], p0["point_idx"])
            k1 = (p1["series_idx"], p1["point_idx"])
            if k0 in accepted_keys and k1 in accepted_keys:
                x1, y1 = p0["cx"], p0["cy"]
                x2, y2 = p1["cx"], p1["cy"]
                length = math.hypot(x2 - x1, y2 - y1)
                if length >= MIN_SEG_LEN:
                    segments.append({
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "series_idx": si,
                        "length": round(length, 2)
                    })

    bbox  = ax.get_position()
    pa_x0 = int(round(bbox.x0 * W_px))
    pa_y0 = int(round((1 - bbox.y1) * H_px))
    pa_x1 = int(round(bbox.x1 * W_px))
    pa_y1 = int(round((1 - bbox.y0) * H_px))

    plt.close(fig)

    img_path = Path(out_dir) / f"plot_{idx:05d}.png"
    gt_path  = Path(out_dir) / f"gt_{idx:05d}.json"
    cv2.imwrite(str(img_path), img_bgr)
    with open(gt_path, "w") as f:
        json.dump({
            "plot_w": W_px, "plot_h": H_px,
            "pa":   {"x0": pa_x0, "y0": pa_y0, "x1": pa_x1, "y1": pa_y1},
            "points":     gt_points,
            "all_points": raw_points,
            "segments":   segments,
        }, f)
    return str(img_path), str(gt_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DISTANCE & ANGLE FIELD COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_distance_angle_fields(
    segments: list[dict],
    h: int, w: int,
    max_dist: float = MAX_DIST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast distance and angle field computation using Euclidean Distance Transform.

    Algorithm (~9x faster than a per-segment NumPy loop):
      1. Rasterise all segments onto a binary canvas with cv2.line.
         Each pixel records which segment index painted it.
      2. Run scipy EDT once on the whole canvas to get exact per-pixel
         distances and the coordinates of the nearest segment pixel.
      3. Look up the segment index at each nearest pixel to get the angle.

    Shapes:
      dist_field  (h, w) float32 — distance to nearest segment
      angle_field (h, w) float32 — orientation of nearest segment (radians, [0, π))
      mask        (h, w) bool    — True where dist < max_dist
    """
    from scipy.ndimage import distance_transform_edt

    if not segments:
        return (np.full((h, w), max_dist, dtype=np.float32),
                np.zeros((h, w), dtype=np.float32),
                np.zeros((h, w), dtype=bool))

    # Pre-compute segment angles  (N,)
    angles = np.array([
        math.atan2(s["y2"] - s["y1"], s["x2"] - s["x1"]) % math.pi
        for s in segments
    ], dtype=np.float32)

    # Rasterise all segments; record which segment index owns each pixel
    canvas      = np.zeros((h, w), dtype=np.uint8)
    seg_idx_map = np.full((h, w), -1, dtype=np.int32)
    tmp         = np.zeros((h, w), dtype=np.uint8)

    for si, seg in enumerate(segments):
        x1 = int(round(seg["x1"])); y1 = int(round(seg["y1"]))
        x2 = int(round(seg["x2"])); y2 = int(round(seg["y2"]))
        tmp[:] = 0
        cv2.line(tmp, (x1, y1), (x2, y2), 1, 1)
        painted = tmp == 1
        canvas[painted] = 1
        seg_idx_map[painted] = si   # last segment wins on overlap (fine for training)

    # Single EDT call: O(h*w) — far cheaper than 132 separate NumPy passes
    dist_field, (nr, nc) = distance_transform_edt(canvas == 0, return_indices=True)
    dist_field = dist_field.astype(np.float32)

    # Map each pixel to the angle of its nearest segment
    nearest_seg = seg_idx_map[nr, nc]                        # (h, w)
    safe_idx    = np.clip(nearest_seg, 0, len(angles) - 1)  # guard -1 entries
    angle_field = np.where(nearest_seg >= 0,
                           angles[safe_idx],
                           0.0).astype(np.float32)

    # Clamp to max_dist so that dist_field / max_dist is always in [0, 1].
    # Without this clamp, background pixels have dist >> max_dist, which means
    # dist_t = dist/MAX_DIST >> 1 while sigmoid(model output) is in [0,1].
    # The L1 loss is then dominated by background pixels where the target is
    # e.g. 8.4 but the model can only predict 1.0 — the model learns to always
    # predict 1.0 everywhere and never learns the actual distance field shape.
    dist_field = np.minimum(dist_field, max_dist)
    mask = dist_field < max_dist
    return dist_field, angle_field, mask


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SegmentDataset(Dataset):
    """
    Each sample is one plotting-area crop.
    Returns:
      image  (3, H_crop, W_crop) float32 normalised
      dist   (1, H_crop, W_crop) float32  distance field / MAX_DIST
      angle  (1, H_crop, W_crop) float32  angle field in [0, π)
      mask   (1, H_crop, W_crop) float32  1 where dist < MAX_DIST else 0
    """
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _cache_key(gt_paths: list[Path]) -> str:
        """
        Build a short hash that uniquely identifies this exact set of GT files
        and their last-modified timestamps.  If any file changes or the set
        changes, the hash changes and the cache is rebuilt automatically.
        """
        import hashlib
        h = hashlib.md5()
        for p in sorted(gt_paths):
            h.update(str(p).encode())
            try:
                h.update(str(p.stat().st_mtime).encode())
            except OSError:
                pass
        # Also include MAX_DIST so changing it invalidates the cache
        h.update(str(MAX_DIST).encode())
        return h.hexdigest()[:16]

    def __init__(self, gt_paths: list[Path], augment: bool = True,
                 label: str = "",
                 cache_dir: Path | None = None):
        self.augment = augment
        self.records: list[tuple[Path, dict]] = []   # kept for n_segs reporting
        self.samples: list[tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]] = []

        # ── Try to load from cache ────────────────────────────────────────────
        cache_file: Path | None = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._cache_key(gt_paths)
            cache_file = cache_dir / f"{label}_{key}.pt"

            if cache_file.exists():
                tag = f" [{label}]" if label else ""
                print(f"  Loading{tag} dataset from cache ({cache_file.name}) ...",
                      end="", flush=True)
                t0 = time.time()
                payload = torch.load(cache_file, weights_only=False)
                self.samples = payload["samples"]
                self.records = payload["records"]
                print(f" done in {time.time()-t0:.1f}s  "
                      f"({len(self.samples)} samples)")
                return   # ← skip all computation below

        # ── Cache miss: compute from scratch ──────────────────────────────────
        _dummy = (torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE),
                  torch.zeros(1, IMG_SIZE, IMG_SIZE), torch.zeros(1, IMG_SIZE, IMG_SIZE))

        n_total = len(gt_paths)
        t0 = time.time()
        prefix = f"  Building {label} dataset: " if label else "  Building dataset: "

        for i, p in enumerate(gt_paths):
            _pbar(i + 1, n_total, t0, prefix=prefix)

            gt = _load_gt(p)
            img_path = p.parent / p.name.replace("gt_", "plot_").replace(".json", ".png")
            self.records.append((img_path, gt))

            pa = gt["pa"]
            x0, y0, x1, y1 = pa["x0"], pa["y0"], pa["x1"], pa["y1"]

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                self.samples.append(_dummy)
                continue

            crop = img_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                self.samples.append(_dummy)
                continue

            h_crop, w_crop = crop.shape[:2]

            # Scale segment coordinates to match the resized crop
            sx = IMG_SIZE / w_crop
            sy = IMG_SIZE / h_crop
            segs_local = [
                {"x1": (s["x1"] - x0) * sx, "y1": (s["y1"] - y0) * sy,
                 "x2": (s["x2"] - x0) * sx, "y2": (s["y2"] - y0) * sy}
                for s in gt.get("segments", [])
            ]

            # Resize crop to fixed IMG_SIZE × IMG_SIZE
            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_LINEAR)

            dist, angle, mask = compute_distance_angle_fields(
                segs_local, IMG_SIZE, IMG_SIZE, max_dist=MAX_DIST
            )

            img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_rgb = (img_rgb - self._MEAN) / self._STD
            img_t   = torch.from_numpy(img_rgb.transpose(2, 0, 1))          # (3,IMG_SIZE,IMG_SIZE)
            dist_t  = torch.from_numpy(dist[None]  / MAX_DIST)              # (1,IMG_SIZE,IMG_SIZE)
            angle_t = torch.from_numpy(angle[None])                          # (1,IMG_SIZE,IMG_SIZE)
            mask_t  = torch.from_numpy(mask[None].astype(np.float32))       # (1,IMG_SIZE,IMG_SIZE)

            self.samples.append((img_t, dist_t, angle_t, mask_t))

        elapsed = time.time() - t0
        print(f"  Done — {n_total} samples pre-computed in {elapsed:.1f}s")

        # ── Save to cache for next run ─────────────────────────────────────────
        if cache_file is not None:
            print(f"  Saving dataset cache to {cache_file.name} ...",
                  end="", flush=True)
            ts = time.time()
            torch.save({"samples": self.samples, "records": self.records},
                       cache_file)
            print(f" done in {time.time()-ts:.1f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_t, dist_t, angle_t, mask_t = self.samples[idx]
        if self.augment and random.random() < 0.5:
            # Random horizontal flip: flip image and fields spatially.
            # Angle must also be reflected: angle -> pi - angle.
            img_t   = img_t.flip(-1)
            dist_t  = dist_t.flip(-1)
            mask_t  = mask_t.flip(-1)
            angle_t = (math.pi - angle_t.flip(-1)) % math.pi
        return img_t, dist_t, angle_t, mask_t


def _collate_pad(batch):
    """
    All samples are pre-resized to IMG_SIZE×IMG_SIZE, so a plain stack suffices.
    The padding logic is kept as a fallback for edge cases (e.g. dummy tensors).
    """
    imgs, dists, angles, masks = zip(*batch)
    # Fast path: all same size (normal case after resize)
    if all(x.shape == imgs[0].shape for x in imgs):
        return (torch.stack(list(imgs)),
                torch.stack(list(dists)),
                torch.stack(list(angles)),
                torch.stack(list(masks)))
    # Fallback: pad to max size (should not happen in normal training)
    max_h = max(x.shape[1] for x in imgs)
    max_w = max(x.shape[2] for x in imgs)
    def pad(t, fill=0.0):
        _, h, w = t.shape
        return F.pad(t, (0, max_w - w, 0, max_h - h), value=fill)
    return (torch.stack([pad(x) for x in imgs]),
            torch.stack([pad(x, fill=1.0) for x in dists]),
            torch.stack([pad(x) for x in angles]),
            torch.stack([pad(x) for x in masks]))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL  (Deep-LSD-style encoder–decoder)
# ══════════════════════════════════════════════════════════════════════════════
class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

class _UpBlock(nn.Module):
    """Upsample × 2 then two conv-bn-relu."""
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_c + skip_c, out_c),
            _ConvBnRelu(out_c, out_c),
        )
    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DeepLSDNet(nn.Module):
    """
    Lightweight U-Net encoder–decoder that predicts:
      dist_pred  (B, 1, H, W)  — normalised distance field in [0, 1]
      angle_pred (B, 2, H, W)  — (cos θ, sin θ) for angle field
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(_ConvBnRelu(3,  32), _ConvBnRelu(32,  32))
        self.enc2 = nn.Sequential(_ConvBnRelu(32, 64), _ConvBnRelu(64,  64))
        self.enc3 = nn.Sequential(_ConvBnRelu(64,128), _ConvBnRelu(128,128))
        self.enc4 = nn.Sequential(_ConvBnRelu(128,256),_ConvBnRelu(256,256))
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _ConvBnRelu(256, 512),
            _ConvBnRelu(512, 512),
        )

        # Decoder
        self.dec4 = _UpBlock(512, 256, 256)
        self.dec3 = _UpBlock(256, 128, 128)
        self.dec2 = _UpBlock(128,  64,  64)
        self.dec1 = _UpBlock( 64,  32,  32)

        # Output heads
        self.head_dist  = nn.Conv2d(32, 1, 1)   # distance field
        self.head_angle = nn.Conv2d(32, 2, 1)   # (cos θ, sin θ)

        # Initialise dist head bias to +2.0 so the model starts by predicting
        # ~2.0 everywhere (before ReLU+clamp → clamped to 1.0 = background).
        # This is the correct prior: almost all pixels are background (dist=1.0).
        # The model then only needs to learn to suppress the output near segments.
        nn.init.constant_(self.head_dist.bias, 2.0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        # Use ReLU+clamp instead of sigmoid for the distance head.
        # sigmoid maps 0 → 0.5 at random init, so the model starts by predicting
        # ~0.46 everywhere and never produces values near 0 (the segment region).
        # ReLU+clamp maps 0 → 0 at random init, giving the model a natural
        # starting point and allowing sharp near-zero predictions near segments.
        dist_pred  = torch.clamp(F.relu(self.head_dist(d1)), 0.0, 1.0)  # (B,1,H,W) ∈ [0,1]
        angle_pred = self.head_angle(d1)                                  # (B,2,H,W) raw
        return dist_pred, angle_pred


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def dist_loss(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Two-component distance field loss:

    1. Near-segment L1 (mask region only):
       Trains the model to predict the exact distance valley near segments.
       This is the primary learning signal.

    2. Background MSE (outside mask):
       Trains the model to predict 1.0 for background pixels.
       MSE is used (not L1) so the gradient is proportional to the error
       magnitude, giving a soft push toward 1.0 without dominating.

    The two components are balanced so the near-segment region receives
    ~10x more gradient than the background, regardless of mask coverage.
    """
    if mask is None:
        return (pred - target).abs().mean()

    near = mask                         # (B,1,H,W) float32 in {0,1}
    bg   = 1.0 - near

    # Component 1: near-segment L1
    n_near = near.sum().clamp(min=1)
    l_near = ((pred - target).abs() * near).sum() / n_near

    # Component 2: background MSE (soft push toward 1.0)
    n_bg = bg.sum().clamp(min=1)
    l_bg = ((pred - target).pow(2) * bg).sum() / n_bg

    # Weight near-segment 10x relative to background
    return l_near + 0.1 * l_bg


def angle_loss(pred_cs: torch.Tensor,
               target_angle: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    Angular loss using (cos, sin) representation.
    pred_cs     : (B, 2, H, W)  raw logits → normalised to unit vector
    target_angle: (B, 1, H, W)  angle in [0, π)
    mask        : (B, 1, H, W)  1 where segment is nearby

    Loss = mean(1 - |cos(pred_angle - target_angle)|) over masked pixels.
    The absolute cosine handles the π-ambiguity of line orientation.
    """
    n_pixels = mask.sum().clamp(min=1)
    # Normalise prediction to unit vector
    pred_norm = F.normalize(pred_cs, dim=1)   # (B,2,H,W)
    cos_p = pred_norm[:, 0:1]
    sin_p = pred_norm[:, 1:2]
    # Target unit vector
    cos_t = torch.cos(target_angle)
    sin_t = torch.sin(target_angle)
    # dot product = cos(pred - target)
    dot = cos_p * cos_t + sin_p * sin_t   # (B,1,H,W)
    # Use |dot| to handle π-ambiguity
    loss = (1.0 - dot.abs()) * mask
    return loss.sum() / n_pixels


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT EXTRACTION FROM PREDICTED FIELDS  (inference post-processing)
# ══════════════════════════════════════════════════════════════════════════════
def extract_segments_from_fields(
    dist_field: np.ndarray,
    angle_field: np.ndarray,
    dist_thresh: float = 0.3,
    min_len: float = 10.0,
    step: float = 1.0,
    n_seeds: int = 500,
) -> list[tuple[float, float, float, float]]:
    """
    Robust segment extraction from predicted distance and angle fields.

    Strategy:
      1. Threshold the distance field to get a binary close-to-segment mask.
      2. Find connected components in the mask.
      3. For each component, fit a line using PCA on the component pixels.
         PCA gives the principal axis = the segment direction, which is
         robust to noisy angle fields and works even with imperfect predictions.
      4. Project all component pixels onto the principal axis to find endpoints.
      5. Return segments longer than min_len pixels.

    This is much more robust than the seed-trace approach because:
      - It does not depend on the angle field being accurate (PCA uses positions)
      - It does not require the model to produce sharp near-zero values
        (any threshold < 0.5 will find the right region once training starts)
      - It handles curved/bent predictions gracefully

    Returns list of (x1, y1, x2, y2) in image (crop) coordinates.
    """
    h, w = dist_field.shape
    close_mask = (dist_field < dist_thresh).astype(np.uint8)
    if close_mask.sum() == 0:
        return []

    # Morphological closing to connect nearby blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_mask = cv2.morphologyEx(close_mask, cv2.MORPH_CLOSE, kernel)

    # Connected components
    n_labels, labels = cv2.connectedComponents(close_mask)

    segments = []
    for label in range(1, n_labels):  # skip background (0)
        ys, xs = np.where(labels == label)
        if len(xs) < 5:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)  # (N,2)

        # PCA: find principal axis
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]  # principal direction (unit vector)

        # Project all points onto the principal axis
        proj = centered @ axis  # (N,)
        i_min = proj.argmin()
        i_max = proj.argmax()

        x1, y1 = mean + proj[i_min] * axis
        x2, y2 = mean + proj[i_max] * axis

        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_len:
            segments.append((float(x1), float(y1), float(x2), float(y2)))

    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def _pixel_metrics(
    pred_dist: np.ndarray,
    gt_dist:   np.ndarray,
    threshold: float = 0.3,
) -> dict:
    """
    Pixel-level binary classification metrics:
      Positive = pixel is within MAX_DIST * threshold of a segment.
    """
    pred_pos = pred_dist < threshold
    gt_pos   = gt_dist   < threshold
    TP = int(np.logical_and( pred_pos,  gt_pos).sum())
    FP = int(np.logical_and( pred_pos, ~gt_pos).sum())
    FN = int(np.logical_and(~pred_pos,  gt_pos).sum())
    TN = int(np.logical_and(~pred_pos, ~gt_pos).sum())
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "precision": prec, "recall": rec, "f1": f1}


# ══════════════════════════════════════════════════════════════════════════════
#  EPOCH LOG: SAVE ANNOTATED VALIDATION IMAGES
# ══════════════════════════════════════════════════════════════════════════════
def _save_epoch_val_images(
    epoch: int,
    model: DeepLSDNet,
    val_gt_paths: list[Path],
    device: torch.device,
    n_save: int = 8,
) -> None:
    """
    Run inference on up to n_save validation plots and save:
      • The plotting-area crop with GT segments (green) and predicted segments (red)
      • The predicted distance field as a heatmap
    """
    out_dir = EPOCH_LOG_DIR / f"epoch_{epoch:03d}" / "val_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    _MEAN_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD_t  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    chosen = val_gt_paths[:n_save]
    MEAN_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for gt_path in chosen:
        img_path = gt_path.parent / gt_path.name.replace("gt_", "plot_").replace(".json", ".png")
        gt = _load_gt(gt_path)
        pa = gt["pa"]
        x0, y0, x1, y1 = pa["x0"], pa["y0"], pa["x1"], pa["y1"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        crop = img_bgr[y0:y1, x0:x1].copy()
        h_c, w_c = crop.shape[:2]

        # ── Resize to IMG_SIZE (MUST match training preprocessing) ────────────
        # The model was trained on IMG_SIZE×IMG_SIZE inputs.  Passing the raw
        # crop at its native resolution produces garbage predictions because the
        # receptive field and distance scale are calibrated for IMG_SIZE.
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - MEAN_np) / STD_np
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(enabled=(device.type == "cuda")):
                dist_pred, angle_pred = model(img_t)

        dist_np     = dist_pred[0, 0].cpu().float().numpy()   # (IMG_SIZE,IMG_SIZE)
        angle_np_cs = angle_pred[0].cpu().float().numpy()     # (2,IMG_SIZE,IMG_SIZE)
        angle_np    = np.arctan2(angle_np_cs[1], angle_np_cs[0]) % math.pi

        # ── Scale factors: IMG_SIZE coords → original crop coords ─────────────
        sx_back = w_c / IMG_SIZE
        sy_back = h_c / IMG_SIZE
        # Scale factors: full-image coords → original crop coords
        sx_gt = 1.0  # GT segs are already in full-image coords; subtract x0/y0
        sy_gt = 1.0

        # ── Draw GT segments (green) on original-size crop ────────────────────
        vis = crop.copy()
        for seg in gt.get("segments", []):
            gx1, gy1 = int(seg["x1"] - x0), int(seg["y1"] - y0)
            gx2, gy2 = int(seg["x2"] - x0), int(seg["y2"] - y0)
            cv2.line(vis, (gx1, gy1), (gx2, gy2), (0, 200, 0), 1)

        # ── Extract predicted segments (in IMG_SIZE space) ────────────────────
        # Use dist_thresh=0.5 so segments are found even when the model is
        # still learning (values between 0.3 and 0.5 near segments).
        pred_segs_scaled = extract_segments_from_fields(
            dist_np, angle_np, dist_thresh=0.5, min_len=10.0)

        # Scale back to original crop size and draw (red)
        for (px1, py1, px2, py2) in pred_segs_scaled:
            cv2.line(vis,
                     (int(px1 * sx_back), int(py1 * sy_back)),
                     (int(px2 * sx_back), int(py2 * sy_back)),
                     (0, 0, 220), 1)

        # ── Distance field heatmap (resize back to crop size) ─────────────────
        dist_heat_small = cv2.resize(
            (dist_np * 255).clip(0, 255).astype(np.uint8),
            (w_c, h_c), interpolation=cv2.INTER_LINEAR)
        dist_heat = cv2.applyColorMap(dist_heat_small, cv2.COLORMAP_JET)

        # ── Side-by-side: original crop | GT+pred overlay | dist heatmap ──────
        panel = np.hstack([crop, vis, dist_heat])
        stem  = gt_path.stem.replace("gt_", "")
        cv2.imwrite(str(out_dir / f"plot_{stem}.png"), panel)

    print(f"  Val images saved → {out_dir}  ({len(chosen)} plots)")


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT-LEVEL MATCHING  (used for validation metrics)
# ══════════════════════════════════════════════════════════════════════════════
MATCH_DIST_PX  = 10.0   # midpoint distance tolerance (pixels in crop space)
MATCH_ANGLE_DEG = 15.0  # undirected angular tolerance (degrees)


def _seg_midpoint(x1: float, y1: float,
                  x2: float, y2: float) -> tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _seg_angle_deg(x1: float, y1: float,
                   x2: float, y2: float) -> float:
    """Undirected angle in [0, 90) degrees."""
    dx, dy = x2 - x1, y2 - y1
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def _angle_diff_deg(a1: float, a2: float) -> float:
    """Smallest difference between two undirected angles (both in [0, 90])."""
    d = abs(a1 - a2)
    return 180.0 - d if d > 90.0 else d


def _match_segments(
    pred_segs: list[tuple[float, float, float, float]],
    gt_segs:   list[tuple[float, float, float, float]],
    match_dist:  float = MATCH_DIST_PX,
    match_angle: float = MATCH_ANGLE_DEG,
) -> tuple[int, int, int]:
    """
    Greedy one-to-one matching between predicted and GT segments.
    A predicted segment is a TP if its midpoint is within `match_dist` px of
    an unmatched GT midpoint AND the undirected angle difference is within
    `match_angle` degrees.
    Returns (TP, FP, FN).
    """
    matched_gt: set[int] = set()
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
            d = math.hypot(pmx - gmx, pmy - gmy)
            if d > match_dist:
                continue
            ga = _seg_angle_deg(gx1, gy1, gx2, gy2)
            if _angle_diff_deg(pa, ga) > match_angle:
                continue
            if d < best_dist:
                best_dist = d
                best_idx  = gi
        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
    fp = len(pred_segs) - tp
    fn = len(gt_segs)   - tp
    return tp, fp, fn


# ══════════════════════════════════════════════════════════════════════════════
#  GT MIGRATION  (inject 'segments' into existing marker-detector GT files)
# ══════════════════════════════════════════════════════════════════════════════
def _segments_from_points(points: list[dict]) -> list[dict]:
    """
    Compute segment GT from a list of marker points.

    Compatible with GT files from both chart_marker_detector_v3.py and
    chart_segment_detector.py.

    chart_marker_detector_v3.py stores only {cx, cy, class_idx, class_name}.
    In that script each series uses a unique symbol class, so
    class_idx == series_idx.  point_idx is inferred from the order of
    appearance within each class.

    chart_segment_detector.py stores {cx, cy, class_idx, class_name,
    series_idx, point_idx} explicitly.
    """
    # Normalise: ensure every point has series_idx and point_idx.
    # If missing, derive from class_idx (series) and order (point_idx).
    by_series: dict[int, list] = {}
    for pt in points:
        si = pt.get("series_idx", pt.get("class_idx"))
        if si is None:
            continue
        by_series.setdefault(si, []).append(pt)

    segments = []
    for si, pts_s in by_series.items():
        # Assign point_idx by order of appearance if not already present.
        # Points are stored in the order they were rendered (x ascending
        # along the dose axis), so consecutive entries are consecutive
        # on the curve.
        for order, pt in enumerate(pts_s):
            pt["_pi"] = pt.get("point_idx", order)

        pts_s.sort(key=lambda p: p["_pi"])

        for i in range(len(pts_s) - 1):
            p0, p1 = pts_s[i], pts_s[i + 1]
            # Only connect truly consecutive indices
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


def _segments_from_series_pixels(series_pixels: list[list[dict]]) -> list[dict]:
    """
    Derive GT segments from series_pixels (the ordered pixel sequences saved
    by chart_marker_detector_v3.py).  ALL consecutive pairs are included;
    the MIN_SEP filter only governs ViT subimage patch selection and must
    not gate segment GT coverage.
    """
    segments: list[dict] = []
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


def _load_gt(gt_path: Path) -> dict:
    """
    Load a GT JSON and ensure it has a non-empty 'segments' list.
    Priority order:
      1. Use existing 'segments' key if non-empty.
      2. Use 'series_pixels' + 'points' (correct ordering, both-endpoint filter).
      3. Fallback: reconstruct from 'points' using class_idx as series_idx.
    Derived segments are written back to the file for future use.
    """
    with open(gt_path, encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("segments"):
        if data.get("series_pixels"):
            segs = _segments_from_series_pixels(data["series_pixels"])
        elif data.get("points"):
            segs = _segments_from_points(data["points"])
        else:
            segs = []
        data["segments"] = segs
        try:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except PermissionError:
            pass

    return data


def _migrate_gt_segments(synth_dir: Path) -> None:
    """
    Scan all gt_*.json files in synth_dir.  For any file that lacks the
    'segments' key (or has an empty list), compute segments using the
    3-priority logic and write them back in-place.  Files that already
    have a non-empty 'segments' list are skipped.  Prints a summary.
    """
    gt_files = sorted(synth_dir.glob("gt_*.json"))
    if not gt_files:
        return

    n_migrated = 0
    n_skipped  = 0
    t0 = time.time()
    print("\n" + "=" * 60)
    print("STEP 1b — Migrating GT files (injecting 'segments' key)")
    print("=" * 60)

    for i, gt_path in enumerate(gt_files):
        _pbar(i + 1, len(gt_files), t0, prefix="  Migrating: ")
        try:
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
        except Exception:
            n_skipped += 1
            continue

        # Skip only if segments is already a non-empty list
        if gt.get("segments"):
            n_skipped += 1
            continue

        # Priority 2: series_pixels (all pairs, no endpoint filter)
        if gt.get("series_pixels"):
            gt["segments"] = _segments_from_series_pixels(gt["series_pixels"])
        else:
            # Priority 3: fallback from points only
            gt["segments"] = _segments_from_points(gt.get("points", []))

        try:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt, f)
            n_migrated += 1
        except PermissionError:
            print(f"  [warn] Could not write {gt_path.name} — skipping.")
            n_skipped += 1

    print(f"  Migrated: {n_migrated}  |  Already had segments: {n_skipped}  "
          f"(took {time.time()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train(n_plots: int = N_PLOTS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("\n" + "!" * 60)
        print("WARNING: No CUDA GPU detected — training on CPU.")
        print("  On CPU, one epoch takes ~20-30 min (vs ~10-30s on GPU).")
        print("  Install GPU PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("!" * 60)

    # ── STEP 1: generate synthetic plots ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Generating synthetic plots")
    print("=" * 60)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(SYNTH_DIR.glob("plot_*.png")))
    if existing >= n_plots:
        print(f"  {existing} plots already exist — skipping generation.")
    else:
        seeds = [random.randint(0, 2**31) for _ in range(n_plots)]
        args  = [(i, str(SYNTH_DIR), seeds[i]) for i in range(n_plots)]
        n_cpu = max(1, WORKERS)
        print(f"  Generating {n_plots} plots using {n_cpu} CPU workers...")
        t0 = time.time()
        with mp.Pool(n_cpu) as pool:
            results = pool.map(generate_one_plot, args)
        print(f"  Done in {time.time()-t0:.1f}s — {len(results)} plots saved.")

    # ── STEP 1b: inject 'segments' into GT files that lack it ────────────────
    # GT files created by chart_marker_detector_v3.py already contain
    # 'points' (with series_idx and point_idx) but no 'segments' key.
    # This one-time migration computes segments from the existing data and
    # writes them back in-place so no re-generation is needed.
    _migrate_gt_segments(SYNTH_DIR)

    # ── STEP 2: build dataset split ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Building dataset split")
    print("=" * 60)
    all_gt = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not all_gt:
        raise RuntimeError(f"No GT files found in {SYNTH_DIR}")

    random.shuffle(all_gt)
    n_val   = max(1, int(len(all_gt) * VAL_FRAC))
    val_gt  = all_gt[:n_val]
    train_gt = all_gt[n_val:]
    print(f"  Total plots: {len(all_gt)}  |  Train: {len(train_gt)}  |  Val: {len(val_gt)}")

    # ── STEP 3: build model ─────────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Building model")
    print("=" * 60)
    model = DeepLSDNet().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Parameters : {trainable:,} trainable / {total_p:,} total")

    if USE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Warmup for 5 epochs then cosine decay: prevents large early updates from
    # destabilising the randomly initialised distance-field head.
    def _lr_lambda(epoch_0idx):
        warmup_epochs = 5
        if epoch_0idx < warmup_epochs:
            return (epoch_0idx + 1) / warmup_epochs
        progress = (epoch_0idx - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # ── STEP 3b: pre-compute dataset tensors ─────────────────────────────────
    # All distance/angle/mask fields are computed once here and cached to disk.
    # First run: takes ~20-30s for 1000 plots, then saves a .pt cache file.
    # Subsequent runs: loads from cache in ~2-3s (no recomputation needed).
    # Cache is auto-invalidated if any GT file changes or MAX_DIST changes.
    print("\n" + "=" * 60)
    print("STEP 3b — Loading dataset tensors (cached after first run)")
    print("=" * 60)
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = SegmentDataset(train_gt, augment=True,  label="train",
                              cache_dir=DATASET_CACHE_DIR)
    val_ds   = SegmentDataset(val_gt,   augment=False, label="val",
                              cache_dir=DATASET_CACHE_DIR)
    n_segs = sum(len(r[1].get("segments", [])) for r in train_ds.records + val_ds.records)
    print(f"  Total GT segments across all plots: {n_segs:,}")
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=_collate_pad,
                          pin_memory=(device.type == "cuda"))
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=_collate_pad,
                          pin_memory=(device.type == "cuda"))

    best_val_loss = float("inf")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EPOCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # CSV for epoch-level metrics
    metrics_csv = EPOCH_LOG_DIR / "training_metrics.csv"
    try:
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_dist_loss,val_angle_loss,"
                    "val_seg_precision,val_seg_recall,val_seg_f1\n")
    except PermissionError:
        pass

    # ── STEP 4: training loop ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Training")
    print("=" * 60)

    # Enable cuDNN auto-tuner: finds the fastest conv algorithm on first batch.
    # This causes a one-time delay on the very first batch (kernel benchmarking)
    # but speeds up every subsequent batch. The delay is expected — not a hang.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("  Note: first batch may take 10-30s while cuDNN benchmarks kernels.")
        print("  Subsequent batches will be fast.")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        # Accumulate loss as a GPU tensor — avoid .item() inside the loop
        # which forces a GPU→CPU sync (stalls the GPU) on every batch.
        tr_loss_t = torch.zeros(1, device=device)
        n_tr_batches = len(train_ld)
        t0 = time.time()
        if epoch == 1:
            print(f"  Starting epoch 1 — waiting for first batch...")
        for bi, (imgs, dists, angles, masks) in enumerate(train_ld):
            if epoch == 1 and bi == 0:
                print(f"  First batch received. Training started.")
            imgs   = imgs.to(device, non_blocking=True)
            dists  = dists.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                dist_pred, angle_pred = model(imgs)
                l_dist  = dist_loss(dist_pred, dists, masks)
                l_angle = angle_loss(angle_pred, angles, masks)
                loss    = l_dist + l_angle

            scaler.scale(loss).backward()
            # Gradient clipping: prevents exploding gradients during early training
            # when the distance-field head is randomly initialised.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            # Detach before accumulating to avoid holding the full graph
            tr_loss_t += loss.detach()

        # Single .item() call per epoch — one GPU→CPU sync total
        tr_loss = tr_loss_t.item() / max(1, n_tr_batches)
        elapsed_tr = time.time() - t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} train: {n_tr_batches} batches in {elapsed_tr:.1f}s  "
              f"({elapsed_tr/n_tr_batches*1000:.0f}ms/batch)  loss={tr_loss:.4f}")

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        # ── Phase A: loss accumulation (GPU tensors, no .item() in loop) ────────
        va_loss_t  = torch.zeros(1, device=device)
        va_dist_t  = torch.zeros(1, device=device)
        va_angle_t = torch.zeros(1, device=device)
        n_va_batches = len(val_ld)
        t0_val = time.time()

        # Store per-sample predicted fields for segment-level matching below.
        # We keep them as CPU float32 numpy arrays to avoid holding GPU memory.
        pred_dist_list:  list[np.ndarray] = []   # (H,W) in [0,1]
        pred_angle_list: list[np.ndarray] = []   # (H,W) in radians

        with torch.no_grad():
            for bi, (imgs, dists, angles, masks) in enumerate(val_ld):
                imgs   = imgs.to(device, non_blocking=True)
                dists  = dists.to(device, non_blocking=True)
                angles = angles.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    dist_pred, angle_pred = model(imgs)
                    l_dist  = dist_loss(dist_pred, dists, masks)
                    l_angle = angle_loss(angle_pred, angles, masks)
                    loss    = l_dist + l_angle
                va_loss_t  += loss.detach()
                va_dist_t  += l_dist.detach()
                va_angle_t += l_angle.detach()
                # Move predictions to CPU for segment extraction (done outside loop)
                dp = dist_pred.cpu().float().numpy()   # (B,1,H,W)
                ap = angle_pred.cpu().float().numpy()  # (B,2,H,W)
                for b in range(dp.shape[0]):
                    pred_dist_list.append(dp[b, 0])
                    # Convert (cos,sin) to angle in [0,π)
                    pred_angle_list.append(
                        np.arctan2(ap[b, 1], ap[b, 0]) % math.pi
                    )

        # Single GPU→CPU sync for losses
        va_loss       = va_loss_t.item()  / max(1, n_va_batches)
        va_dist_loss  = va_dist_t.item()  / max(1, n_va_batches)
        va_angle_loss = va_angle_t.item() / max(1, n_va_batches)
        elapsed_val   = time.time() - t0_val

        # ── Phase B: segment-level matching (CPU, after GPU loop) ───────────────
        # For each val sample, extract predicted segments from the predicted
        # distance/angle fields, then match against GT segments using greedy
        # midpoint + angle matching (same protocol as chart_lsd_eval.py).
        #
        # GT segments are in full-image pixel coordinates; we convert them to
        # crop-local coordinates scaled to IMG_SIZE × IMG_SIZE.
        agg_TP = agg_FP = agg_FN = 0
        for si, (img_path, gt) in enumerate(val_ds.records):
            if si >= len(pred_dist_list):
                break
            dist_np  = pred_dist_list[si]   # (IMG_SIZE, IMG_SIZE)
            angle_np = pred_angle_list[si]  # (IMG_SIZE, IMG_SIZE)

            # Extract predicted segments.
            # dist_thresh=0.5 means: extract any pixel predicted to be within
            # 0.5 * MAX_DIST = 4px of a segment.  This is permissive enough to
            # find segments even when the model is still learning, while being
            # strict enough to avoid extracting random background noise.
            pred_segs = extract_segments_from_fields(
                dist_np, angle_np,
                dist_thresh=0.5, min_len=10.0,
            )

            # Build GT segments in crop-local IMG_SIZE coordinates
            pa = gt["pa"]
            x0, y0, x1_pa, y1_pa = pa["x0"], pa["y0"], pa["x1"], pa["y1"]
            w_crop = max(1, x1_pa - x0)
            h_crop = max(1, y1_pa - y0)
            sx = IMG_SIZE / w_crop
            sy = IMG_SIZE / h_crop
            gt_segs: list[tuple[float, float, float, float]] = [
                ((s["x1"] - x0) * sx, (s["y1"] - y0) * sy,
                 (s["x2"] - x0) * sx, (s["y2"] - y0) * sy)
                for s in gt.get("segments", [])
            ]

            tp, fp, fn = _match_segments(pred_segs, gt_segs)
            agg_TP += tp
            agg_FP += fp
            agg_FN += fn

        prec = agg_TP / (agg_TP + agg_FP) if (agg_TP + agg_FP) > 0 else 0.0
        rec  = agg_TP / (agg_TP + agg_FN) if (agg_TP + agg_FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        scheduler.step()

        # ── Save best model ────────────────────────────────────────────────────
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
            saved_str = "  ← best"
        else:
            saved_str = ""

        # ── Console summary ──────────────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch {epoch:3d}/{EPOCHS}  "
              f"train={elapsed_tr:.1f}s  val={elapsed_val:.1f}s  "
              f"lr={current_lr:.2e}  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"dist={va_dist_loss:.4f}  angle={va_angle_loss:.4f}{saved_str}")
        print(f"  Segment metrics (match≪{MATCH_DIST_PX:.0f}px, ⊢{MATCH_ANGLE_DEG:.0f}°) — "
              f"TP={agg_TP:,}  FP={agg_FP:,}  FN={agg_FN:,}  "
              f"Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")

        # ── Metrics table (console) ──────────────────────────────────────────────────────────────
        header = (f"{'Metric':<12} {'Value':>10}")
        rows = [
            ("train_loss",  f"{tr_loss:.4f}"),
            ("val_loss",    f"{va_loss:.4f}"),
            ("dist_loss",   f"{va_dist_loss:.4f}"),
            ("angle_loss",  f"{va_angle_loss:.4f}"),
            ("precision",   f"{prec:.4f}"),
            ("recall",      f"{rec:.4f}"),
            ("F1",          f"{f1:.4f}"),
        ]
        print(f"  {header}")
        print(f"  {'-'*24}")
        for name, val in rows:
            print(f"  {name:<12} {val:>10}")

        # ── CSV log ────────────────────────────────────────────────────────────
        try:
            with open(metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{tr_loss:.6f},{va_loss:.6f},"
                        f"{va_dist_loss:.6f},{va_angle_loss:.6f},"
                        f"{prec:.4f},{rec:.4f},{f1:.4f}\n")
        except PermissionError:
            pass

        # ── Save epoch-level val images every 2 epochs ─────────────────────────
        if epoch % 2 == 0 or epoch == 1:
            _save_epoch_val_images(epoch, model, val_gt, device, n_save=8)

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved → {MODEL_SAVE_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE / DETECT
# ══════════════════════════════════════════════════════════════════════════════
def detect(
    image_path: str | Path,
    model_path: str | Path = MODEL_SAVE_PATH,
    dist_thresh: float = 0.5,
    min_len: float = 10.0,
) -> list[tuple[float, float, float, float]]:
    """
    Detect line segments in a plotting-area image.

    Parameters
    ----------
    image_path : path to the plotting-area PNG (already cropped to plotting area)
    model_path : path to the saved model weights
    dist_thresh: distance threshold (normalised, 0–1) for segment extraction.
                 0.5 means "within 0.5 * MAX_DIST pixels of a segment".
    min_len    : minimum segment length in pixels (in original image coordinates)

    Returns
    -------
    List of (x1, y1, x2, y2) segment endpoints in original image coordinates.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLSDNet().to(device)
    state = torch.load(str(model_path), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Resize to IMG_SIZE (same as training) ─────────────────────────────────
    # The model was trained on IMG_SIZE × IMG_SIZE crops.  Running inference on
    # a different size gives wrong predictions because the receptive field and
    # distance scale are mismatched.
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - MEAN) / STD
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(enabled=(device.type == "cuda")):
            dist_pred, angle_pred = model(img_t)

    dist_np  = dist_pred[0, 0].cpu().float().numpy()   # (IMG_SIZE, IMG_SIZE) in [0,1]
    angle_cs = angle_pred[0].cpu().float().numpy()      # (2, IMG_SIZE, IMG_SIZE)
    angle_np = np.arctan2(angle_cs[1], angle_cs[0]) % math.pi

    # Extract segments in IMG_SIZE coordinates
    segs_scaled = extract_segments_from_fields(
        dist_np, angle_np,
        dist_thresh=dist_thresh,
        min_len=min_len * (IMG_SIZE / max(orig_w, orig_h)),  # scale min_len too
    )

    # ── Scale segment coordinates back to original image size ─────────────────
    sx = orig_w / IMG_SIZE
    sy = orig_h / IMG_SIZE
    segments = [
        (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        for (x1, y1, x2, y2) in segs_scaled
    ]

    # ── Visualise: original image + detected segments + distance field ─────────
    vis = img_bgr.copy()
    for (x1, y1, x2, y2) in segments:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 220), 2)
        cv2.circle(vis, (int(x1), int(y1)), 3, (0, 220, 0), -1)
        cv2.circle(vis, (int(x2), int(y2)), 3, (0, 220, 0), -1)

    # Distance field heatmap (rescaled to original size for inspection)
    dist_vis = cv2.resize((dist_np * 255).astype(np.uint8),
                          (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)

    # Side-by-side: original+segments | distance field
    side_by_side = np.hstack([vis, dist_color])

    out_path  = Path(image_path).with_suffix(".detected_segments.png")
    dist_path = Path(image_path).with_suffix(".dist_field.png")
    cv2.imwrite(str(out_path), side_by_side)
    cv2.imwrite(str(dist_path), dist_color)
    print(f"  Detected {len(segments)} segments")
    print(f"  Visualisation → {out_path}")
    print(f"  Distance field → {dist_path}")
    print(f"  (Blue=near segment, Red=far from segment in distance field)")

    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chart segment detector (Deep-LSD style)")
    parser.add_argument("--mode",   choices=["train", "detect"], default="train")
    parser.add_argument("--plots",  type=int, default=N_PLOTS,
                        help="Number of synthetic plots to generate (train mode)")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to plotting-area image (detect mode)")
    parser.add_argument("--model",  type=str, default=str(MODEL_SAVE_PATH),
                        help="Path to model weights (detect mode)")
    parser.add_argument("--thresh", type=float, default=0.3,
                        help="Distance threshold for segment extraction (detect mode)")
    parser.add_argument("--minlen", type=float, default=10.0,
                        help="Minimum segment length in pixels (detect mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train(n_plots=args.plots)
    elif args.mode == "detect":
        if args.image is None:
            parser.error("--image is required for detect mode")
        segs = detect(args.image, model_path=args.model,
                      dist_thresh=args.thresh, min_len=args.minlen)
        print(f"Detected {len(segs)} segments:")
        for i, (x1, y1, x2, y2) in enumerate(segs):
            print(f"  [{i:3d}]  ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})"
                  f"  len={math.hypot(x2-x1, y2-y1):.1f}")
