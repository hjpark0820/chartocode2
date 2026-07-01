"""
chart_segment_detector_binseg.py
================================
Working directory : <project>/src/
Model save path   : ../models/chart_segment_binseg.pth  (relative to src/)

OPTION B: Binary Segmentation approach.

Instead of predicting a continuous distance field (which merges parallel lines
into blobs), this model predicts a **binary line mask** — a 1-pixel-wide
rasterisation of each GT segment.  The model output is a single-channel
probability map where each pixel indicates whether it lies on a segment.

Advantages over the distance-field approach:
  • Parallel lines 3px apart remain separate connected components
  • No distance-field merging — each segment is a thin, distinct line
  • PCA extraction works correctly because components are thin and elongated
  • Simpler loss (binary cross-entropy with positive-class weighting)
  • Faster training convergence

Ground truth: 1px-wide Bresenham lines drawn between consecutive marker pairs
of the same series (using ALL series_pixels, not MIN_SEP-filtered).

USAGE
-----
  # Generate plots + train
  python chart_segment_detector_binseg.py --mode train

  # Detect segments in a plotting-area image
  python chart_segment_detector_binseg.py --mode detect --image path/to/plotting_area.png

  # Generate plots only (no training)
  python chart_segment_detector_binseg.py --mode generate --plots 1000

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
MODEL_SAVE_PATH = _SRC_DIR / ".." / "models" / "chart_segment_binseg.pth"
SYNTH_DIR       = _SRC_DIR / ".." / "data" / "synthetic_plots"
EPOCH_LOG_DIR   = _SRC_DIR / ".." / "data" / "binseg_epoch_logs"
DATASET_CACHE_DIR = _SRC_DIR / ".." / "data" / "binseg_dataset_cache"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
N_PLOTS      = 1000
MARKER_PT    = 8
DPI          = 100
PLOT_W_IN    = 5.6
PLOT_H_IN    = 4.2
PLOT_W_PX    = int(PLOT_W_IN * DPI)
PLOT_H_PX    = int(PLOT_H_IN * DPI)
N_POINTS     = 12

EPOCHS       = 100
LR           = 3e-4
BATCH_SIZE   = 8
IMG_SIZE     = 320
VAL_FRAC     = 0.15
WORKERS      = min(4, mp.cpu_count())
USE_COMPILE  = False

# Line rasterisation width for GT mask (1 = single pixel Bresenham line)
LINE_WIDTH   = 1

# Dilation radius for GT mask — makes the target slightly wider so the model
# gets a stronger gradient signal. 2px is a good balance: wide enough to give
# a strong gradient, narrow enough that the predicted blob skeletonizes into a
# clean 1px line without branching artifacts. Parallel lines in a 320px image
# are typically 8-15px apart, so 2px dilation keeps them well separated.
DILATE_RADIUS = 2

# Minimum segment length (pixels)
MIN_SEG_LEN  = 5.0

# ══════════════════════════════════════════════════════════════════════════════
#  SYMBOL / SERIES CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "filled_circle", "open_circle",
    "filled_square",  "open_square",
    "open_triangle",  "open_inv_triangle",
    "filled_triangle","filled_inv_triangle",
    "open_rhombus",   "filled_rhombus",
    "x_marker",       "plus_marker",
]
N_SYMBOLS = len(CLASS_NAMES)

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
#  AUGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _add_paper_texture(img, strength, rng):
    grain = rng.normal(0, strength * 30, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)

def _add_yellowing(img, strength):
    out = img.astype(np.float32)
    out[:, :, 0] -= strength * 40
    out[:, :, 1] -= strength * 10
    out[:, :, 2] += strength * 20
    out += strength * 15
    return np.clip(out, 0, 255).astype(np.uint8)

def _add_blur(img, strength):
    k = max(1, int(strength * 3))
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), strength * 0.8)

def _add_elastic_warp(img, strength, rng):
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

def _apply_scan_noise(img, level, rng):
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
#  HILL FUNCTION
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
#  MEASURE SYMBOL DIAMETER
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
#  PLOT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def generate_one_plot(args_tuple):
    """Generate one synthetic dose-response plot with GT segments."""
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

    # Build raw_points
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

    # MIN_SEP filter (for marker detection, NOT for segment GT)
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

    # Build segment GT from ALL raw_points (not MIN_SEP-filtered)
    segments = []
    by_series: dict[int, list] = {}
    for pt in raw_points:
        by_series.setdefault(pt["series_idx"], []).append(pt)
    for si, pts_s in by_series.items():
        pts_s.sort(key=lambda p: p["point_idx"])
        for i in range(len(pts_s) - 1):
            p0, p1 = pts_s[i], pts_s[i + 1]
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

    gt_data = {
        "plot_w": W_px, "plot_h": H_px,
        "pa": {"x0": pa_x0, "y0": pa_y0, "x1": pa_x1, "y1": pa_y1},
        "points": gt_points,
        "all_points": raw_points,
        "segments": segments,
    }
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f)

    return str(gt_path)


# ══════════════════════════════════════════════════════════════════════════════
#  GT LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _segments_from_points(points: list[dict]) -> list[dict]:
    by_series: dict[int, list] = {}
    for pt in points:
        si = pt.get("series_idx", pt.get("class_idx"))
        if si is None:
            continue
        by_series.setdefault(si, []).append(pt)
    segments = []
    for si, pts_s in by_series.items():
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
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "series_idx": si, "length": round(length, 2),
                })
    return segments


def _segments_from_series_pixels(series_pixels: list[list[dict]]) -> list[dict]:
    segments: list[dict] = []
    for si, sp in enumerate(series_pixels):
        for i in range(len(sp) - 1):
            p0, p1 = sp[i], sp[i + 1]
            length = math.hypot(p1["cx"] - p0["cx"], p1["cy"] - p0["cy"])
            if length >= MIN_SEG_LEN:
                segments.append({
                    "x1": p0["cx"], "y1": p0["cy"],
                    "x2": p1["cx"], "y2": p1["cy"],
                    "series_idx": si, "length": round(length, 2),
                })
    return segments


def _load_gt(gt_path: Path) -> dict:
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
    gt_files = sorted(synth_dir.glob("gt_*.json"))
    if not gt_files:
        return
    n_migrated = 0
    n_skipped  = 0
    t0 = time.time()
    print("\n" + "=" * 60)
    print("STEP 1b \u2014 Migrating GT files (injecting 'segments' key)")
    print("=" * 60)
    for i, gt_path in enumerate(gt_files):
        _pbar(i + 1, len(gt_files), t0, prefix="  Migrating: ")
        try:
            with open(gt_path, encoding="utf-8") as f:
                gt = json.load(f)
        except Exception:
            n_skipped += 1
            continue
        if gt.get("segments"):
            n_skipped += 1
            continue
        if gt.get("series_pixels"):
            gt["segments"] = _segments_from_series_pixels(gt["series_pixels"])
        else:
            gt["segments"] = _segments_from_points(gt.get("points", []))
        try:
            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gt, f)
            n_migrated += 1
        except PermissionError:
            n_skipped += 1
    print(f"  Migrated: {n_migrated}  |  Already had segments: {n_skipped}  "
          f"(took {time.time()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════════════════════
#  BINARY LINE MASK COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_line_mask(segments: list[dict], h: int, w: int,
                      line_width: int = LINE_WIDTH) -> np.ndarray:
    """
    Rasterise GT segments as thin lines on a binary mask.

    Returns:
        mask (h, w) float32 in {0, 1}
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in segments:
        x1 = int(round(seg["x1"])); y1 = int(round(seg["y1"]))
        x2 = int(round(seg["x2"])); y2 = int(round(seg["y2"]))
        cv2.line(mask, (x1, y1), (x2, y2), 1, line_width)
    return mask.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SegmentDataset(Dataset):
    """
    Each sample is one plotting-area crop.
    Returns:
      image  (3, IMG_SIZE, IMG_SIZE) float32 normalised
      mask   (1, IMG_SIZE, IMG_SIZE) float32 binary line mask
    """
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _cache_key(gt_paths: list[Path]) -> str:
        import hashlib
        h = hashlib.md5()
        for p in sorted(gt_paths):
            h.update(str(p).encode())
            try:
                h.update(str(p.stat().st_mtime).encode())
            except OSError:
                pass
        h.update(f"binseg_lw{LINE_WIDTH}_dr{DILATE_RADIUS}".encode())
        return h.hexdigest()[:16]

    def __init__(self, gt_paths: list[Path], augment: bool = True,
                 label: str = "", cache_dir: Path | None = None):
        self.augment = augment
        self.records: list[tuple[Path, dict]] = []
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Try to load from cache
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
                return

        # Cache miss: compute from scratch
        _dummy = (torch.zeros(3, IMG_SIZE, IMG_SIZE),
                  torch.zeros(1, IMG_SIZE, IMG_SIZE))

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

            # Scale segment coordinates to IMG_SIZE
            sx = IMG_SIZE / w_crop
            sy = IMG_SIZE / h_crop
            segs_local = [
                {"x1": (s["x1"] - x0) * sx, "y1": (s["y1"] - y0) * sy,
                 "x2": (s["x2"] - x0) * sx, "y2": (s["y2"] - y0) * sy}
                for s in gt.get("segments", [])
            ]

            # Resize crop
            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                      interpolation=cv2.INTER_LINEAR)

            # Compute binary line mask
            line_mask = compute_line_mask(segs_local, IMG_SIZE, IMG_SIZE)

            # Dilate mask to give the model a wider gradient signal.
            # DILATE_RADIUS controls the half-width of the dilation kernel.
            dk = 2 * DILATE_RADIUS + 1
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
            line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)

            img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_rgb = (img_rgb - self._MEAN) / self._STD
            img_t  = torch.from_numpy(img_rgb.transpose(2, 0, 1))    # (3,H,W)
            mask_t = torch.from_numpy(line_mask[None])                 # (1,H,W)

            self.samples.append((img_t, mask_t))

        elapsed = time.time() - t0
        print(f"  Done \u2014 {n_total} samples pre-computed in {elapsed:.1f}s")

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
        img_t, mask_t = self.samples[idx]
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                img_t  = img_t.flip(-1)
                mask_t = mask_t.flip(-1)
            # Vertical flip
            if random.random() < 0.5:
                img_t  = img_t.flip(-2)
                mask_t = mask_t.flip(-2)
            # 90-degree rotation (0, 90, 180, 270 with equal probability)
            k = random.randint(0, 3)
            if k > 0:
                img_t  = torch.rot90(img_t,  k, dims=(-2, -1))
                mask_t = torch.rot90(mask_t, k, dims=(-2, -1))
            # Brightness / contrast jitter (image only, not mask)
            if random.random() < 0.5:
                brightness = random.uniform(0.7, 1.3)
                contrast   = random.uniform(0.8, 1.2)
                img_t = img_t * contrast + (brightness - 1.0)
        return img_t, mask_t


def _collate(batch):
    imgs, masks = zip(*batch)
    return torch.stack(list(imgs)), torch.stack(list(masks))


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL  (U-Net with pretrained ResNet-34 encoder)
# ══════════════════════════════════════════════════════════════════════════════
class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

class _UpBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_c + skip_c, out_c),
            _ConvBnRelu(out_c, out_c),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class BinSegNet(nn.Module):
    """
    U-Net with a pretrained ResNet-34 encoder.

    Using a pretrained backbone gives the model strong low-level edge and
    texture features from the start, so it can find thin lines much faster
    than training from random weights.  This is the primary fix for the
    'good initial F1 but no growth' problem.

    Output: (B, 1, H, W) raw logits — apply sigmoid for probabilities.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        weights = tvm.ResNet34_Weights.DEFAULT if pretrained else None
        backbone = tvm.resnet34(weights=weights)

        # Encoder stages (reuse ResNet-34 layers)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                   backbone.relu)          # /2  -> 64ch
        self.pool0 = backbone.maxpool                       # /4
        self.enc1  = backbone.layer1                        # /4  -> 64ch
        self.enc2  = backbone.layer2                        # /8  -> 128ch
        self.enc3  = backbone.layer3                        # /16 -> 256ch
        self.enc4  = backbone.layer4                        # /32 -> 512ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            _ConvBnRelu(512, 512),
            _ConvBnRelu(512, 512),
        )

        # Decoder
        self.dec4 = _UpBlock(512, 256, 256)   # /16
        self.dec3 = _UpBlock(256, 128, 128)   # /8
        self.dec2 = _UpBlock(128,  64,  64)   # /4
        self.dec1 = _UpBlock( 64,  64,  64)   # /2  (skip from enc0, 64ch)
        self.dec0 = nn.Sequential(             # /1  (no skip)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            _ConvBnRelu(64, 32),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)          # (B, 64,  H/2,  W/2)
        p0 = self.pool0(e0)        # (B, 64,  H/4,  W/4)
        e1 = self.enc1(p0)         # (B, 64,  H/4,  W/4)
        e2 = self.enc2(e1)         # (B,128,  H/8,  W/8)
        e3 = self.enc3(e2)         # (B,256,  H/16, W/16)
        e4 = self.enc4(e3)         # (B,512,  H/32, W/32)
        b  = self.bottleneck(e4)   # (B,512,  H/32, W/32)
        d4 = self.dec4(b,  e3)     # (B,256,  H/16, W/16)
        d3 = self.dec3(d4, e2)     # (B,128,  H/8,  W/8)
        d2 = self.dec2(d3, e1)     # (B, 64,  H/4,  W/4)
        d1 = self.dec1(d2, e0)     # (B, 64,  H/2,  W/2)
        d0 = self.dec0(d1)         # (B, 32,  H,    W)
        return self.head(d0)       # (B,  1,  H,    W) raw logits


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS  (focal BCE — handles extreme class imbalance)
# ══════════════════════════════════════════════════════════════════════════════
def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 0.85, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy loss.

    With DILATE_RADIUS=2, the GT mask has ~1-3% positive pixels (line pixels)
    and ~97-99% negative pixels (background).  Standard BCE is dominated by
    the easy negatives.  Focal loss down-weights easy examples and focuses on
    hard positives.

    alpha=0.85 : higher weight for positive class to compensate for the very
                 small fraction of positive pixels with thin dilation
    gamma=2.0  : focusing parameter (higher = more focus on hard examples)
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    # p_t = probability of the correct class
    p_t = probs * targets + (1 - probs) * (1 - targets)
    # Focal modulating factor
    focal_weight = (1 - p_t) ** gamma
    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce
    return loss.mean()


# Dice loss for additional shape-aware supervision
def dice_loss(logits: torch.Tensor, targets: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT EXTRACTION FROM BINARY MASK
# ══════════════════════════════════════════════════════════════════════════════
def _find_junction_pixels(skeleton: np.ndarray) -> np.ndarray:
    """
    Find junction (branch-point) pixels in a 1px-wide skeleton.

    A junction pixel is a skeleton pixel that has 3 or more skeleton
    neighbours in its 3×3 neighbourhood.  Removing these pixels splits
    the skeleton into individual line branches.

    Returns a boolean mask of junction pixels.
    """
    # Count neighbours using convolution with a 3x3 kernel (excluding centre)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    skel_u8 = (skeleton > 0).astype(np.uint8)
    neighbour_count = cv2.filter2D(skel_u8, -1, kernel)
    # Junction = skeleton pixel with >= 3 neighbours
    junctions = (skel_u8 > 0) & (neighbour_count >= 3)
    return junctions


def extract_segments_from_mask(
    prob_map: np.ndarray,
    threshold: float = 0.5,
    min_len: float = 10.0,
    hough_threshold: int = 6,
    hough_max_gap: int = 6,
) -> list[tuple[float, float, float, float]]:
    """
    Extract line segments from a predicted probability map.

    Strategy (skeleton → split at junctions → PCA per branch):
      1. Threshold to binary mask
      2. Skeletonize the entire mask to 1px-wide lines
      3. Detect junction pixels (pixels with ≥3 skeleton neighbours)
      4. Remove **only** the junction pixels themselves (no dilation) to
         split the skeleton into individual branches while preserving
         as many pixels as possible for recall
      5. Find connected components on the split skeleton
      6. For each branch with ≥3 pixels:
         - Fit a PCA line (project all pixels onto principal axis, take
           the two extreme projected points as endpoints)
      7. Merge collinear overlapping segments from the same series

    Why junction-splitting + PCA works:
      With 12 series packed into 320×320, dilation causes lines to merge
      into one giant blob.  Even after skeletonization, the skeleton is
      still ONE connected component.  Running Hough on this giant skeleton
      produces random diagonal segments (the FP problem).  By removing
      junction pixels, we split the skeleton into ~100-200 individual
      branches, each of which is a clean straight line that PCA can fit.
      Using PCA instead of Hough avoids the Hough parameter-sensitivity
      problem entirely.

    Returns list of (x1, y1, x2, y2) in image coordinates.
    """
    binary = (prob_map > threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # ── Step 1: Skeletonize to 1px lines ─────────────────────────────────
    try:
        from skimage.morphology import skeletonize as _skeletonize
        skeleton = _skeletonize(binary > 0).astype(np.uint8)
    except ImportError:
        if hasattr(cv2, 'ximgproc'):
            skeleton = cv2.ximgproc.thinning(binary * 255)
            skeleton = (skeleton > 0).astype(np.uint8)
        else:
            skeleton = binary

    if skeleton.sum() == 0:
        return []

    # ── Step 2: Split skeleton at junctions ────────────────────────────
    junctions = _find_junction_pixels(skeleton)
    # Remove ONLY the junction pixels (no dilation) to preserve maximum
    # branch length for recall.  This is less aggressive than dilating
    # the junction mask.
    split_skeleton = skeleton.copy()
    split_skeleton[junctions] = 0

    # ── Step 3: Connected components on the split skeleton ───────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        split_skeleton, connectivity=8)

    raw_segs: list[tuple[float, float, float, float]] = []

    for lbl in range(1, n_labels):  # skip background (label 0)
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < 3:  # ignore tiny noise specks
            continue

        # Extract component pixels
        ys, xs = np.where(labels == lbl)
        pts = np.column_stack([xs, ys]).astype(np.float32)  # (N, 2)

        if len(pts) < 2:
            continue

        # PCA to find the principal axis
        mean = pts.mean(axis=0)
        centered = pts - mean
        try:
            _, svals, vt = np.linalg.svd(centered, full_matrices=False)
            axis = vt[0]  # principal direction
        except np.linalg.LinAlgError:
            continue

        proj = centered @ axis
        proj_range = proj.max() - proj.min()

        if proj_range < 3.0:
            continue  # too short even for merging

        # Fit PCA line: endpoints are the extreme projections
        i_min = proj.argmin()
        i_max = proj.argmax()
        p1 = mean + proj[i_min] * axis
        p2 = mean + proj[i_max] * axis
        raw_segs.append((float(p1[0]), float(p1[1]),
                         float(p2[0]), float(p2[1])))

    # ── Step 4: Merge collinear overlapping segments ─────────────────────
    merged = _merge_collinear_segments(raw_segs, dist_tol=8.0, angle_tol=12.0)
    # Final length filter
    return [(x1, y1, x2, y2) for (x1, y1, x2, y2) in merged
            if math.hypot(x2-x1, y2-y1) >= min_len]


def _merge_collinear_segments(
    segs: list[tuple[float, float, float, float]],
    dist_tol: float = 6.0,
    angle_tol: float = 10.0,
    max_passes: int = 3,
) -> list[tuple[float, float, float, float]]:
    """
    Merge collinear overlapping / nearby segments.

    After junction-splitting, a single GT segment may become 2-4 short
    fragments.  This function merges them back by checking:
      1. Angle similarity (within *angle_tol* degrees)
      2. Perpendicular distance of ALL FOUR endpoints of the two segments
         to the line through the first segment (within *dist_tol* pixels)

    Using endpoint-to-line distance (instead of midpoint-to-midpoint)
    correctly handles the case where two collinear fragments are far apart
    along the line direction but perfectly aligned perpendicularly.

    Runs multiple passes so that chain-merging (A+B, then AB+C) works.
    """
    if not segs:
        return []

    def _point_to_line_dist(px, py, lx, ly, dx, dy):
        """Perpendicular distance from point (px,py) to line through (lx,ly) with direction (dx,dy)."""
        nx, ny = -dy, dx  # normal
        return abs((px - lx) * nx + (py - ly) * ny)

    current = list(segs)
    for _pass in range(max_passes):
        n = len(current)
        used = [False] * n
        merged = []
        changed = False

        for i in range(n):
            if used[i]:
                continue
            used[i] = True
            group = [current[i]]

            # Compute the line direction for the seed segment
            sx1, sy1, sx2, sy2 = current[i]
            smx, smy = (sx1+sx2)/2, (sy1+sy2)/2
            sa = math.atan2(sy2-sy1, sx2-sx1) % math.pi
            sdx, sdy = math.cos(sa), math.sin(sa)

            for j in range(i + 1, n):
                if used[j]:
                    continue
                jx1, jy1, jx2, jy2 = current[j]
                ja = math.atan2(jy2-jy1, jx2-jx1) % math.pi

                # Angle check
                da = abs(sa - ja)
                if da > math.pi / 2:
                    da = math.pi - da
                if math.degrees(da) > angle_tol:
                    continue

                # Perpendicular distance: check all 4 endpoints against
                # the line through the seed segment's midpoint
                d1 = _point_to_line_dist(jx1, jy1, smx, smy, sdx, sdy)
                d2 = _point_to_line_dist(jx2, jy2, smx, smy, sdx, sdy)
                d3 = _point_to_line_dist(sx1, sy1, (jx1+jx2)/2, (jy1+jy2)/2,
                                         math.cos(ja), math.sin(ja))
                d4 = _point_to_line_dist(sx2, sy2, (jx1+jx2)/2, (jy1+jy2)/2,
                                         math.cos(ja), math.sin(ja))
                max_perp = min(max(d1, d2), max(d3, d4))  # take the better fit
                if max_perp > dist_tol:
                    continue

                used[j] = True
                group.append(current[j])
                changed = True

            # Merge group: project all endpoints onto PCA axis, take extremes
            all_pts = []
            for s in group:
                all_pts.append((s[0], s[1]))
                all_pts.append((s[2], s[3]))
            pts = np.array(all_pts, dtype=np.float32)
            mean = pts.mean(axis=0)
            centered = pts - mean
            try:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                axis = vt[0]
            except np.linalg.LinAlgError:
                axis = np.array([sdx, sdy])
            proj = centered @ axis
            i_min, i_max = proj.argmin(), proj.argmax()
            p1 = mean + proj[i_min] * axis
            p2 = mean + proj[i_max] * axis
            merged.append((float(p1[0]), float(p1[1]),
                           float(p2[0]), float(p2[1])))

        current = merged
        if not changed:
            break  # converged

    return current


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT-LEVEL MATCHING
# ══════════════════════════════════════════════════════════════════════════════
MATCH_DIST_PX  = 10.0
MATCH_ANGLE_DEG = 15.0

def _seg_midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def _seg_angle_deg(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    return math.degrees(math.atan2(abs(dy), abs(dx)))

def _angle_diff_deg(a1, a2):
    d = abs(a1 - a2)
    return 180.0 - d if d > 90.0 else d

def _point_to_seg_dist(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to the line segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _match_segments(pred_segs, gt_segs, match_dist=MATCH_DIST_PX,
                    match_angle=MATCH_ANGLE_DEG):
    """
    Coverage-based segment matching.

    A GT segment is matched (TP) if there exists a predicted segment such that:
      1. Both GT endpoints are within match_dist of the predicted segment line
      2. The angle difference is within match_angle degrees

    This allows one long predicted segment to cover multiple short GT segments
    (which is the correct behaviour for Hough-extracted segments vs point-to-point GT).

    A predicted segment is a FP if it does not cover any unmatched GT segment.
    """
    matched_gt: set[int] = set()
    pred_matched_any = [False] * len(pred_segs)

    for pi, (px1, py1, px2, py2) in enumerate(pred_segs):
        pa = _seg_angle_deg(px1, py1, px2, py2)
        for gi, (gx1, gy1, gx2, gy2) in enumerate(gt_segs):
            if gi in matched_gt:
                continue
            ga = _seg_angle_deg(gx1, gy1, gx2, gy2)
            if _angle_diff_deg(pa, ga) > match_angle:
                continue
            # Check if both GT endpoints are close to the predicted segment
            d1 = _point_to_seg_dist(gx1, gy1, px1, py1, px2, py2)
            d2 = _point_to_seg_dist(gx2, gy2, px1, py1, px2, py2)
            if d1 <= match_dist and d2 <= match_dist:
                matched_gt.add(gi)
                pred_matched_any[pi] = True

    tp = len(matched_gt)
    fp = sum(1 for m in pred_matched_any if not m)
    fn = len(gt_segs) - tp
    return tp, fp, fn


# ══════════════════════════════════════════════════════════════════════════════
#  EPOCH VAL IMAGES
# ══════════════════════════════════════════════════════════════════════════════
def _save_epoch_val_images(epoch, model, val_gt_paths, device, n_save=8):
    out_dir = EPOCH_LOG_DIR / f"epoch_{epoch:03d}" / "val_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    chosen = val_gt_paths[:n_save]
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

        # Resize to IMG_SIZE
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_norm = (img_rgb - _MEAN) / _STD
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast(enabled=(device.type == "cuda")):
                logits = model(img_t)

        prob_np = torch.sigmoid(logits[0, 0]).cpu().float().numpy()

        # Scale factors
        sx_back = w_c / IMG_SIZE
        sy_back = h_c / IMG_SIZE

        # Draw GT segments (green)
        vis = crop.copy()
        for seg in gt.get("segments", []):
            gx1, gy1 = int(seg["x1"] - x0), int(seg["y1"] - y0)
            gx2, gy2 = int(seg["x2"] - x0), int(seg["y2"] - y0)
            cv2.line(vis, (gx1, gy1), (gx2, gy2), (0, 200, 0), 1)

        # Extract predicted segments
        pred_segs = extract_segments_from_mask(prob_np, threshold=0.5, min_len=10.0)
        for (px1, py1, px2, py2) in pred_segs:
            cv2.line(vis,
                     (int(px1 * sx_back), int(py1 * sy_back)),
                     (int(px2 * sx_back), int(py2 * sy_back)),
                     (0, 0, 220), 1)

        # Probability heatmap
        prob_vis = cv2.resize(
            (prob_np * 255).clip(0, 255).astype(np.uint8),
            (w_c, h_c), interpolation=cv2.INTER_LINEAR)
        prob_heat = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)

        # GT mask visualisation
        gt_mask = compute_line_mask(
            [{"x1": (s["x1"] - x0) * (IMG_SIZE / w_c),
              "y1": (s["y1"] - y0) * (IMG_SIZE / h_c),
              "x2": (s["x2"] - x0) * (IMG_SIZE / w_c),
              "y2": (s["y2"] - y0) * (IMG_SIZE / h_c)}
             for s in gt.get("segments", [])],
            IMG_SIZE, IMG_SIZE
        )
        gt_mask_vis = cv2.resize(
            (gt_mask * 255).astype(np.uint8),
            (w_c, h_c), interpolation=cv2.INTER_NEAREST)
        gt_mask_color = cv2.applyColorMap(gt_mask_vis, cv2.COLORMAP_JET)

        panel = np.hstack([crop, vis, gt_mask_color, prob_heat])
        stem  = gt_path.stem.replace("gt_", "")
        cv2.imwrite(str(out_dir / f"plot_{stem}.png"), panel)

    print(f"  Val images saved \u2192 {out_dir}  ({len(chosen)} plots)")


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train(n_plots: int = N_PLOTS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("\n" + "!" * 60)
        print("WARNING: No CUDA GPU detected \u2014 training on CPU.")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("!" * 60)

    # STEP 1: generate synthetic plots
    print("\n" + "=" * 60)
    print("STEP 1 \u2014 Generating synthetic plots")
    print("=" * 60)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(SYNTH_DIR.glob("plot_*.png")))
    if existing >= n_plots:
        print(f"  {existing} plots already exist \u2014 skipping generation.")
    else:
        seeds = [random.randint(0, 2**31) for _ in range(n_plots)]
        args  = [(i, str(SYNTH_DIR), seeds[i]) for i in range(n_plots)]
        n_cpu = max(1, WORKERS)
        print(f"  Generating {n_plots} plots using {n_cpu} CPU workers...")
        t0 = time.time()
        with mp.Pool(n_cpu) as pool:
            results = pool.map(generate_one_plot, args)
        print(f"  Done in {time.time()-t0:.1f}s \u2014 {len(results)} plots saved.")

    _migrate_gt_segments(SYNTH_DIR)

    # STEP 2: dataset split
    print("\n" + "=" * 60)
    print("STEP 2 \u2014 Building dataset split")
    print("=" * 60)
    all_gt = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not all_gt:
        raise RuntimeError(f"No GT files found in {SYNTH_DIR}")
    random.shuffle(all_gt)
    n_val   = max(1, int(len(all_gt) * VAL_FRAC))
    val_gt  = all_gt[:n_val]
    train_gt = all_gt[n_val:]
    print(f"  Total: {len(all_gt)}  |  Train: {len(train_gt)}  |  Val: {len(val_gt)}")

    # STEP 3: model
    print("\n" + "=" * 60)
    print("STEP 3 \u2014 Building model")
    print("=" * 60)
    model = BinSegNet().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  Parameters : {trainable:,} trainable / {total_p:,} total")

    # Differential learning rates: pretrained backbone gets 10x lower LR
    # to preserve ImageNet features while the decoder learns fast.
    backbone_params = list(model.enc0.parameters()) + \
                      list(model.enc1.parameters()) + \
                      list(model.enc2.parameters()) + \
                      list(model.enc3.parameters()) + \
                      list(model.enc4.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    decoder_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},
        {'params': decoder_params,  'lr': LR},
    ], weight_decay=1e-4)
    def _lr_lambda(epoch_0idx):
        warmup = 5
        if epoch_0idx < warmup:
            return (epoch_0idx + 1) / warmup
        progress = (epoch_0idx - warmup) / max(1, EPOCHS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # STEP 3b: pre-compute dataset
    print("\n" + "=" * 60)
    print("STEP 3b \u2014 Loading dataset tensors (cached after first run)")
    print("=" * 60)
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = SegmentDataset(train_gt, augment=True,  label="train",
                              cache_dir=DATASET_CACHE_DIR)
    val_ds   = SegmentDataset(val_gt,   augment=False, label="val",
                              cache_dir=DATASET_CACHE_DIR)
    n_segs = sum(len(r[1].get("segments", [])) for r in train_ds.records + val_ds.records)
    print(f"  Total GT segments: {n_segs:,}")

    # Check positive pixel ratio
    pos_pixels = sum(s[1].sum().item() for s in train_ds.samples)
    total_pixels = len(train_ds.samples) * IMG_SIZE * IMG_SIZE
    pos_ratio = pos_pixels / total_pixels if total_pixels > 0 else 0
    print(f"  Positive pixel ratio: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=_collate,
                          pin_memory=(device.type == "cuda"))
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, collate_fn=_collate,
                          pin_memory=(device.type == "cuda"))

    best_val_loss = float("inf")
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EPOCH_LOG_DIR.mkdir(parents=True, exist_ok=True)

    metrics_csv = EPOCH_LOG_DIR / "training_metrics.csv"
    try:
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_seg_precision,val_seg_recall,val_seg_f1\n")
    except PermissionError:
        pass

    # STEP 4: training loop
    print("\n" + "=" * 60)
    print("STEP 4 \u2014 Training")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss_t = torch.zeros(1, device=device)
        n_tr = len(train_ld)
        t0 = time.time()

        for bi, (imgs, masks) in enumerate(train_ld):
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = focal_bce_loss(logits, masks) + dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss_t += loss.detach()

        tr_loss = tr_loss_t.item() / max(1, n_tr)
        elapsed_tr = time.time() - t0

        # Validate
        model.eval()
        va_loss_t = torch.zeros(1, device=device)
        n_va = len(val_ld)
        t0_val = time.time()

        pred_prob_list: list[np.ndarray] = []

        with torch.no_grad():
            for imgs, masks in val_ld:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = focal_bce_loss(logits, masks) + dice_loss(logits, masks)
                va_loss_t += loss.detach()
                probs = torch.sigmoid(logits).cpu().float().numpy()
                for b in range(probs.shape[0]):
                    pred_prob_list.append(probs[b, 0])

        va_loss = va_loss_t.item() / max(1, n_va)
        elapsed_val = time.time() - t0_val

        # Segment-level matching
        agg_TP = agg_FP = agg_FN = 0
        for si, (img_path, gt) in enumerate(val_ds.records):
            if si >= len(pred_prob_list):
                break
            prob_np = pred_prob_list[si]
            pred_segs = extract_segments_from_mask(prob_np, threshold=0.5, min_len=10.0)

            pa = gt["pa"]
            x0, y0, x1_pa, y1_pa = pa["x0"], pa["y0"], pa["x1"], pa["y1"]
            w_crop = max(1, x1_pa - x0)
            h_crop = max(1, y1_pa - y0)
            sx = IMG_SIZE / w_crop
            sy = IMG_SIZE / h_crop
            gt_segs = [
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

        # Save best
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
            saved_str = "  \u2190 best"
        else:
            saved_str = ""

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{EPOCHS}  "
              f"train={elapsed_tr:.1f}s  val={elapsed_val:.1f}s  "
              f"lr={current_lr:.2e}  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"TP={agg_TP}  FP={agg_FP}  FN={agg_FN}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}{saved_str}")

        try:
            with open(metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{tr_loss:.6f},{va_loss:.6f},"
                        f"{prec:.4f},{rec:.4f},{f1:.4f}\n")
        except PermissionError:
            pass

        if epoch % 2 == 0 or epoch == 1:
            _save_epoch_val_images(epoch, model, val_gt, device, n_save=8)

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved \u2192 {MODEL_SAVE_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE / DETECT
# ══════════════════════════════════════════════════════════════════════════════
def detect(image_path, model_path=MODEL_SAVE_PATH,
           threshold=0.5, min_len=10.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinSegNet().to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - MEAN) / STD
    img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(enabled=(device.type == "cuda")):
            logits = model(img_t)

    prob_np = torch.sigmoid(logits[0, 0]).cpu().float().numpy()

    segs_scaled = extract_segments_from_mask(
        prob_np, threshold=threshold,
        min_len=min_len * (IMG_SIZE / max(orig_w, orig_h)),
    )

    sx = orig_w / IMG_SIZE
    sy = orig_h / IMG_SIZE
    segments = [
        (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        for (x1, y1, x2, y2) in segs_scaled
    ]

    vis = img_bgr.copy()
    for (x1, y1, x2, y2) in segments:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 220), 2)
        cv2.circle(vis, (int(x1), int(y1)), 3, (0, 220, 0), -1)
        cv2.circle(vis, (int(x2), int(y2)), 3, (0, 220, 0), -1)

    prob_vis = cv2.resize((prob_np * 255).astype(np.uint8),
                          (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)

    side_by_side = np.hstack([vis, prob_color])
    out_path  = Path(image_path).with_suffix(".detected_segments.png")
    prob_path = Path(image_path).with_suffix(".prob_field.png")
    cv2.imwrite(str(out_path), side_by_side)
    cv2.imwrite(str(prob_path), prob_color)
    print(f"  Detected {len(segments)} segments")
    print(f"  Visualisation \u2192 {out_path}")
    print(f"  Probability field \u2192 {prob_path}")

    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE ONLY
# ══════════════════════════════════════════════════════════════════════════════
def generate_plots(n_plots: int = N_PLOTS):
    """Generate synthetic plots without training."""
    print("\n" + "=" * 60)
    print("Generating synthetic plots")
    print("=" * 60)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(SYNTH_DIR.glob("plot_*.png")))
    if existing >= n_plots:
        print(f"  {existing} plots already exist \u2014 skipping generation.")
        print(f"  To regenerate, delete {SYNTH_DIR} first.")
    else:
        seeds = [random.randint(0, 2**31) for _ in range(n_plots)]
        args  = [(i, str(SYNTH_DIR), seeds[i]) for i in range(n_plots)]
        n_cpu = max(1, WORKERS)
        print(f"  Generating {n_plots} plots using {n_cpu} CPU workers...")
        t0 = time.time()
        with mp.Pool(n_cpu) as pool:
            results = pool.map(generate_one_plot, args)
        print(f"  Done in {time.time()-t0:.1f}s \u2014 {len(results)} plots saved.")

    _migrate_gt_segments(SYNTH_DIR)
    print(f"\n  Plots directory: {SYNTH_DIR.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chart segment detector (binary segmentation)")
    parser.add_argument("--mode",
                        choices=["train", "detect", "generate"],
                        default="train")
    parser.add_argument("--plots",  type=int, default=N_PLOTS,
                        help="Number of synthetic plots to generate")
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to plotting-area image (detect mode)")
    parser.add_argument("--model",  type=str, default=str(MODEL_SAVE_PATH),
                        help="Path to model weights (detect mode)")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Probability threshold for segment extraction")
    parser.add_argument("--minlen", type=float, default=10.0,
                        help="Minimum segment length in pixels")
    args = parser.parse_args()

    if args.mode == "train":
        train(n_plots=args.plots)
    elif args.mode == "detect":
        if args.image is None:
            parser.error("--image is required for detect mode")
        segs = detect(args.image, model_path=args.model,
                      threshold=args.thresh, min_len=args.minlen)
        print(f"Detected {len(segs)} segments:")
        for i, (x1, y1, x2, y2) in enumerate(segs):
            print(f"  [{i:3d}]  ({x1:.1f}, {y1:.1f}) \u2192 ({x2:.1f}, {y2:.1f})"
                  f"  len={math.hypot(x2-x1, y2-y1):.1f}")
    elif args.mode == "generate":
        generate_plots(n_plots=args.plots)
