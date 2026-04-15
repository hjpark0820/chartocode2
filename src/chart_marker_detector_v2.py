"""
chart_marker_detector_v2.py
============================
Working directory : C:\\Users\\ziola\\OneDrive\\Documents\\GitHub\\chartocode\\src
Model save path   : ../models/chart_marker_net_v2.pth   (relative to src/)

Subimage storage  : D:\\chartocode_subimages\\
  All subimage patches are pre-generated ONCE and saved as a single
  memory-mapped NumPy file on D:\\ to avoid slow on-the-fly rendering
  during training.  This makes each epoch ~10-20× faster.

USAGE
-----
  # Train (generates 2000+ synthetic plots, extracts subimages, trains ViT)
  python chart_marker_detector_v2.py --mode train

  # Detect markers in a plotting-area image
  python chart_marker_detector_v2.py --mode detect --image path/to/plotting_area.png

REQUIREMENTS
------------
  pip install timm torch torchvision opencv-python matplotlib scikit-learn numpy
  For GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"""

from __future__ import annotations
import argparse, json, math, os, random, time, warnings
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
import timm
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
mp.freeze_support()

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
_SRC_DIR        = Path(__file__).parent
MODEL_SAVE_PATH = _SRC_DIR / ".." / "models" / "chart_marker_net_v2.pth"
SYNTH_DIR       = _SRC_DIR / ".." / "data" / "synthetic_plots"

# Subimage storage on D:\ to avoid filling C:\
SUBIMG_DIR      = Path(r"D:\chartocode_subimages")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
N_PLOTS         = 500
MARKER_PT       = 8
DPI             = 100
PLOT_W_IN       = 5.6
PLOT_H_IN       = 4.2
PLOT_W_PX       = int(PLOT_W_IN * DPI)   # 560
PLOT_H_PX       = int(PLOT_H_IN * DPI)   # 420
N_POINTS        = 12

# P and HALF are computed at module import time so DataLoader worker
# subprocesses (Windows spawn) always have valid values.
def _compute_p_at_import() -> tuple[int, int]:
    """Measure symbol diameter and return (P, HALF). Runs at import time."""
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import cv2 as _cv2, numpy as _np, math as _math
    fig, ax = _plt.subplots(figsize=(1.0, 1.0), dpi=DPI)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.plot(0.5, 0.5, marker='o', markersize=MARKER_PT,
            markerfacecolor='black', markeredgecolor='black', linestyle='none')
    fig.canvas.draw()
    buf = _np.frombuffer(fig.canvas.buffer_rgba(), dtype=_np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    _plt.close(fig)
    gray = _cv2.cvtColor(buf, _cv2.COLOR_RGBA2GRAY)
    _, bw = _cv2.threshold(gray, 200, 255, _cv2.THRESH_BINARY_INV)
    coords = _np.argwhere(bw > 0)
    diam = int(max(coords.max(axis=0) - coords.min(axis=0)) + 1) if len(coords) else MARKER_PT
    p    = int(_math.ceil(diam * 1.20)) | 1
    return p, p // 2

P, HALF = _compute_p_at_import()


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS BAR HELPER  (no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def _pbar(done: int, total: int, t0: float, width: int = 40,
          prefix: str = "") -> None:
    """
    Print an in-place progress bar to stdout.
    Example:  [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░]  512/2100  24%  ETA 3m12s
    """
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
        print()   # newline when complete


VIT_INPUT       = 64
BATCH_SIZE      = 512           # larger batch — data is now cheap to load
EPOCHS          = 40
LR              = 3e-4
USE_COMPILE     = False         # disabled — hangs on some Windows/GPU combos
CONF_THRESH     = 0.65
STRIDE          = 2
NMS_RADIUS_FACTOR = 1.5
UNKNOWN_THRESH  = 0.40
MIN_DARK_FRAC   = 0.03
WORKERS         = min(8, mp.cpu_count())

# 11 symbol classes + background
CLASS_NAMES = [
    "filled_circle",        # 0
    "open_circle",          # 1
    "filled_square",        # 2
    "open_square",          # 3
    "open_triangle",        # 4
    "open_inv_triangle",    # 5
    "filled_triangle",      # 6
    "filled_inv_triangle",  # 7
    "open_rhombus",         # 8
    "filled_rhombus",       # 9
    "x_marker",             # 10
    "background",           # 11
]
N_CLASSES  = len(CLASS_NAMES)
N_SYMBOLS  = N_CLASSES - 1   # 11

_MPL_MARKERS = [
    ('o', True),    # 0  filled_circle
    ('o', False),   # 1  open_circle
    ('s', True),    # 2  filled_square
    ('s', False),   # 3  open_square
    ('^', False),   # 4  open_triangle
    ('v', False),   # 5  open_inv_triangle
    ('^', True),    # 6  filled_triangle
    ('v', True),    # 7  filled_inv_triangle
    ('D', False),   # 8  open_rhombus
    ('D', True),    # 9  filled_rhombus
    ('x', True),    # 10 x_marker
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  HILL EQUATION
# ══════════════════════════════════════════════════════════════════════════════

def hill(x: np.ndarray, bottom: float, top: float,
         ec50: float, n: float) -> np.ndarray:
    return bottom + (top - bottom) / (1.0 + (ec50 / x) ** n)


def make_series_x(n_pts: int, log_min: float = -15.0,
                  log_max: float = -7.0) -> np.ndarray:
    return np.logspace(log_min, log_max, n_pts)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC PLOT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_one_plot(args_tuple):
    """Worker: generate one synthetic concentration-efficacy plot."""
    idx, out_dir, seed = args_tuple
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    LOG_MIN, LOG_MAX = -15.0, -7.0
    Y_MIN,   Y_MAX   =  0.0,   1.05
    X_MARGIN_FRAC = 0.04
    Y_MARGIN_FRAC = 0.04
    LOG_RANGE = LOG_MAX - LOG_MIN
    Y_RANGE   = Y_MAX - Y_MIN
    LOG_MIN_PLOT = LOG_MIN - LOG_RANGE * X_MARGIN_FRAC
    LOG_MAX_PLOT = LOG_MAX + LOG_RANGE * X_MARGIN_FRAC
    Y_MIN_PLOT   = Y_MIN   - Y_RANGE   * Y_MARGIN_FRAC
    Y_MAX_PLOT   = Y_MAX   + Y_RANGE   * Y_MARGIN_FRAC

    series_data = []
    for si in range(N_SYMBOLS):
        bottom = np_rng.uniform(0.05, 0.15)
        top    = np_rng.uniform(0.80, 1.00)
        ec50   = 10 ** np_rng.uniform(-12, -8)
        n_hill = np_rng.uniform(0.8, 2.5)
        x_vals = make_series_x(N_POINTS, LOG_MIN, LOG_MAX)
        y_vals = hill(x_vals, bottom, top, ec50, n_hill)
        y_vals += np_rng.normal(0, 0.01, N_POINTS)
        y_vals  = np.clip(y_vals, Y_MIN + 0.02, Y_MAX - 0.02)
        series_data.append((x_vals, y_vals))

    z_order = list(range(N_SYMBOLS))
    rng.shuffle(z_order)

    fig, ax = plt.subplots(figsize=(PLOT_W_IN, PLOT_H_IN), dpi=DPI)
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.patch.set_alpha(0.0)

    for si in z_order:
        x_vals, y_vals = series_data[si]
        ax.plot(x_vals, y_vals, color='black', linewidth=0.8,
                marker='none', zorder=si)

    for z, si in enumerate(z_order):
        x_vals, y_vals = series_data[si]
        mcode, filled  = _MPL_MARKERS[si]
        fc = 'black' if filled else 'white'
        ax.plot(x_vals, y_vals,
                color='black',
                marker=mcode,
                markersize=MARKER_PT,
                markerfacecolor=fc,
                markeredgecolor='black',
                markeredgewidth=0.8,
                linestyle='none',
                zorder=N_SYMBOLS + z)

    ax.set_xscale('log')
    ax.set_xlim(10**LOG_MIN_PLOT, 10**LOG_MAX_PLOT)
    ax.set_ylim(Y_MIN_PLOT, Y_MAX_PLOT)
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    H_px, W_px = img_bgr.shape[:2]

    gt_points = []
    for si in range(N_SYMBOLS):
        x_vals, y_vals = series_data[si]
        for xi, yi in zip(x_vals, y_vals):
            disp = ax.transData.transform((xi, yi))
            px = int(round(disp[0]))
            py = int(round(H_px - disp[1]))
            px = max(0, min(W_px - 1, px))
            py = max(0, min(H_px - 1, py))
            gt_points.append({
                "cx": px, "cy": py,
                "class_idx": si,
                "class_name": CLASS_NAMES[si]
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
            "pa": {"x0": pa_x0, "y0": pa_y0, "x1": pa_x1, "y1": pa_y1},
            "points": gt_points
        }, f)
    return str(img_path), str(gt_path)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBIMAGE EXTRACTION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def extract_patch_padded(gray: np.ndarray, cx: int, cy: int,
                         p: int = None) -> np.ndarray:
    """Extract p×p patch centred at (cx, cy), padding with 255 at borders."""
    if p is None: p = P
    half = p // 2
    H, W = gray.shape
    x0 = cx - half; y0 = cy - half
    x1 = x0 + p;    y1 = y0 + p
    sx0 = max(x0, 0); sy0 = max(y0, 0)
    sx1 = min(x1, W); sy1 = min(y1, H)
    dx0 = sx0 - x0; dy0 = sy0 - y0
    dx1 = dx0 + (sx1 - sx0); dy1 = dy0 + (sy1 - sy0)
    patch = np.full((p, p), 255, dtype=np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        patch[dy0:dy1, dx0:dx1] = gray[sy0:sy1, sx0:sx1]
    return patch


def patch_to_tensor(patch: np.ndarray) -> np.ndarray:
    """Convert grayscale patch to normalised 3-channel float32 CHW."""
    r = cv2.resize(patch, (VIT_INPUT, VIT_INPUT), interpolation=cv2.INTER_LINEAR)
    t = np.stack([r, r, r], axis=0).astype(np.float32) / 255.0
    return (t - _MEAN) / _STD


def patch_to_uint8(patch: np.ndarray) -> np.ndarray:
    """Resize patch to VIT_INPUT×VIT_INPUT and return as uint8 HWC.

    Storing uint8 instead of float32 reduces disk usage 4×.
    Normalisation is applied at batch-load time in uint8_batch_to_tensor().
    """
    r = cv2.resize(patch, (VIT_INPUT, VIT_INPUT), interpolation=cv2.INTER_LINEAR)
    return np.stack([r, r, r], axis=2)  # HWC uint8


def uint8_batch_to_tensor(batch: np.ndarray, device) -> torch.Tensor:
    """Convert (N, H, W, 3) uint8 numpy array to normalised float32 NCHW tensor."""
    # batch: (N, H, W, 3) uint8  →  (N, 3, H, W) float32 normalised
    t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cpu').view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device='cpu').view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.to(device, non_blocking=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBIMAGE PRE-GENERATION  (write once to D:\, load cheaply every epoch)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_subimages_worker(args_tuple):
    """
    Worker: extract all subimage patches from one plot and write them as
    a chunk .npy file directly to disk.  Returns (chunk_t_path, chunk_l_path)
    so the main process only receives two short strings through the IPC pipe
    — avoiding the MemoryError caused by sending large arrays over the pipe.
    """
    gt_path_str, seed, chunk_dir_str = args_tuple
    rng  = random.Random(seed)
    rng2 = random.Random(seed + 99999)

    gt_path  = Path(gt_path_str)
    img_path = gt_path.parent / gt_path.name.replace("gt_", "plot_").replace(".json", ".png")
    chunk_dir = Path(chunk_dir_str)

    if not img_path.exists():
        return None, None

    with open(gt_path) as f:
        gt = json.load(f)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pa   = gt["pa"]
    pts  = gt["points"]
    sym_centers = [(p["cx"], p["cy"]) for p in pts]

    tensors_list = []
    labels_list  = []

    for pt in pts:
        cx, cy, ci = pt["cx"], pt["cy"], pt["class_idx"]
        # Case 1 — 2 centred samples
        for _ in range(2):
            ox = rng.randint(-2, 2)
            oy = rng.randint(-2, 2)
            patch = extract_patch_padded(gray, cx + ox, cy + oy)
            tensors_list.append(patch_to_uint8(patch))
            labels_list.append(ci)
        # Case 2 — 1 slightly-off sample (background label)
        angle = rng.uniform(0, 2 * math.pi)
        dist  = rng.uniform(3, 5)
        ox = int(round(dist * math.cos(angle)))
        oy = int(round(dist * math.sin(angle)))
        patch = extract_patch_padded(gray, cx + ox, cy + oy)
        tensors_list.append(patch_to_uint8(patch))
        labels_list.append(N_SYMBOLS)

    # Case 3 — random background patches
    target_bg = len(pts) * 3
    added = attempts = 0
    while added < target_bg and attempts < target_bg * 20:
        attempts += 1
        bx = rng2.randint(pa["x0"], pa["x1"])
        by = rng2.randint(pa["y0"], pa["y1"])
        if any(math.sqrt((bx - sx)**2 + (by - sy)**2) < HALF + 2
               for sx, sy in sym_centers):
            continue
        patch = extract_patch_padded(gray, bx, by)
        tensors_list.append(patch_to_uint8(patch))
        labels_list.append(N_SYMBOLS)
        added += 1

    # Write chunk directly to disk — avoids sending large arrays over IPC pipe.
    # Stored as uint8 HWC (not float32 CHW) to reduce disk usage 4×.
    stem = gt_path.stem  # e.g. "gt_00042"
    ct_path = chunk_dir / f"{stem}_t.npy"
    cl_path = chunk_dir / f"{stem}_l.npy"
    np.save(str(ct_path), np.stack(tensors_list, axis=0).astype(np.uint8))
    np.save(str(cl_path), np.array(labels_list, dtype=np.int32))
    return str(ct_path), str(cl_path)


def build_subimage_dataset(synth_dir: Path, subimg_dir: Path,
                           n_plots: int = N_PLOTS,
                           n_workers: int = WORKERS,
                           force_rebuild: bool = False) -> None:
    """
    Pre-generate subimage patches for the first n_plots plots and save to
    subimg_dir as:
      tensors.npy  — shape (N, H, W, 3) uint8
      labels.npy   — shape (N,) int32

    Only the first n_plots GT files (sorted by name) are used, so setting
    N_PLOTS=500 will extract from exactly 500 plots even if more exist on disk.
    Workers write per-plot chunk files directly to a temp subfolder on disk
    (avoiding IPC MemoryError), then the main process merges them into the
    final tensors.npy / labels.npy and deletes the chunks.
    """
    subimg_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = subimg_dir / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    t_path = subimg_dir / "tensors.npy"
    l_path = subimg_dir / "labels.npy"

    all_gt_files = sorted(synth_dir.glob("gt_*.json"))
    if not all_gt_files:
        raise FileNotFoundError(f"No GT JSON files found in {synth_dir}")

    # Limit to first n_plots so N_PLOTS controls dataset size regardless
    # of how many plots already exist on disk.
    gt_files = all_gt_files[:n_plots]
    print(f"  Extracting subimages from {len(gt_files)}/{len(all_gt_files)} plots "
          f"using {n_workers} workers ...")
    t0    = time.time()
    seeds = [random.randint(0, 2**31) for _ in gt_files]
    args  = [(str(gf), s, str(chunk_dir)) for gf, s in zip(gt_files, seeds)]

    chunk_t_paths: list[str] = []
    chunk_l_paths: list[str] = []

    n_plots_total = len(args)
    done_plots    = 0

    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            for ct, cl in pool.imap_unordered(
                    _extract_subimages_worker, args,
                    chunksize=max(1, len(args) // (n_workers * 4))):
                if ct is not None:
                    chunk_t_paths.append(ct)
                    chunk_l_paths.append(cl)
                done_plots += 1
                _pbar(done_plots, n_plots_total, t0, prefix="Extracting: ")
    else:
        for a in args:
            ct, cl = _extract_subimages_worker(a)
            if ct is not None:
                chunk_t_paths.append(ct)
                chunk_l_paths.append(cl)
            done_plots += 1
            _pbar(done_plots, n_plots_total, t0, prefix="Extracting: ")

    # ── streaming merge into pre-allocated memory-mapped files ──────────────
    # This avoids loading all chunks into RAM at once (prevents OOM).
    sorted_t = sorted(chunk_t_paths)
    sorted_l = sorted(chunk_l_paths)

    # count total samples without loading data
    n_total = sum(np.load(p, mmap_mode='r').shape[0] for p in sorted_t)
    print(f"  Merging {len(sorted_t)} chunk files ({n_total:,} samples) ...")

    # create output mmap arrays
    # uint8 HWC layout: 4× smaller than float32 CHW (20 GB vs 81 GB)
    t_mmap = np.lib.format.open_memmap(
        str(t_path), mode='w+', dtype=np.uint8,
        shape=(n_total, VIT_INPUT, VIT_INPUT, 3))
    l_mmap = np.lib.format.open_memmap(
        str(l_path), mode='w+', dtype=np.int32,
        shape=(n_total,))

    # write chunks sequentially with live progress
    t_merge = time.time()
    offset  = 0
    for mi, (tp, lp) in enumerate(zip(sorted_t, sorted_l), 1):
        tc = np.load(tp)
        lc = np.load(lp)
        n  = len(lc)
        t_mmap[offset:offset+n] = tc
        l_mmap[offset:offset+n] = lc
        offset += n
        del tc, lc
        _pbar(mi, len(sorted_t), t_merge, prefix="Merging:    ")

    del t_mmap, l_mmap  # flush to disk

    # Save a shuffled index file instead of shuffling the data in-place.
    # Shuffling a multi-GB mmap file requires thousands of random disk seeks
    # and is extremely slow.  Instead we save a permuted index array (tiny,
    # ~2 MB) and apply it in NpyDataset / DataLoader at load time.
    idx = np.random.permutation(n_total).astype(np.int32)
    np.save(str(subimg_dir / "shuffle_idx.npy"), idx)

    # clean up chunk files
    for p in chunk_t_paths + chunk_l_paths:
        try: Path(p).unlink()
        except Exception: pass
    try: chunk_dir.rmdir()
    except Exception: pass

    elapsed = time.time() - t0
    size_gb = (t_path.stat().st_size + l_path.stat().st_size) / 1e9
    print(f"  Done in {elapsed:.1f}s  ({size_gb:.2f} GB written)")


# ══════════════════════════════════════════════════════════════════════════════
#  FAST DISK-BACKED DATASET  (memory-mapped .npy files)
# ══════════════════════════════════════════════════════════════════════════════

# How many samples to load into RAM at once during training.
# Each uint8 sample = 64*64*3 = 12,288 bytes.
# With 500 plots the full train set is ~4.1 GB — fits in one chunk.
# Increase this if you scale up N_PLOTS later.
CHUNK_SAMPLES = 400_000  # covers the full 500-plot train set in one pass


def _sorted_mmap_read(mmap_arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """
    Read rows from a memory-mapped array using SORTED indices, then restore
    the original order.  Sorting turns random disk seeks into sequential reads,
    which is orders of magnitude faster on both HDDs and SSDs.
    """
    sort_order  = np.argsort(idx)          # positions that sort idx ascending
    unsort_order = np.argsort(sort_order)  # inverse permutation
    data = mmap_arr[idx[sort_order]].copy()  # sequential read
    return data[unsort_order]              # restore original order


def _load_val_to_ram(tensors_path: str, labels_path: str,
                     va_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load only the validation subset into RAM.
    Uses sorted index reads to avoid random disk seeks on the 20 GB mmap file.
    Returns (va_x_uint8, va_y) where va_x_uint8 is (N,H,W,3) uint8.
    """
    print("  Loading val set into RAM ...", end="", flush=True)
    t0  = time.time()
    t_m = np.load(tensors_path, mmap_mode='r')  # (N, H, W, 3) uint8
    l_m = np.load(labels_path,  mmap_mode='r')  # (N,) int32
    va_t = torch.from_numpy(_sorted_mmap_read(t_m, va_idx))
    va_l = torch.from_numpy(_sorted_mmap_read(l_m, va_idx).astype(np.int64))
    del t_m, l_m
    print(f" done in {time.time()-t0:.1f}s  "
          f"({va_t.nbytes/1e9:.2f} GB)")
    return va_t, va_l


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(n_classes: int = N_CLASSES) -> nn.Module:
    model = timm.create_model(
        'vit_tiny_patch16_224', pretrained=True,
        num_classes=n_classes, img_size=VIT_INPUT
    )
    # freeze all blocks except the last 4
    for blk in list(model.blocks)[:-4]:
        for p in blk.parameters():
            p.requires_grad = False
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(n_plots: int = N_PLOTS):
    print(f"  Symbol diameter: {P} px  →  p = {P} px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── STEP 1: generate synthetic plots ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Generating synthetic plots")
    print("="*60)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(SYNTH_DIR.glob("plot_*.png")))
    if existing >= n_plots:
        print(f"  {existing} plots already exist — skipping generation.")
    else:
        seeds  = [random.randint(0, 2**31) for _ in range(n_plots)]
        args   = [(i, str(SYNTH_DIR), seeds[i]) for i in range(n_plots)]
        n_cpu  = max(1, WORKERS)
        print(f"  Generating {n_plots} plots using {n_cpu} CPU workers...")
        t0 = time.time()
        with mp.Pool(n_cpu) as pool:
            results = pool.map(generate_one_plot, args)
        print(f"  Done in {time.time()-t0:.1f}s — {len(results)} plots saved.")

    # ── STEP 2: pre-generate subimage patches (once) ──────────────────────────
    print("\n" + "="*60)
    print("STEP 2 — Pre-generating subimage patches → D:\\chartocode_subimages\\")
    print("="*60)

    # Check if we need to rebuild (e.g. number of plots changed)
    t_path = SUBIMG_DIR / "tensors.npy"
    l_path = SUBIMG_DIR / "labels.npy"
    force  = False
    if t_path.exists() and l_path.exists():
        # Sanity check: saved count must be close to what n_plots would produce.
        # Trigger rebuild if too few (incomplete) OR too many (n_plots was reduced).
        try:
            n_saved = np.load(str(l_path), mmap_mode='r').shape[0]
            expected_min = n_plots * N_POINTS * N_SYMBOLS * 2
            expected_max = n_plots * N_POINTS * N_SYMBOLS * 6 + n_plots * 100
            if n_saved < expected_min * 0.9:
                print(f"  Saved count {n_saved:,} < expected min {expected_min:,} — rebuilding.")
                force = True
            elif n_saved > expected_max * 1.1:
                print(f"  Saved count {n_saved:,} > expected max {expected_max:,} — "
                      f"N_PLOTS was reduced, rebuilding.")
                force = True
            else:
                print(f"  Found {n_saved:,} saved samples — skipping extraction.")
        except Exception:
            force = True

    if force or not (t_path.exists() and l_path.exists()):
        build_subimage_dataset(
            SYNTH_DIR, SUBIMG_DIR, n_plots=n_plots,
            n_workers=WORKERS, force_rebuild=True
        )

    n_total = np.load(str(l_path), mmap_mode='r').shape[0]
    print(f"  Total samples: {n_total:,}")

    # train / val split
    all_idx  = np.arange(n_total)
    val_size = max(1, int(n_total * 0.15))
    tr_idx   = all_idx[val_size:]
    va_idx   = all_idx[:val_size]
    print(f"  Train: {len(tr_idx):,}  |  Val: {len(va_idx):,}")

    # ── STEP 3: train ViT ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Training ViT  (GPU-optimised)")
    print("="*60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    sidx_path   = SUBIMG_DIR / "shuffle_idx.npy"
    shuffle_idx = np.load(str(sidx_path)) if sidx_path.exists() else None

    # Build the full shuffled index, then split into train / val
    base = shuffle_idx if shuffle_idx is not None else np.arange(n_total, dtype=np.int32)
    tr_full_idx = base[tr_idx]   # shuffled train indices into tensors.npy
    va_full_idx = base[va_idx]   # shuffled val   indices into tensors.npy

    # ── Load val set into RAM once (small: ~15% of 20 GB = ~3 GB uint8) ────────
    t_size_gb = t_path.stat().st_size / 1e9
    print(f"  tensors.npy: {t_size_gb:.1f} GB on disk  "
          f"(uint8, {n_total:,} samples)")
    va_t_u8, va_l = _load_val_to_ram(str(t_path), str(l_path), va_full_idx)
    va_ds = TensorDataset(va_t_u8, va_l)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Build chunk boundaries for training set ────────────────────────────────
    # Training data is too large to fit in RAM all at once.
    # We split tr_full_idx into chunks of CHUNK_SAMPLES, load each chunk
    # sequentially, train on it, then release it before loading the next.
    n_tr = len(tr_full_idx)
    chunk_starts = list(range(0, n_tr, CHUNK_SAMPLES))
    n_chunks     = len(chunk_starts)
    print(f"  Train: {n_tr:,} samples split into {n_chunks} chunks "
          f"of ~{CHUNK_SAMPLES:,} each  "
          f"(~{CHUNK_SAMPLES*VIT_INPUT*VIT_INPUT*3/1e9:.1f} GB/chunk)")

    model = build_model().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    if USE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        print("  Compiling model with torch.compile() ...")
        model = torch.compile(model)
    else:
        print("  torch.compile() skipped.")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_start = time.time()

    # Open mmap handles once (read-only, shared across chunks)
    t_mmap = np.load(str(t_path), mmap_mode='r')  # (N, H, W, 3) uint8
    l_mmap = np.load(str(l_path), mmap_mode='r')  # (N,) int32

    n_va_batches = len(va_ld)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        epoch_t0 = time.time()

        # Re-shuffle chunk order each epoch so model sees data in different order
        chunk_order = list(range(n_chunks))
        random.shuffle(chunk_order)

        # Count total batches across all chunks for progress bar
        n_tr_batches = math.ceil(n_tr / BATCH_SIZE)
        batch_global = 0

        for ci in chunk_order:
            cs = chunk_starts[ci]
            ce = min(cs + CHUNK_SAMPLES, n_tr)
            chunk_idx = tr_full_idx[cs:ce]

            # Load this chunk using sorted indices → sequential disk reads (fast)
            chunk_t = torch.from_numpy(_sorted_mmap_read(t_mmap, chunk_idx))   # (N,H,W,3) uint8
            chunk_l = torch.from_numpy(_sorted_mmap_read(l_mmap, chunk_idx).astype(np.int64))
            chunk_ds = TensorDataset(chunk_t, chunk_l)
            chunk_ld = DataLoader(chunk_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0,
                                  pin_memory=(device.type == "cuda"))

            for xb_u8, yb in chunk_ld:
                # Convert uint8 HWC → float32 NCHW normalised on-the-fly
                xb = uint8_batch_to_tensor(xb_u8.numpy(), device)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda")):
                    out  = model(xb)
                    loss = criterion(out, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss    += loss.item() * len(yb)
                tr_correct += (out.argmax(1) == yb).sum().item()
                tr_total   += len(yb)
                batch_global += 1
                _pbar(batch_global, n_tr_batches, epoch_t0,
                      prefix=f"Epoch {epoch:3d}/{EPOCHS} train: ")

            del chunk_t, chunk_l, chunk_ds, chunk_ld  # free RAM before next chunk

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        va_correct = va_total = 0
        val_t0 = time.time()
        with torch.no_grad():
            for bi, (xb_u8, yb) in enumerate(va_ld, 1):
                xb = uint8_batch_to_tensor(xb_u8.numpy(), device)
                yb = yb.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    out = model(xb)
                va_correct += (out.argmax(1) == yb).sum().item()
                va_total   += len(yb)
                _pbar(bi, n_va_batches, val_t0,
                      prefix=f"Epoch {epoch:3d}/{EPOCHS} val:   ")
        va_acc = va_correct / va_total

        if va_acc > best_acc:
            best_acc = va_acc
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(save_model.state_dict(), str(MODEL_SAVE_PATH))

        epoch_elapsed = time.time() - epoch_t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"loss={tr_loss/tr_total:.4f} | "
              f"val_acc={va_acc:.4f} | best={best_acc:.4f} | "
              f"{epoch_elapsed:.0f}s")

    del t_mmap, l_mmap
    train_elapsed = time.time() - train_start
    mins, secs = divmod(int(train_elapsed), 60)
    hrs,  mins = divmod(mins, 60)
    time_str = (f"{hrs}h {mins:02d}m {secs:02d}s" if hrs
                else f"{mins}m {secs:02d}s" if mins
                else f"{secs}s")
    print(f"\n  Training complete. Best val acc: {best_acc:.4f}")
    print(f"  Total training time : {time_str}  ({train_elapsed:.1f}s)")
    print(f"  Model saved → {MODEL_SAVE_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(model_path: str | Path) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model()
    m.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    m.eval().to(device)
    return m, device


def _per_class_nms(dets: list[dict], radius: float) -> list[dict]:
    by_class = defaultdict(list)
    for d in dets: by_class[d["class_idx"]].append(d)
    kept = []
    for ci, cls_dets in by_class.items():
        cls_dets = sorted(cls_dets, key=lambda x: -x["confidence"])
        suppressed = set()
        for i, d in enumerate(cls_dets):
            if i in suppressed: continue
            kept.append(d)
            for j in range(i+1, len(cls_dets)):
                if j in suppressed: continue
                dist = math.sqrt((d["cx"]-cls_dets[j]["cx"])**2 +
                                  (d["cy"]-cls_dets[j]["cy"])**2)
                if dist < radius: suppressed.add(j)
    return kept


def _estimate_center_in_patch(window_cx: int, window_cy: int,
                               patch_gray: np.ndarray, p: int) -> tuple[int, int]:
    _, bw = cv2.threshold(patch_gray, 180, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return window_cx, window_cy
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    est_cx = int(round(centroids[best][0])) - p // 2
    est_cy = int(round(centroids[best][1])) - p // 2
    return window_cx + est_cx, window_cy + est_cy


def detect(image_path: str,
           model_path: str | Path = MODEL_SAVE_PATH,
           conf_thresh: float = CONF_THRESH,
           stride: int = STRIDE,
           unknown_thresh: float = UNKNOWN_THRESH,
           min_dark_frac: float = MIN_DARK_FRAC,
           p: int = None) -> list[dict]:
    """
    Detect markers in a plotting-area image.

    Parameters
    ----------
    image_path    : path to the plotting-area image (entire image = plotting area)
    model_path    : path to trained weights
    conf_thresh   : minimum confidence to keep a detection
    stride        : sliding window stride (px)
    unknown_thresh: if max class probability < this, label as "unknown"
    min_dark_frac : minimum fraction of p×p pixels that must be dark
    p             : window size (px); if None, uses module-level P

    Returns
    -------
    List of dicts, each with:
      cx, cy       — estimated symbol centre in image coordinates
      class_idx    — 0-10 (symbol) or -1 (unknown)
      class_name   — e.g. "filled_circle" or "unknown"
      confidence   — softmax probability of the predicted class
    """
    if p is None:
        p = P
    half = p // 2
    nms_radius = p * NMS_RADIUS_FACTOR

    model, device = _load_model(model_path)

    img_bgr  = cv2.imread(str(image_path))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W     = img_gray.shape

    min_dark_pixels = int(p * p * min_dark_frac)
    raw_dets: list[dict] = []

    batch_coords: list[tuple[int,int]] = []
    batch_tensors: list[np.ndarray]    = []

    def flush_batch():
        if not batch_tensors: return
        t = torch.tensor(np.stack(batch_tensors), dtype=torch.float32).to(device)
        with torch.no_grad():
            with autocast(enabled=(device.type == "cuda")):
                probs = torch.softmax(model(t), dim=1).cpu().numpy()
        for (bx, by), prob in zip(batch_coords, probs):
            max_prob = float(prob.max())
            ci       = int(prob.argmax())
            if ci == N_SYMBOLS:
                pass
            elif max_prob < unknown_thresh:
                patch = extract_patch_padded(img_gray, bx, by, p)
                ecx, ecy = _estimate_center_in_patch(bx, by, patch, p)
                raw_dets.append({
                    "cx": ecx, "cy": ecy,
                    "class_idx": -1, "class_name": "unknown",
                    "confidence": round(max_prob, 4)
                })
            elif max_prob >= conf_thresh:
                patch = extract_patch_padded(img_gray, bx, by, p)
                ecx, ecy = _estimate_center_in_patch(bx, by, patch, p)
                raw_dets.append({
                    "cx": ecx, "cy": ecy,
                    "class_idx": ci, "class_name": CLASS_NAMES[ci],
                    "confidence": round(max_prob, 4)
                })
        batch_coords.clear(); batch_tensors.clear()

    for cy_w in range(0, H, stride):
        for cx_w in range(0, W, stride):
            patch = extract_patch_padded(img_gray, cx_w, cy_w, p)
            _, bw = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
            if np.count_nonzero(bw) < min_dark_pixels: continue
            batch_coords.append((cx_w, cy_w))
            batch_tensors.append(patch_to_tensor(patch))
            if len(batch_tensors) == 512: flush_batch()
    flush_batch()

    symbol_dets  = [d for d in raw_dets if d["class_idx"] >= 0]
    unknown_dets = [d for d in raw_dets if d["class_idx"] == -1]

    kept         = _per_class_nms(symbol_dets, nms_radius)
    unknown_kept = _per_class_nms(
        [{**d, "class_idx": 999} for d in unknown_dets], nms_radius
    )
    for d in unknown_kept: d["class_idx"] = -1

    results = sorted(kept + unknown_kept,
                     key=lambda d: (d["class_idx"], d["cy"], d["cx"]))

    found_classes = set(d["class_name"] for d in results if d["class_idx"] >= 0)
    print(f"  Detected {len(results)} markers across {len(found_classes)} symbol type(s):")
    for cn in sorted(found_classes):
        n = sum(1 for d in results if d["class_name"] == cn)
        print(f"    {cn}: {n}")
    n_unk = sum(1 for d in results if d["class_idx"] == -1)
    if n_unk: print(f"    unknown: {n_unk}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

_CLASS_COLORS = {
    "filled_circle":       (0,   0, 220),
    "open_circle":         (0, 140, 255),
    "filled_square":       (0, 180,   0),
    "open_square":         (180, 200,  0),
    "open_triangle":       (0, 200, 180),
    "open_inv_triangle":   (0, 160, 160),
    "filled_triangle":     (200, 100,  0),
    "filled_inv_triangle": (180,  60,  0),
    "open_rhombus":        (180,   0, 180),
    "filled_rhombus":      (140,   0, 140),
    "x_marker":            (0,   0,   0),
    "unknown":             (128, 128, 128),
}


def visualise(image_path: str, detections: list[dict],
              out_path: str | None = None) -> np.ndarray:
    img  = cv2.imread(str(image_path))
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    for d in detections:
        cx, cy = d["cx"], d["cy"]
        cn     = d["class_name"]
        conf   = d["confidence"]
        color  = _CLASS_COLORS.get(cn, (100, 100, 100))
        r      = HALF + 2
        cv2.circle(img, (cx, cy), r, color, 1)
        cv2.putText(img, f"{conf:.2f}", (cx + r + 1, cy + 4),
                    FONT, 0.25, color, 1, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(str(out_path), img)
        print(f"  Visualisation saved → {out_path}")
    return img


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chart marker detector (ViT)")
    parser.add_argument("--mode",   choices=["train", "detect"], required=True)
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to plotting-area image (detect mode)")
    parser.add_argument("--model",  type=str, default=str(MODEL_SAVE_PATH),
                        help="Path to model weights")
    parser.add_argument("--plots",  type=int, default=N_PLOTS,
                        help="Number of synthetic plots to generate")
    parser.add_argument("--conf",   type=float, default=CONF_THRESH)
    parser.add_argument("--stride", type=int,   default=STRIDE)
    parser.add_argument("--out",    type=str,   default=None,
                        help="Output path for detection visualisation")
    args = parser.parse_args()

    if args.mode == "train":
        train(n_plots=args.plots)

    elif args.mode == "detect":
        if not args.image:
            parser.error("--image is required for detect mode")
        dets = detect(
            image_path  = args.image,
            model_path  = args.model,
            conf_thresh = args.conf,
            stride      = args.stride,
        )
        out_vis  = args.out or str(Path(args.image).with_suffix("")) + "_detections.png"
        visualise(args.image, dets, out_path=out_vis)
        out_json = str(Path(args.image).with_suffix("")) + "_markers.json"
        with open(out_json, "w") as f:
            json.dump({"n_detections": len(dets), "detections": dets}, f, indent=2)
        print(f"  JSON saved → {out_json}")
