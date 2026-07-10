"""
chart_marker_detector_v3.py
============================
Working directory : C:\\Users\\ziola\\OneDrive\\Documents\\GitHub\\chartocode\\src
Model save path   : ../models/chart_marker_net_v3.pth   (relative to src/)

Subimage storage  : <project>/data/subimages/
  All subimage patches are pre-generated ONCE and saved as a single
  memory-mapped NumPy file alongside the project to avoid slow
  on-the-fly rendering during training.  Each epoch is ~10-20× faster.

USAGE
-----
  # Train (generates 2000+ synthetic plots, extracts subimages, trains ViT)
  python chart_marker_detector_v3.py --mode train

  # Detect markers in a plotting-area image
  python chart_marker_detector_v3.py --mode detect --image path/to/plotting_area.png

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
MODEL_SAVE_PATH = _SRC_DIR / ".." / "models" / "chart_marker_net_v3.pth"
SYNTH_DIR       = _SRC_DIR / ".." / "data" / "synthetic_plots"

# Subimage storage alongside the project (same drive as source)
SUBIMG_DIR      = _SRC_DIR / ".." / "data" / "subimages"

# Per-epoch log storage alongside the project
# <project>/data/epoch_logs/
#   epoch_001/
#     train_subimages/   ← sampled training subimage PNGs (class_name_NNNNN.png)
#     val_detections/    ← annotated validation plot PNGs with colour legend
EPOCH_LOG_DIR   = _SRC_DIR / ".." / "data" / "epoch_logs"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
N_PLOTS         = 1000
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
    # Account for max size variation (1.20) + margin (1.20)
    p    = int(_math.ceil(diam * 1.20 * 1.20)) | 1
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
EPOCHS          = 100
LR              = 3e-4
USE_COMPILE     = False         # disabled — hangs on some Windows/GPU combos
CONF_THRESH     = 0.65
STRIDE          = 2
NMS_RADIUS_FACTOR = 2.5    # ≈ 47 px at P=19; suppress same-class FP within ~2.5 marker widths
XCOL_NMS_WIDTH_FACTOR = 2.5  # x-column bin width = P * factor ≈ 47px; ~1 marker spacing wide
UNKNOWN_THRESH  = 0.40
MIN_DARK_FRAC   = 0.03
WORKERS         = min(8, mp.cpu_count())

# 12 symbol classes + background
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
    "plus_marker",          # 11
    "background",           # 12
]
N_CLASSES  = len(CLASS_NAMES)   # 13
N_SYMBOLS  = N_CLASSES - 1      # 12 (background is the last class)

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
    ('+', True),    # 11 plus_marker
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION PIPELINE (NOISE & SYMBOL VARIATION)
# ══════════════════════════════════════════════════════════════════════════════

def add_paper_texture(img: np.ndarray, strength: float, np_rng) -> np.ndarray:
    grain = np_rng.normal(0, strength * 30, img.shape).astype(np.float32)
    out = img.astype(np.float32) + grain
    return np.clip(out, 0, 255).astype(np.uint8)

def add_yellowing(img: np.ndarray, strength: float) -> np.ndarray:
    out = img.astype(np.float32)
    out[:, :, 0] -= strength * 40   # blue down
    out[:, :, 1] -= strength * 10   # green slight down
    out[:, :, 2] += strength * 20   # red up
    out += strength * 15            # brighten slightly
    return np.clip(out, 0, 255).astype(np.uint8)

def add_blur(img: np.ndarray, strength: float) -> np.ndarray:
    k = max(1, int(strength * 3))
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), strength * 0.8)

def add_elastic_warp(img: np.ndarray, strength: float, np_rng) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    h, w = img.shape[:2]
    sigma = 20
    alpha = strength * 8
    dx = gaussian_filter(np_rng.random((h, w)).astype(np.float32) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(np_rng.random((h, w)).astype(np.float32) * 2 - 1, sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def add_slight_rotation(img: np.ndarray, max_deg: float, np_rng) -> np.ndarray:
    angle = np_rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_scan_noise(img: np.ndarray, level: str, np_rng) -> np.ndarray:
    if level == 'none':
        return img
    if level == 'light':
        img = add_paper_texture(img, 0.15, np_rng)
        img = add_blur(img, 0.4)
        return img
    if level == 'medium':
        img = add_yellowing(img, 0.4)
        img = add_paper_texture(img, 0.35, np_rng)
        img = add_blur(img, 0.8)
        img = add_elastic_warp(img, 0.5, np_rng)
        return img
    if level == 'heavy':
        img = add_yellowing(img, 0.8)
        img = add_paper_texture(img, 0.6, np_rng)
        img = add_blur(img, 1.2)
        img = add_elastic_warp(img, 1.0, np_rng)
        img = add_slight_rotation(img, 1.5, np_rng)
        return img
    return img

def sample_marker_size(base: float, variation: float, np_rng) -> float:
    factor = float(np_rng.uniform(1.0 - variation, 1.0 + variation))
    return base * factor

def sample_linewidth(base: float, variation: float, np_rng) -> float:
    factor = float(np_rng.uniform(1.0 - variation, 1.0 + variation))
    return max(0.3, base * factor)


def round_polygon_corners(verts: np.ndarray, r: float,
                          n_arc: int = 16) -> np.ndarray:
    """
    Return a dense polygon with worn, convex-rounded corners.

    Each sharp vertex is replaced by a circular arc that is tangent to both
    adjacent edges and bulges OUTWARD (convex), as if the corner tip has been
    worn smooth.  r=0 gives the original sharp polygon; r=0.5 is the maximum
    used in training data.
    """
    if r <= 0.0:
        return verts.copy()
    N = len(verts)
    pts = []
    for i in range(N):
        A = verts[(i - 1) % N]
        B = verts[i]
        C = verts[(i + 1) % N]
        BA = A - B;  len_BA = np.linalg.norm(BA)
        BC = C - B;  len_BC = np.linalg.norm(BC)
        if len_BA < 1e-9 or len_BC < 1e-9:
            pts.append(B); continue
        uBA = BA / len_BA
        uBC = BC / len_BC
        cos_half = np.clip(np.dot(uBA, uBC), -1.0, 1.0)
        half_angle = np.arccos(cos_half) / 2.0
        if half_angle < 1e-6:
            pts.append(B); continue
        tan_half = np.tan(half_angle)
        if tan_half < 1e-9:
            pts.append(B); continue
        max_t = min(len_BA, len_BC) * 0.45 * r
        r_arc = max_t / tan_half
        bisector = uBA + uBC
        bisector_len = np.linalg.norm(bisector)
        if bisector_len < 1e-9:
            pts.append(B); continue
        bisector = bisector / bisector_len
        centre = B + bisector * (r_arc / np.sin(half_angle))
        T1 = B + uBA * max_t
        T2 = B + uBC * max_t
        a1 = np.arctan2(T1[1] - centre[1], T1[0] - centre[0])
        a2 = np.arctan2(T2[1] - centre[1], T2[0] - centre[0])
        diff = (a2 - a1) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        arc_angles = np.linspace(a1, a1 + diff, n_arc)
        arc_pts = centre + r_arc * np.column_stack(
            [np.cos(arc_angles), np.sin(arc_angles)])
        pts.extend(arc_pts.tolist())
    return np.array(pts)


def _regular_polygon(n: int, angle_offset: float = 0.0) -> np.ndarray:
    """Return (n, 2) unit-radius regular polygon vertices."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([np.cos(angles), np.sin(angles)])


# Only squares and rhombuses go through draw_rounded_marker (corner rounding).
# Triangles (4,5,6,7), circles (0,1), x_marker (10), plus_marker (11) are
# all drawn by matplotlib directly with perfectly sharp native markers.
_POLY_VERTS = {
    2: lambda: _regular_polygon(4, angle_offset=np.pi / 4),   # filled_square
    3: lambda: _regular_polygon(4, angle_offset=np.pi / 4),   # open_square
    8: lambda: _regular_polygon(4, angle_offset=0.0),         # open_rhombus
    9: lambda: _regular_polygon(4, angle_offset=0.0),         # filled_rhombus
}

# Triangle class indices — always drawn via matplotlib (sharp native markers)
_TRIANGLE_IDXS = {4, 5, 6, 7}


def draw_rounded_marker(ax, x_data: float, y_data: float,
                        class_idx: int, filled: bool,
                        r: float, ms_pt: float, lw: float,
                        zorder: int) -> None:
    """
    Draw a single rounded-polygon marker at data coordinates (x_data, y_data).

    The polygon is scaled so that its bounding radius (max distance from
    centre to any point on the outline) always equals ms_pt/2 display pixels,
    regardless of the rounding level r.  This prevents rounded polygons from
    appearing smaller than their sharp counterparts.
    """
    verts = _POLY_VERTS[class_idx]()
    pts   = round_polygon_corners(verts, r)          # rounded in unit space

    # Normalise: rescale pts so that the maximum radius of the rounded shape
    # equals 1.0 (same as the original unit polygon whose vertices sit on the
    # unit circle).  Without this, rounding trims the corners inward and the
    # shape appears smaller.
    max_r_rounded = np.max(np.linalg.norm(pts, axis=1))
    if max_r_rounded > 1e-9:
        pts = pts / max_r_rounded   # now max radius == 1.0

    # Convert the data point to display (pixel) coordinates
    disp_centre = ax.transData.transform((x_data, y_data))

    # Scale: ms_pt is the marker diameter in points; 1 pt = 1/72 inch.
    # ax.get_figure().get_dpi() gives pixels-per-inch.
    pt_to_px = ax.get_figure().get_dpi() / 72.0
    radius_px = (ms_pt / 2.0) * pt_to_px

    # Scale normalised polygon to radius_px and shift to display centre
    disp_pts = pts * radius_px + disp_centre

    # Convert back to data coordinates
    data_pts = ax.transData.inverted().transform(disp_pts)
    xs = np.append(data_pts[:, 0], data_pts[0, 0])
    ys = np.append(data_pts[:, 1], data_pts[0, 1])

    fc = 'black' if filled else 'white'
    ax.fill(xs, ys, color=fc, zorder=zorder)
    ax.plot(xs, ys, color='black', linewidth=lw, zorder=zorder + 0.1)


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

    # ── Overlap-minimised series generation ────────────────────────────────
    # Three strategies to reduce symbol overlap:
    #
    #  1. EC50 slot partitioning (horizontal spread): the log-EC50 range is
    #     widened to cover the full visible x-axis and divided into N_SYMBOLS
    #     equal slots.  Each series is assigned a unique slot, guaranteeing
    #     that the sigmoid transitions are spread evenly across the x-axis.
    #
    #  2. Staggered vertical bands (vertical spread): the [0, 1] y-range is
    #     divided into N_SYMBOLS equal bands.  Each series is assigned a unique
    #     band and its bottom/top are constrained to that band, so curves
    #     occupy different vertical regions and cross less often.
    #
    #  3. Tiny x-jitter (±5% of sub-interval): symbols from different series
    #     are never exactly co-located in the same pixel column.

    # --- EC50 slots (horizontal) ---
    # Narrower range (3 decades) so more curves transition at similar x-positions
    # → higher overlap in the steep region.
    LOG_EC50_MIN = -12.5
    LOG_EC50_MAX = -9.5
    ec50_slot_w  = (LOG_EC50_MAX - LOG_EC50_MIN) / N_SYMBOLS
    ec50_slots   = np_rng.permutation(N_SYMBOLS)

    # --- No vertical band constraints ---
    # bottom/top sampled freely from [0.02, 1.00] so curves can overlap vertically.

    # --- Shared nominal x-positions with tiny per-series jitter ---
    x_sub_w   = (LOG_MAX - LOG_MIN) / N_POINTS
    x_nominal = np.array([
        10 ** (LOG_MIN + (k + 0.5) * x_sub_w) for k in range(N_POINTS)
    ])

    series_data = []
    for si in range(N_SYMBOLS):
        # EC50: unique slot within the narrow range
        slot   = ec50_slots[si]
        log_ec = LOG_EC50_MIN + ec50_slot_w * slot + np_rng.uniform(0.05, 0.95) * ec50_slot_w
        ec50   = 10 ** log_ec

        # bottom/top free — allows vertical overlap between series
        bottom = np_rng.uniform(0.02, 0.20)
        top    = np_rng.uniform(0.75, 1.00)

        n_hill = np_rng.uniform(0.8, 4.0)

        # Tiny x-jitter: ±5% of sub-interval width in log space
        jitter = np_rng.uniform(-0.05, 0.05, N_POINTS) * x_sub_w
        x_vals = np.array([
            10 ** (np.log10(x_nominal[k]) + jitter[k]) for k in range(N_POINTS)
        ])
        y_vals = hill(x_vals, bottom, top, ec50, n_hill)
        y_vals += np_rng.normal(0, 0.005, N_POINTS)
        y_vals  = np.clip(y_vals, Y_MIN + 0.01, Y_MAX - 0.01)
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

    # Sample per-series perturbation parameters once per plot.
    #
    # Size rule: open and filled variants of the same shape are always the
    # same size within a plot.  Shape groups (by class index):
    #   circle  : 0 (filled_circle),  1 (open_circle)
    #   square  : 2 (filled_square),  3 (open_square)
    #   triangle: 4 (open_triangle),  6 (filled_triangle)
    #   inv_tri : 5 (open_inv_tri),   7 (filled_inv_tri)
    #   rhombus : 8 (open_rhombus),   9 (filled_rhombus)
    #   x       : 10 (x_marker)  -- no pair
    #   plus    : 11 (plus_marker) -- no pair
    #
    # One plot-level base size is drawn from MARKER_PT ±20% so different
    # plots still have different overall symbol sizes.  Each shape group
    # then gets its own size sampled at ±5% from that base, shared by both
    # the open and filled variant in the group.
    _SHAPE_GROUPS = {
        0: 'circle',   1: 'circle',
        2: 'square',   3: 'square',
        4: 'triangle', 5: 'triangle',   # all four triangle types share one size
        6: 'triangle', 7: 'triangle',
        8: 'rhombus',  9: 'rhombus',
        10: 'x',
        11: 'plus',
    }
    plot_base_ms = sample_marker_size(base=MARKER_PT, variation=0.20, np_rng=np_rng)
    # One size per shape group
    unique_groups = list(dict.fromkeys(_SHAPE_GROUPS.values()))  # preserves order
    group_ms = {
        g: sample_marker_size(base=plot_base_ms, variation=0.02, np_rng=np_rng)
        for g in unique_groups
    }
    series_params = []
    for si in range(N_SYMBOLS):
        ms_pt    = group_ms[_SHAPE_GROUPS[si]]
        lw       = sample_linewidth(base=0.8, variation=0.40, np_rng=np_rng)
        # Triangles always sharp; circles/x/plus use matplotlib (r irrelevant)
        r_corner = (float(np_rng.uniform(0.0, 0.5))
                    if si in _POLY_VERTS and si not in _TRIANGLE_IDXS
                    else 0.0)
        series_params.append((ms_pt, lw, r_corner))

    ax.set_xscale('log')
    ax.set_xlim(10**LOG_MIN_PLOT, 10**LOG_MAX_PLOT)
    ax.set_ylim(Y_MIN_PLOT, Y_MAX_PLOT)
    ax.axis('off')   # no axes, ticks, labels or spines — plotting area only
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()   # needed so transData is valid before drawing markers

    for z, si in enumerate(z_order):
        x_vals, y_vals = series_data[si]
        mcode, filled  = _MPL_MARKERS[si]
        ms_pt, lw, r_corner = series_params[si]
        zbase = N_SYMBOLS + z * 3

        if si in _POLY_VERTS:
            # Draw each point as a rounded polygon via draw_rounded_marker
            for xi, yi in zip(x_vals, y_vals):
                draw_rounded_marker(ax, xi, yi,
                                    class_idx=si, filled=filled,
                                    r=r_corner, ms_pt=ms_pt, lw=lw,
                                    zorder=zbase)
        else:
            # Circles, x_marker, plus_marker — use matplotlib directly
            fc = 'black' if filled else 'white'
            ax.plot(x_vals, y_vals,
                    color='black',
                    marker=mcode,
                    markersize=ms_pt,
                    markerfacecolor=fc,
                    markeredgecolor='black',
                    markeredgewidth=lw,
                    linestyle='none',
                    zorder=zbase)

    fig.canvas.draw()   # final redraw with all markers in place

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Composite RGBA onto a white background before converting to BGR.
    # Without this, open symbols rendered with markerfacecolor='white' on a
    # transparent figure get alpha=0 for their white-fill pixels, which then
    # become indistinguishable from the background after RGBA→BGR conversion.
    rgba_f   = buf.astype(np.float32) / 255.0
    alpha    = rgba_f[:, :, 3:4]
    white_bg = np.ones_like(rgba_f[:, :, :3])
    rgb_comp = rgba_f[:, :, :3] * alpha + white_bg * (1.0 - alpha)
    img_bgr  = cv2.cvtColor((rgb_comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Apply scan noise (randomly choose level per plot)
    noise_level = rng.choices(['none', 'light', 'medium', 'heavy'], 
                              weights=[0.1, 0.3, 0.4, 0.2])[0]
    img_bgr = apply_scan_noise(img_bgr, noise_level, np_rng)
    
    H_px, W_px = img_bgr.shape[:2]

    # Build raw gt_points (all symbol centres)
    # Build raw_points (all symbol centres, in series order) and
    # series_pixels (ordered pixel sequences per series for GT segments).
    raw_points: list = []
    series_pixels: list = []   # list of lists: series_pixels[si] = [{cx,cy}, ...]
    for si in range(N_SYMBOLS):
        x_vals, y_vals = series_data[si]
        sp: list = []
        for xi, yi in zip(x_vals, y_vals):
            disp = ax.transData.transform((xi, yi))
            px = int(round(disp[0]))
            py = int(round(H_px - disp[1]))
            px = max(0, min(W_px - 1, px))
            py = max(0, min(H_px - 1, py))
            raw_points.append({"cx": px, "cy": py,
                               "class_idx": si,
                               "class_name": CLASS_NAMES[si]})
            sp.append({"cx": px, "cy": py})
        series_pixels.append(sp)

    # Minimum-distance filter: drop any point whose centre is within
    # MIN_SEP pixels of an already-accepted point.  This removes the
    # patches where two symbols are so close that neither can be
    # classified cleanly.  MIN_SEP = 1.5 × symbol diameter.
    MIN_SEP = int(round(P * 1.5))
    accepted = []   # list of (cx, cy) already kept
    gt_points = []
    for pt in raw_points:
        cx, cy = pt["cx"], pt["cy"]
        too_close = any(
            (cx - ax2) ** 2 + (cy - ay2) ** 2 < MIN_SEP ** 2
            for ax2, ay2 in accepted
        )
        if not too_close:
            accepted.append((cx, cy))
            gt_points.append(pt)

    bbox  = ax.get_position()
    pa_x0 = int(round(bbox.x0 * W_px))
    pa_y0 = int(round((1 - bbox.y1) * H_px))
    pa_x1 = int(round(bbox.x1 * W_px))
    pa_y1 = int(round((1 - bbox.y0) * H_px))

    plt.close(fig)

    img_path = Path(out_dir) / f"plot_{idx:05d}.png"
    gt_path  = Path(out_dir) / f"gt_{idx:05d}.json"
    cv2.imwrite(str(img_path), img_bgr)
    # Build a set of accepted (cx, cy) for fast lookup
    accepted_set = {(pt["cx"], pt["cy"]) for pt in gt_points}

    # GT segments: connect ALL consecutive pairs within each series.
    # The MIN_SEP filter is only for selecting clean ViT subimage patches;
    # it must NOT gate segment GT — every rendered line should be covered.
    MIN_SEG_LEN_PX = 5.0
    gt_segments: list = []
    for si, sp in enumerate(series_pixels):
        for i in range(len(sp) - 1):
            p0, p1 = sp[i], sp[i + 1]
            length = math.hypot(p1["cx"] - p0["cx"], p1["cy"] - p0["cy"])
            if length >= MIN_SEG_LEN_PX:
                gt_segments.append({
                    "x1": p0["cx"], "y1": p0["cy"],
                    "x2": p1["cx"], "y2": p1["cy"],
                    "series_idx": si,
                    "length": round(length, 2),
                })

    with open(gt_path, "w") as f:
        json.dump({
            "plot_w": W_px, "plot_h": H_px,
            "pa": {"x0": pa_x0, "y0": pa_y0, "x1": pa_x1, "y1": pa_y1},
            "points": gt_points,
            # all_points: every rendered symbol (before MIN_SEP filter).
            # Used by the background-patch sampler.
            "all_points": raw_points,
            # series_pixels: ordered pixel sequences for each series.
            # Used by segment detectors to derive correct GT segments.
            "series_pixels": series_pixels,
            # segments: GT line segments (both endpoints survived MIN_SEP).
            "segments": gt_segments,
        }, f)
    return str(img_path), str(gt_path)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBIMAGE EXTRACTION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def extract_patch_padded(gray: np.ndarray, cx: int, cy: int,
                         p: int = None) -> np.ndarray:
    """Extract p×p patch centred at (cx, cy), padding with 255 at borders.
    Always returns exactly p×p — any arithmetic edge cases are cropped."""
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
    # Strict size guarantee: crop to exactly p×p in case of any off-by-one
    return patch[:p, :p]


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
    # Use all_points (every rendered symbol, before MIN_SEP filter) for the
    # exclusion check so that dropped symbols — still visible in the image —
    # are also excluded from background patches.
    all_pts = gt.get("all_points", pts)  # fallback to pts for old GT files
    sym_centers  = [(p["cx"], p["cy"]) for p in all_pts]
    tensors_list = []
    labels_list  = []

    # Helper: returns True if the background patch centred at (px, py) would
    # contain any symbol centre.
    #
    # Uses Chebyshev (L∞) distance: max(|dx|, |dy|) < t.
    # This matches the square patch geometry exactly — a symbol centre at
    # (sx, sy) is inside the P×P patch iff max(|px-sx|,|py-sy|) < HALF.
    # t = P/2 = HALF: the exclusion boundary exactly coincides with the
    # patch edge, so no symbol centre can appear inside a background patch.
    _EXCL = P / 2   # t = P/2 = HALF  (≈ 9.5 px at P=19)
    def _patch_contains_marker(px: int, py: int) -> bool:
        return any(
            max(abs(px - sx), abs(py - sy)) < _EXCL
            for sx, sy in sym_centers
        )

    for pt in pts:
        cx, cy, ci = pt["cx"], pt["cy"], pt["class_idx"]
        # Case 1 — 2 centred samples (symbol label)
        for _ in range(2):
            ox = rng.randint(-2, 2)
            oy = rng.randint(-2, 2)
            patch = extract_patch_padded(gray, cx + ox, cy + oy)
            tensors_list.append(patch_to_uint8(patch))
            labels_list.append(ci)
        # Case 2 — background patch placed near this marker but strictly far
        # enough from ALL markers so that no symbol centre falls within the patch.
        # Exclusion radius is _EXCL = P.  We sample from [P*1.05, P*2.0]
        # to stay just outside the exclusion zone while remaining close to the
        # symbol (providing useful near-symbol background context).
        max_tries = 20
        for _try in range(max_tries):
            angle = rng.uniform(0, 2 * math.pi)
            dist  = rng.uniform(_EXCL * 1.05, P * 2.0)
            ox = int(round(dist * math.cos(angle)))
            oy = int(round(dist * math.sin(angle)))
            bx2, by2 = cx + ox, cy + oy
            if not _patch_contains_marker(bx2, by2):
                patch = extract_patch_padded(gray, bx2, by2)
                tensors_list.append(patch_to_uint8(patch))
                labels_list.append(N_SYMBOLS)  # background label
                break
        # If no valid position found in max_tries, skip this background sample

    # Case 3 — random background patches: patch centre must be at least _EXCL
    # pixels from every symbol centre (guaranteed by _patch_contains_marker).
    target_bg = len(pts) * 3
    added = attempts = 0
    while added < target_bg and attempts < target_bg * 20:
        attempts += 1
        bx = rng2.randint(pa["x0"], pa["x1"])
        by = rng2.randint(pa["y0"], pa["y1"])
        if _patch_contains_marker(bx, by):
            continue
        patch = extract_patch_padded(gray, bx, by)
        tensors_list.append(patch_to_uint8(patch))
        labels_list.append(N_SYMBOLS)  # background label
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
CHUNK_SAMPLES = 400_000  # covers ~half the 1000-plot train set per pass (2 chunks)


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

def build_model(n_classes: int = N_CLASSES, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        'vit_tiny_patch16_224', pretrained=pretrained,
        num_classes=n_classes, img_size=VIT_INPUT
    )
    # freeze all blocks except the last 4
    for blk in list(model.blocks)[:-4]:
        for p in blk.parameters():
            p.requires_grad = False
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  EPOCH LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

EPOCH_LOG_SAMPLE_RATE = 0.01   # fraction of subimages / val plots to save


def _save_epoch_subimages(
        epoch: int,
        chunk_t: torch.Tensor,   # (N, H, W, 3) uint8
        chunk_l: torch.Tensor,   # (N,) int64
) -> None:
    """
    Save a random 1% of training subimages in this chunk to:
      <project>/data/epoch_logs/epoch_XXX/train_subimages/<class_name>_NNNNN.png

    Each image is the 64×64 uint8 patch (grayscale stored as RGB).
    """
    out_dir = EPOCH_LOG_DIR / f"epoch_{epoch:03d}" / "train_subimages"
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs  = chunk_t.numpy()   # (N, H, W, 3) uint8
    lbls  = chunk_l.numpy()   # (N,) int64
    N     = len(imgs)
    # Randomly select 1% of indices
    rng_save = np.random.default_rng(epoch * 7919)  # deterministic per epoch
    n_save   = max(1, int(round(N * EPOCH_LOG_SAMPLE_RATE)))
    chosen   = rng_save.choice(N, size=n_save, replace=False)
    class_counters: dict[int, int] = {}
    for i in chosen:
        ci   = int(lbls[i])
        name = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else f"class_{ci}"
        cnt  = class_counters.get(ci, 0)
        class_counters[ci] = cnt + 1
        fname = out_dir / f"{name}_{cnt:05d}.png"
        # imgs[i] is HWC uint8 — cv2.imwrite expects BGR; since it is grayscale
        # replicated to 3 channels, RGB == BGR, so no conversion needed.
        cv2.imwrite(str(fname), imgs[i])


def _legend_panel(height: int) -> np.ndarray:
    """Build a colour-legend panel of the given height."""
    LEGEND_W  = 200
    ROW_H     = 22
    PADDING   = 10
    SWATCH_W  = 14
    SWATCH_H  = 14
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    classes   = list(CLASS_NAMES[:-1]) + ["unknown"]  # exclude background
    legend_h  = max(height, PADDING * 2 + len(classes) * ROW_H + ROW_H)
    panel     = np.full((legend_h, LEGEND_W, 3), 245, dtype=np.uint8)
    cv2.putText(panel, "Legend", (PADDING, PADDING + 14),
                FONT, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(panel, (PADDING, PADDING + 20),
             (LEGEND_W - PADDING, PADDING + 20), (180, 180, 180), 1)
    for i, cls_name in enumerate(classes):
        y_top = PADDING + ROW_H + i * ROW_H
        color = _CLASS_COLORS.get(cls_name, (100, 100, 100))
        sx0 = PADDING; sy0 = y_top + (ROW_H - SWATCH_H) // 2
        sx1 = sx0 + SWATCH_W; sy1 = sy0 + SWATCH_H
        cv2.rectangle(panel, (sx0, sy0), (sx1, sy1), color, -1)
        cv2.rectangle(panel, (sx0, sy0), (sx1, sy1), (80, 80, 80), 1)
        cv2.putText(panel, cls_name.replace("_", " "),
                    (sx1 + 6, y_top + ROW_H // 2 + 5),
                    FONT, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def _save_epoch_val_detections(
        epoch: int,
        model: nn.Module,
        device: torch.device,
        synth_dir: Path,
        n_plots: int,
) -> None:
    """
    Run detect() on every validation synthetic plot and save the annotated
    image with a colour legend to:
      <project>/data/epoch_logs/epoch_XXX/val_detections/plot_NNNNN.png

    Validation plots are the last 15% of the sorted plot list.
    """
    out_dir = EPOCH_LOG_DIR / f"epoch_{epoch:03d}" / "val_detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Identify validation plots (last 15% of the n_plots sorted list)
    all_plots = sorted(synth_dir.glob("plot_*.png"))[:n_plots]
    val_start = int(len(all_plots) * 0.85)
    val_plots = all_plots[val_start:]

    # Randomly sample 1% of validation plots to save
    rng_val = np.random.default_rng(epoch * 3571)
    n_save  = max(1, int(round(len(val_plots) * EPOCH_LOG_SAMPLE_RATE)))
    save_plots = list(rng_val.choice(len(val_plots), size=min(n_save, len(val_plots)),
                                     replace=False))
    save_set   = set(save_plots)

    model.eval()
    for idx, img_path in enumerate(val_plots):
        if idx not in save_set:
            continue
        # Run sliding-window detection using the current model weights
        dets = detect(str(img_path), model_path=None, _model=model, _device=device)

        # Draw detections on the image
        img_bgr = cv2.imread(str(img_path))
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        for d in dets:
            cx, cy = d["cx"], d["cy"]
            cn     = d["class_name"]
            conf   = d["confidence"]
            color  = _CLASS_COLORS.get(cn, (100, 100, 100))
            r      = HALF + 2
            cv2.circle(img_bgr, (cx, cy), r, color, 1)
            cv2.putText(img_bgr, f"{conf:.2f}", (cx + r + 1, cy + 4),
                        FONT, 0.25, color, 1, cv2.LINE_AA)

        # Attach legend panel
        H_img = img_bgr.shape[0]
        legend = _legend_panel(H_img)
        if img_bgr.shape[0] < legend.shape[0]:
            pad = np.full((legend.shape[0] - img_bgr.shape[0],
                           img_bgr.shape[1], 3), 255, dtype=np.uint8)
            img_bgr = np.vstack([img_bgr, pad])
        combined = np.hstack([img_bgr, legend])
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), combined)

        # Compute per-plot detection metrics against ground truth
        gt_path = img_path.parent / img_path.name.replace("plot_", "gt_").replace(".png", ".json")
        if gt_path.exists():
            with open(gt_path) as f:
                gt = json.load(f)
            
            # Ground truth points by class
            gt_by_cls = {c: [] for c in range(N_SYMBOLS)}
            for pt in gt["points"]:
                if 0 <= pt["class_idx"] < N_SYMBOLS:
                    gt_by_cls[pt["class_idx"]].append((pt["cx"], pt["cy"]))
            
            # Detected points by class
            det_by_cls = {c: [] for c in range(N_SYMBOLS)}
            for d in dets:
                if 0 <= d["class_idx"] < N_SYMBOLS:
                    det_by_cls[d["class_idx"]].append((d["cx"], d["cy"]))
            
            # Match detections to GT (greedy distance matching within MIN_SEP)
            MIN_SEP = int(round(P * 1.5))
            plot_metrics = []
            
            for c in range(N_SYMBOLS):
                gts = list(gt_by_cls[c])
                dts = list(det_by_cls[c])
                
                tp = 0
                for dx, dy in dts:
                    best_dist = float('inf')
                    best_gi = -1
                    for gi, (gx, gy) in enumerate(gts):
                        dist = math.hypot(dx - gx, dy - gy)
                        if dist < best_dist and dist < MIN_SEP:
                            best_dist = dist
                            best_gi = gi
                    if best_gi >= 0:
                        tp += 1
                        gts.pop(best_gi)  # matched, remove from pool
                
                fp = len(dts) - tp
                fn = len(gt_by_cls[c]) - tp
                # For per-plot detection, TN is not well-defined, leave as 0 or N/A
                tn = 0
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                plot_metrics.append({
                    "class": CLASS_NAMES[c],
                    "TP": tp, "FP": fp, "FN": fn,
                    "precision": prec, "recall": rec, "f1": f1
                })
            
            # Save per-plot CSV (skip silently if the file is locked, e.g. open in Excel)
            csv_path = out_dir / img_path.name.replace(".png", "_metrics.csv")
            try:
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write("class,TP,FP,FN,precision,recall,f1\n")
                    for row in plot_metrics:
                        f.write(f"{row['class']},{row['TP']},{row['FP']},{row['FN']},{row['precision']:.4f},{row['recall']:.4f},{row['f1']:.4f}\n")
            except PermissionError:
                print(f"  [warn] Could not write {csv_path.name} — file may be open in another program.")

    print(f"  Val detections saved → {out_dir}  ({len(save_set)}/{len(val_plots)} plots sampled)")


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
    print(f"STEP 2 — Pre-generating subimage patches → {SUBIMG_DIR}")
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
            # 2 symbol + ~4 background samples per point
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

            # Save all training subimages for this chunk to D:\chartocode_epoch_logs
            _save_epoch_subimages(epoch, chunk_t, chunk_l)

            del chunk_t, chunk_l, chunk_ds, chunk_ld  # free RAM before next chunk

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_t0 = time.time()
        # Collect all predictions and ground-truth labels for F1 computation
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for bi, (xb_u8, yb) in enumerate(va_ld, 1):
                xb = uint8_batch_to_tensor(xb_u8.numpy(), device)
                yb = yb.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    out = model(xb)
                preds = out.argmax(1).cpu().numpy()
                trues = yb.cpu().numpy()
                all_preds.append(preds)
                all_trues.append(trues)
                _pbar(bi, n_va_batches, val_t0,
                      prefix=f"Epoch {epoch:3d}/{EPOCHS} val:   ")

        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)

        # Macro F1: average F1 over all classes (penalises FP and FN equally)
        # Compute per-class TP, FP, FN, TN
        per_f1 = []
        metrics_table = []
        for c in range(N_CLASSES):
            tp = int(((all_preds == c) & (all_trues == c)).sum())
            fp = int(((all_preds == c) & (all_trues != c)).sum())
            fn = int(((all_preds != c) & (all_trues == c)).sum())
            tn = int(((all_preds != c) & (all_trues != c)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_f1.append(f1)
            metrics_table.append({
                "class": CLASS_NAMES[c],
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": prec, "recall": rec, "f1": f1
            })
            
        macro_f1 = float(np.mean(per_f1))
        # Keep symbol-only macro F1 (exclude background class index N_SYMBOLS)
        sym_f1   = float(np.mean(per_f1[:N_SYMBOLS]))

        improved = macro_f1 > best_acc
        if improved:
            best_acc = macro_f1
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(save_model.state_dict(), str(MODEL_SAVE_PATH))

        # Save validation detection visualisations for this epoch
        _save_epoch_val_detections(
            epoch,
            model._orig_mod if hasattr(model, "_orig_mod") else model,
            device,
            SYNTH_DIR,
            n_plots,
        )

        epoch_elapsed = time.time() - epoch_t0
        stop_info = "  [saved]" if improved else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"loss={tr_loss/tr_total:.4f} | "
              f"macro_f1={macro_f1:.4f} | sym_f1={sym_f1:.4f} | best={best_acc:.4f} | "
              f"{epoch_elapsed:.0f}s{stop_info}")

        # Save per-class metrics to CSV
        # Columns: class, TP, FP, FN, TN,
        #          TP+FP (prec denom), TP+FN (rec denom),
        #          precision = TP/(TP+FP), recall = TP/(TP+FN),
        #          2*P*R (F1 numerator), P+R (F1 denominator), F1 = 2PR/(P+R)
        csv_dir = EPOCH_LOG_DIR / f"epoch_{epoch:03d}"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "val_metrics.csv"
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("class,TP,FP,FN,TN,"
                        "TP+FP,TP+FN,"
                        "precision=TP/(TP+FP),recall=TP/(TP+FN),"
                        "2*P*R,P+R,F1=2PR/(P+R)\n")
                for row in metrics_table:
                    tp_fp = row['TP'] + row['FP']
                    tp_fn = row['TP'] + row['FN']
                    two_pr = 2.0 * row['precision'] * row['recall']
                    p_plus_r = row['precision'] + row['recall']
                    f.write(
                        f"{row['class']},{row['TP']},{row['FP']},{row['FN']},{row['TN']},"
                        f"{tp_fp},{tp_fn},"
                        f"{row['precision']:.4f},{row['recall']:.4f},"
                        f"{two_pr:.4f},{p_plus_r:.4f},{row['f1']:.4f}\n"
                    )
        except PermissionError:
            print(f"  [warn] Could not write {csv_path.name} — file may be open in another program.")

        # Print full metrics table to console
        # Layout:
        #  Class | TP | FP | FN | TN | TP+FP | TP+FN | Prec=TP/(TP+FP) | Rec=TP/(TP+FN) | 2PR | P+R | F1=2PR/(P+R)
        hdr = (f"  {'Class':<22} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'TN':>8} "
               f"| {'TP+FP':>7} | {'TP+FN':>7} "
               f"| {'Prec':>7} | {'Rec':>7} "
               f"| {'2·P·R':>7} | {'P+R':>7} | {'F1':>7}")
        print(f"\n  Epoch {epoch} Validation Metrics:")
        print(f"  Formula: Prec = TP/(TP+FP)   Rec = TP/(TP+FN)   F1 = 2·Prec·Rec / (Prec+Rec)")
        print(hdr)
        print("  " + "-" * len(hdr))
        for row in metrics_table:
            tp_fp = row['TP'] + row['FP']
            tp_fn = row['TP'] + row['FN']
            two_pr = 2.0 * row['precision'] * row['recall']
            p_plus_r = row['precision'] + row['recall']
            print(
                f"  {row['class']:<22} | {row['TP']:6d} | {row['FP']:6d} | {row['FN']:6d} | {row['TN']:8d} "
                f"| {tp_fp:7d} | {tp_fn:7d} "
                f"| {row['precision']:7.4f} | {row['recall']:7.4f} "
                f"| {two_pr:7.4f} | {p_plus_r:7.4f} | {row['f1']:7.4f}"
            )
        # Summary rows
        print("  " + "-" * len(hdr))
        print(f"  {'macro_f1 (all classes)':<22}   {'':>6}   {'':>6}   {'':>6}   {'':>8} "
              f"  {'':>7}   {'':>7} "
              f"  {'':>7}   {'':>7} "
              f"  {'':>7}   {'':>7}   {macro_f1:7.4f}")
        print(f"  {'sym_f1  (symbols only)':<22}   {'':>6}   {'':>6}   {'':>6}   {'':>8} "
              f"  {'':>7}   {'':>7} "
              f"  {'':>7}   {'':>7} "
              f"  {'':>7}   {'':>7}   {sym_f1:7.4f}")
        print()


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

def _load_model(model_path: str | Path | None = None) -> tuple[nn.Module, torch.device]:
    """
    Load the chart marker ViT model.

    Priority:
      1. model_path argument (if given and file exists)
      2. Default MODEL_SAVE_PATH  (chartocode2/models/chart_marker_net_v3.pth)
      3. timm pretrained ImageNet weights (fallback — no fine-tuning)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve which weight file to use
    candidates = []
    if model_path is not None:
        candidates.append(Path(model_path))
    candidates.append(MODEL_SAVE_PATH)   # default: ../models/chart_marker_net_v3.pth

    for p in candidates:
        if Path(p).exists():
            print(f"[model] Loading fine-tuned weights: {p}")
            m = build_model(pretrained=False)   # don't re-download ImageNet weights
            m.load_state_dict(
                torch.load(str(p), map_location=device, weights_only=True)
            )
            m.eval().to(device)
            return m, device

    # Fallback: timm pretrained (ImageNet) — no chart-specific fine-tuning
    print("[model] WARNING: chart_marker_net_v3.pth not found.")
    print("[model] Falling back to timm pretrained ImageNet weights (vit_tiny_patch16_224).")
    print("[model] Detection accuracy will be lower without fine-tuned weights.")
    m = build_model(pretrained=True)
    m.eval().to(device)
    return m, device


def _per_class_nms(dets: list[dict], radius: float) -> list[dict]:
    """Spatial NMS: suppress same-class detections within `radius` pixels."""
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


def _per_class_xcol_nms(dets: list[dict], bin_width: float) -> list[dict]:
    """
    X-column NMS: for each class, partition detections into x-bins of width
    `bin_width`. Within each bin keep only the highest-confidence detection.
    This enforces the constraint that each class has at most one marker per
    x-interval [a, a+bin_width].
    """
    by_class = defaultdict(list)
    for d in dets:
        by_class[d["class_idx"]].append(d)
    kept = []
    for ci, cls_dets in by_class.items():
        # Assign each detection to an x-bin index
        bins: dict[int, list[dict]] = defaultdict(list)
        for d in cls_dets:
            bin_idx = int(d["cx"] // bin_width)
            bins[bin_idx].append(d)
        # Keep only the highest-confidence detection per bin
        for bin_dets in bins.values():
            best = max(bin_dets, key=lambda x: x["confidence"])
            kept.append(best)
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
           p: int = None,
           _model: nn.Module = None,
           _device: torch.device = None) -> list[dict]:
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

    if _model is not None and _device is not None:
        model, device = _model, _device
    else:
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
            if ci == N_SYMBOLS:          # background — skip
                continue
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

    # X-column NMS only — keep at most one detection per class per x-bin
    # (spatial NMS disabled; x-column constraint is the sole suppression step)
    xcol_bin_width = p * XCOL_NMS_WIDTH_FACTOR
    kept         = _per_class_xcol_nms(symbol_dets, xcol_bin_width)
    unknown_kept_list = [{**d, "class_idx": 999} for d in unknown_dets]
    unknown_kept_list = _per_class_xcol_nms(unknown_kept_list, xcol_bin_width)
    for d in unknown_kept_list: d["class_idx"] = -1
    unknown_kept = unknown_kept_list

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
    "plus_marker":         (60,  60, 200),
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
    parser.add_argument("--mode",   choices=["train", "detect", "generate"], required=True)
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

    elif args.mode == "generate":
        # Generate synthetic plots only — no subimage extraction or training.
        # Useful for refreshing the dataset before running segment detectors.
        n = args.plots
        SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        existing = len(list(SYNTH_DIR.glob("plot_*.png")))
        if existing >= n:
            print(f"  {existing} plots already exist — nothing to do.")
            print(f"  Delete {SYNTH_DIR} to force regeneration.")
        else:
            seeds = [random.randint(0, 2**31) for _ in range(n)]
            args_list = [(i, str(SYNTH_DIR), seeds[i]) for i in range(n)]
            n_cpu = max(1, WORKERS)
            print(f"  Generating {n} plots using {n_cpu} CPU workers...")
            t0 = time.time()
            with mp.Pool(n_cpu) as pool:
                results = pool.map(generate_one_plot, args_list)
            print(f"  Done in {time.time()-t0:.1f}s — {len(results)} plots saved.")
            print(f"  Output: {SYNTH_DIR}")

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
