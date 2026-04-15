"""
chart_marker_detector_v2.py
============================
Working directory : C:\\Users\\ziola\\OneDrive\\Documents\\GitHub\\chartocode\\src
Model save path   : ../models/chart_marker_net_v2.pth   (relative to src/)

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
import argparse, io, json, math, os, random, time, warnings
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
from torch.utils.data import DataLoader, Dataset
import timm
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
mp.freeze_support()

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  (relative to src/)
# ══════════════════════════════════════════════════════════════════════════════
_SRC_DIR        = Path(__file__).parent
MODEL_SAVE_PATH = _SRC_DIR / ".." / "models" / "chart_marker_net_v2.pth"
SYNTH_DIR       = _SRC_DIR / ".." / "data" / "synthetic_plots"
SUBIMG_DIR      = _SRC_DIR / ".." / "data" / "subimages"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
N_PLOTS         = 2100          # synthetic plots to generate
SYMBOL_DIAM     = 12            # base symbol diameter in pixels
P_FACTOR        = 1.20          # p = ceil(SYMBOL_DIAM * P_FACTOR) → must be odd
P               = int(math.ceil(SYMBOL_DIAM * P_FACTOR)) | 1  # ensure odd
HALF            = P // 2
VIT_INPUT       = 64            # ViT input resolution
PLOT_W, PLOT_H  = 560, 420      # synthetic plot canvas (plotting area only)
N_POINTS        = 12            # data points per series
BATCH_SIZE      = 256
EPOCHS          = 40
LR              = 3e-4
USE_COMPILE     = True          # set False if triton not installed
CONF_THRESH     = 0.65          # detection confidence threshold
STRIDE          = 2             # sliding window stride
NMS_RADIUS      = P * 1.5       # per-class NMS suppression radius
UNKNOWN_THRESH  = 0.40          # below this max-prob → "unknown"
MIN_DARK_FRAC   = 0.03          # ignore windows with too few dark pixels
WORKERS         = min(8, mp.cpu_count())

# 11 symbol classes + background
CLASS_NAMES = [
    "filled_circle",        # 0  → matplotlib marker 'o', filled black
    "open_circle",          # 1  → matplotlib marker 'o', open (white fill)
    "filled_square",        # 2  → matplotlib marker 's', filled black
    "open_square",          # 3  → matplotlib marker 's', open (white fill)
    "open_triangle",        # 4  → matplotlib marker '^', open
    "open_inv_triangle",    # 5  → matplotlib marker 'v', open
    "filled_triangle",      # 6  → matplotlib marker '^', filled black
    "filled_inv_triangle",  # 7  → matplotlib marker 'v', filled black
    "open_rhombus",         # 8  → matplotlib marker 'D', open
    "filled_rhombus",       # 9  → matplotlib marker 'D', filled black
    "x_marker",             # 10 → matplotlib marker 'x'
    "background",           # 11
]
N_CLASSES  = len(CLASS_NAMES)
N_SYMBOLS  = N_CLASSES - 1   # 11

# matplotlib marker codes and fill styles for each symbol class
# (marker, filled)  — filled=True → facecolor black, filled=False → facecolor white
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
    ('x', True),    # 10 x_marker  (x has no fill concept — always black)
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# ══════════════════════════════════════════════════════════════════════════════
#  SYMBOL RENDERING  — uses matplotlib default markers
# ══════════════════════════════════════════════════════════════════════════════

# Pre-rendered symbol cache: sym_id → RGBA uint8 (SYMBOL_DIAM × SYMBOL_DIAM)
_SYMBOL_CACHE: dict[int, np.ndarray] = {}


def render_symbol(sym_id: int, d: int = SYMBOL_DIAM,
                  corner_density: float = 0.0) -> np.ndarray:
    """
    Render symbol sym_id using matplotlib's default marker into a d×d RGBA image.
    corner_density is accepted for API compatibility but ignored (matplotlib handles
    corner rounding automatically for square markers).
    Returns RGBA uint8 array.
    """
    cache_key = (sym_id, d)
    if cache_key in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[cache_key].copy()

    marker, filled = _MPL_MARKERS[sym_id]

    # DPI and figure size chosen so the rendered symbol is exactly d px
    dpi    = 100
    fig_px = d * 4          # render at 4× then downscale for anti-aliasing
    fig_in = fig_px / dpi

    fig, ax = plt.subplots(figsize=(fig_in, fig_in), dpi=dpi)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.axis("off")

    # marker size in points: fill ~70% of the figure
    ms = fig_px * 0.65

    fc = "black" if filled else "white"
    ec = "black"
    lw = max(1.0, d * 0.10)

    ax.plot(0, 0, marker=marker,
            markersize=ms,
            markerfacecolor=fc,
            markeredgecolor=ec,
            markeredgewidth=lw,
            linestyle="none")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", pad_inches=0,
                transparent=True)
    plt.close(fig)
    buf.seek(0)

    # decode PNG → RGBA
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    rgba_full = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)   # H×W×4
    if rgba_full is None or rgba_full.ndim < 3:
        # fallback: white square
        rgba_full = np.zeros((fig_px, fig_px, 4), dtype=np.uint8)
        rgba_full[:, :, :3] = 255

    # downscale to d×d with INTER_AREA for clean anti-aliasing
    rgba_d = cv2.resize(rgba_full, (d, d), interpolation=cv2.INTER_AREA)

    # ensure open symbols have fully opaque white interior (not semi-transparent)
    if not filled and marker != 'x':
        # flood-fill the interior region with white+opaque
        gray = rgba_d[:, :, 3]
        _, bw = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # find interior: invert mask, flood from centre
        interior = np.zeros((d + 2, d + 2), dtype=np.uint8)
        cv2.floodFill(255 - bw, interior, (d // 2, d // 2), 255)
        interior = interior[1:-1, 1:-1]
        mask = interior > 0
        rgba_d[mask, 0] = 255  # R
        rgba_d[mask, 1] = 255  # G
        rgba_d[mask, 2] = 255  # B
        rgba_d[mask, 3] = 255  # A (fully opaque white)

    _SYMBOL_CACHE[cache_key] = rgba_d.copy()
    return rgba_d


def composite_symbol(canvas_bgr: np.ndarray, sym_rgba: np.ndarray,
                     cx: int, cy: int) -> None:
    """Alpha-composite sym_rgba centred at (cx,cy) onto canvas_bgr (in-place)."""
    d  = sym_rgba.shape[0]
    hd = d // 2
    H, W = canvas_bgr.shape[:2]
    sx0, sy0 = 0, 0
    sx1, sy1 = d, d
    dx0 = cx - hd; dy0 = cy - hd
    dx1 = dx0 + d; dy1 = dy0 + d
    if dx0 < 0: sx0 -= dx0; dx0 = 0
    if dy0 < 0: sy0 -= dy0; dy0 = 0
    if dx1 > W: sx1 -= (dx1 - W); dx1 = W
    if dy1 > H: sy1 -= (dy1 - H); dy1 = H
    if dx0 >= dx1 or dy0 >= dy1: return
    patch = sym_rgba[sy0:sy1, sx0:sx1]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    # sym_rgba is BGRA (OpenCV convention after cv2.imdecode)
    rgb   = patch[:, :, :3].astype(np.float32)
    roi   = canvas_bgr[dy0:dy1, dx0:dx1].astype(np.float32)
    canvas_bgr[dy0:dy1, dx0:dx1] = (rgb * alpha + roi * (1 - alpha)).clip(0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  HILL EQUATION  (concentration-efficacy curve)
# ══════════════════════════════════════════════════════════════════════════════

def hill(x: np.ndarray, bottom: float, top: float,
         ec50: float, n: float) -> np.ndarray:
    return bottom + (top - bottom) / (1.0 + (ec50 / x) ** n)


def make_series_x(n_pts: int, log_min: float = -15.0,
                  log_max: float = -7.0) -> np.ndarray:
    """Equally log-spaced x-coordinates."""
    return np.logspace(log_min, log_max, n_pts)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC PLOT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _log_to_px(log_x: float, log_min: float, log_max: float, W: int) -> int:
    frac = (log_x - log_min) / (log_max - log_min)
    return int(round(frac * (W - 1)))


def _y_to_px(y: float, y_min: float, y_max: float, H: int) -> int:
    frac = (y - y_min) / (y_max - y_min)
    return int(round((1.0 - frac) * (H - 1)))


def generate_one_plot(args_tuple):
    """
    Worker function for multiprocessing.
    Returns (img_path, gt_path).
    """
    idx, out_dir, seed = args_tuple
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    W, H = PLOT_W, PLOT_H
    LOG_MIN, LOG_MAX = -15.0, -7.0
    Y_MIN, Y_MAX = 0.0, 1.05

    # --- canvas (white background) ---
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    # --- plotting area margins ---
    margin_l, margin_r = 40, 20
    margin_b, margin_t = 30, 20
    pa_x0, pa_y0 = margin_l, margin_t
    pa_x1, pa_y1 = W - margin_r, H - margin_b
    pa_w = pa_x1 - pa_x0
    pa_h = pa_y1 - pa_y0
    cv2.rectangle(canvas, (pa_x0, pa_y0), (pa_x1, pa_y1), (0, 0, 0), 1)

    # --- generate 11 series ---
    gt_points = []
    z_order   = list(range(N_SYMBOLS))
    rng.shuffle(z_order)

    # pre-render all symbols using matplotlib default markers
    sym_imgs = [render_symbol(si, SYMBOL_DIAM) for si in range(N_SYMBOLS)]

    # generate curve parameters per series
    series_data = []
    for si in range(N_SYMBOLS):
        bottom = np_rng.uniform(0.05, 0.15)
        top    = np_rng.uniform(0.80, 1.00)
        ec50   = 10 ** np_rng.uniform(-12, -8)
        n_hill = np_rng.uniform(0.8, 2.5)
        x_vals = make_series_x(N_POINTS)
        y_vals = hill(x_vals, bottom, top, ec50, n_hill)
        y_vals += np_rng.normal(0, 0.01, N_POINTS)
        y_vals  = np.clip(y_vals, Y_MIN + 0.01, Y_MAX - 0.01)
        series_data.append((x_vals, y_vals))

    # Step 1: draw all connecting lines (lowest z-order)
    for si in z_order:
        x_vals, y_vals = series_data[si]
        pts = []
        for xi, yi in zip(x_vals, y_vals):
            px = pa_x0 + _log_to_px(math.log10(xi), LOG_MIN, LOG_MAX, pa_w)
            py = pa_y0 + _y_to_px(yi, Y_MIN, Y_MAX, pa_h)
            pts.append((px, py))
        for k in range(len(pts) - 1):
            cv2.line(canvas, pts[k], pts[k+1], (0, 0, 0), 1)

    # Step 2: draw symbols in z-order (on top of lines)
    for si in z_order:
        x_vals, y_vals = series_data[si]
        for xi, yi in zip(x_vals, y_vals):
            px = pa_x0 + _log_to_px(math.log10(xi), LOG_MIN, LOG_MAX, pa_w)
            py = pa_y0 + _y_to_px(yi, Y_MIN, Y_MAX, pa_h)
            # clamp to plotting area
            px = max(pa_x0, min(pa_x1, px))
            py = max(pa_y0, min(pa_y1, py))
            composite_symbol(canvas, sym_imgs[si], px, py)
            gt_points.append({
                "cx": px, "cy": py,
                "class_idx": si,
                "class_name": CLASS_NAMES[si]
            })

    # add mild noise
    noise  = np_rng.integers(0, 6, canvas.shape, dtype=np.uint8)
    canvas = np.clip(canvas.astype(np.int16) + noise - 3, 0, 255).astype(np.uint8)

    # save
    img_path = Path(out_dir) / f"plot_{idx:05d}.png"
    gt_path  = Path(out_dir) / f"gt_{idx:05d}.json"
    cv2.imwrite(str(img_path), canvas)
    with open(gt_path, "w") as f:
        json.dump({
            "plot_w": W, "plot_h": H,
            "pa": {"x0": pa_x0, "y0": pa_y0, "x1": pa_x1, "y1": pa_y1},
            "points": gt_points
        }, f)
    return str(img_path), str(gt_path)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBIMAGE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_patch_padded(gray: np.ndarray, cx: int, cy: int,
                         p: int = P) -> np.ndarray:
    """Extract p×p patch centred at (cx,cy), padding with 255 if needed."""
    half = p // 2
    H, W = gray.shape
    pad  = np.full((H + p, W + p), 255, dtype=np.uint8)
    pad[half:half+H, half:half+W] = gray
    x = cx + half; y = cy + half
    return pad[y-half:y+half+1, x-half:x+half+1]


def patch_to_tensor(patch: np.ndarray) -> np.ndarray:
    """Convert grayscale patch to normalised 3-channel float32 CHW."""
    r = cv2.resize(patch, (VIT_INPUT, VIT_INPUT), interpolation=cv2.INTER_LINEAR)
    t = np.stack([r, r, r], axis=0).astype(np.float32) / 255.0
    return (t - _MEAN) / _STD


def extract_subimages(synth_dir: Path, out_dir: Path):
    """
    Extract subimages from all synthetic plots using 3 sampling methods.
    Saves tensors.npy and labels.npy to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(synth_dir.glob("gt_*.json"))
    print(f"  Extracting subimages from {len(gt_files)} plots...")

    case1_count = 0
    case2_count = 0
    bg_coords_all = []
    records = []

    # --- pass 1: cases 1 and 2 ---
    for gt_file in gt_files:
        with open(gt_file) as f:
            gt = json.load(f)
        img_file = gt_file.parent / gt_file.name.replace("gt_", "plot_").replace(".json", ".png")
        if not img_file.exists(): continue
        gray = cv2.cvtColor(cv2.imread(str(img_file)), cv2.COLOR_BGR2GRAY)
        pa   = gt["pa"]
        pts  = gt["points"]
        symbol_centers = [(p["cx"], p["cy"]) for p in pts]

        for pt in pts:
            cx, cy, ci = pt["cx"], pt["cy"], pt["class_idx"]
            # Case 1: centred (offset < 3 px) — 2 samples per point
            for _ in range(2):
                ox = random.randint(-2, 2); oy = random.randint(-2, 2)
                patch = extract_patch_padded(gray, cx+ox, cy+oy)
                records.append((patch_to_tensor(patch), ci))
                case1_count += 1
            # Case 2: slightly off (3-5 px) → background label — 1 sample per point
            angle = random.uniform(0, 2*math.pi)
            dist  = random.uniform(3, 5)
            ox = int(round(dist * math.cos(angle)))
            oy = int(round(dist * math.sin(angle)))
            patch = extract_patch_padded(gray, cx+ox, cy+oy)
            records.append((patch_to_tensor(patch), N_SYMBOLS))  # background
            case2_count += 1

        bg_coords_all.append((gray, pa, symbol_centers))

    print(f"  Case 1 (positive):      {case1_count}")
    print(f"  Case 2 (slightly-off):  {case2_count}")

    # --- pass 2: case 3 (random background) ---
    target_bg = case1_count + case2_count
    bg_count  = 0
    rng = random.Random(42)
    for gray, pa, sym_centers in bg_coords_all:
        n_needed = max(1, target_bg // len(bg_coords_all))
        attempts = 0; added = 0
        while added < n_needed and attempts < n_needed * 20:
            attempts += 1
            cx = rng.randint(pa["x0"], pa["x1"])
            cy = rng.randint(pa["y0"], pa["y1"])
            too_close = any(
                math.sqrt((cx-sx)**2 + (cy-sy)**2) < HALF + 2
                for sx, sy in sym_centers
            )
            if too_close: continue
            patch = extract_patch_padded(gray, cx, cy)
            records.append((patch_to_tensor(patch), N_SYMBOLS))
            bg_count += 1; added += 1
        if bg_count >= target_bg: break

    print(f"  Case 3 (random bg):     {bg_count}")
    print(f"  Total subimages:        {len(records)}")

    tensors = np.stack([r[0] for r in records], axis=0).astype(np.float32)
    labels  = np.array([r[1] for r in records], dtype=np.int64)
    np.save(str(out_dir / "tensors.npy"), tensors)
    np.save(str(out_dir / "labels.npy"),  labels)
    print(f"  Saved tensors.npy {tensors.shape} and labels.npy")
    return tensors, labels


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SubimageDataset(Dataset):
    def __init__(self, tensors: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(tensors)
        self.y = torch.from_numpy(labels)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def build_model(n_classes: int = N_CLASSES) -> nn.Module:
    model = timm.create_model(
        'vit_tiny_patch16_224', pretrained=True,
        num_classes=n_classes, img_size=VIT_INPUT
    )
    # unfreeze last 4 transformer blocks + classification head
    blocks = list(model.blocks)
    for blk in blocks[:-4]:
        for p in blk.parameters():
            p.requires_grad = False
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(n_plots: int = N_PLOTS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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
        # Pre-render all symbols once in main process to populate cache
        for si in range(N_SYMBOLS):
            render_symbol(si, SYMBOL_DIAM)
        t0 = time.time()
        with mp.Pool(n_cpu) as pool:
            results = pool.map(generate_one_plot, args)
        print(f"  Done in {time.time()-t0:.1f}s — {len(results)} plots saved.")

    print("\n" + "="*60)
    print("STEP 2 — Extracting subimages")
    print("="*60)
    tensors_path = SUBIMG_DIR / "tensors.npy"
    labels_path  = SUBIMG_DIR / "labels.npy"
    if tensors_path.exists() and labels_path.exists():
        print("  Loading cached subimages...")
        tensors = np.load(str(tensors_path))
        labels  = np.load(str(labels_path))
        print(f"  Loaded {len(labels)} subimages.")
    else:
        tensors, labels = extract_subimages(SYNTH_DIR, SUBIMG_DIR)

    print("\n" + "="*60)
    print("STEP 3 — Training ViT  (GPU-optimised)")
    print("="*60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    X_tr, X_va, y_tr, y_va = train_test_split(
        tensors, labels, test_size=0.15, random_state=42, stratify=labels
    )
    tr_ds = SubimageDataset(X_tr, y_tr)
    va_ds = SubimageDataset(X_va, y_va)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=WORKERS, pin_memory=(device.type=="cuda"),
                       persistent_workers=(WORKERS > 0),
                       prefetch_factor=4 if WORKERS > 0 else None)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=WORKERS, pin_memory=(device.type=="cuda"),
                       persistent_workers=(WORKERS > 0),
                       prefetch_factor=4 if WORKERS > 0 else None)

    model = build_model().to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # torch.compile — CUDA only (avoids MSVC requirement on CPU/Windows)
    if USE_COMPILE and hasattr(torch, "compile") and device.type == "cuda":
        print("  Compiling model with torch.compile() ...")
        model = torch.compile(model)
    else:
        print("  torch.compile() skipped (CPU or CUDA unavailable / USE_COMPILE=False)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in tr_ld:
            xb = xb.to(device, non_blocking=True)
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
        scheduler.step()

        model.eval()
        va_correct = va_total = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    out = model(xb)
                va_correct += (out.argmax(1) == yb).sum().item()
                va_total   += len(yb)
        va_acc = va_correct / va_total

        if va_acc > best_acc:
            best_acc = va_acc
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(save_model.state_dict(), str(MODEL_SAVE_PATH))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"loss={tr_loss/tr_total:.4f} | "
                  f"val_acc={va_acc:.4f} | best={best_acc:.4f}")

    print(f"\n  Training complete. Best val acc: {best_acc:.4f}")
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
    """Apply NMS independently per class."""
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


def _estimate_center(window_cx: int, window_cy: int,
                     patch_gray: np.ndarray) -> tuple[int, int]:
    """
    Estimate the true symbol centre within the p×p patch via connected-component centroid.
    """
    _, bw = cv2.threshold(patch_gray, 180, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return window_cx, window_cy
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    est_cx = int(round(centroids[best][0])) - P // 2
    est_cy = int(round(centroids[best][1])) - P // 2
    return window_cx + est_cx, window_cy + est_cy


def detect(image_path: str,
           model_path: str | Path = MODEL_SAVE_PATH,
           conf_thresh: float = CONF_THRESH,
           stride: int = STRIDE,
           nms_radius: float = NMS_RADIUS,
           unknown_thresh: float = UNKNOWN_THRESH,
           symbol_diam: int = SYMBOL_DIAM,
           min_dark_frac: float = MIN_DARK_FRAC) -> list[dict]:
    """
    Detect markers in a plotting-area image.

    Parameters
    ----------
    image_path    : path to the plotting-area image (entire image = plotting area)
    model_path    : path to trained weights
    conf_thresh   : minimum confidence to keep a detection
    stride        : sliding window stride (px)
    nms_radius    : per-class NMS suppression radius (px)
    unknown_thresh: if max class probability < this, label as "unknown"
    symbol_diam   : expected symbol diameter (px)
    min_dark_frac : minimum fraction of p×p pixels that must be dark

    Returns
    -------
    List of dicts, each with:
      cx, cy       — estimated symbol centre in image coordinates
      class_idx    — 0-10 (symbol) or -1 (unknown)
      class_name   — e.g. "filled_circle" or "unknown"
      confidence   — softmax probability of the predicted class
    """
    model, device = _load_model(model_path)

    img_bgr  = cv2.imread(str(image_path))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W     = img_gray.shape

    min_dark_pixels = int(P * P * min_dark_frac)
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
                pass  # background — skip
            elif max_prob < unknown_thresh:
                patch = extract_patch_padded(img_gray, bx, by)
                ecx, ecy = _estimate_center(bx, by, patch)
                raw_dets.append({
                    "cx": ecx, "cy": ecy,
                    "class_idx": -1, "class_name": "unknown",
                    "confidence": round(max_prob, 4)
                })
            elif max_prob >= conf_thresh:
                patch = extract_patch_padded(img_gray, bx, by)
                ecx, ecy = _estimate_center(bx, by, patch)
                raw_dets.append({
                    "cx": ecx, "cy": ecy,
                    "class_idx": ci, "class_name": CLASS_NAMES[ci],
                    "confidence": round(max_prob, 4)
                })
        batch_coords.clear(); batch_tensors.clear()

    # sliding window — centre must be inside plotting area (full image)
    for cy_w in range(0, H, stride):
        for cx_w in range(0, W, stride):
            patch = extract_patch_padded(img_gray, cx_w, cy_w)
            _, bw = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
            if np.count_nonzero(bw) < min_dark_pixels: continue
            batch_coords.append((cx_w, cy_w))
            batch_tensors.append(patch_to_tensor(patch))
            if len(batch_tensors) == 512: flush_batch()
    flush_batch()

    # per-class NMS (unknowns handled separately)
    symbol_dets  = [d for d in raw_dets if d["class_idx"] >= 0]
    unknown_dets = [d for d in raw_dets if d["class_idx"] == -1]
    kept         = _per_class_nms(symbol_dets, nms_radius)
    unknown_kept = _per_class_nms(
        [{**d, "class_idx": 999} for d in unknown_dets], nms_radius
    )
    for d in unknown_kept: d["class_idx"] = -1

    results = sorted(kept + unknown_kept, key=lambda d: (d["class_idx"], d["cy"], d["cx"]))

    found_classes = set(d["class_name"] for d in results if d["class_idx"] >= 0)
    print(f"  Detected {len(results)} markers across {len(found_classes)} symbol type(s):")
    for cn in sorted(found_classes):
        n = sum(1 for d in results if d["class_name"] == cn)
        print(f"    {cn}: {n}")
    n_unk = sum(1 for d in results if d["class_idx"] == -1)
    if n_unk: print(f"    unknown: {n_unk}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPER
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
    """Draw detected markers on the image and optionally save."""
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