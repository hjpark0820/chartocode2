"""
Chart Marker Detector — Full Pipeline  (GPU-optimised for RTX 5060 + 16-core CPU)
==================================================================================
Key design choices
------------------
  • All symbols are rendered in BLACK on a WHITE plot background — matching
    typical scientific publication style.
  • Open symbols (open_circle, open_square, open_triangle, etc.) use a
    TRANSPARENT interior: the white interior fill is NOT stamped onto the
    canvas, so underlying line segments remain visible through the symbol.
    Only the outline strokes are composited onto the plot.
  • Filled symbols stamp their solid black shape directly onto the canvas.
  • Subimage patches therefore contain the real mixed pixel content
    (symbol outline + whatever line/background is underneath), not an
    artificial white-blocked region.

GPU speedup techniques (RTX 5060 + 16-core CPU)
-------------------------------------------------
  1. torch.cuda.amp  — Automatic Mixed Precision (FP16)
  2. num_workers=8   — Parallel DataLoader workers
  3. pin_memory=True + persistent_workers + prefetch_factor=4
  4. torch.compile() — TorchInductor (CUDA only; skipped on CPU/Windows)
  5. Batch size 128  — Better GPU utilisation
  6. cudnn.benchmark — Auto-selects fastest conv algorithm
  7. Parallel plot generation via multiprocessing.Pool (16 cores)
  8. Pre-built TensorDataset — patches converted to tensors once

11 symbol classes + background
-------------------------------
  0  filled_circle          6  filled_triangle
  1  open_circle            7  filled_inv_triangle
  2  filled_square          8  open_rhombus
  3  open_square            9  filled_rhombus
  4  open_triangle         10  x_marker
  5  open_inv_triangle     11  background

Usage
-----
  python chart_marker_detector.py --mode train
  python chart_marker_detector.py --mode detect --image path/to/plotting_area.png
"""

import argparse
import os
import json
import math
import random
import warnings
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_MODEL_DIR = r"C:\Users\ziola\OneDrive\Documents\GitHub\chartocode\models"
_model_dir = Path(OUTPUT_MODEL_DIR)
if not _model_dir.exists():
    _model_dir = Path(__file__).parent / "models"
_model_dir.mkdir(parents=True, exist_ok=True)

WORK_DIR   = Path(__file__).parent
SYNTH_DIR  = WORK_DIR / "synthetic_plots"
GT_DIR     = SYNTH_DIR / "ground_truth"
MODEL_PATH = _model_dir / "chart_marker_net.pth"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

# Plot canvas
PLOT_W, PLOT_H = 640, 480

# Symbol & window
SYM_RADIUS = 6          # half-size of rendered symbol in pixels
P          = 19         # sliding window / subimage size (must be odd)
HALF       = P // 2
VIT_INPUT  = 64         # ViT input resolution

# Dataset
N_PLOTS        = 520
N_SERIES       = 11
N_POINTS       = 8
PATCHES_POS    = 60     # positive patches per symbol per plot
PATCHES_OFF    = 40     # slightly-off patches per symbol per plot
PATCHES_BG     = 80     # random background patches per plot

# Training
EPOCHS          = 40
BATCH_SIZE      = 128
LR              = 3e-4
WEIGHT_DECAY    = 0.01
NUM_WORKERS     = 8
PREFETCH_FACTOR = 4
USE_AMP         = True
USE_COMPILE     = True   # activated only when CUDA is available

# Inference
STRIDE      = 2
CONF_THRESH = 0.65
NMS_RADIUS  = P * 1.5

CLASS_NAMES = [
    "filled_circle", "open_circle",
    "filled_square", "open_square",
    "open_triangle", "open_inv_triangle",
    "filled_triangle", "filled_inv_triangle",
    "open_rhombus", "filled_rhombus",
    "x_marker",
    "background",
]
N_CLASSES = len(CLASS_NAMES)   # 12

# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — SYMBOL RENDERING  (RGBA — transparent interior for open shapes)
# ═══════════════════════════════════════════════════════════════════════════

def render_symbol_rgba(cls_idx: int, size: int = P,
                       corner_density: float = 0.0) -> np.ndarray:
    """
    Render symbol cls_idx into a (size × size, 4) RGBA uint8 image.
    Alpha channel:
      • Filled symbols  → fully opaque black shape (alpha=255).
      • Open symbols    → white-filled interior + opaque black outline.
                          The white fill occludes any lines beneath the symbol,
                          matching real scientific plot appearance.
    Background pixels (outside the symbol) are always transparent.
    """
    img = np.zeros((size, size, 4), dtype=np.uint8)   # fully transparent
    cx = cy = size // 2
    r  = max(2, size // 3)
    thick = max(1, size // 10)
    BLACK_OPAQUE = (0, 0, 0, 255)

    # ── helper: square corner points with optional Bézier rounding ─────────
    def square_pts(smooth=0.0):
        if smooth < 0.05:
            return np.array([[cx-r, cy-r], [cx+r, cy-r],
                              [cx+r, cy+r], [cx-r, cy+r]], dtype=np.float32)
        k = smooth * r * 0.6
        corners = [(-r, -r), (r, -r), (r, r), (-r, r)]
        pts = []
        for i, (dx, dy) in enumerate(corners):
            nx1, ny1 = corners[(i - 1) % 4]
            nx2, ny2 = corners[(i + 1) % 4]
            tx1 = 0 if dx == nx1 else np.sign(nx1 - dx)
            ty1 = 0 if dy == ny1 else np.sign(ny1 - dy)
            tx2 = 0 if dx == nx2 else np.sign(nx2 - dx)
            ty2 = 0 if dy == ny2 else np.sign(ny2 - dy)
            p0 = np.array([cx+dx+tx1*k, cy+dy+ty1*k])
            p1 = np.array([cx+dx,       cy+dy      ])
            p2 = np.array([cx+dx+tx2*k, cy+dy+ty2*k])
            for t in np.linspace(0, 1, 8):
                pts.append((1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2)
        return np.array(pts, dtype=np.float32)

    def tri_pts(inv=False):
        s = -1 if not inv else 1
        return np.array([[cx, cy-s*r], [cx+r, cy+s*r], [cx-r, cy+s*r]],
                        dtype=np.float32)

    def rhombus_pts():
        return np.array([[cx, cy-r], [cx+r, cy], [cx, cy+r], [cx-r, cy]],
                        dtype=np.float32)

    # ── filled symbols: stamp solid black shape ─────────────────────────────
    if cls_idx == 0:   # filled_circle
        cv2.circle(img, (cx, cy), r, BLACK_OPAQUE, -1)

    elif cls_idx == 2:  # filled_square
        cv2.fillPoly(img, [square_pts(corner_density).astype(np.int32)], BLACK_OPAQUE)

    elif cls_idx == 6:  # filled_triangle
        cv2.fillPoly(img, [tri_pts(False).astype(np.int32)], BLACK_OPAQUE)

    elif cls_idx == 7:  # filled_inv_triangle
        cv2.fillPoly(img, [tri_pts(True).astype(np.int32)], BLACK_OPAQUE)

    elif cls_idx == 9:  # filled_rhombus
        cv2.fillPoly(img, [rhombus_pts().astype(np.int32)], BLACK_OPAQUE)

    elif cls_idx == 10:  # x_marker
        d = max(2, r - 1)
        cv2.line(img, (cx-d, cy-d), (cx+d, cy+d), BLACK_OPAQUE, thick+1)
        cv2.line(img, (cx+d, cy-d), (cx-d, cy+d), BLACK_OPAQUE, thick+1)

    # ── open symbols: transparent interior, opaque outline only ────────────
    elif cls_idx == 1:  # open_circle — white fill + black outline
        cv2.circle(img, (cx, cy), r, (255, 255, 255, 255), -1)   # white fill
        cv2.circle(img, (cx, cy), r, BLACK_OPAQUE, thick)         # black outline

    elif cls_idx == 3:  # open_square — white fill + black outline
        pts = square_pts(corner_density).astype(np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255, 255))             # white fill
        cv2.polylines(img, [pts], True, BLACK_OPAQUE, thick)       # black outline

    elif cls_idx == 4:  # open_triangle — white fill + black outline
        pts = tri_pts(False).astype(np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255, 255))             # white fill
        cv2.polylines(img, [pts], True, BLACK_OPAQUE, thick)       # black outline

    elif cls_idx == 5:  # open_inv_triangle — white fill + black outline
        pts = tri_pts(True).astype(np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255, 255))             # white fill
        cv2.polylines(img, [pts], True, BLACK_OPAQUE, thick)       # black outline

    elif cls_idx == 8:  # open_rhombus — white fill + black outline
        pts = rhombus_pts().astype(np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255, 255))             # white fill
        cv2.polylines(img, [pts], True, BLACK_OPAQUE, thick)       # black outline

    return img   # RGBA uint8


def composite_symbol(canvas_rgb: np.ndarray,
                     sym_rgba: np.ndarray,
                     cx: int, cy: int) -> None:
    """
    Alpha-composite sym_rgba (RGBA) onto canvas_rgb (RGB, white background)
    centred at (cx, cy).  Transparent pixels in sym_rgba leave the canvas
    unchanged — so lines drawn beneath open symbols remain visible.
    """
    H, W = canvas_rgb.shape[:2]
    sh, sw = sym_rgba.shape[:2]
    hh, hw = sh // 2, sw // 2

    # Source region within sym_rgba
    sx1 = max(0, -( cx - hw));  sy1 = max(0, -(cy - hh))
    sx2 = sw - max(0, (cx + hw + (sw % 2)) - W)
    sy2 = sh - max(0, (cy + hh + (sh % 2)) - H)
    # Destination region on canvas
    dx1 = max(0, cx - hw);  dy1 = max(0, cy - hh)
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if dx2 <= dx1 or dy2 <= dy1:
        return

    src_rgb = sym_rgba[sy1:sy2, sx1:sx2, :3].astype(np.float32)
    alpha   = sym_rgba[sy1:sy2, sx1:sx2,  3].astype(np.float32) / 255.0
    dst     = canvas_rgb[dy1:dy2, dx1:dx2].astype(np.float32)

    alpha3  = alpha[:, :, np.newaxis]
    blended = src_rgb * alpha3 + dst * (1.0 - alpha3)
    canvas_rgb[dy1:dy2, dx1:dx2] = np.clip(blended, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — SYNTHETIC PLOT GENERATION  (parallelised)
# ═══════════════════════════════════════════════════════════════════════════

def _sigmoid_curve(x, ec50, hill, top, bottom):
    return bottom + (top - bottom) / (1.0 + (ec50 / (x + 1e-9)) ** hill)


def _gen_plot_worker(args):
    """Worker — picklable for multiprocessing.Pool."""
    plot_idx, synth_dir, gt_dir = args
    random.seed(plot_idx)
    np.random.seed(plot_idx)

    W, H   = PLOT_W, PLOT_H
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)   # white background
    margin = HALF + SYM_RADIUS + 4
    x_min_px, x_max_px = margin, W - margin
    y_min_px, y_max_px = margin, H - margin

    corner_densities = [random.uniform(0.0, 1.0) for _ in range(N_SERIES)]

    x_log_vals = np.linspace(-3.0, 1.0, N_POINTS)
    x_conc     = 10.0 ** x_log_vals

    series_params = [
        (random.uniform(0.05, 0.8), random.uniform(0.8, 3.0),
         random.uniform(0.6, 1.0),  random.uniform(0.0, 0.15))
        for _ in range(N_SERIES)
    ]

    # Compute pixel coordinates for every series
    all_px = []
    for si in range(N_SERIES):
        ec50, hill, top, bot = series_params[si]
        y = _sigmoid_curve(x_conc, ec50, hill, top, bot)
        y += np.random.normal(0, 0.02, N_POINTS)
        y  = np.clip(y, 0.0, 1.0)
        px_x = np.linspace(x_min_px, x_max_px, N_POINTS).astype(int)
        px_y = np.clip((y_max_px - y * (y_max_px - y_min_px)).astype(int),
                       y_min_px, y_max_px)
        all_px.append((px_x, px_y))

    draw_order = list(range(N_SERIES))
    random.shuffle(draw_order)

    # ── Step 1: draw all connecting lines first (lowest z-order) ───────────
    for si in draw_order:
        px_x, px_y = all_px[si]
        for i in range(len(px_x) - 1):
            cv2.line(canvas,
                     (int(px_x[i]),   int(px_y[i])),
                     (int(px_x[i+1]), int(px_y[i+1])),
                     (0, 0, 0), 1)   # black line, 1 px

    # ── Step 2: composite symbols on top (randomised z-order) ──────────────
    gt_series = []
    for si in draw_order:
        px_x, px_y = all_px[si]
        sym_rgba   = render_symbol_rgba(si, size=P,
                                        corner_density=corner_densities[si])
        pts_gt = []
        for i in range(N_POINTS):
            cx_g, cy_g = int(px_x[i]), int(px_y[i])
            composite_symbol(canvas, sym_rgba, cx_g, cy_g)
            pts_gt.append({"px": cx_g, "py": cy_g})
        gt_series.append({
            "series_idx":    si,
            "symbol_class":  si,
            "symbol_name":   CLASS_NAMES[si],
            "corner_density": float(corner_densities[si]),
            "points":        pts_gt,
        })

    # Light Gaussian noise to mimic scan/print artefacts
    noise  = np.random.normal(0, 2, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img_path = Path(synth_dir) / f"plot_{plot_idx:04d}.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    gt = {
        "plot_idx":   plot_idx,
        "image_path": str(img_path),
        "image_size": {"w": W, "h": H},
        "n_series":   N_SERIES,
        "n_points":   N_POINTS,
        "series":     gt_series,
    }
    with open(Path(gt_dir) / f"gt_{plot_idx:04d}.json", "w") as f:
        json.dump(gt, f, indent=2)
    return gt


def generate_all_plots(n_plots: int, n_workers: int = None) -> list:
    """Generate all synthetic plots in parallel."""
    if n_workers is None:
        n_workers = min(16, mp.cpu_count())
    args = [(i, str(SYNTH_DIR), str(GT_DIR)) for i in range(n_plots)]
    print(f"  Generating {n_plots} plots using {n_workers} CPU workers ...")
    with mp.Pool(processes=n_workers) as pool:
        gt_list = pool.map(_gen_plot_worker, args)
    return gt_list


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — SUBIMAGE EXTRACTION  (3 sampling methods)
# ═══════════════════════════════════════════════════════════════════════════

def extract_patch(img_gray: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Extract a P×P grayscale patch centred at (cx, cy); pads with 255 (white)."""
    H, W  = img_gray.shape
    patch = np.full((P, P), 255, dtype=np.uint8)
    x1, y1 = cx - HALF, cy - HALF
    x2, y2 = x1 + P,   y1 + P
    sx1 = max(0, -x1);  sy1 = max(0, -y1)
    sx2 = P - max(0, x2 - W);  sy2 = P - max(0, y2 - H)
    dx1 = max(0, x1);   dy1 = max(0, y1)
    dx2 = min(W, x2);   dy2 = min(H, y2)
    if dx2 > dx1 and dy2 > dy1:
        patch[sy1:sy2, sx1:sx2] = img_gray[dy1:dy2, dx1:dx2]
    return patch


def build_subimage_dataset(gt_list: list) -> tuple:
    patches, labels = [], []
    pos_count = [0] * N_SERIES
    off_count = [0] * N_SERIES
    bg_count  = 0

    for gt in gt_list:
        img_bgr = cv2.imread(gt["image_path"])
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W     = img_gray.shape
        all_centres = [(pt["px"], pt["py"])
                       for s in gt["series"] for pt in s["points"]]

        for series in gt["series"]:
            cls = series["symbol_class"]
            for pt in series["points"]:
                cx, cy = pt["px"], pt["py"]

                # Method 1: positive — centre within ≤2 px jitter
                for _ in range(3):
                    if pos_count[cls] >= PATCHES_POS:
                        break
                    dx = random.randint(-2, 2)
                    dy = random.randint(-2, 2)
                    patches.append(extract_patch(img_gray, cx+dx, cy+dy))
                    labels.append(cls)
                    pos_count[cls] += 1

                # Method 2: slightly-off (3–5 px) → background label
                if off_count[cls] < PATCHES_OFF:
                    angle = random.uniform(0, 2 * math.pi)
                    dist  = random.uniform(3, 5)
                    ox = cx + int(round(dist * math.cos(angle)))
                    oy = cy + int(round(dist * math.sin(angle)))
                    patches.append(extract_patch(img_gray, ox, oy))
                    labels.append(N_SERIES)   # background
                    off_count[cls] += 1

        # Method 3: random background — no centred symbol
        attempts = 0
        while bg_count < PATCHES_BG * len(gt_list) and attempts < 2000:
            attempts += 1
            rx = random.randint(HALF, W - HALF - 1)
            ry = random.randint(HALF, H - HALF - 1)
            if any(abs(rx - cx) < HALF and abs(ry - cy) < HALF
                   for cx, cy in all_centres):
                continue
            patch = extract_patch(img_gray, rx, ry)
            _, pb = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
            if np.sum(pb > 0) / (P * P) < 0.5:
                patches.append(patch)
                labels.append(N_SERIES)
                bg_count += 1

    print(f"  Positive: {sum(pos_count)}  |  "
          f"Slightly-off: {sum(off_count)}  |  "
          f"Background: {bg_count}  |  Total: {len(patches)}")
    return patches, labels


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — TENSOR CACHE
# ═══════════════════════════════════════════════════════════════════════════

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def patches_to_tensor(patches: list) -> torch.Tensor:
    """Convert list of (P×P) uint8 grayscale arrays → (N,3,VIT_INPUT,VIT_INPUT) float32."""
    out = np.empty((len(patches), 3, VIT_INPUT, VIT_INPUT), dtype=np.float32)
    for i, p in enumerate(patches):
        r = cv2.resize(p, (VIT_INPUT, VIT_INPUT))
        t = np.stack([r] * 3, axis=0).astype(np.float32) / 255.0
        out[i] = (t - _MEAN) / _STD
    return torch.from_numpy(out)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — VIT MODEL
# ═══════════════════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=N_CLASSES,
        img_size=VIT_INPUT,
    )
    # Freeze all, then unfreeze last 3 transformer blocks + head
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(9, 12)) or \
           "head" in name or "norm" in name:
            param.requires_grad = True
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train(n_plots: int = N_PLOTS):
    print("=" * 60)
    print("STEP 1 — Generating synthetic plots (parallel)")
    print("=" * 60)
    gt_list = generate_all_plots(n_plots)
    print(f"  Done. {len(gt_list)} plots saved to {SYNTH_DIR}")

    print("\n" + "=" * 60)
    print("STEP 2 — Extracting subimages (3 sampling methods)")
    print("=" * 60)
    patches, labels = build_subimage_dataset(gt_list)

    print("\n" + "=" * 60)
    print("STEP 3 — Building tensor cache")
    print("=" * 60)
    print(f"  Converting {len(patches)} patches to tensors ...")
    X_tensor = patches_to_tensor(patches)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    print(f"  Tensor shape: {X_tensor.shape}")

    idx = list(range(len(patches)))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2,
                                       random_state=42, stratify=labels)
    train_ds = TensorDataset(X_tensor[tr_idx], y_tensor[tr_idx])
    val_ds   = TensorDataset(X_tensor[va_idx], y_tensor[va_idx])

    dl_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    )
    train_dl = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_dl   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)

    print("\n" + "=" * 60)
    print("STEP 4 — Training ViT  (GPU-optimised)")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model = build_model().to(device)

    # torch.compile — CUDA only (CPU/Windows requires MSVC cl.exe)
    _can_compile = (USE_COMPILE
                    and hasattr(torch, "compile")
                    and device.type == "cuda")
    if _can_compile:
        print("  Compiling model with torch.compile() ...")
        model = torch.compile(model)
    else:
        print("  torch.compile() skipped (CPU or CUDA unavailable)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    weights      = torch.ones(N_CLASSES, device=device)
    weights[-1]  = 0.5
    criterion    = nn.CrossEntropyLoss(weight=weights)
    optimizer    = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler(
        enabled=(USE_AMP and device.type == "cuda"))

    best_val_acc = 0.0
    best_state   = None
    train_accs, val_accs = [], []

    import time
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        tc = tt = 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                    enabled=(USE_AMP and device.type == "cuda")):
                out  = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tc += (out.argmax(1) == yb).sum().item()
            tt += len(xb)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(
                        enabled=(USE_AMP and device.type == "cuda")):
                    out = model(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(xb)

        ta = tc / tt;  va = vc / vt
        train_accs.append(ta);  val_accs.append(va)

        if va > best_val_acc:
            best_val_acc = va
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.clone() for k, v in raw.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}:  "
                  f"train={ta:.4f}  val={va:.4f}  "
                  f"best={best_val_acc:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # Save best weights
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.load_state_dict(best_state)
    torch.save(best_state, str(MODEL_PATH))
    print(f"\n  Best val_acc = {best_val_acc:.4f}")
    print(f"  Weights saved → {MODEL_PATH}")
    print(f"  Total training time: {time.time()-t0:.0f}s")

    # Confusion matrix
    raw.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                    enabled=(USE_AMP and device.type == "cuda")):
                out = raw(xb)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    cm = confusion_matrix(all_true, all_preds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_accs, label="Train")
    axes[0].plot(val_accs,   label="Val")
    axes[0].axhline(best_val_acc, linestyle="--", color="green",
                    label=f"Best={best_val_acc:.3f}")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    short = [c[:7] for c in CLASS_NAMES]
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(N_CLASSES)); axes[1].set_yticks(range(N_CLASSES))
    axes[1].set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticklabels(short, fontsize=7)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=6,
                         color="white" if cm[i, j] > cm.max()/2 else "black")
    axes[1].set_title(f"Confusion Matrix  (val_acc={best_val_acc:.3f})")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    out_png = WORK_DIR / "training_results.png"
    fig.savefig(str(out_png), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Training plot → {out_png}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 7 — INFERENCE
#  Input: path to a plotting-area image (entire image = plotting area)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_path: str = str(MODEL_PATH)) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    state  = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def detect(image_path: str,
           model_path: str = str(MODEL_PATH),
           conf_thresh: float = CONF_THRESH,
           stride: int = STRIDE,
           nms_radius: float = NMS_RADIUS,
           save_vis: bool = True) -> list:
    """
    Detect symbols in a plotting-area image.
    The entire input image is treated as the plotting area.

    Returns list of dicts: {cx, cy, class_idx, class_name, confidence}
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = load_model(model_path)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W     = img_gray.shape

    print(f"[detect] {W}×{H} px  |  p={P}  stride={stride}  "
          f"conf≥{conf_thresh}  NMS_r={nms_radius:.1f}")

    MIN_PIX = int(P * P * 0.04)
    batch_coords, batch_patches = [], []
    raw_dets = []

    def _flush():
        if not batch_patches:
            return
        t = torch.tensor(np.stack(batch_patches),
                         dtype=torch.float32).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                probs = torch.softmax(model(t), dim=1).cpu().numpy()
        for (bx, by), prob in zip(batch_coords, probs):
            ci = int(prob.argmax())
            if ci < N_SERIES and prob[ci] >= conf_thresh:
                raw_dets.append((bx, by, ci, float(prob[ci])))
        batch_coords.clear(); batch_patches.clear()

    for cy in range(0, H, stride):
        for cx in range(0, W, stride):
            patch = extract_patch(img_gray, cx, cy)
            _, pb = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
            if np.sum(pb > 0) < MIN_PIX:
                continue
            r = cv2.resize(patch, (VIT_INPUT, VIT_INPUT))
            t = np.stack([r] * 3, axis=0).astype(np.float32) / 255.0
            t = (t - _MEAN) / _STD
            batch_coords.append((cx, cy))
            batch_patches.append(t)
            if len(batch_patches) == 512:
                _flush()
    _flush()

    print(f"  Raw detections: {len(raw_dets)}")

    # Best-of-group NMS
    raw_dets.sort(key=lambda d: -d[3])
    kept = []; suppressed = set()
    for i, det in enumerate(raw_dets):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(raw_dets)):
            if j in suppressed:
                continue
            dist = math.sqrt((det[0]-raw_dets[j][0])**2 +
                             (det[1]-raw_dets[j][1])**2)
            if dist < nms_radius:
                suppressed.add(j)

    markers = [
        {"cx": d[0], "cy": d[1],
         "class_idx": d[2], "class_name": CLASS_NAMES[d[2]],
         "confidence": d[3]}
        for d in kept
    ]
    print(f"  After NMS: {len(markers)} markers")
    for cls in CLASS_NAMES[:N_SERIES]:
        cnt = sum(1 for m in markers if m["class_name"] == cls)
        if cnt:
            print(f"    {cls}: {cnt}")

    if save_vis:
        vis = img_rgb.copy()
        for m in markers:
            cv2.circle(vis, (m["cx"], m["cy"]), HALF, (220, 30, 30), 2)
            cv2.putText(vis, f"{m['confidence']:.2f}",
                        (m["cx"] + HALF + 2, m["cy"] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 30, 30), 1)
        vis_path = str(Path(image_path).with_suffix("")) + "_detections.png"
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  Visualisation → {vis_path}")

    return markers


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mp.freeze_support()   # required for multiprocessing on Windows

    parser = argparse.ArgumentParser(description="Chart Marker Detector")
    parser.add_argument("--mode",    choices=["train", "detect"], required=True)
    parser.add_argument("--image",   type=str,   default=None)
    parser.add_argument("--model",   type=str,   default=str(MODEL_PATH))
    parser.add_argument("--plots",   type=int,   default=N_PLOTS)
    parser.add_argument("--conf",    type=float, default=CONF_THRESH)
    parser.add_argument("--stride",  type=int,   default=STRIDE)
    parser.add_argument("--workers", type=int,   default=NUM_WORKERS,
                        help="DataLoader worker processes (default 8)")
    args = parser.parse_args()

    if args.mode == "train":
        NUM_WORKERS = args.workers
        train(n_plots=args.plots)
    elif args.mode == "detect":
        if args.image is None:
            parser.error("--image is required for --mode detect")
        results = detect(image_path=args.image, model_path=args.model,
                         conf_thresh=args.conf, stride=args.stride)
        print("\nDetected markers:")
        for m in results:
            print(f"  ({m['cx']:4d},{m['cy']:4d})  "
                  f"{m['class_name']:<22s}  conf={m['confidence']:.3f}")
