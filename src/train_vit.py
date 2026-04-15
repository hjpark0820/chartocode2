"""
ViT Symbol Detector — Full Pipeline
====================================
Covers:
  1. Synthetic plot generation  (500+ concentration-efficacy curves)
  2. Ground truth export        (JSON per plot)
  3. p×p subimage extraction    (3 sampling methods → balanced dataset)
  4. ViT fine-tuning            (pretrained vit_tiny_patch16_224 via timm)
  5. Weight saving              (to OUTPUT_MODEL_DIR)
  6. Inference function         (input = plotting-area image file path)

11 symbol classes
-----------------
  0  filled_circle
  1  open_circle
  2  filled_square
  3  open_square
  4  open_triangle
  5  open_inv_triangle
  6  filled_triangle
  7  filled_inv_triangle
  8  open_rhombus
  9  filled_rhombus
 10  x_marker
 11  background          (negative class)

Usage
-----
  # Training (run once):
      python train_vit.py --mode train

  # Inference on a plotting-area image:
      python train_vit.py --mode detect --image path/to/plotting_area.png
"""

import argparse
import os
import json
import math
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT_MODEL_DIR = r"C:\Users\ziola\OneDrive\Documents\GitHub\chartocode\models"
# On Linux/Mac this path will not exist; the script falls back to ./models/
_model_dir = Path(OUTPUT_MODEL_DIR)
if not _model_dir.exists():
    _model_dir = Path(__file__).parent / "models"
_model_dir.mkdir(parents=True, exist_ok=True)

WORK_DIR        = Path(__file__).parent
SYNTH_DIR       = WORK_DIR / "synthetic_plots"
SUBIMG_DIR      = WORK_DIR / "subimages"
MODEL_PATH      = _model_dir / "vit_symbol_detector.pth"
GT_DIR          = SYNTH_DIR / "ground_truth"

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
SUBIMG_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

# Plot canvas
PLOT_W, PLOT_H  = 640, 480       # plotting-area image size (pixels)

# Symbol & window
SYM_RADIUS      = 6              # half-size of rendered symbol (pixels)
P               = 19             # sliding-window / subimage size (odd)
HALF            = P // 2
VIT_INPUT       = 64             # ViT input resolution

# Dataset
N_PLOTS         = 520            # synthetic plots to generate
N_SERIES        = 11             # one series per symbol type
N_POINTS        = 8              # data points per series
PATCHES_POS     = 60             # positive patches per symbol class
PATCHES_OFF     = 40             # slightly-off patches per symbol class
PATCHES_BG      = 80             # background patches per symbol class (random)

# Training
EPOCHS          = 40
BATCH_SIZE      = 64
LR              = 3e-4
WEIGHT_DECAY    = 0.01

# Inference
STRIDE          = 2
CONF_THRESH     = 0.65
NMS_RADIUS      = P * 1.5

CLASS_NAMES = [
    "filled_circle", "open_circle",
    "filled_square", "open_square",
    "open_triangle", "open_inv_triangle",
    "filled_triangle", "filled_inv_triangle",
    "open_rhombus", "filled_rhombus",
    "x_marker",
    "background",
]
N_CLASSES = len(CLASS_NAMES)      # 12

# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — SYMBOL RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def _pts_to_mask(pts, size):
    """Rasterise a polygon into a binary mask of given size."""
    mask = np.zeros((size, size), dtype=np.uint8)
    pts_i = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts_i], 255)
    return mask


def render_symbol(cls_idx: int, size: int = P,
                  corner_density: float = 0.0,
                  color: tuple = (0, 0, 0),
                  bg_color: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Render symbol cls_idx into a (size × size) RGB image.
    corner_density ∈ [0, 1]: smoothness of square corners (0 = sharp, 1 = very round).
    Returns uint8 RGB array.
    """
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    cx = cy = size // 2
    r  = max(2, size // 3)

    def circle_pts(n=64):
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)

    def square_pts(smooth=0.0):
        """Square with rounded corners. smooth ∈ [0,1]."""
        if smooth < 0.05:
            return np.array([[cx-r, cy-r], [cx+r, cy-r],
                              [cx+r, cy+r], [cx-r, cy+r]], dtype=np.float32)
        # Bézier-rounded corners
        k = smooth * r * 0.6
        corners = [(-r, -r), (r, -r), (r, r), (-r, r)]
        pts = []
        for i, (dx, dy) in enumerate(corners):
            nx1, ny1 = corners[(i - 1) % 4]
            nx2, ny2 = corners[(i + 1) % 4]
            # tangent directions
            tx1 = (0 if dx == nx1 else np.sign(nx1 - dx))
            ty1 = (0 if dy == ny1 else np.sign(ny1 - dy))
            tx2 = (0 if dx == nx2 else np.sign(nx2 - dx))
            ty2 = (0 if dy == ny2 else np.sign(ny2 - dy))
            p0 = np.array([cx + dx + tx1 * k, cy + dy + ty1 * k])
            p1 = np.array([cx + dx, cy + dy])
            p2 = np.array([cx + dx + tx2 * k, cy + dy + ty2 * k])
            for t in np.linspace(0, 1, 8):
                b = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                pts.append(b)
        return np.array(pts, dtype=np.float32)

    def triangle_pts(inverted=False):
        sign = -1 if not inverted else 1
        return np.array([
            [cx,       cy - sign * r],
            [cx + r,   cy + sign * r],
            [cx - r,   cy + sign * r],
        ], dtype=np.float32)

    def rhombus_pts():
        return np.array([
            [cx,     cy - r],
            [cx + r, cy    ],
            [cx,     cy + r],
            [cx - r, cy    ],
        ], dtype=np.float32)

    c = color
    thick = max(1, size // 10)

    if cls_idx == 0:   # filled_circle
        cv2.circle(img, (cx, cy), r, c, -1)
    elif cls_idx == 1: # open_circle
        cv2.circle(img, (cx, cy), r, bg_color, -1)
        cv2.circle(img, (cx, cy), r, c, thick)
    elif cls_idx == 2: # filled_square
        pts = square_pts(corner_density)
        cv2.fillPoly(img, [pts.astype(np.int32)], c)
    elif cls_idx == 3: # open_square
        pts = square_pts(corner_density)
        cv2.fillPoly(img, [pts.astype(np.int32)], bg_color)
        cv2.polylines(img, [pts.astype(np.int32)], True, c, thick)
    elif cls_idx == 4: # open_triangle
        pts = triangle_pts(False)
        cv2.fillPoly(img, [pts.astype(np.int32)], bg_color)
        cv2.polylines(img, [pts.astype(np.int32)], True, c, thick)
    elif cls_idx == 5: # open_inv_triangle
        pts = triangle_pts(True)
        cv2.fillPoly(img, [pts.astype(np.int32)], bg_color)
        cv2.polylines(img, [pts.astype(np.int32)], True, c, thick)
    elif cls_idx == 6: # filled_triangle
        pts = triangle_pts(False)
        cv2.fillPoly(img, [pts.astype(np.int32)], c)
    elif cls_idx == 7: # filled_inv_triangle
        pts = triangle_pts(True)
        cv2.fillPoly(img, [pts.astype(np.int32)], c)
    elif cls_idx == 8: # open_rhombus
        pts = rhombus_pts()
        cv2.fillPoly(img, [pts.astype(np.int32)], bg_color)
        cv2.polylines(img, [pts.astype(np.int32)], True, c, thick)
    elif cls_idx == 9: # filled_rhombus
        pts = rhombus_pts()
        cv2.fillPoly(img, [pts.astype(np.int32)], c)
    elif cls_idx == 10: # x_marker
        d = max(2, r - 1)
        cv2.line(img, (cx-d, cy-d), (cx+d, cy+d), c, thick+1)
        cv2.line(img, (cx+d, cy-d), (cx-d, cy+d), c, thick+1)

    return img


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — SYNTHETIC PLOT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _sigmoid_curve(x_vals, ec50, hill, top, bottom):
    """4-parameter logistic (Hill equation)."""
    return bottom + (top - bottom) / (1.0 + (ec50 / (x_vals + 1e-9)) ** hill)


def generate_synthetic_plot(plot_idx: int,
                             save_dir: Path,
                             gt_dir: Path) -> dict:
    """
    Generate one concentration-efficacy plot with all 11 symbol types.
    Each series uses equally-spaced x-coordinates.
    Returns ground-truth dict.
    """
    W, H = PLOT_W, PLOT_H
    img  = np.full((H, W, 3), 255, dtype=np.uint8)

    # Axis margins (the plotting area IS the full image here)
    margin = HALF + SYM_RADIUS + 4
    x_min_px = margin;  x_max_px = W - margin
    y_min_px = margin;  y_max_px = H - margin

    # Random curve parameters per series
    series_params = []
    for _ in range(N_SERIES):
        ec50  = random.uniform(0.05, 0.8)
        hill  = random.uniform(0.8, 3.0)
        top   = random.uniform(0.6, 1.0)
        bottom = random.uniform(0.0, 0.15)
        series_params.append((ec50, hill, top, bottom))

    # Equally-spaced x-coordinates in log space (common for conc-efficacy)
    x_log_min, x_log_max = -3.0, 1.0
    x_log_vals = np.linspace(x_log_min, x_log_max, N_POINTS)
    x_conc     = 10.0 ** x_log_vals          # actual concentration values

    # Assign random colours and corner densities
    palette = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44),
        (214, 39, 40),  (148, 103, 189), (140, 86, 75),
        (227, 119, 194),(127, 127, 127), (188, 189, 34),
        (23, 190, 207), (0, 0, 0),
    ]
    random.shuffle(palette)
    corner_densities = [random.uniform(0.0, 1.0) for _ in range(N_SERIES)]

    # Ground truth storage
    gt_series = []

    # Draw order: shuffle series z-order to create overlaps
    draw_order = list(range(N_SERIES))
    random.shuffle(draw_order)

    # Pre-compute pixel coordinates for all series
    all_px_coords = []
    for si in range(N_SERIES):
        ec50, hill, top, bottom = series_params[si]
        y_vals = _sigmoid_curve(x_conc, ec50, hill, top, bottom)
        # Add small noise
        y_vals += np.random.normal(0, 0.02, N_POINTS)
        y_vals  = np.clip(y_vals, 0.0, 1.0)

        # Map to pixel space
        px_x = np.linspace(x_min_px, x_max_px, N_POINTS).astype(int)
        px_y = (y_max_px - y_vals * (y_max_px - y_min_px)).astype(int)
        px_y = np.clip(px_y, y_min_px, y_max_px)
        all_px_coords.append((px_x, px_y))

    # Draw lines first (lowest z-order), then symbols in shuffled order
    for si in draw_order:
        px_x, px_y = all_px_coords[si]
        color = palette[si]
        # Draw connecting line
        for i in range(len(px_x) - 1):
            cv2.line(img, (px_x[i], px_y[i]),
                     (px_x[i+1], px_y[i+1]), color, 1)

    # Draw symbols in shuffled z-order
    for si in draw_order:
        px_x, px_y = all_px_coords[si]
        color = palette[si]
        cd    = corner_densities[si]
        sym_img = render_symbol(si, size=P, corner_density=cd,
                                color=color, bg_color=(255, 255, 255))

        pts_gt = []
        for i in range(N_POINTS):
            cx_g = int(px_x[i]); cy_g = int(px_y[i])
            # Paste symbol (white fill ensures occlusion of lines behind)
            x1 = cx_g - HALF; y1 = cy_g - HALF
            x2 = x1 + P;      y2 = y1 + P
            # Clip to image bounds
            sx1 = max(0, -x1); sy1 = max(0, -y1)
            sx2 = P - max(0, x2 - W); sy2 = P - max(0, y2 - H)
            dx1 = max(0, x1); dy1 = max(0, y1)
            dx2 = min(W, x2); dy2 = min(H, y2)
            if dx2 > dx1 and dy2 > dy1:
                img[dy1:dy2, dx1:dx2] = sym_img[sy1:sy2, sx1:sx2]
            pts_gt.append({"px": cx_g, "py": cy_g})

        gt_series.append({
            "series_idx":     si,
            "symbol_class":   si,
            "symbol_name":    CLASS_NAMES[si],
            "color":          list(color),
            "corner_density": float(cd),
            "points":         pts_gt,
        })

    # Add light Gaussian noise
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save image
    img_path = save_dir / f"plot_{plot_idx:04d}.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Save ground truth
    gt = {
        "plot_idx":    plot_idx,
        "image_path":  str(img_path),
        "image_size":  {"w": W, "h": H},
        "n_series":    N_SERIES,
        "n_points":    N_POINTS,
        "series":      gt_series,
    }
    gt_path = gt_dir / f"gt_{plot_idx:04d}.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    return gt


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — SUBIMAGE EXTRACTION (3 SAMPLING METHODS)
# ═══════════════════════════════════════════════════════════════════════════

def extract_patch(img_gray: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Extract p×p patch centred at (cx, cy), padding with white if needed."""
    H, W = img_gray.shape
    patch = np.full((P, P), 255, dtype=np.uint8)
    x1 = cx - HALF; y1 = cy - HALF
    x2 = x1 + P;    y2 = y1 + P
    sx1 = max(0, -x1); sy1 = max(0, -y1)
    sx2 = P - max(0, x2 - W); sy2 = P - max(0, y2 - H)
    dx1 = max(0, x1); dy1 = max(0, y1)
    dx2 = min(W, x2); dy2 = min(H, y2)
    if dx2 > dx1 and dy2 > dy1:
        patch[sy1:sy2, sx1:sx2] = img_gray[dy1:dy2, dx1:dx2]
    return patch


def build_subimage_dataset(gt_list: list) -> tuple:
    """
    Build balanced subimage dataset from ground-truth list.

    Returns (patches, labels) where:
      patches : list of (P×P) uint8 arrays
      labels  : list of int (0-10 = symbol class, 11 = background)
    """
    patches, labels = [], []

    # Counters per class
    pos_count  = [0] * N_SERIES
    off_count  = [0] * N_SERIES
    bg_count   = 0

    for gt in gt_list:
        img_bgr = cv2.imread(gt["image_path"])
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W     = img_gray.shape

        # Collect all symbol centres for this plot (for background exclusion)
        all_centres = []
        for series in gt["series"]:
            for pt in series["points"]:
                all_centres.append((pt["px"], pt["py"]))

        for series in gt["series"]:
            cls = series["symbol_class"]

            for pt in series["points"]:
                cx, cy = pt["px"], pt["py"]

                # ── Method 1: Positive (centre within 3 px) ──────────────
                if pos_count[cls] < PATCHES_POS:
                    for _ in range(3):           # a few jittered positives
                        dx = random.randint(-2, 2)
                        dy = random.randint(-2, 2)
                        patch = extract_patch(img_gray, cx + dx, cy + dy)
                        patches.append(patch)
                        labels.append(cls)
                        pos_count[cls] += 1
                        if pos_count[cls] >= PATCHES_POS:
                            break

                # ── Method 2: Slightly-off (3-5 px offset → background) ──
                if off_count[cls] < PATCHES_OFF:
                    angle  = random.uniform(0, 2 * math.pi)
                    dist   = random.uniform(3, 5)
                    ox = cx + int(round(dist * math.cos(angle)))
                    oy = cy + int(round(dist * math.sin(angle)))
                    patch  = extract_patch(img_gray, ox, oy)
                    patches.append(patch)
                    labels.append(N_SERIES)      # background
                    off_count[cls] += 1

        # ── Method 3: Random background patches ──────────────────────────
        attempts = 0
        while bg_count < PATCHES_BG * len(gt_list) and attempts < 2000:
            attempts += 1
            rx = random.randint(HALF, W - HALF - 1)
            ry = random.randint(HALF, H - HALF - 1)
            # Ensure not centred on any symbol
            too_close = any(
                abs(rx - cx) < HALF and abs(ry - cy) < HALF
                for cx, cy in all_centres
            )
            if too_close:
                continue
            patch = extract_patch(img_gray, rx, ry)
            # Only keep patches with some content (not pure white)
            _, pb = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
            if np.sum(pb > 0) / (P * P) < 0.5:
                patches.append(patch)
                labels.append(N_SERIES)          # background
                bg_count += 1

    print(f"  Positive patches: {sum(pos_count)} "
          f"(per class: {[c for c in pos_count]})")
    print(f"  Slightly-off patches: {sum(off_count)}")
    print(f"  Random background patches: {bg_count}")
    print(f"  Total: {len(patches)}")
    return patches, labels


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — VIT DATASET & MODEL
# ═══════════════════════════════════════════════════════════════════════════

class SymbolDataset(Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels  = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = cv2.resize(self.patches[idx], (VIT_INPUT, VIT_INPUT))
        p3    = np.stack([patch] * 3, axis=0).astype(np.float32) / 255.0
        mean  = np.array([0.485, 0.456, 0.406])[:, None, None]
        std   = np.array([0.229, 0.224, 0.225])[:, None, None]
        p3    = (p3 - mean) / std
        return torch.tensor(p3, dtype=torch.float32), int(self.labels[idx])


def build_model() -> nn.Module:
    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=True,
        num_classes=N_CLASSES,
        img_size=VIT_INPUT,
    )
    # Freeze early blocks; fine-tune last 3 blocks + head
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(9, 12)) or \
           "head" in name or "norm" in name:
            param.requires_grad = True
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train(n_plots: int = N_PLOTS):
    print("=" * 60)
    print("STEP 1 — Generating synthetic plots")
    print("=" * 60)
    gt_list = []
    for i in range(n_plots):
        gt = generate_synthetic_plot(i, SYNTH_DIR, GT_DIR)
        gt_list.append(gt)
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{n_plots} plots")
    print(f"  Done. Saved to {SYNTH_DIR}")

    print("\n" + "=" * 60)
    print("STEP 2 — Extracting subimages (3 sampling methods)")
    print("=" * 60)
    patches, labels = build_subimage_dataset(gt_list)

    print("\n" + "=" * 60)
    print("STEP 3 — Training ViT")
    print("=" * 60)
    X_tr, X_va, y_tr, y_va = train_test_split(
        patches, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_dl = DataLoader(SymbolDataset(X_tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(SymbolDataset(X_va, y_va),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model  = build_model().to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Slightly upweight symbol classes vs background
    weights = torch.ones(N_CLASSES, device=device)
    weights[-1] = 0.5                            # background weight
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state   = None
    train_accs, val_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        tc = tt = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tc += (out.argmax(1) == yb).sum().item()
            tt += len(xb)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(xb)

        ta = tc / tt; va = vc / vt
        train_accs.append(ta); val_accs.append(va)

        if va > best_val_acc:
            best_val_acc = va
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS}:  "
                  f"train={ta:.4f}  val={va:.4f}  best={best_val_acc:.4f}")

    # Save best weights
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), str(MODEL_PATH))
    print(f"\n  Best val_acc = {best_val_acc:.4f}")
    print(f"  Weights saved → {MODEL_PATH}")

    # Confusion matrix
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            all_preds.extend(model(xb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    cm = confusion_matrix(all_true, all_preds)

    # Training curve plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_accs, label="Train"); axes[0].plot(val_accs, label="Val")
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
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_title(f"Confusion Matrix  (val_acc={best_val_acc:.3f})")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    fig.savefig(str(WORK_DIR / "training_results.png"), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Training plot saved → {WORK_DIR / 'training_results.png'}")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 — INFERENCE
#  Input: path to a plotting-area image file (the full image IS the plot area)
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_path: str = str(MODEL_PATH)) -> nn.Module:
    """Load trained ViT weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model()
    state  = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def detect(image_path: str,
           model_path: str = str(MODEL_PATH),
           conf_thresh: float = CONF_THRESH,
           stride: int = STRIDE,
           nms_radius: float = NMS_RADIUS,
           save_vis: bool = True) -> list:
    """
    Detect symbols in a plotting-area image.

    Parameters
    ----------
    image_path : str
        Path to the plotting-area image. The entire image is treated as the
        plotting area — no axis/region detection is performed.
    model_path : str
        Path to trained ViT weights (.pth).
    conf_thresh : float
        Minimum softmax confidence to accept a detection.
    stride : int
        Sliding window stride in pixels.
    nms_radius : float
        NMS suppression radius in pixels.
    save_vis : bool
        If True, save a visualisation PNG next to the input image.

    Returns
    -------
    list of dict  — each dict has keys:
        cx, cy        : pixel centre of detected symbol
        class_idx     : int (0-10)
        class_name    : str
        confidence    : float
    """
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = load_model(model_path)

    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W     = img_gray.shape

    print(f"[detect] Image: {W}×{H} px  |  p={P}  stride={stride}  "
          f"conf≥{conf_thresh}  NMS_r={nms_radius:.1f}")

    # ── Sliding window ────────────────────────────────────────────────────
    raw_dets = []
    MIN_PIX  = int(P * P * 0.04)

    # Batch windows for speed
    batch_coords = []
    batch_patches = []

    for cy in range(0, H, stride):
        for cx in range(0, W, stride):
            patch = extract_patch(img_gray, cx, cy)
            _, pb = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
            if np.sum(pb > 0) < MIN_PIX:
                continue
            batch_coords.append((cx, cy))
            resized = cv2.resize(patch, (VIT_INPUT, VIT_INPUT))
            p3 = np.stack([resized]*3, axis=0).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std  = np.array([0.229, 0.224, 0.225])[:, None, None]
            p3   = (p3 - mean) / std
            batch_patches.append(p3)

            if len(batch_patches) == 256:
                t = torch.tensor(np.stack(batch_patches), dtype=torch.float32).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(t), dim=1).cpu().numpy()
                for (bx, by), prob in zip(batch_coords, probs):
                    ci = prob.argmax()
                    if ci < N_SERIES and prob[ci] >= conf_thresh:
                        raw_dets.append((bx, by, int(ci), float(prob[ci])))
                batch_coords.clear(); batch_patches.clear()

    # Flush remainder
    if batch_patches:
        t = torch.tensor(np.stack(batch_patches), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1).cpu().numpy()
        for (bx, by), prob in zip(batch_coords, probs):
            ci = prob.argmax()
            if ci < N_SERIES and prob[ci] >= conf_thresh:
                raw_dets.append((bx, by, int(ci), float(prob[ci])))

    print(f"  Raw detections: {len(raw_dets)}")

    # ── NMS ───────────────────────────────────────────────────────────────
    raw_dets.sort(key=lambda d: -d[3])
    kept = []; suppressed = set()
    for i, det in enumerate(raw_dets):
        if i in suppressed:
            continue
        kept.append(det)
        for j in range(i + 1, len(raw_dets)):
            if j in suppressed:
                continue
            dist = math.sqrt((det[0] - raw_dets[j][0])**2 +
                             (det[1] - raw_dets[j][1])**2)
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

    # ── Visualisation ─────────────────────────────────────────────────────
    if save_vis:
        _colors = plt.cm.tab20.colors
        sym_colors = {CLASS_NAMES[i]: tuple(int(c*255) for c in _colors[i % 20])
                      for i in range(N_SERIES)}

        vis = img_rgb.copy()
        for m in markers:
            cx, cy = m["cx"], m["cy"]
            col    = sym_colors[m["class_name"]]
            # Draw circle marker in red (per preference)
            cv2.circle(vis, (cx, cy), HALF, (220, 30, 30), 2)
            # Confidence label
            cv2.putText(vis, f"{m['confidence']:.2f}",
                        (cx + HALF + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 30, 30), 1)

        vis_path = str(Path(image_path).with_suffix("")) + "_detections.png"
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  Visualisation saved → {vis_path}")

    return markers


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT Symbol Detector")
    parser.add_argument("--mode",   choices=["train", "detect"], required=True)
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to plotting-area image (--mode detect)")
    parser.add_argument("--model",  type=str, default=str(MODEL_PATH),
                        help="Path to model weights (.pth)")
    parser.add_argument("--plots",  type=int, default=N_PLOTS,
                        help="Number of synthetic plots to generate (--mode train)")
    parser.add_argument("--conf",   type=float, default=CONF_THRESH,
                        help="Confidence threshold for detection")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help="Sliding window stride")
    args = parser.parse_args()

    if args.mode == "train":
        train(n_plots=args.plots)

    elif args.mode == "detect":
        if args.image is None:
            parser.error("--image is required for --mode detect")
        results = detect(
            image_path=args.image,
            model_path=args.model,
            conf_thresh=args.conf,
            stride=args.stride,
        )
        print("\nDetected markers:")
        for m in results:
            print(f"  ({m['cx']:4d}, {m['cy']:4d})  "
                  f"{m['class_name']:<22s}  conf={m['confidence']:.3f}")
