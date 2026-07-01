"""
chart_twobranch_train.py
========================
Working directory : <project>/src/
Weights directory : <project>/weights/   (deeplsd_md.tar for LSD branch init)
Checkpoints       : <project>/weights/twobranch/

Two-Branch segment detection model that fuses:
  Branch 1 — Original grayscale image   (VGGUNet backbone, optionally init from DeepLSD)
  Branch 2 — LSD rasterised maps        (heatmap + angle map + distance field, 3ch)

Both branches are fused and decoded into:
  • heatmap  (B,1,H,W)  — per-pixel line probability
  • angle    (B,2,H,W)  — (cos θ, sin θ) per pixel

Training uses the synthetic plots produced by chart_marker_detector_v3.py.
GT segments are loaded from gt_*.json files (same format as chart_deeplsd_eval.py).

OUTPUT
------
  data/twobranch_train/
    per_epoch_metrics.csv      — train/val loss + P/R/F1 per epoch
    best_model.pt              — checkpoint with best val F1
    last_model.pt              — checkpoint after final epoch
    images/epoch_NNN/          — annotated val images (GT green, pred red)

USAGE
-----
  # Train from scratch:
  python chart_twobranch_train.py

  # Resume from checkpoint:
  python chart_twobranch_train.py --resume weights/twobranch/last_model.pt

  # Evaluate a saved checkpoint (no training):
  python chart_twobranch_train.py --eval-only --resume weights/twobranch/best_model.pt

  # Quick smoke-test on 20 plots:
  python chart_twobranch_train.py --n-plots 20 --epochs 2

REQUIREMENTS
------------
  pip install torch torchvision opencv-python matplotlib numpy
"""

from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time, warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
_SRC_DIR      = Path(__file__).parent
SYNTH_DIR     = _SRC_DIR / ".." / "data" / "synthetic_plots"
WEIGHTS_DIR   = _SRC_DIR / ".." / "weights"
CKPT_DIR      = WEIGHTS_DIR / "twobranch"
TRAIN_OUT_DIR = _SRC_DIR / ".." / "data" / "twobranch_train"

DEFAULT_DEEPLSD_WEIGHT = "deeplsd_md.tar"

# ══════════════════════════════════════════════════════════════════════════════
#  HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE        = 256       # crops are resized to this before model input
BATCH_SIZE      = 8
LR              = 1e-4
EPOCHS          = 50
VAL_SPLIT       = 0.15      # fraction of data used for validation
HEATMAP_RADIUS  = 2         # px radius when rasterising GT segments to heatmap
MIN_SEG_LEN     = 5.0       # px — ignore GT segments shorter than this
MATCH_DIST      = 10.0      # px — midpoint distance tolerance for TP matching
MATCH_ANGLE_DEG = 15.0      # degrees — angular tolerance for TP matching
LINE_NEIGHBORHOOD = 5       # DeepLSD normalise_df parameter (kept for compat.)
POS_WEIGHT      = 10.0      # BCE pos_weight: compensates for sparse line pixels
SEED            = 42

# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS BAR  (identical helper to chart_deeplsd_eval.py)
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
    line = (f"\r  {prefix}[{bar}] {done:>{len(str(total))}}/{total}"
            f"  {frac*100:5.1f}%  ETA {eta}")
    print(line, end="", flush=True)
    if done == total:
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  LSD RASTERISER  — converts OpenCV LSD segments → 3-channel map
# ══════════════════════════════════════════════════════════════════════════════
def lsd_to_maps(h: int, w: int, lines: np.ndarray | None) -> np.ndarray:
    """
    Convert raw LSD line segments to a (3, H, W) float32 array:
      ch0 — binary heatmap  (1 on detected line pixels, else 0)
      ch1 — angle map       (line orientation in [0, π], mapped to [0, 1])
      ch2 — distance field  (soft exp(-dist/10) — high near lines)

    lines : (N, 4) float array x1,y1,x2,y2  OR None
    """
    heatmap  = np.zeros((h, w), dtype=np.float32)
    anglemap = np.zeros((h, w), dtype=np.float32)

    if lines is not None and len(lines) > 0:
        for seg in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(int, seg[:4])
            cv2.line(heatmap, (x1, y1), (x2, y2), 1.0, thickness=1)
            angle_norm = (math.atan2(y2 - y1, x2 - x1) % math.pi) / math.pi
            cv2.line(anglemap, (x1, y1), (x2, y2), float(angle_norm), thickness=1)

    # Distance field: high (≈1) near lines, decays away
    bin_inv  = (1 - heatmap.astype(np.uint8))
    dist     = cv2.distanceTransform(bin_inv, cv2.DIST_L2, 3)
    distmap  = np.exp(-dist / 10.0).astype(np.float32)

    return np.stack([heatmap, anglemap, distmap], axis=0)   # (3, H, W)


def run_lsd(gray_u8: np.ndarray) -> np.ndarray | None:
    """Run OpenCV LSD on a uint8 grayscale image. Returns (N,4) array or None."""
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
    lines, *_ = lsd.detect(gray_u8)
    if lines is None or len(lines) == 0:
        return None
    return lines.reshape(-1, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  GT HEATMAP BUILDER  — rasterises GT segment list to (1,H,W) ground truth
# ══════════════════════════════════════════════════════════════════════════════
def build_gt_maps(h: int, w: int, segs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ground-truth heatmap and angle maps from segment dicts.

    Returns:
      gt_heat  : (1, H, W) float32 in [0, 1]
      gt_angle : (2, H, W) float32 — (cos θ, sin θ) per pixel (on line pixels)
    """
    gt_heat  = np.zeros((h, w), dtype=np.float32)
    gt_cos   = np.zeros((h, w), dtype=np.float32)
    gt_sin   = np.zeros((h, w), dtype=np.float32)

    for s in segs:
        x1, y1, x2, y2 = int(round(s["x1"])), int(round(s["y1"])), \
                          int(round(s["x2"])), int(round(s["y2"]))
        cv2.line(gt_heat, (x1, y1), (x2, y2), 1.0, thickness=HEATMAP_RADIUS * 2 + 1)
        c = math.cos(s["angle_rad"])
        si = math.sin(s["angle_rad"])
        cv2.line(gt_cos,  (x1, y1), (x2, y2), float(c),  thickness=HEATMAP_RADIUS * 2 + 1)
        cv2.line(gt_sin,  (x1, y1), (x2, y2), float(si), thickness=HEATMAP_RADIUS * 2 + 1)

    gt_heat  = np.clip(gt_heat, 0, 1)[None]            # (1, H, W)
    gt_angle = np.stack([gt_cos, gt_sin], axis=0)       # (2, H, W)
    return gt_heat, gt_angle


# ══════════════════════════════════════════════════════════════════════════════
#  GT SEGMENT LOADING  (identical logic to chart_deeplsd_eval.py)
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
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class ChartSegmentDataset(Dataset):
    """
    Loads synthetic chart images + GT JSON files.

    Each sample returns:
      image   : (1, IMG_SIZE, IMG_SIZE) float32 grayscale in [0, 1]
      lsd_map : (3, IMG_SIZE, IMG_SIZE) float32 — LSD rasterised maps
      gt_heat : (1, IMG_SIZE, IMG_SIZE) float32 — GT line heatmap
      gt_angle: (2, IMG_SIZE, IMG_SIZE) float32 — GT (cos θ, sin θ)
      meta    : dict with plot_id, gt_segs (in resized coords), scale_x, scale_y
    """

    def __init__(self, gt_files: list[Path], augment: bool = False):
        self.gt_files = gt_files
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, idx: int) -> dict:
        gt_path = self.gt_files[idx]
        img_path = (gt_path.parent /
                    gt_path.name.replace("gt_", "plot_").replace(".json", ".png"))

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Return blank sample on read failure
            z = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            return {"image": torch.zeros(1, IMG_SIZE, IMG_SIZE),
                    "lsd_map": torch.zeros(3, IMG_SIZE, IMG_SIZE),
                    "gt_heat": torch.zeros(1, IMG_SIZE, IMG_SIZE),
                    "gt_angle": torch.zeros(2, IMG_SIZE, IMG_SIZE),
                    "meta": {"plot_id": gt_path.stem, "gt_segs": [],
                             "scale_x": 1.0, "scale_y": 1.0}}

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        pa = gt["pa"]
        x0, y0, x1, y1 = pa["x0"], pa["y0"], pa["x1"], pa["y1"]

        crop_bgr = img_bgr[y0:y1, x0:x1]
        if crop_bgr.size == 0:
            crop_bgr = img_bgr

        oh, ow = crop_bgr.shape[:2]
        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # ── Resize to fixed size ────────────────────────────────────────────
        scale_x = IMG_SIZE / max(ow, 1)
        scale_y = IMG_SIZE / max(oh, 1)
        gray_rs  = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE),
                               interpolation=cv2.INTER_LINEAR)

        # ── Augmentation (horizontal flip only — safe for line symmetry) ───
        if self.augment and random.random() < 0.5:
            gray_rs = cv2.flip(gray_rs, 1)
            flip = True
        else:
            flip = False

        # ── LSD on resized image ────────────────────────────────────────────
        lsd_lines = run_lsd(gray_rs)
        lsd_map   = lsd_to_maps(IMG_SIZE, IMG_SIZE, lsd_lines)  # (3, H, W)

        # ── GT segments (scale + optionally flip) ──────────────────────────
        segs_orig = load_gt_segments(gt_path, x0, y0)
        segs_scaled = []
        for s in segs_orig:
            sx1 = s["x1"] * scale_x
            sy1 = s["y1"] * scale_y
            sx2 = s["x2"] * scale_x
            sy2 = s["y2"] * scale_y
            if flip:
                sx1 = IMG_SIZE - 1 - sx1
                sx2 = IMG_SIZE - 1 - sx2
            length = math.hypot(sx2 - sx1, sy2 - sy1)
            if length < 1.0:
                continue
            segs_scaled.append({
                "x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2,
                "length": length,
                "angle_rad": math.atan2(sy2 - sy1, sx2 - sx1) % math.pi,
            })

        gt_heat, gt_angle = build_gt_maps(IMG_SIZE, IMG_SIZE, segs_scaled)

        # ── Normalise image to [0, 1] ──────────────────────────────────────
        image_t = torch.tensor(gray_rs, dtype=torch.float32)[None] / 255.0

        return {
            "image":    image_t,
            "lsd_map":  torch.tensor(lsd_map,   dtype=torch.float32),
            "gt_heat":  torch.tensor(gt_heat,   dtype=torch.float32),
            "gt_angle": torch.tensor(gt_angle,  dtype=torch.float32),
            "meta": {
                "plot_id":  gt_path.stem.replace("gt_", ""),
                "gt_segs":  segs_scaled,
                "scale_x":  scale_x,
                "scale_y":  scale_y,
            },
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, keep meta as list."""
    return {
        "image":    torch.stack([b["image"]    for b in batch]),
        "lsd_map":  torch.stack([b["lsd_map"]  for b in batch]),
        "gt_heat":  torch.stack([b["gt_heat"]  for b in batch]),
        "gt_angle": torch.stack([b["gt_angle"] for b in batch]),
        "meta":     [b["meta"] for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VGGUNet BACKBONE  (identical to chart_deeplsd_eval.py — weight-compatible)
# ══════════════════════════════════════════════════════════════════════════════
class _VGGUNet(nn.Module):
    """
    VGG-style U-Net backbone.
    Input:  (B, in_ch, H, W)
    Output: (B, 64, H, W)
    """
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def _enc(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co),
                nn.Conv2d(co, co, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co),
            )

        def _dec(ci, cmid, co):
            return nn.Sequential(
                nn.Conv2d(ci,   cmid, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(cmid),
                nn.Conv2d(cmid, co,   3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co),
            )

        self.block1   = _enc(in_ch, 64)
        self.block2   = _enc(64,   128)
        self.block3   = _enc(128,  256)
        self.block4   = _enc(256,  512)
        self.deblock4 = _dec(512,       256, 256)
        self.deblock3 = _dec(256 + 256, 256, 128)
        self.deblock2 = _dec(128 + 128, 128,  64)
        self.deblock1 = _dec( 64 +  64,  64,  64)

    def forward(self, x):
        f1  = self.block1(x)
        f2  = self.block2(self.pool(f1))
        f3  = self.block3(self.pool(f2))
        f4  = self.block4(self.pool(f3))
        out = self.deblock4(f4)
        out = self.deblock3(torch.cat([
            F.interpolate(out, f3.shape[2:], mode="bilinear", align_corners=False), f3], dim=1))
        out = self.deblock2(torch.cat([
            F.interpolate(out, f2.shape[2:], mode="bilinear", align_corners=False), f2], dim=1))
        out = self.deblock1(torch.cat([
            F.interpolate(out, f1.shape[2:], mode="bilinear", align_corners=False), f1], dim=1))
        return out   # (B, 64, H, W)


# ══════════════════════════════════════════════════════════════════════════════
#  TWO-BRANCH MODEL
# ══════════════════════════════════════════════════════════════════════════════
class TwoBranchSegNet(nn.Module):
    """
    Two-Branch line segment detector.

    Branch 1 — Image branch:
        VGGUNet(in_ch=1) → (B, 64, H, W)
        Optionally initialised from pretrained DeepLSD backbone weights.

    Branch 2 — LSD branch:
        VGGUNet(in_ch=3) → (B, 64, H, W)
        Input: [heatmap | angle_map | distance_field] from OpenCV LSD

    Fusion:
        Concatenate → (B, 128, H, W) → 1×1 conv → (B, 64, H, W)

    Heads:
        heatmap_head  → (B, 1, H, W)  sigmoid  — per-pixel line prob
        angle_head    → (B, 2, H, W)  tanh     — (cos θ, sin θ)
    """

    def __init__(self):
        super().__init__()
        # Branch 1: grayscale image
        self.img_branch = _VGGUNet(in_ch=1)
        # Branch 2: LSD rasterised maps (3ch)
        self.lsd_branch = _VGGUNet(in_ch=3)
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
        )
        # Output heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 1,  1),
            nn.Sigmoid(),
        )
        self.angle_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.Conv2d(64, 2,  1),
            nn.Tanh(),    # (cos, sin) both in [-1, 1]
        )

    def forward(self, image: torch.Tensor,
                lsd_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        image   : (B, 1, H, W) float32 in [0, 1]
        lsd_map : (B, 3, H, W) float32 — LSD rasterised maps

        Returns:
          heatmap : (B, 1, H, W) in [0, 1]
          angle   : (B, 2, H, W) in [-1, 1]  — (cos θ, sin θ)
        """
        img_feat = self.img_branch(image)       # (B, 64, H, W)
        lsd_feat = self.lsd_branch(lsd_map)     # (B, 64, H, W)
        fused    = self.fusion(
            torch.cat([img_feat, lsd_feat], dim=1))   # (B, 64, H, W)
        heatmap  = self.heatmap_head(fused)     # (B, 1, H, W)
        angle    = self.angle_head(fused)       # (B, 2, H, W)
        return heatmap, angle


# ══════════════════════════════════════════════════════════════════════════════
#  PRETRAINED WEIGHT LOADER  (DeepLSD → img_branch only)
# ══════════════════════════════════════════════════════════════════════════════
def load_deeplsd_into_branch(net: TwoBranchSegNet,
                              weight_file: str = DEFAULT_DEEPLSD_WEIGHT) -> None:
    """
    Copy DeepLSD backbone weights into net.img_branch where key names match.
    Keys in the .tar follow 'backbone.*' which maps to img_branch.* here.
    """
    ckpt_path = WEIGHTS_DIR / weight_file
    if not ckpt_path.exists():
        print(f"  [WARN] DeepLSD weights not found at {ckpt_path} — "
              "img_branch will train from scratch.")
        return

    ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    # Remap 'backbone.*' → 'img_branch.*'
    remapped = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            new_k = k.replace("backbone.", "img_branch.", 1)
            remapped[new_k] = v

    missing, unexpected = net.load_state_dict(remapped, strict=False)
    loaded = len(remapped) - len([k for k in missing if k in remapped])
    print(f"  Loaded {len(remapped)} DeepLSD backbone keys → img_branch  "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def total_loss(pred_heat: torch.Tensor,
               pred_angle: torch.Tensor,
               gt_heat: torch.Tensor,
               gt_angle: torch.Tensor,
               pos_weight: float = POS_WEIGHT) -> tuple[torch.Tensor, dict]:
    """
    pred_heat  : (B, 1, H, W) in [0,1]
    pred_angle : (B, 2, H, W) tanh output  (cos, sin)
    gt_heat    : (B, 1, H, W) in {0, 1}
    gt_angle   : (B, 2, H, W) (cos θ, sin θ) — valid only where gt_heat==1

    Returns scalar loss + dict of component losses for logging.
    """
    # ── Heatmap loss (weighted BCE) ─────────────────────────────────────────
    pw = torch.tensor([pos_weight], device=pred_heat.device)
    bce = F.binary_cross_entropy(
        pred_heat, gt_heat,
        weight=(gt_heat * (pos_weight - 1) + 1),   # manual pos_weight
        reduction="mean",
    )

    # ── Angle loss (cosine similarity on line pixels only) ──────────────────
    mask = (gt_heat[:, 0] > 0.5)   # (B, H, W)
    if mask.any():
        # Normalise predictions to unit circle
        pred_norm = F.normalize(pred_angle, dim=1)  # (B, 2, H, W)
        gt_norm   = F.normalize(gt_angle,   dim=1)  # (B, 2, H, W)
        cos_sim   = (pred_norm * gt_norm).sum(dim=1)  # (B, H, W)  in [-1,1]
        ang_loss  = (1.0 - cos_sim)[mask].mean()
    else:
        ang_loss = pred_angle.sum() * 0.0   # zero, keeps gradient graph

    loss = bce + 0.5 * ang_loss
    return loss, {"bce": bce.item(), "angle": ang_loss.item()
                  if isinstance(ang_loss, torch.Tensor) else ang_loss,
                  "total": loss.item()}


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT EXTRACTION  — heatmap + angle → line segments (NMS + grouping)
# ══════════════════════════════════════════════════════════════════════════════
def heatmap_to_segments(heatmap: np.ndarray,
                        angle_map: np.ndarray,
                        thresh: float = 0.35,
                        min_len: float = MIN_SEG_LEN) -> np.ndarray:
    """
    Convert predicted heatmap + angle map to line segment array.

    heatmap   : (H, W) float32 in [0, 1]
    angle_map : (2, H, W) float32  (cos θ, sin θ)

    Returns (N, 2, 2) float32 array: [[x1,y1],[x2,y2]]
    """
    h, w = heatmap.shape
    # Threshold
    binary = (heatmap > thresh).astype(np.uint8)

    # Skeletonise to thin predicted lines
    try:
        skel = cv2.ximgproc.thinning(binary * 255)
        binary = (skel > 0).astype(np.uint8)
    except AttributeError:
        pass  # ximgproc not available — skip thinning

    # Connected components → fit line per component
    n_labels, labels = cv2.connectedComponents(binary)
    segments = []

    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if len(xs) < 3:
            continue

        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        # PCA to find dominant direction
        mean = pts.mean(axis=0)
        centered = pts - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]   # principal axis

        # Project all points onto the direction
        proj = centered @ direction
        p_min, p_max = proj.min(), proj.max()
        if (p_max - p_min) < min_len:
            continue

        x1 = float(mean[0] + direction[0] * p_min)
        y1 = float(mean[1] + direction[1] * p_min)
        x2 = float(mean[0] + direction[0] * p_max)
        y2 = float(mean[1] + direction[1] * p_max)

        # Clamp to image bounds
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w - 1), x2))
        y2 = max(0.0, min(float(h - 1), y2))

        segments.append([[x1, y1], [x2, y2]])

    if not segments:
        return np.zeros((0, 2, 2), dtype=np.float32)
    return np.array(segments, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENT MATCHING  (identical to chart_deeplsd_eval.py)
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
#  VISUALISATION  (mirrors chart_deeplsd_eval.py draw_segments)
# ══════════════════════════════════════════════════════════════════════════════
def draw_segments(gray_u8: np.ndarray,
                  gt_segs: list[dict],
                  pred_lines: np.ndarray,
                  metrics: dict) -> np.ndarray:
    vis = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
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
#  ONE EPOCH: TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(net: TwoBranchSegNet,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int) -> dict:
    net.train()
    total = len(loader)
    sum_loss = sum_bce = sum_ang = 0.0
    t0 = time.time()

    for i, batch in enumerate(loader):
        image    = batch["image"].to(device)
        lsd_map  = batch["lsd_map"].to(device)
        gt_heat  = batch["gt_heat"].to(device)
        gt_angle = batch["gt_angle"].to(device)

        pred_heat, pred_angle = net(image, lsd_map)
        loss, parts = total_loss(pred_heat, pred_angle, gt_heat, gt_angle)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

        sum_loss += parts["total"]
        sum_bce  += parts["bce"]
        sum_ang  += parts["angle"]

        _pbar(i + 1, total, t0,
              prefix=f"  [Epoch {epoch}/{total_epochs}] Train: ")

    n = max(total, 1)
    return {"loss": sum_loss / n, "bce": sum_bce / n, "angle": sum_ang / n}


# ══════════════════════════════════════════════════════════════════════════════
#  ONE EPOCH: VALIDATE
# ══════════════════════════════════════════════════════════════════════════════
def validate(net: TwoBranchSegNet,
             loader: DataLoader,
             device: torch.device,
             epoch: int,
             total_epochs: int,
             save_images: bool = True,
             img_out_dir: Path | None = None) -> dict:
    net.eval()
    total = len(loader)
    sum_loss = sum_bce = sum_ang = 0.0
    all_metrics: list[dict] = []
    t0 = time.time()

    if save_images and img_out_dir is not None:
        img_out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            image    = batch["image"].to(device)
            lsd_map  = batch["lsd_map"].to(device)
            gt_heat  = batch["gt_heat"].to(device)
            gt_angle = batch["gt_angle"].to(device)
            metas    = batch["meta"]

            pred_heat, pred_angle = net(image, lsd_map)
            loss, parts = total_loss(pred_heat, pred_angle, gt_heat, gt_angle)

            sum_loss += parts["total"]
            sum_bce  += parts["bce"]
            sum_ang  += parts["angle"]

            # ── Per-sample metrics ────────────────────────────────────────────
            # metas is a list of per-sample dicts (from collate_fn)
            for b in range(image.size(0)):
                heat_np    = pred_heat[b, 0].cpu().numpy()
                angle_np   = pred_angle[b].cpu().numpy()    # (2, H, W)
                pred_lines = heatmap_to_segments(heat_np, angle_np)
                gt_segs    = metas[b]["gt_segs"]
                metrics    = match_segments(pred_lines, gt_segs)
                metrics["plot_id"]     = metas[b]["plot_id"]
                metrics["n_gt_segs"]   = len(gt_segs)
                metrics["n_pred_segs"] = len(pred_lines)
                all_metrics.append(metrics)

                # Save first image of each batch for visualisation
                if save_images and img_out_dir is not None and b == 0:
                    gray_u8  = (image[b, 0].cpu().numpy() * 255).astype(np.uint8)
                    vis      = draw_segments(gray_u8, gt_segs, pred_lines, metrics)
                    # Also show GT heatmap side-by-side
                    gt_h_u8  = (gt_heat[b, 0].cpu().numpy() * 255).astype(np.uint8)
                    gt_h_bgr = cv2.cvtColor(gt_h_u8, cv2.COLOR_GRAY2BGR)
                    pred_h_u8  = (heat_np * 255).astype(np.uint8)
                    pred_h_bgr = cv2.cvtColor(pred_h_u8, cv2.COLOR_GRAY2BGR)
                    panel = np.hstack([
                        cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR),
                        gt_h_bgr, pred_h_bgr, vis
                    ])
                    try:
                        fname = f"val_{metrics['plot_id']}_b{i:04d}.png"
                        cv2.imwrite(str(img_out_dir / fname), panel)
                    except Exception:
                        pass

            _pbar(i + 1, total, t0,
                  prefix=f"  [Epoch {epoch}/{total_epochs}] Val:   ")

    n = max(total, 1)
    # Aggregate metrics
    if all_metrics:
        total_tp = sum(m["TP"] for m in all_metrics)
        total_fp = sum(m["FP"] for m in all_metrics)
        total_fn = sum(m["FN"] for m in all_metrics)
        micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)
    else:
        micro_p = micro_r = micro_f1 = 0.0

    return {
        "loss":      sum_loss / n,
        "bce":       sum_bce  / n,
        "angle":     sum_ang  / n,
        "precision": micro_p,
        "recall":    micro_r,
        "F1":        micro_f1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(net: TwoBranchSegNet,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    metrics: dict,
                    path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics":   metrics,
    }, str(path))


def load_checkpoint(net: TwoBranchSegNet,
                    optimizer: torch.optim.Optimizer | None,
                    path: Path,
                    device: torch.device) -> tuple[int, dict]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    epoch   = ckpt.get("epoch", 0)
    metrics = ckpt.get("metrics", {})
    print(f"  Resumed from {path.name}  (epoch {epoch}, "
          f"val F1={metrics.get('F1', 0):.4f})")
    return epoch, metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train(n_plots: int | None = None,
          epochs: int = EPOCHS,
          resume: str | None = None,
          init_deeplsd: bool = True,
          eval_only: bool = False,
          save_images: bool = True) -> None:

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Data ─────────────────────────────────────────────────────────────────
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    gt_files = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not gt_files:
        print(f"\n[ERROR] No GT files found in {SYNTH_DIR}")
        sys.exit(1)
    if n_plots is not None:
        gt_files = gt_files[:n_plots]

    random.shuffle(gt_files)
    n_val   = max(1, int(len(gt_files) * VAL_SPLIT))
    val_files   = gt_files[:n_val]
    train_files = gt_files[n_val:]

    print(f"\n  Dataset : {len(train_files)} train  |  {len(val_files)} val")

    train_ds = ChartSegmentDataset(train_files, augment=True)
    val_ds   = ChartSegmentDataset(val_files,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              drop_last=len(train_ds) > BATCH_SIZE)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = TwoBranchSegNet().to(device)
    print(f"  Device  : {device}")
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Params  : {n_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    start_epoch = 1
    best_f1     = 0.0

    # ── Optional: init img_branch from DeepLSD ────────────────────────────
    if init_deeplsd and resume is None:
        load_deeplsd_into_branch(net)

    # ── Optional: resume ──────────────────────────────────────────────────
    if resume:
        rpath = Path(resume)
        if rpath.exists():
            start_epoch, prev_metrics = load_checkpoint(
                net, optimizer, rpath, device)
            best_f1     = prev_metrics.get("F1", 0.0)
            start_epoch += 1
        else:
            print(f"  [WARN] Checkpoint not found: {resume} — starting fresh.")

    # ── Output dirs & CSV ─────────────────────────────────────────────────
    TRAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    epoch_csv_path = TRAIN_OUT_DIR / "per_epoch_metrics.csv"
    csv_fields = ["epoch", "train_loss", "train_bce", "train_angle",
                  "val_loss", "val_bce", "val_angle",
                  "val_precision", "val_recall", "val_F1", "lr"]

    csv_mode = "a" if resume and epoch_csv_path.exists() else "w"

    # ── Eval-only mode ────────────────────────────────────────────────────
    if eval_only:
        print("\n  Running validation only …")
        img_dir = TRAIN_OUT_DIR / "images" / "eval"
        val_met = validate(net, val_loader, device,
                           epoch=0, total_epochs=0,
                           save_images=save_images, img_out_dir=img_dir)
        print(f"\n  Val Loss={val_met['loss']:.4f}  "
              f"P={val_met['precision']:.4f}  R={val_met['recall']:.4f}  "
              f"F1={val_met['F1']:.4f}")
        return

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n  Training for {epochs} epochs …\n")

    with open(epoch_csv_path, csv_mode, newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        if csv_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, start_epoch + epochs):
            # Train
            train_met = train_one_epoch(
                net, train_loader, optimizer, device, epoch,
                start_epoch + epochs - 1)

            # Validate
            img_dir = (TRAIN_OUT_DIR / "images" / f"epoch_{epoch:03d}"
                       if save_images else None)
            val_met = validate(
                net, val_loader, device, epoch,
                start_epoch + epochs - 1,
                save_images=save_images, img_out_dir=img_dir)

            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step(val_met["F1"])

            # ── Print summary line ────────────────────────────────────────
            print(f"\n  Epoch {epoch:3d}  "
                  f"train_loss={train_met['loss']:.4f}  "
                  f"val_loss={val_met['loss']:.4f}  "
                  f"P={val_met['precision']:.4f}  "
                  f"R={val_met['recall']:.4f}  "
                  f"F1={val_met['F1']:.4f}  "
                  f"lr={lr_now:.2e}")

            # ── CSV logging ───────────────────────────────────────────────
            try:
                writer.writerow({
                    "epoch":          epoch,
                    "train_loss":     f"{train_met['loss']:.6f}",
                    "train_bce":      f"{train_met['bce']:.6f}",
                    "train_angle":    f"{train_met['angle']:.6f}",
                    "val_loss":       f"{val_met['loss']:.6f}",
                    "val_bce":        f"{val_met['bce']:.6f}",
                    "val_angle":      f"{val_met['angle']:.6f}",
                    "val_precision":  f"{val_met['precision']:.6f}",
                    "val_recall":     f"{val_met['recall']:.6f}",
                    "val_F1":         f"{val_met['F1']:.6f}",
                    "lr":             f"{lr_now:.2e}",
                })
                csvf.flush()
            except PermissionError:
                pass

            # ── Save checkpoints ──────────────────────────────────────────
            last_path = CKPT_DIR / "last_model.pt"
            save_checkpoint(net, optimizer, epoch, val_met, last_path)

            if val_met["F1"] > best_f1:
                best_f1 = val_met["F1"]
                best_path = CKPT_DIR / "best_model.pt"
                save_checkpoint(net, optimizer, epoch, val_met, best_path)
                print(f"  ★  New best val F1={best_f1:.4f}  → {best_path}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n  Training complete.")
    print(f"  Best val F1 : {best_f1:.4f}")
    print(f"  Best model  → {CKPT_DIR / 'best_model.pt'}")
    print(f"  Last model  → {CKPT_DIR / 'last_model.pt'}")
    print(f"  Epoch CSV   → {epoch_csv_path}")
    if save_images:
        print(f"  Val images  → {TRAIN_OUT_DIR / 'images'}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Two-Branch segment detector on synthetic charts.")
    parser.add_argument("--n-plots", type=int, default=None,
                        help="Limit total plots used (default: all).")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS}).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--no-deeplsd-init", action="store_true",
                        help="Do NOT initialise img_branch from DeepLSD weights.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, run validation only (requires --resume).")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip saving annotated validation images.")
    args = parser.parse_args()

    train(
        n_plots     = args.n_plots,
        epochs      = args.epochs,
        resume      = args.resume,
        init_deeplsd= not args.no_deeplsd_init,
        eval_only   = args.eval_only,
        save_images = not args.no_images,
    )
