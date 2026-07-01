"""
chart_hawp_detector.py
======================
HAWP-style (Holistic Attraction Wire-frame Parser) line segment detector
trained and validated on the synthetic chart plots produced by
chart_marker_detector_v3.py.

Architecture
------------
The model follows the HAWP pipeline:
  1. Stacked-Hourglass (or lightweight CNN) backbone → 256-channel feature map
  2. Two prediction heads:
       - Junction heatmap head  : (H, W)  – sigmoid probability of a junction at each pixel
       - Junction offset head   : (2,H,W) – sub-pixel offset from pixel centre to junction
       - Attraction field head  : (2,H,W) – unit vector pointing from each pixel toward
                                            the nearest line segment (HAT field)
       - Distance field head    : (H, W)  – normalised distance to nearest line segment
  3. LOI (Line-Of-Interest) pooling head:
       - Samples features along candidate line segments
       - MLP classifier: is this a real segment or not?

Training
--------
  GT is built from the "segments" key in each plot's GT JSON file.
  The HAT field and junction heatmap are computed on-the-fly per image.
  Loss = BCE(junction heatmap) + SmoothL1(junction offset) +
         SmoothL1(attraction field) + BCE(distance field) +
         BCE(LOI classifier)

Validation
----------
  At each epoch:
    - Metrics table (TP, FP, FN, precision, recall, F1) written to CSV
    - Annotated validation images saved every 2 epochs

Usage
-----
  # Train (reuses existing synthetic plots)
  python chart_hawp_detector.py --mode train --plots 1000 --epochs 100

  # Detect segments in a single plotting-area crop
  python chart_hawp_detector.py --mode detect --image path/to/crop.png

Reference
---------
  Zhou et al. "HAWP: A Universal Framework for Parametric Image Parsing"
  IEEE TPAMI 2023.  https://arxiv.org/abs/2209.12213
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths  (relative to this script's directory)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

SYNTH_DIR   = _PROJECT_DIR / "data" / "synthetic_plots"
MODEL_DIR   = _PROJECT_DIR / "models"
LOG_DIR     = _PROJECT_DIR / "data" / "hawp_epoch_logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "hawp_best.pth"
LAST_MODEL_PATH = MODEL_DIR / "hawp_last.pth"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
IMG_SIZE        = 512          # resize plotting-area crop to this before feeding model
BATCH_SIZE      = 4
LR              = 1e-3
LR_STEP         = 30           # cosine restart period (epochs)
WEIGHT_DECAY    = 1e-4
EPOCHS_DEFAULT  = 100
PLOTS_DEFAULT   = 1000
VAL_FRAC        = 0.15
NUM_WORKERS     = 0            # set >0 on Linux; keep 0 on Windows to avoid spawn issues
PIN_MEMORY      = False

# HAT field parameters
DIS_TH          = 10.0         # pixels: attraction field reach
ANG_TH          = 0.9          # cosine threshold for line verification
JHM_SIGMA       = 2.0          # Gaussian sigma for junction heatmap
JHM_THRESHOLD   = 0.01         # minimum junction heatmap value to keep

# LOI sampling
N_PTS           = 32           # number of points sampled along each candidate line
LOI_DIM         = 64           # LOI feature dimension

# Matching thresholds (validation)
MATCH_DIST_PX   = 10.0         # max midpoint distance for TP (pixels in original crop)
MATCH_ANGLE_DEG = 15.0         # max angular difference for TP

# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
def _pbar(iterable, total: int, prefix: str = "", width: int = 40):
    """Yield items from *iterable* while printing a progress bar."""
    start = time.time()
    for i, item in enumerate(iterable, 1):
        yield item
        filled = int(width * i / total)
        bar    = "█" * filled + "░" * (width - filled)
        eta    = (time.time() - start) / i * (total - i)
        print(f"\r  {prefix}: [{bar}] {i:>{len(str(total))}}/{total}"
              f"  {100*i/total:5.1f}%  ETA {eta:.0f}s",
              end="", flush=True)
    print()


# ===========================================================================
# GT Migration  (inject "segments" key into v3 GT files if missing)
# ===========================================================================

_MIN_SEG_LEN = 5.0   # pixels — shorter segments are ignored in GT


def _segments_from_points(points: List[dict]) -> List[dict]:
    """
    Build segment list from a list of point dicts (fallback for old GT files).

    Supports two GT formats:
      - New format: each point has 'series_idx' and 'point_idx' keys.
      - v3 format:  each point has only 'cx','cy','class_idx','class_name'.
                    In this case class_idx is used as series_idx and points
                    are ordered by their appearance in the list.
    """
    from collections import defaultdict
    has_series = all("series_idx" in p and "point_idx" in p for p in points)

    series: Dict[int, List[dict]] = defaultdict(list)
    if has_series:
        for p in points:
            series[p["series_idx"]].append(p)
        for s in series.values():
            s.sort(key=lambda p: p["point_idx"])
    else:
        # v3 format: group by class_idx, preserve list order as point_idx
        for p in points:
            series[p.get("class_idx", 0)].append(p)

    segments = []
    for s_pts in series.values():
        for i in range(len(s_pts) - 1):
            a, b = s_pts[i], s_pts[i + 1]
            length = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
            if length >= _MIN_SEG_LEN:
                segments.append({
                    "x1": a["cx"], "y1": a["cy"],
                    "x2": b["cx"], "y2": b["cy"],
                    "series_idx": a.get("series_idx", a.get("class_idx", 0)),
                    "length": round(length, 2),
                })
    return segments


def _segments_from_series_pixels(series_pixels: List[List[dict]]) -> List[dict]:
    """
    Derive GT segments from series_pixels (the ordered pixel sequences saved
    by chart_marker_detector_v3.py).  ALL consecutive pairs are included;
    the MIN_SEP filter only governs ViT subimage patch selection and must
    not gate segment GT coverage.
    """
    segments: List[dict] = []
    for si, sp in enumerate(series_pixels):
        for i in range(len(sp) - 1):
            p0, p1 = sp[i], sp[i + 1]
            length = math.hypot(p1["cx"] - p0["cx"], p1["cy"] - p0["cy"])
            if length >= _MIN_SEG_LEN:
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
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    if not data.get("segments"):
        if data.get("series_pixels"):
            segs = _segments_from_series_pixels(data["series_pixels"])
        elif data.get("points"):
            segs = _segments_from_points(data.get("all_points") or data["points"])
        else:
            segs = []
        data["segments"] = segs
        try:
            gt_path.write_text(json.dumps(data), encoding="utf-8")
        except PermissionError:
            pass
    return data


def _migrate_gt_segments(synth_dir: Path) -> None:
    """
    Inject 'segments' key into every GT JSON that is missing it or has an
    empty list (from a previous broken migration run).
    """
    gt_files = sorted(synth_dir.glob("gt_*.json"))
    if not gt_files:
        return

    need_migration = [
        f for f in gt_files
        if not json.loads(f.read_text(encoding="utf-8")).get("segments")
    ]

    if not need_migration:
        print("  GT migration: all files already have segments.")
        return

    print(f"  GT migration: updating {len(need_migration)} / {len(gt_files)} files …")
    for f in _pbar(need_migration, len(need_migration), prefix="  Migrating"):
        _load_gt(f)  # writes back in-place using 3-priority logic


# ===========================================================================
# Dataset
# ===========================================================================

def _compute_hat_fields(
    segments: List[dict],
    pa: dict,
    img_h: int,
    img_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute HAT (Holistic Attraction) fields for one image.

    Returns (in crop-coordinate space, resized to IMG_SIZE × IMG_SIZE):
      jmap  : (H, W)   junction heatmap (Gaussian blobs at endpoints)
      joff  : (2,H,W)  sub-pixel offset from pixel centre to nearest junction
      afmap : (2,H,W)  unit attraction vector toward nearest line segment
      dfmap : (H, W)   normalised distance to nearest segment ∈ [0,1]
    """
    H, W = IMG_SIZE, IMG_SIZE
    x0, y0 = pa["x0"], pa["y0"]
    cw = pa["x1"] - x0
    ch = pa["y1"] - y0
    sx = W / max(cw, 1)
    sy = H / max(ch, 1)

    # Translate and scale segments to IMG_SIZE space
    segs_scaled = []
    for seg in segments:
        x1s = (seg["x1"] - x0) * sx
        y1s = (seg["y1"] - y0) * sy
        x2s = (seg["x2"] - x0) * sx
        y2s = (seg["y2"] - y0) * sy
        segs_scaled.append((x1s, y1s, x2s, y2s))

    # --- Distance field and attraction field ---
    dfmap  = np.full((H, W), DIS_TH, dtype=np.float32)
    afmap  = np.zeros((2, H, W), dtype=np.float32)

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ys += 0.5
    xs += 0.5

    for (x1s, y1s, x2s, y2s) in segs_scaled:
        dx = x2s - x1s
        dy = y2s - y1s
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-3:
            continue
        # Project each pixel onto the segment
        t = ((xs - x1s) * dx + (ys - y1s) * dy) / (seg_len * seg_len)
        t = np.clip(t, 0.0, 1.0)
        px = x1s + t * dx
        py = y1s + t * dy
        dist = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
        # Update where this segment is closer
        closer = dist < dfmap
        dfmap[closer] = dist[closer]
        # Attraction vector: from pixel toward nearest point on segment
        vx = (px - xs) / np.maximum(dist, 1e-6)
        vy = (py - ys) / np.maximum(dist, 1e-6)
        afmap[0][closer] = vx[closer]
        afmap[1][closer] = vy[closer]

    dfmap = np.clip(dfmap / DIS_TH, 0.0, 1.0)

    # --- Junction heatmap and offset ---
    jmap  = np.zeros((H, W), dtype=np.float32)
    joff  = np.zeros((2, H, W), dtype=np.float32)

    junctions = []
    for (x1s, y1s, x2s, y2s) in segs_scaled:
        junctions.extend([(x1s, y1s), (x2s, y2s)])

    # Deduplicate junctions within 2 px
    unique_juncs: List[Tuple[float, float]] = []
    for jx, jy in junctions:
        if all(math.hypot(jx - ux, jy - uy) > 2.0 for ux, uy in unique_juncs):
            unique_juncs.append((jx, jy))

    for jx, jy in unique_juncs:
        xi, yi = int(jx), int(jy)
        if not (0 <= xi < W and 0 <= yi < H):
            continue
        # Gaussian blob
        r = int(3 * JHM_SIGMA)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px2, py2 = xi + dx, yi + dy
                if 0 <= px2 < W and 0 <= py2 < H:
                    g = math.exp(-(dx * dx + dy * dy) / (2 * JHM_SIGMA ** 2))
                    if g > jmap[py2, px2]:
                        jmap[py2, px2] = g
        # Sub-pixel offset at the integer pixel location
        joff[0, yi, xi] = jx - (xi + 0.5)
        joff[1, yi, xi] = jy - (yi + 0.5)

    return jmap, joff, afmap, dfmap


class ChartSegDataset(Dataset):
    def __init__(self, gt_files: List[Path]):
        self.gt_files = gt_files

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx: int):
        data     = _load_gt(self.gt_files[idx])
        img_path = Path(data["image_path"])
        if not img_path.exists():
            img_path = SYNTH_DIR / img_path.name

        pa       = data.get("pa") or data.get("plot_area")
        segments = data.get("segments", [])

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((100, 100), dtype=np.uint8)

        # Crop to plotting area
        x0, y0, x1, y1 = int(pa["x0"]), int(pa["y0"]), int(pa["x1"]), int(pa["y1"])
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = img

        # Resize to IMG_SIZE
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(crop_resized.astype(np.float32) / 255.0).unsqueeze(0)

        # Compute GT fields
        jmap, joff, afmap, dfmap = _compute_hat_fields(segments, pa,
                                                        crop.shape[0], crop.shape[1])

        target = {
            "jmap":  torch.from_numpy(jmap).unsqueeze(0),   # (1,H,W)
            "joff":  torch.from_numpy(joff),                 # (2,H,W)
            "afmap": torch.from_numpy(afmap),                # (2,H,W)
            "dfmap": torch.from_numpy(dfmap).unsqueeze(0),   # (1,H,W)
            # Raw segments in crop-pixel space (for validation matching)
            "segments": segments,
            "pa":       pa,
            "crop_h":   crop.shape[0],
            "crop_w":   crop.shape[1],
        }
        return img_t, target


def _collate(batch):
    imgs    = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


# ===========================================================================
# Model
# ===========================================================================

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _HourglassBlock(nn.Module):
    """Single hourglass block (encode → bottleneck → decode with skip)."""
    def __init__(self, ch: int, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.enc = nn.ModuleList([_ConvBnRelu(ch, ch) for _ in range(depth)])
        self.bot = _ConvBnRelu(ch, ch)
        self.dec = nn.ModuleList([_ConvBnRelu(ch, ch) for _ in range(depth)])
        self.pool = nn.MaxPool2d(2, 2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bot(x)
        for dec, skip in zip(self.dec, reversed(skips)):
            x = self.up(x)
            x = dec(x + skip)
        return x


class HAWPBackbone(nn.Module):
    """
    Lightweight two-stack hourglass backbone.
    Input : (B, 1, H, W) grayscale
    Output: (B, 256, H, W) feature map at full resolution
    """
    def __init__(self, ch: int = 128, n_stacks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBnRelu(1, 64, k=7, s=2, p=3),
            _ConvBnRelu(64, 128, k=3, s=2, p=1),
            _ConvBnRelu(128, ch, k=3, s=1, p=1),
        )
        self.hg = nn.ModuleList([_HourglassBlock(ch) for _ in range(n_stacks)])
        self.out_conv = nn.Conv2d(ch, 256, 1)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # (B, ch, H/4, W/4)
        for hg in self.hg:
            x = hg(x)
        x = self.up4(self.out_conv(x))  # (B, 256, H, W)
        return x


class HAWPHeads(nn.Module):
    """
    Prediction heads on top of the 256-channel feature map.
    Outputs:
      jmap  : (B, 1, H, W)  junction heatmap logits
      joff  : (B, 2, H, W)  junction sub-pixel offset
      afmap : (B, 2, H, W)  attraction field (unit vectors)
      dfmap : (B, 1, H, W)  distance field logits
    """
    def __init__(self, in_ch: int = 256):
        super().__init__()
        self.jmap_head  = nn.Sequential(_ConvBnRelu(in_ch, 64), nn.Conv2d(64, 1, 1))
        self.joff_head  = nn.Sequential(_ConvBnRelu(in_ch, 64), nn.Conv2d(64, 2, 1))
        self.afmap_head = nn.Sequential(_ConvBnRelu(in_ch, 64), nn.Conv2d(64, 2, 1))
        self.dfmap_head = nn.Sequential(_ConvBnRelu(in_ch, 64), nn.Conv2d(64, 1, 1))

    def forward(self, feat: torch.Tensor):
        jmap  = self.jmap_head(feat)
        joff  = self.joff_head(feat)
        afmap = self.afmap_head(feat)
        dfmap = self.dfmap_head(feat)
        return jmap, joff, afmap, dfmap


class LOIHead(nn.Module):
    """
    Line-Of-Interest pooling + MLP classifier.
    Given feature map and candidate line endpoints, samples N_PTS features
    along each line and classifies it as real / background.
    """
    def __init__(self, in_ch: int = 256, n_pts: int = N_PTS, loi_dim: int = LOI_DIM):
        super().__init__()
        self.n_pts   = n_pts
        self.loi_dim = loi_dim
        self.feat_proj = nn.Conv2d(in_ch, loi_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(n_pts * loi_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )
        t = torch.linspace(0, 1, n_pts)
        self.register_buffer("tspan", t)

    def _sample_features(self, feat: torch.Tensor,
                         lines: torch.Tensor) -> torch.Tensor:
        """
        feat  : (C, H, W)
        lines : (N, 4)  [x1,y1,x2,y2] in pixel coords
        returns (N, n_pts * C)
        """
        C, H, W = feat.shape
        N = lines.shape[0]
        if N == 0:
            return feat.new_zeros((0, self.n_pts * C))

        t = self.tspan.to(feat.device)                          # (n_pts,)
        x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
        # (N, n_pts)
        px = x1[:, None] + t[None, :] * (x2 - x1)[:, None]
        py = y1[:, None] + t[None, :] * (y2 - y1)[:, None]
        # Normalise to [-1, 1] for grid_sample
        px_n = (px / (W - 1)) * 2 - 1
        py_n = (py / (H - 1)) * 2 - 1
        grid = torch.stack([px_n, py_n], dim=-1).unsqueeze(0)   # (1, N, n_pts, 2)
        feat_4d = feat.unsqueeze(0)                              # (1, C, H, W)
        sampled = F.grid_sample(feat_4d, grid, align_corners=True,
                                mode="bilinear", padding_mode="border")
        # sampled: (1, C, N, n_pts) → (N, n_pts * C)
        return sampled[0].permute(1, 2, 0).reshape(N, -1)

    def forward(self, feat: torch.Tensor,
                lines: torch.Tensor) -> torch.Tensor:
        """
        feat  : (C, H, W)
        lines : (N, 4)
        returns logits (N,)
        """
        proj = self.feat_proj(feat.unsqueeze(0))[0]             # (loi_dim, H, W)
        sampled = self._sample_features(proj, lines)            # (N, n_pts*loi_dim)
        return self.mlp(sampled).squeeze(-1)                    # (N,)


class HAWPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HAWPBackbone()
        self.heads    = HAWPHeads()
        self.loi      = LOIHead()

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        jmap, joff, afmap, dfmap = self.heads(feat)
        return feat, jmap, joff, afmap, dfmap


# ===========================================================================
# Loss
# ===========================================================================

def _weighted_bce(pred: torch.Tensor, target: torch.Tensor,
                  pos_weight: float = 5.0) -> torch.Tensor:
    """BCE with higher weight on positive pixels."""
    w = torch.where(target > 0.5,
                    torch.full_like(target, pos_weight),
                    torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(pred, target, weight=w)


def compute_loss(
    jmap_pred: torch.Tensor, joff_pred: torch.Tensor,
    afmap_pred: torch.Tensor, dfmap_pred: torch.Tensor,
    targets: List[dict],
) -> Tuple[torch.Tensor, dict]:
    B = jmap_pred.shape[0]
    device = jmap_pred.device

    loss_jmap = jmap_pred.new_zeros(())
    loss_joff = jmap_pred.new_zeros(())
    loss_af   = jmap_pred.new_zeros(())
    loss_df   = jmap_pred.new_zeros(())

    for i in range(B):
        t = targets[i]
        jmap_gt  = t["jmap"].to(device)   # (1,H,W)
        joff_gt  = t["joff"].to(device)   # (2,H,W)
        afmap_gt = t["afmap"].to(device)  # (2,H,W)
        dfmap_gt = t["dfmap"].to(device)  # (1,H,W)

        # Junction heatmap loss
        loss_jmap = loss_jmap + _weighted_bce(jmap_pred[i], jmap_gt)

        # Junction offset loss (masked to junction pixels)
        jmask = (jmap_gt > 0.5).float()
        if jmask.sum() > 0:
            loss_joff = loss_joff + (
                F.smooth_l1_loss(joff_pred[i], joff_gt, reduction="none") * jmask
            ).sum() / jmask.sum()

        # Attraction field loss (masked to pixels near segments)
        near_mask = (dfmap_gt < 0.5).float()   # dfmap_gt < 0.5 → within DIS_TH/2
        if near_mask.sum() > 0:
            loss_af = loss_af + (
                F.smooth_l1_loss(afmap_pred[i], afmap_gt, reduction="none") * near_mask
            ).sum() / near_mask.sum()

        # Distance field loss
        loss_df = loss_df + F.binary_cross_entropy_with_logits(
            dfmap_pred[i], 1.0 - dfmap_gt  # dfmap_gt=0 means on segment → target=1
        )

    loss_jmap = loss_jmap / B
    loss_joff = loss_joff / B
    loss_af   = loss_af   / B
    loss_df   = loss_df   / B

    total = loss_jmap + 0.5 * loss_joff + loss_af + loss_df
    return total, {
        "jmap": loss_jmap.item(),
        "joff": loss_joff.item(),
        "af":   loss_af.item(),
        "df":   loss_df.item(),
    }


# ===========================================================================
# Inference: extract line segments from predictions
# ===========================================================================

def _extract_junctions(jmap: np.ndarray, joff: np.ndarray,
                       threshold: float = JHM_THRESHOLD) -> np.ndarray:
    """
    Extract junction candidates from heatmap + offset.
    Returns (N, 2) array of (x, y) in pixel coords.
    """
    # Simple NMS: suppress pixels not the local maximum in 3×3 window
    from scipy.ndimage import maximum_filter
    jmap_nms = jmap * (jmap == maximum_filter(jmap, size=3))
    ys, xs = np.where(jmap_nms > threshold)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Apply sub-pixel offset
    jx = xs + 0.5 + joff[0, ys, xs]
    jy = ys + 0.5 + joff[1, ys, xs]
    conf = jmap_nms[ys, xs]
    # Sort by confidence, keep top 300
    order = np.argsort(-conf)[:300]
    return np.stack([jx[order], jy[order]], axis=1).astype(np.float32)


def _verify_line(x1: float, y1: float, x2: float, y2: float,
                 afmap: np.ndarray, dfmap: np.ndarray,
                 n_pts: int = 16) -> float:
    """
    Score a candidate line segment using the attraction field.
    Returns a score in [0, 1]: fraction of sampled points whose attraction
    vector aligns with the line direction.
    """
    H, W = afmap.shape[1], afmap.shape[2]
    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)
    if seg_len < 2.0:
        return 0.0
    ux, uy = dx / seg_len, dy / seg_len  # unit direction

    ts = np.linspace(0, 1, n_pts)
    xs = np.clip(x1 + ts * dx, 0, W - 1).astype(int)
    ys = np.clip(y1 + ts * dy, 0, H - 1).astype(int)

    # Attraction vector at each sampled point
    ax = afmap[0, ys, xs]
    ay = afmap[1, ys, xs]
    # Cosine similarity with line direction (use absolute value for undirected)
    cos_sim = np.abs(ax * ux + ay * uy)
    # Also check distance field: points should be near a segment
    near = dfmap[0, ys, xs] < 0.5
    if near.sum() == 0:
        return 0.0
    return float(cos_sim[near].mean())


def run_hawp(net: HAWPNet, gray_crop: np.ndarray,
             device: torch.device,
             score_threshold: float = ANG_TH) -> np.ndarray:
    """
    Run HAWP inference on a grayscale plotting-area crop.
    Returns (N, 4) array of [x1, y1, x2, y2] in crop-pixel coords.
    """
    H, W = gray_crop.shape
    img_resized = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
    img_t = img_t.unsqueeze(0).unsqueeze(0).to(device)

    net.eval()
    with torch.no_grad():
        feat, jmap_l, joff_p, afmap_p, dfmap_l = net(img_t)

    jmap  = torch.sigmoid(jmap_l)[0, 0].cpu().numpy()
    joff  = joff_p[0].cpu().numpy()
    afmap = afmap_p[0].cpu().numpy()
    dfmap = torch.sigmoid(dfmap_l)[0].cpu().numpy()

    # Extract junctions
    juncs = _extract_junctions(jmap, joff)
    if len(juncs) < 2:
        return np.zeros((0, 4), dtype=np.float32)

    # Enumerate all junction pairs, verify via attraction field
    lines = []
    for i in range(len(juncs)):
        for j in range(i + 1, len(juncs)):
            x1, y1 = juncs[i]
            x2, y2 = juncs[j]
            score = _verify_line(x1, y1, x2, y2, afmap, dfmap)
            if score >= score_threshold:
                lines.append((x1, y1, x2, y2, score))

    if not lines:
        return np.zeros((0, 4), dtype=np.float32)

    lines_arr = np.array(lines, dtype=np.float32)

    # Scale back to original crop size
    lines_arr[:, 0] *= W / IMG_SIZE
    lines_arr[:, 2] *= W / IMG_SIZE
    lines_arr[:, 1] *= H / IMG_SIZE
    lines_arr[:, 3] *= H / IMG_SIZE

    return lines_arr[:, :4]


# ===========================================================================
# Validation metrics
# ===========================================================================

def _seg_midpoint(seg) -> Tuple[float, float]:
    if isinstance(seg, dict):
        return (seg["x1"] + seg["x2"]) / 2, (seg["y1"] + seg["y2"]) / 2
    return (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2


def _seg_angle(seg) -> float:
    if isinstance(seg, dict):
        dx, dy = seg["x2"] - seg["x1"], seg["y2"] - seg["y1"]
    else:
        dx, dy = seg[2] - seg[0], seg[3] - seg[1]
    return math.degrees(math.atan2(dy, dx)) % 180


def _match_segments(pred_segs: np.ndarray, gt_segs: List[dict],
                    dist_th: float = MATCH_DIST_PX,
                    angle_th: float = MATCH_ANGLE_DEG) -> Tuple[int, int, int]:
    """Return (TP, FP, FN)."""
    matched_gt = set()
    TP = 0
    for ps in pred_segs:
        mx, my = _seg_midpoint(ps)
        pa = _seg_angle(ps)
        for gi, gs in enumerate(gt_segs):
            if gi in matched_gt:
                continue
            gx, gy = _seg_midpoint(gs)
            ga = _seg_angle(gs)
            if (math.hypot(mx - gx, my - gy) < dist_th and
                    min(abs(pa - ga), 180 - abs(pa - ga)) < angle_th):
                TP += 1
                matched_gt.add(gi)
                break
    FP = len(pred_segs) - TP
    FN = len(gt_segs) - len(matched_gt)
    return TP, FP, FN


# ===========================================================================
# Epoch visualisation
# ===========================================================================

def _save_epoch_val_images(
    epoch: int,
    net: HAWPNet,
    val_files: List[Path],
    device: torch.device,
    n_samples: int = 8,
) -> None:
    epoch_dir = LOG_DIR / f"epoch_{epoch:03d}" / "val_images"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    sample_files = rng.sample(val_files, min(n_samples, len(val_files)))

    for gt_file in sample_files:
        data     = json.loads(gt_file.read_text(encoding="utf-8"))
        img_path = Path(data["image_path"])
        if not img_path.exists():
            img_path = SYNTH_DIR / img_path.name
        pa       = data.get("pa") or data.get("plot_area")
        segments = data.get("segments", [])

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        x0, y0, x1, y1 = int(pa["x0"]), int(pa["y0"]), int(pa["x1"]), int(pa["y1"])
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        pred_lines = run_hawp(net, crop, device)

        # Draw
        vis_gt   = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        vis_pred = vis_gt.copy()
        for seg in segments:
            cv2.line(vis_gt,
                     (int(seg["x1"] - x0), int(seg["y1"] - y0)),
                     (int(seg["x2"] - x0), int(seg["y2"] - y0)),
                     (0, 200, 0), 2)
        for seg in pred_lines:
            cv2.line(vis_pred,
                     (int(seg[0]), int(seg[1])),
                     (int(seg[2]), int(seg[3])),
                     (0, 0, 220), 2)

        panel = np.hstack([vis_gt, vis_pred])
        stem  = gt_file.stem.replace("gt_", "")
        cv2.imwrite(str(epoch_dir / f"{stem}.png"), panel)


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    n_plots: int = PLOTS_DEFAULT,
    epochs:  int = EPOCHS_DEFAULT,
) -> None:
    # ------------------------------------------------------------------
    # Step 1 — ensure synthetic plots exist
    # ------------------------------------------------------------------
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    gt_files_all = sorted(SYNTH_DIR.glob("gt_*.json"))

    if len(gt_files_all) < n_plots:
        print(f"  Only {len(gt_files_all)} plots found in {SYNTH_DIR}.")
        print(f"  Please run chart_marker_detector_v3.py first to generate "
              f"{n_plots} synthetic plots.")
        sys.exit(1)

    gt_files_all = gt_files_all[:n_plots]

    # ------------------------------------------------------------------
    # Step 1b — migrate GT files (inject 'segments' key if missing)
    # ------------------------------------------------------------------
    _migrate_gt_segments(SYNTH_DIR)

    # ------------------------------------------------------------------
    # Step 2 — train/val split
    # ------------------------------------------------------------------
    rng = random.Random(42)
    shuffled = list(gt_files_all)
    rng.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * VAL_FRAC))
    val_files   = shuffled[:n_val]
    train_files = shuffled[n_val:]
    print(f"\n  Dataset: {len(train_files)} train  |  {len(val_files)} val")

    train_ds = ChartSegDataset(train_files)
    val_ds   = ChartSegDataset(val_files)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          collate_fn=_collate)
    val_dl   = DataLoader(val_ds, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                          collate_fn=_collate)

    # ------------------------------------------------------------------
    # Step 3 — model, optimiser, scheduler
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    net = HAWPNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=LR_STEP, T_mult=1, eta_min=LR * 0.01)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume if checkpoint exists
    start_epoch = 1
    best_val_loss = float("inf")
    if LAST_MODEL_PATH.exists():
        ckpt = torch.load(str(LAST_MODEL_PATH), map_location=device,
                          weights_only=False)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  Resumed from epoch {start_epoch - 1}")

    # CSV metrics file
    csv_path = LOG_DIR / "training_metrics.csv"
    if not csv_path.exists() or start_epoch == 1:
        csv_path.write_text(
            "epoch,train_loss,val_loss,jmap_loss,joff_loss,af_loss,df_loss,"
            "val_TP,val_FP,val_FN,val_precision,val_recall,val_F1\n",
            encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        # ---- Train ----
        net.train()
        t_loss = 0.0
        t_sub  = {"jmap": 0.0, "joff": 0.0, "af": 0.0, "df": 0.0}
        n_steps = len(train_dl)

        for step, (imgs, targets) in _pbar(
                enumerate(train_dl, 1), n_steps,
                prefix=f"  Epoch {epoch:3d}/{epochs} train"):
            imgs = imgs.to(device)
            opt.zero_grad()
            with autocast(device_type=device.type,
                          enabled=(device.type == "cuda")):
                feat, jmap_p, joff_p, afmap_p, dfmap_p = net(imgs)
                loss, sub = compute_loss(jmap_p, joff_p, afmap_p, dfmap_p, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            t_loss += loss.item()
            for k in t_sub:
                t_sub[k] += sub[k]

        scheduler.step()
        t_loss /= n_steps
        for k in t_sub:
            t_sub[k] /= n_steps

        # ---- Validate ----
        net.eval()
        v_loss = 0.0
        v_sub  = {"jmap": 0.0, "joff": 0.0, "af": 0.0, "df": 0.0}
        total_TP = total_FP = total_FN = 0

        with torch.no_grad():
            for imgs, targets in _pbar(
                    val_dl, len(val_dl),
                    prefix=f"  Epoch {epoch:3d}/{epochs} val  "):
                imgs = imgs.to(device)
                with autocast(device_type=device.type,
                              enabled=(device.type == "cuda")):
                    feat, jmap_p, joff_p, afmap_p, dfmap_p = net(imgs)
                    loss, sub = compute_loss(jmap_p, joff_p, afmap_p, dfmap_p, targets)
                v_loss += loss.item()
                for k in v_sub:
                    v_sub[k] += sub[k]

                # Segment-level metrics
                t = targets[0]
                pa       = t["pa"]
                gt_segs  = t["segments"]
                crop_h   = t["crop_h"]
                crop_w   = t["crop_w"]

                jmap_np  = torch.sigmoid(jmap_p)[0, 0].cpu().numpy()
                joff_np  = joff_p[0].cpu().numpy()
                afmap_np = afmap_p[0].cpu().numpy()
                dfmap_np = torch.sigmoid(dfmap_p)[0].cpu().numpy()

                juncs = _extract_junctions(jmap_np, joff_np)
                pred_lines_scaled = []
                for i in range(len(juncs)):
                    for j in range(i + 1, len(juncs)):
                        x1, y1 = juncs[i]
                        x2, y2 = juncs[j]
                        score = _verify_line(x1, y1, x2, y2, afmap_np, dfmap_np)
                        if score >= ANG_TH:
                            # Scale to crop coords
                            pred_lines_scaled.append([
                                x1 * crop_w / IMG_SIZE,
                                y1 * crop_h / IMG_SIZE,
                                x2 * crop_w / IMG_SIZE,
                                y2 * crop_h / IMG_SIZE,
                            ])

                pred_arr = np.array(pred_lines_scaled, dtype=np.float32) \
                    if pred_lines_scaled else np.zeros((0, 4), dtype=np.float32)

                # Translate GT segments to crop coords
                x0, y0 = pa["x0"], pa["y0"]
                gt_crop = [
                    {"x1": s["x1"] - x0, "y1": s["y1"] - y0,
                     "x2": s["x2"] - x0, "y2": s["y2"] - y0}
                    for s in gt_segs
                ]
                TP, FP, FN = _match_segments(pred_arr, gt_crop)
                total_TP += TP
                total_FP += FP
                total_FN += FN

        v_loss /= len(val_dl)
        for k in v_sub:
            v_sub[k] /= len(val_dl)

        prec = total_TP / max(total_TP + total_FP, 1)
        rec  = total_TP / max(total_TP + total_FN, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)

        # ---- Print metrics ----
        print(f"\n  Epoch {epoch:3d}/{epochs}  "
              f"train={t_loss:.4f}  val={v_loss:.4f}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
        print(f"  {'Metric':<20} {'Train':>10} {'Val':>10}")
        print(f"  {'-'*42}")
        for k in ("jmap", "joff", "af", "df"):
            print(f"  {k+'_loss':<20} {t_sub[k]:>10.4f} {v_sub[k]:>10.4f}")
        print(f"  {'TP':<20} {total_TP:>10}")
        print(f"  {'FP':<20} {total_FP:>10}")
        print(f"  {'FN':<20} {total_FN:>10}")
        print(f"  {'Precision':<20} {prec:>10.4f}")
        print(f"  {'Recall':<20} {rec:>10.4f}")
        print(f"  {'F1':<20} {f1:>10.4f}")

        # ---- CSV ----
        try:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{t_loss:.6f},{v_loss:.6f},"
                        f"{v_sub['jmap']:.6f},{v_sub['joff']:.6f},"
                        f"{v_sub['af']:.6f},{v_sub['df']:.6f},"
                        f"{total_TP},{total_FP},{total_FN},"
                        f"{prec:.6f},{rec:.6f},{f1:.6f}\n")
        except PermissionError:
            print("  [WARN] Could not write metrics CSV (file locked).")

        # ---- Save images every 2 epochs ----
        if epoch % 2 == 0 or epoch == 1:
            _save_epoch_val_images(epoch, net, val_files, device)

        # ---- Checkpoints ----
        ckpt = {
            "epoch": epoch,
            "model": net.state_dict(),
            "opt":   opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }
        try:
            torch.save(ckpt, str(LAST_MODEL_PATH))
        except PermissionError:
            pass

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            try:
                torch.save(ckpt, str(BEST_MODEL_PATH))
            except PermissionError:
                pass

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {BEST_MODEL_PATH}")


# ===========================================================================
# Detect mode
# ===========================================================================

def detect(image_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HAWPNet().to(device)

    ckpt_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else LAST_MODEL_PATH
    if not ckpt_path.exists():
        print(f"[ERROR] No trained model found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    print(f"  Loaded model from {ckpt_path}  (epoch {ckpt['epoch']})")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    lines = run_hawp(net, img, device)
    print(f"  Detected {len(lines)} line segments")

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for seg in lines:
        cv2.line(vis, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])),
                 (0, 0, 220), 2)

    out_path = Path(image_path).with_suffix(".hawp_detected.png")
    cv2.imwrite(str(out_path), vis)
    print(f"  Saved to {out_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAWP-style chart segment detector")
    parser.add_argument("--mode",   choices=["train", "detect"], default="train")
    parser.add_argument("--plots",  type=int, default=PLOTS_DEFAULT,
                        help="Number of synthetic plots to use for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--image",  type=str, default=None,
                        help="Path to plotting-area crop (detect mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train(n_plots=args.plots, epochs=args.epochs)
    elif args.mode == "detect":
        if args.image is None:
            parser.error("--image is required for detect mode")
        detect(args.image)
