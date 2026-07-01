"""
chart_letr_detector.py
======================
LETR-style Line Segment Detection for synthetic dose-response charts.

Architecture (self-contained, no external LETR package needed):
  • ResNet-50 backbone (torchvision pretrained) → multi-scale features
  • Transformer encoder-decoder (DETR-style)
  • Segment prediction MLP: predicts N_QUERIES line segments as (x1,y1,x2,y2)
    normalised to [0,1] relative to the plotting-area crop
  • Hungarian matching loss (L1 endpoint distance + focal classification)
  • Two-stage training: coarse (stage 1) then fine (stage 2) — both in one script

Training data:
  • Reuses synthetic plots from chart_marker_detector_v3.py
    (data/synthetic_plots/*.png + gt_*.json)
  • GT segments: consecutive marker pairs within each series where both
    endpoints survived the MIN_SEP filter
  • GT JSONs are migrated in-place to add a "segments" key if missing

Validation:
  • Per-epoch CSV: TP, FP, FN, precision, recall, F1 (structural AP)
  • Every SAVE_EVERY epochs: annotated images (GT green, predicted red)

Usage:
  python chart_letr_detector.py --mode train [--plots N] [--epochs E]
  python chart_letr_detector.py --mode detect --image path/to/crop.png

Reference:
  Xu et al. "Line Segment Detection Using Transformers Without Edges."
  CVPR 2021. https://arxiv.org/abs/2101.01909
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
from typing import List, Tuple, Dict, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ── optional torchvision ResNet backbone ─────────────────────────────────────
try:
    import torchvision.models as tvm
    _TORCHVISION_OK = True
except ImportError:
    _TORCHVISION_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  (relative to this script's directory)
# ══════════════════════════════════════════════════════════════════════════════
_HERE        = Path(__file__).resolve().parent
SYNTH_DIR    = _HERE / ".." / "data" / "synthetic_plots"
MODEL_DIR    = _HERE / ".." / "models"
EPOCH_DIR    = _HERE / ".." / "data" / "letr_epoch_logs"

# ══════════════════════════════════════════════════════════════════════════════
#  HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE      = 512          # crop is resized to IMG_SIZE × IMG_SIZE for the model
N_QUERIES     = 200          # number of segment queries (≥ max segments per plot)
HIDDEN_DIM    = 256          # transformer hidden dimension
N_HEADS       = 8           # attention heads
N_ENC_LAYERS  = 6           # transformer encoder layers
N_DEC_LAYERS  = 6           # transformer decoder layers
DIM_FFN       = 1024        # feedforward dim in transformer
DROPOUT       = 0.1

LR            = 1e-4
LR_BACKBONE   = 1e-5
WEIGHT_DECAY  = 1e-4
EPOCHS        = 100
BATCH_SIZE    = 4
VAL_FRAC      = 0.15
SAVE_EVERY    = 2            # save annotated val images every N epochs

# Matching / loss weights
COST_CLASS    = 1.0
COST_LINE     = 5.0
LOSS_CLASS    = 1.0
LOSS_LINE     = 5.0
EOS_COEF      = 0.1          # weight for "no-segment" class

# Evaluation matching thresholds
MATCH_DIST_PX    = 10.0      # midpoint distance tolerance (pixels in crop space)
MATCH_ANGLE_DEG  = 15.0      # angular tolerance

# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESS BAR
# ══════════════════════════════════════════════════════════════════════════════
def _pbar(done: int, total: int, t0: float, width: int = 40,
          prefix: str = "") -> None:
    frac    = done / total if total else 1.0
    filled  = int(width * frac)
    bar     = "█" * filled + "░" * (width - filled)
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
#  GT MIGRATION  (inject "segments" key into existing GT JSONs)
# ══════════════════════════════════════════════════════════════════════════════
_MIN_SEG_LEN = 5.0   # pixels — shorter segments are ignored in GT


def _segments_from_points(points: List[dict]) -> List[dict]:
    """
    Build GT segments from a list of point dicts (fallback for old GT files).
    Each point must have keys: cx, cy, class_idx (used as series_idx).
    Segments connect consecutive points within the same series.
    """
    from collections import defaultdict
    series: Dict[int, List[dict]] = defaultdict(list)
    for p in points:
        si = p.get("series_idx", p.get("class_idx", 0))
        series[si].append(p)
    segs = []
    for si, pts in series.items():
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            length = math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])
            if length >= _MIN_SEG_LEN:
                segs.append({
                    "x1": a["cx"], "y1": a["cy"],
                    "x2": b["cx"], "y2": b["cy"],
                    "series_idx": si,
                    "length": round(length, 2),
                })
    return segs


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
    """Add 'segments' key to every GT JSON that lacks it or has an empty list."""
    gt_files = sorted(synth_dir.glob("gt_*.json"))
    if not gt_files:
        return
    need = [f for f in gt_files
            if not json.loads(f.read_text(encoding="utf-8")).get("segments")]
    if not need:
        print(f"  GT migration: all {len(gt_files)} files already have segments.")
        return
    print(f"  Migrating {len(need)}/{len(gt_files)} GT files …")
    t0 = time.time()
    for i, gf in enumerate(need, 1):
        _load_gt(gf)  # writes back in-place
        _pbar(i, len(need), t0, prefix="  Migrating: ")
    print(f"  Migration done ({len(need)} files updated).")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SegmentDataset(Dataset):
    """
    Each sample is one plotting-area crop with its GT segments.
    Returns:
      img   : (1, IMG_SIZE, IMG_SIZE) float32 in [0,1]
      target: dict with keys
                "lines"  : (N, 4) float32 — normalised (x1,y1,x2,y2) in [0,1]
                "labels" : (N,)   int64   — all zeros (segment class)
    """
    def __init__(self, gt_files: List[Path], img_size: int = IMG_SIZE):
        self.gt_files = gt_files
        self.img_size = img_size

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx: int):
        gf   = self.gt_files[idx]
        data = _load_gt(gf)
        pa   = data.get("pa") or data.get("plot_area")  # {x0,y0,x1,y1} in full-image pixels
        segs = data.get("segments", [])

        # Load image and crop to plotting area
        img_path = gf.parent / gf.name.replace("gt_", "plot_").replace(".json", ".png")
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            # Return empty sample
            img_t = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)
            return img_t, {"lines": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}

        x0, y0, x1, y1 = pa["x0"], pa["y0"], pa["x1"], pa["y1"]
        crop = img_bgr[y0:y1, x0:x1]
        h_crop, w_crop = crop.shape[:2]

        # Resize to IMG_SIZE × IMG_SIZE
        crop_rs = cv2.resize(crop, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        gray    = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        img_t   = torch.from_numpy(gray).unsqueeze(0)  # (1, H, W)

        # Build normalised segment coordinates
        sx = self.img_size / max(w_crop, 1)
        sy = self.img_size / max(h_crop, 1)
        lines = []
        for s in segs:
            # Translate from full-image to crop coords, then scale to [0,1]
            nx1 = np.clip((s["x1"] - x0) * sx / self.img_size, 0.0, 1.0)
            ny1 = np.clip((s["y1"] - y0) * sy / self.img_size, 0.0, 1.0)
            nx2 = np.clip((s["x2"] - x0) * sx / self.img_size, 0.0, 1.0)
            ny2 = np.clip((s["y2"] - y0) * sy / self.img_size, 0.0, 1.0)
            lines.append([nx1, ny1, nx2, ny2])

        if lines:
            lines_t  = torch.tensor(lines, dtype=torch.float32)
            labels_t = torch.zeros(len(lines), dtype=torch.long)
        else:
            lines_t  = torch.zeros(0, 4, dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.long)

        return img_t, {"lines": lines_t, "labels": labels_t}


def _collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKBONE  (ResNet-50 or lightweight CNN fallback)
# ══════════════════════════════════════════════════════════════════════════════
class _ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone adapted for grayscale input.
    Returns feature map at 1/32 resolution: (B, 2048, H/32, W/32).
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if _TORCHVISION_OK:
            weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet  = tvm.resnet50(weights=weights)
        else:
            raise ImportError("torchvision is required for the ResNet backbone. "
                              "Install with: pip install torchvision")
        # Adapt first conv for 1-channel input
        orig = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Average RGB weights across channels
            self.conv1.weight.data = orig.weight.data.mean(dim=1, keepdim=True)
        self.bn1    = resnet.bn1
        self.relu   = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.out_channels = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x   # (B, 2048, H/32, W/32)


class _LightBackbone(nn.Module):
    """
    Lightweight CNN backbone (no torchvision dependency).
    Returns (B, 256, H/32, W/32).
    """
    def __init__(self):
        super().__init__()
        def _blk(ci, co, stride=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )
        self.net = nn.Sequential(
            _blk(1,   32, stride=2),
            _blk(32,  64, stride=2),
            _blk(64, 128, stride=2),
            _blk(128,256, stride=2),
            _blk(256,256, stride=2),
        )
        self.out_channels = 256

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════
class _PositionEncoding2D(nn.Module):
    """Sine/cosine 2D positional encoding (DETR-style)."""
    def __init__(self, hidden_dim: int, temperature: float = 10000.0,
                 normalize: bool = True):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.hidden_dim  = hidden_dim
        self.temperature = temperature
        self.normalize   = normalize
        self.scale       = 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → returns (B, hidden_dim, H, W)"""
        B, _, H, W = x.shape
        device = x.device
        y_embed = torch.ones(B, H, W, device=device).cumsum(1).float()
        x_embed = torch.ones(B, H, W, device=device).cumsum(2).float()
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale
        dim_t = torch.arange(self.hidden_dim // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.hidden_dim // 2))
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(0, 3, 1, 2)
        return pos  # (B, hidden_dim, H, W)


# ══════════════════════════════════════════════════════════════════════════════
#  MLP HEAD
# ══════════════════════════════════════════════════════════════════════════════
class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
#  LETR MODEL
# ══════════════════════════════════════════════════════════════════════════════
class LETRModel(nn.Module):
    """
    LETR: Line Segment Detection Using Transformers.

    Input : (B, 1, IMG_SIZE, IMG_SIZE) grayscale float32
    Output: dict with
              "pred_logits" : (B, N_QUERIES, 2)   — segment / no-segment
              "pred_lines"  : (B, N_QUERIES, 4)   — (x1,y1,x2,y2) in [0,1]
    """
    def __init__(self,
                 n_queries:   int = N_QUERIES,
                 hidden_dim:  int = HIDDEN_DIM,
                 n_heads:     int = N_HEADS,
                 n_enc:       int = N_ENC_LAYERS,
                 n_dec:       int = N_DEC_LAYERS,
                 dim_ffn:     int = DIM_FFN,
                 dropout:     float = DROPOUT,
                 use_resnet:  bool = True):
        super().__init__()
        self.n_queries = n_queries

        # Backbone
        if use_resnet and _TORCHVISION_OK:
            self.backbone = _ResNetBackbone(pretrained=True)
        else:
            self.backbone = _LightBackbone()
        backbone_ch = self.backbone.out_channels

        # Project backbone features to hidden_dim
        self.input_proj = nn.Conv2d(backbone_ch, hidden_dim, kernel_size=1)

        # Positional encoding
        self.pos_enc = _PositionEncoding2D(hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=n_enc,
            num_decoder_layers=n_dec,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            batch_first=True,
        )

        # Query embeddings
        self.query_embed = nn.Embedding(n_queries, hidden_dim)

        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, 2)          # segment / no-segment
        self.line_head  = _MLP(hidden_dim, hidden_dim, 4, 3) # (x1,y1,x2,y2)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, 1, H, W)
        feat = self.backbone(x)                              # (B, C, h, w)
        src  = self.input_proj(feat)                         # (B, D, h, w)
        pos  = self.pos_enc(src)                             # (B, D, h, w)
        B, D, h, w = src.shape

        # Flatten spatial dims → sequence
        src_seq = (src + pos).flatten(2).permute(0, 2, 1)   # (B, h*w, D)

        # Queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # Transformer
        hs = self.transformer(src_seq, queries)              # (B, Q, D)

        # Predictions
        logits = self.class_head(hs)                         # (B, Q, 2)
        lines  = self.line_head(hs).sigmoid()                # (B, Q, 4) in [0,1]

        return {"pred_logits": logits, "pred_lines": lines}


# ══════════════════════════════════════════════════════════════════════════════
#  HUNGARIAN MATCHER
# ══════════════════════════════════════════════════════════════════════════════
class _HungarianMatcher(nn.Module):
    """
    Computes optimal bipartite matching between predictions and GT segments.
    Cost = cost_class * class_cost + cost_line * L1_line_cost
    """
    def __init__(self, cost_class: float = COST_CLASS, cost_line: float = COST_LINE):
        super().__init__()
        self.cost_class = cost_class
        self.cost_line  = cost_line

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict]):
        B, Q = outputs["pred_logits"].shape[:2]
        # Flatten batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (B*Q, 2)
        out_line = outputs["pred_lines"].flatten(0, 1)               # (B*Q, 4)

        device   = out_prob.device
        tgt_ids  = torch.cat([t["labels"] for t in targets]).to(device)  # (sum_N,)
        tgt_line = torch.cat([t["lines"]  for t in targets]).to(device)  # (sum_N, 4)

        # Classification cost: negative probability of target class
        cost_class = -out_prob[:, tgt_ids]                           # (B*Q, sum_N)

        # L1 line cost — also consider flipped endpoints
        cost_line_fwd = torch.cdist(out_line, tgt_line, p=1)
        tgt_line_flip = tgt_line[:, [2, 3, 0, 1]]
        cost_line_bwd = torch.cdist(out_line, tgt_line_flip, p=1)
        cost_line = torch.minimum(cost_line_fwd, cost_line_bwd)      # (B*Q, sum_N)

        C = self.cost_class * cost_class + self.cost_line * cost_line
        C = C.view(B, Q, -1).cpu()

        sizes = [len(t["labels"]) for t in targets]
        indices = []
        offset = 0
        for i, sz in enumerate(sizes):
            c = C[i, :, offset:offset + sz]
            if sz == 0:
                indices.append((torch.zeros(0, dtype=torch.long),
                                 torch.zeros(0, dtype=torch.long)))
            else:
                ri, ci = linear_sum_assignment(c.numpy())
                indices.append((torch.as_tensor(ri, dtype=torch.long),
                                 torch.as_tensor(ci, dtype=torch.long)))
            offset += sz
        return indices


# ══════════════════════════════════════════════════════════════════════════════
#  CRITERION
# ══════════════════════════════════════════════════════════════════════════════
class _SetCriterion(nn.Module):
    """
    LETR loss: Hungarian-matched classification + L1 line loss.
    """
    def __init__(self, matcher: _HungarianMatcher,
                 eos_coef: float = EOS_COEF,
                 loss_class: float = LOSS_CLASS,
                 loss_line:  float = LOSS_LINE):
        super().__init__()
        self.matcher    = matcher
        self.eos_coef   = eos_coef
        self.loss_class = loss_class
        self.loss_line  = loss_line
        # Class weights: [segment_weight=1, no-segment_weight=eos_coef]
        self.register_buffer("class_weights",
                             torch.tensor([1.0, eos_coef], dtype=torch.float32))

    def forward(self, outputs: dict, targets: List[dict]):
        indices = self.matcher(outputs, targets)
        B, Q    = outputs["pred_logits"].shape[:2]
        device  = outputs["pred_logits"].device

        # ── Classification loss ───────────────────────────────────────────────
        # Default: all queries assigned to "no-segment" class (index 1)
        tgt_classes = torch.full((B, Q), 1, dtype=torch.long, device=device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx):
                tgt_classes[i, src_idx] = targets[i]["labels"][tgt_idx].to(device)

        loss_ce = F.cross_entropy(
            outputs["pred_logits"].flatten(0, 1),
            tgt_classes.flatten(),
            weight=self.class_weights.to(device),
        )

        # ── Line regression loss (only on matched queries) ────────────────────
        src_lines_all, tgt_lines_all = [], []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            pred = outputs["pred_lines"][i][src_idx]         # (M, 4)
            gt   = targets[i]["lines"][tgt_idx].to(device)   # (M, 4)
            # Consider both endpoint orderings, take minimum
            gt_flip = gt[:, [2, 3, 0, 1]]
            loss_fwd = F.l1_loss(pred, gt,      reduction="none").sum(-1)
            loss_bwd = F.l1_loss(pred, gt_flip, reduction="none").sum(-1)
            src_lines_all.append(torch.minimum(loss_fwd, loss_bwd))

        if src_lines_all:
            n_matched = sum(len(x) for x in src_lines_all)
            loss_l1   = torch.cat(src_lines_all).sum() / max(n_matched, 1)
        else:
            loss_l1 = torch.tensor(0.0, device=device)

        total = self.loss_class * loss_ce + self.loss_line * loss_l1
        return total, loss_ce.detach(), loss_l1.detach()


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════
def _seg_angle(x1, y1, x2, y2) -> float:
    return math.atan2(y2 - y1, x2 - x1)


def _seg_midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2


def _match_segments(pred_segs: np.ndarray, gt_segs: np.ndarray,
                    img_size: int,
                    dist_thr: float = MATCH_DIST_PX,
                    angle_thr_deg: float = MATCH_ANGLE_DEG
                    ) -> Tuple[int, int, int]:
    """
    Match predicted segments to GT segments.
    pred_segs, gt_segs: (N, 4) in pixel coords (x1,y1,x2,y2).
    Returns (TP, FP, FN).
    """
    if len(gt_segs) == 0:
        return 0, len(pred_segs), 0
    if len(pred_segs) == 0:
        return 0, 0, len(gt_segs)

    angle_thr = math.radians(angle_thr_deg)
    matched_gt = set()
    TP = 0
    for p in pred_segs:
        px1, py1, px2, py2 = p
        pmx, pmy = _seg_midpoint(px1, py1, px2, py2)
        pa = _seg_angle(px1, py1, px2, py2)
        for j, g in enumerate(gt_segs):
            if j in matched_gt:
                continue
            gx1, gy1, gx2, gy2 = g
            gmx, gmy = _seg_midpoint(gx1, gy1, gx2, gy2)
            dist = math.hypot(pmx - gmx, pmy - gmy)
            if dist > dist_thr:
                continue
            ga  = _seg_angle(gx1, gy1, gx2, gy2)
            da  = abs(pa - ga) % math.pi
            if da > math.pi / 2:
                da = math.pi - da
            if da <= angle_thr:
                TP += 1
                matched_gt.add(j)
                break
    FP = len(pred_segs) - TP
    FN = len(gt_segs)   - TP
    return TP, FP, FN


def _extract_predictions(outputs: dict, img_size: int,
                         conf_threshold: float = 0.5) -> np.ndarray:
    """
    Extract predicted segments from model output.
    Returns (N, 4) array in pixel coords.
    """
    logits = outputs["pred_logits"][0]   # (Q, 2)
    lines  = outputs["pred_lines"][0]    # (Q, 4)
    probs  = logits.softmax(-1)[:, 0]   # probability of "segment" class
    mask   = probs > conf_threshold
    segs   = lines[mask].cpu().numpy() * img_size
    return segs


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _draw_segments(img_gray: np.ndarray, segs: np.ndarray,
                   color: Tuple, thickness: int = 1) -> np.ndarray:
    """Draw segments on a BGR image."""
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) if img_gray.ndim == 2 else img_gray.copy()
    for s in segs:
        x1, y1, x2, y2 = int(round(s[0])), int(round(s[1])), int(round(s[2])), int(round(s[3]))
        cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def _save_epoch_val_images(epoch: int, model: LETRModel, val_files: List[Path],
                           device: torch.device, epoch_dir: Path,
                           n_samples: int = 8) -> None:
    """Save annotated validation images for the current epoch."""
    img_dir = epoch_dir / f"epoch_{epoch:03d}" / "val_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    sample_files = val_files[:n_samples]

    for gf in sample_files:
        data     = json.loads(gf.read_text(encoding="utf-8"))
        pa       = data.get("pa") or data.get("plot_area")
        gt_segs  = data.get("segments", [])
        img_path = gf.parent / gf.name.replace("gt_", "plot_").replace(".json", ".png")
        img_bgr  = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        x0, y0, x1, y1 = pa["x0"], pa["y0"], pa["x1"], pa["y1"]
        crop = img_bgr[y0:y1, x0:x1]
        h_c, w_c = crop.shape[:2]

        crop_rs = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        gray_rs = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        img_t   = torch.from_numpy(gray_rs).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
        pred_segs = _extract_predictions(out, IMG_SIZE)

        # GT segments in resized pixel coords
        sx, sy = IMG_SIZE / max(w_c, 1), IMG_SIZE / max(h_c, 1)
        gt_px = np.array([
            [(s["x1"] - x0) * sx, (s["y1"] - y0) * sy,
             (s["x2"] - x0) * sx, (s["y2"] - y0) * sy]
            for s in gt_segs
        ], dtype=np.float32) if gt_segs else np.zeros((0, 4), dtype=np.float32)

        # Draw
        canvas = cv2.cvtColor(gray_rs * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        canvas = _draw_segments(canvas, gt_px,   color=(0, 200, 0),  thickness=2)
        canvas = _draw_segments(canvas, pred_segs, color=(0, 0, 220), thickness=1)

        stem = gf.stem.replace("gt_", "")
        cv2.imwrite(str(img_dir / f"{stem}.png"), canvas)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train(n_plots: int = 1000, epochs: int = EPOCHS,
          use_resnet: bool = True) -> None:

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate plots if needed ─────────────────────────────────────
    existing = sorted(SYNTH_DIR.glob("plot_*.png"))
    if len(existing) < n_plots:
        print(f"\n  Generating {n_plots - len(existing)} additional plots …")
        # Import plot generator from chart_marker_detector_v3
        sys.path.insert(0, str(_HERE))
        try:
            from chart_marker_detector_v3 import generate_one_plot
        except ImportError:
            print("[ERROR] chart_marker_detector_v3.py not found in the same directory.")
            sys.exit(1)
        t0 = time.time()
        for i in range(len(existing), n_plots):
            idx = i + 1
            generate_one_plot((idx, str(SYNTH_DIR)))
            _pbar(idx - len(existing), n_plots - len(existing), t0,
                  prefix="  Generating: ")
        print()

    # ── Step 2: Migrate GT segments ───────────────────────────────────────────
    _migrate_gt_segments(SYNTH_DIR)

    # ── Step 3: Build dataset split ───────────────────────────────────────────
    all_gt = sorted(SYNTH_DIR.glob("gt_*.json"))
    if not all_gt:
        print("[ERROR] No GT files found. Run chart_marker_detector_v3.py first.")
        sys.exit(1)

    rng = random.Random(42)
    rng.shuffle(all_gt)
    n_val   = max(1, int(len(all_gt) * VAL_FRAC))
    val_gt  = all_gt[:n_val]
    train_gt = all_gt[n_val:]
    print(f"\n  Dataset: {len(train_gt)} train  |  {len(val_gt)} val")

    train_ds = SegmentDataset(train_gt)
    val_ds   = SegmentDataset(val_gt)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, collate_fn=_collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=1,          shuffle=False,
                          num_workers=0, collate_fn=_collate_fn)

    # ── Step 4: Build model ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = LETRModel(use_resnet=use_resnet).to(device)
    matcher   = _HungarianMatcher()
    criterion = _SetCriterion(matcher).to(device)

    # Separate LR for backbone vs rest
    backbone_params = list(model.backbone.parameters())
    other_params    = [p for p in model.parameters()
                       if not any(p is q for q in backbone_params)]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": other_params,    "lr": LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda")) if device.type == "cuda" else GradScaler(enabled=False)

    # Resume if checkpoint exists
    ckpt_path = MODEL_DIR / "letr_best.pth"
    best_val  = float("inf")
    start_ep  = 1
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best_val = ck.get("best_val", best_val)
        start_ep = ck.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_ep - 1}  (best_val={best_val:.4f})")

    # Metrics CSV
    csv_path = EPOCH_DIR / "training_metrics.csv"
    if not csv_path.exists():
        csv_path.write_text(
            "epoch,train_loss,val_loss,val_ce,val_l1,val_TP,val_FP,val_FN,"
            "val_precision,val_recall,val_F1\n",
            encoding="utf-8"
        )

    # ── Training epochs ───────────────────────────────────────────────────────
    print()
    for epoch in range(start_ep, epochs + 1):
        # Train
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for step, (imgs, targets) in enumerate(train_dl, 1):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss, _, _ = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _pbar(step, len(train_dl), t0,
                  prefix=f"  Epoch {epoch:3d}/{epochs} train: ")
        scheduler.step()
        train_loss = total_loss / len(train_dl)

        # Validate
        model.eval()
        val_loss_sum = val_ce_sum = val_l1_sum = 0.0
        total_TP = total_FP = total_FN = 0
        t0v = time.time()
        with torch.no_grad():
            for vstep, (imgs, targets) in enumerate(val_dl, 1):
                imgs = imgs.to(device)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss, ce, l1 = criterion(outputs, targets)
                val_loss_sum += loss.item()
                val_ce_sum   += ce.item()
                val_l1_sum   += l1.item()

                # Segment-level metrics
                pred_segs = _extract_predictions(outputs, IMG_SIZE)
                gt_lines  = targets[0]["lines"].numpy() * IMG_SIZE
                TP, FP, FN = _match_segments(pred_segs, gt_lines, IMG_SIZE)
                total_TP += TP; total_FP += FP; total_FN += FN
                _pbar(vstep, len(val_dl), t0v,
                      prefix=f"  Epoch {epoch:3d}/{epochs} val:   ")

        val_loss = val_loss_sum / len(val_dl)
        val_ce   = val_ce_sum   / len(val_dl)
        val_l1   = val_l1_sum   / len(val_dl)
        prec = total_TP / max(total_TP + total_FP, 1)
        rec  = total_TP / max(total_TP + total_FN, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)

        # Console summary
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
              f"  CE={val_ce:.4f}  L1={val_l1:.4f}"
              f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        # CSV
        try:
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                        f"{val_ce:.6f},{val_l1:.6f},"
                        f"{total_TP},{total_FP},{total_FN},"
                        f"{prec:.6f},{rec:.6f},{f1:.6f}\n")
        except PermissionError:
            print(f"  [WARN] Cannot write CSV (file locked): {csv_path}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
            }, str(ckpt_path))

        # Save annotated val images
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            _save_epoch_val_images(epoch, model, val_gt, device, EPOCH_DIR)

    print(f"\n  Training complete. Best val loss: {best_val:.4f}")
    print(f"  Model saved to: {ckpt_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DETECT (inference on a single image)
# ══════════════════════════════════════════════════════════════════════════════
def detect(image_path: str, conf_threshold: float = 0.5,
           use_resnet: bool = True) -> np.ndarray:
    """
    Run LETR on a single plotting-area crop.
    Returns (N, 4) array of detected segments in pixel coords.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = LETRModel(use_resnet=use_resnet).to(device)
    ckpt_path = MODEL_DIR / "letr_best.pth"
    if not ckpt_path.exists():
        print(f"[ERROR] No trained model found at {ckpt_path}")
        sys.exit(1)
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        sys.exit(1)

    h, w    = img_bgr.shape[:2]
    crop_rs = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    gray    = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    img_t   = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)

    segs_norm = _extract_predictions(out, IMG_SIZE, conf_threshold)
    # Scale back to original image size
    segs_orig = segs_norm.copy()
    segs_orig[:, [0, 2]] *= w / IMG_SIZE
    segs_orig[:, [1, 3]] *= h / IMG_SIZE
    return segs_orig


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LETR Chart Segment Detector")
    ap.add_argument("--mode",    choices=["train", "detect"], default="train")
    ap.add_argument("--plots",   type=int,   default=1000,
                    help="Number of synthetic plots to generate/use")
    ap.add_argument("--epochs",  type=int,   default=EPOCHS)
    ap.add_argument("--image",   type=str,   default=None,
                    help="Path to plotting-area crop (detect mode)")
    ap.add_argument("--conf",    type=float, default=0.5,
                    help="Confidence threshold for segment prediction")
    ap.add_argument("--no-resnet", action="store_true",
                    help="Use lightweight CNN backbone instead of ResNet-50")
    args = ap.parse_args()

    if args.mode == "train":
        train(n_plots=args.plots, epochs=args.epochs,
              use_resnet=not args.no_resnet)
    else:
        if not args.image:
            print("[ERROR] --image is required in detect mode.")
            sys.exit(1)
        segs = detect(args.image, conf_threshold=args.conf,
                      use_resnet=not args.no_resnet)
        print(f"Detected {len(segs)} segments:")
        for i, s in enumerate(segs):
            print(f"  [{i:3d}]  ({s[0]:.1f},{s[1]:.1f}) → ({s[2]:.1f},{s[3]:.1f})")
