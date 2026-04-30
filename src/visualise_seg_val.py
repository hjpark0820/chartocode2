"""
visualise_seg_val.py
====================
Visualises the validation results saved by chart_segment_detector.py.

Reads the per-epoch CSV metrics and the saved validation images from
  data/seg_epoch_logs/

Produces:
  1. A multi-panel training curve plot (loss, dist loss, angle loss,
     pixel precision, recall, F1) saved as:
       data/seg_epoch_logs/training_curves.png

  2. A contact sheet of the 8 validation images saved at a chosen epoch,
     with GT segments (green) and predicted segments (red), saved as:
       data/seg_epoch_logs/val_sheet_epoch_NNN.png

USAGE
-----
  # Show curves + contact sheet for the latest epoch that has saved images
  python visualise_seg_val.py

  # Show contact sheet for a specific epoch
  python visualise_seg_val.py --epoch 10

  # Show curves only (no contact sheet)
  python visualise_seg_val.py --curves-only

  # Show contact sheet only
  python visualise_seg_val.py --sheet-only --epoch 10
"""

from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
_SRC_DIR      = Path(__file__).parent
EPOCH_LOG_DIR = _SRC_DIR / ".." / "data" / "seg_epoch_logs"
METRICS_CSV   = EPOCH_LOG_DIR / "training_metrics.csv"


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD METRICS CSV
# ══════════════════════════════════════════════════════════════════════════════
def load_metrics(csv_path: Path) -> dict[str, list]:
    """
    Load the training_metrics.csv produced by chart_segment_detector.py.
    Returns a dict mapping column name → list of float values.
    """
    if not csv_path.exists():
        print(f"[ERROR] Metrics file not found: {csv_path}")
        print("Run chart_segment_detector.py first to generate training metrics.")
        sys.exit(1)

    data: dict[str, list] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(v)
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING CURVES PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(data: dict[str, list],
                         out_path: Path) -> None:
    """
    Plot training and validation loss curves plus pixel-level metrics.
    Saves the figure to out_path.
    """
    epochs = data.get("epoch", list(range(1, len(next(iter(data.values()))) + 1)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("chart_segment_detector — Training Curves", fontsize=13)

    def _plot(ax, key_pairs, title, ylabel, ylim=None):
        for key, label, color in key_pairs:
            if key in data:
                ax.plot(epochs, data[key], label=label, color=color, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0],
          [("train_loss", "Train loss", "#1f77b4"),
           ("val_loss",   "Val loss",   "#ff7f0e")],
          "Total Loss", "Loss")

    _plot(axes[0, 1],
          [("val_dist_loss", "Val dist loss", "#2ca02c")],
          "Distance Field Loss", "L1 Loss")

    _plot(axes[0, 2],
          [("val_angle_loss", "Val angle loss", "#d62728")],
          "Angle Field Loss", "Wrapped L1 Loss")

    _plot(axes[1, 0],
          [("val_pixel_precision", "Precision", "#9467bd")],
          "Pixel Precision", "Precision", ylim=(0, 1))

    _plot(axes[1, 1],
          [("val_pixel_recall", "Recall", "#8c564b")],
          "Pixel Recall", "Recall", ylim=(0, 1))

    _plot(axes[1, 2],
          [("val_pixel_f1", "F1", "#e377c2")],
          "Pixel F1 Score", "F1", ylim=(0, 1))

    # Print best epoch table
    if "val_pixel_f1" in data and data["val_pixel_f1"]:
        best_idx  = int(np.argmax(data["val_pixel_f1"]))
        best_ep   = int(epochs[best_idx])
        best_f1   = data["val_pixel_f1"][best_idx]
        best_prec = data.get("val_pixel_precision", [None])[best_idx]
        best_rec  = data.get("val_pixel_recall",    [None])[best_idx]
        best_loss = data.get("val_loss",             [None])[best_idx]
        fig.text(0.5, 0.01,
                 f"Best epoch: {best_ep}  |  "
                 f"Val loss: {best_loss:.4f}  |  "
                 f"Precision: {best_prec:.4f}  |  "
                 f"Recall: {best_rec:.4f}  |  "
                 f"F1: {best_f1:.4f}",
                 ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  EPOCH METRICS TABLE (console)
# ══════════════════════════════════════════════════════════════════════════════
def print_metrics_table(data: dict[str, list]) -> None:
    """Print a formatted table of per-epoch metrics to the console."""
    epochs = data.get("epoch", list(range(1, len(next(iter(data.values()))) + 1)))
    keys   = ["train_loss", "val_loss", "val_dist_loss", "val_angle_loss",
              "val_pixel_precision", "val_pixel_recall", "val_pixel_f1"]
    avail  = [k for k in keys if k in data]

    header = f"  {'Epoch':>6}  " + "  ".join(f"{k:>22}" for k in avail)
    print("\n" + header)
    print("  " + "-" * (8 + 24 * len(avail)))
    for i, ep in enumerate(epochs):
        row = f"  {int(ep):>6}  "
        for k in avail:
            val = data[k][i] if i < len(data[k]) else float("nan")
            row += f"  {val:>22.4f}"
        print(row)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CONTACT SHEET OF VALIDATION IMAGES
# ══════════════════════════════════════════════════════════════════════════════
def make_val_contact_sheet(epoch: int, out_path: Path) -> None:
    """
    Load the 8 validation images saved for the given epoch and arrange them
    in a 2×4 grid contact sheet.  Each image is a side-by-side panel:
      left  = original crop
      right = GT (green) + predicted (red) segments
    """
    img_dir = EPOCH_LOG_DIR / f"epoch_{epoch:03d}" / "val_images"
    if not img_dir.exists():
        print(f"[ERROR] No validation images found for epoch {epoch}: {img_dir}")
        return

    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"[ERROR] No PNG files in {img_dir}")
        return

    imgs = []
    for p in img_files[:8]:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(img)

    if not imgs:
        print(f"[ERROR] Could not load any images from {img_dir}")
        return

    # Resize all to the same height
    target_h = 300
    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_h),
                                  interpolation=cv2.INTER_AREA))

    # Pad to uniform width
    max_w = max(r.shape[1] for r in resized)
    padded = []
    for r in resized:
        h, w = r.shape[:2]
        pad = np.full((h, max_w - w, 3), 220, dtype=np.uint8)
        padded.append(np.hstack([r, pad]))

    # Arrange in 2 rows of 4
    n_cols = 4
    rows   = []
    for row_i in range(0, len(padded), n_cols):
        row_imgs = padded[row_i:row_i + n_cols]
        # Pad row to n_cols if needed
        while len(row_imgs) < n_cols:
            row_imgs.append(np.full_like(padded[0], 220))
        rows.append(np.hstack(row_imgs))

    sheet = np.vstack(rows)

    # Add epoch label at top
    label_bar = np.full((30, sheet.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(label_bar,
                f"Epoch {epoch}  —  Green: GT segments  |  Red: Predicted segments",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)
    sheet = np.vstack([label_bar, sheet])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    print(f"  Contact sheet   → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIND LATEST EPOCH WITH SAVED IMAGES
# ══════════════════════════════════════════════════════════════════════════════
def find_latest_epoch_with_images() -> int | None:
    """Return the highest epoch number that has a val_images directory."""
    epoch_dirs = sorted(EPOCH_LOG_DIR.glob("epoch_*/val_images"))
    if not epoch_dirs:
        return None
    # Extract epoch numbers
    nums = []
    for d in epoch_dirs:
        try:
            nums.append(int(d.parent.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(nums) if nums else None


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise chart_segment_detector validation results.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Epoch to show contact sheet for "
                             "(default: latest epoch with saved images).")
    parser.add_argument("--curves-only", action="store_true",
                        help="Plot training curves only, skip contact sheet.")
    parser.add_argument("--sheet-only", action="store_true",
                        help="Make contact sheet only, skip training curves.")
    args = parser.parse_args()

    # ── Training curves ───────────────────────────────────────────────────────
    if not args.sheet_only:
        data = load_metrics(METRICS_CSV)
        print_metrics_table(data)
        curves_out = EPOCH_LOG_DIR / "training_curves.png"
        plot_training_curves(data, curves_out)

    # ── Contact sheet ─────────────────────────────────────────────────────────
    if not args.curves_only:
        epoch = args.epoch
        if epoch is None:
            epoch = find_latest_epoch_with_images()
            if epoch is None:
                print("\n  No epoch images found yet. "
                      "Run chart_segment_detector.py first.")
                return
            print(f"  Using latest epoch with images: {epoch}")

        sheet_out = EPOCH_LOG_DIR / f"val_sheet_epoch_{epoch:03d}.png"
        make_val_contact_sheet(epoch, sheet_out)


if __name__ == "__main__":
    main()
