"""
chart_preprocessing.py
======================
Grayscale/black-and-white chart noise removal module for chartocode2 pipeline.

Adapted from run_A4_auto_v22.py (colour-based pipeline) to work on
monochrome scientific charts (patent figures, journal plots, etc.) where
all curves, axes, text, and legend are drawn in black/grey on white.

Key operations
--------------
1. detect_axes(img_bgr)
   -> (axis_row, axis_col)  pixel coordinates of x-axis row and y-axis column

2. detect_plot_area(img_bgr, axis_row, axis_col)
   -> (x0, y0, x1, y1)  plotting rectangle (inside the axes)

3. detect_legend_box(img_bgr, plot_area)
   -> (x0, y0, x1, y1) or None

4. detect_lloq_line(binary_mask, plot_area)
   -> row index or None

5. remove_noise_from_binary(binary_mask, img_bgr,
                             axis_row, axis_col, plot_area,
                             legend_box, lloq_row)
   -> cleaned binary mask (axes, legend, LLOQ line, text removed)

6. preprocess(img_bgr)  -- convenience wrapper: runs all steps and returns
   -> dict with keys:
        'plot_area'    : (x0,y0,x1,y1) or None
        'axis_row'     : int or None
        'axis_col'     : int or None
        'legend_box'   : (x0,y0,x1,y1) or None
        'lloq_row'     : int or None
        'clean_fn'     : callable(binary_mask) -> cleaned_mask
        'debug_img'    : annotated BGR image showing detections

Public usage in pipeline
------------------------
    from chart_preprocessing import preprocess
    info = preprocess(img_bgr)
    # pass info['clean_fn'] to segment_detection and point_detection
"""

from __future__ import annotations
import cv2
import numpy as np
import math
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# ── tuneable constants ────────────────────────────────────────────────────────
AXIS_PAD          = 4      # px to blank around detected axis lines
MIN_AXIS_RUN_FRAC = 0.35   # axis line must span >= this fraction of image
TICK_MARGIN       = 18     # px around axis to consider "axis-adjacent"
TICK_AREA_MAX     = 250    # max component area to remove as tick/noise
CHAR_MAX_AREA     = 150    # max component area to consider as a text character
TEXT_CLUSTER_R    = 65     # radius for clustering text characters
LLOQ_DASH_MIN     = 6      # min dash segments to classify a row as LLOQ line
LLOQ_SPAN_FRAC    = 0.50   # LLOQ dashes must span >= this fraction of plot width
LLOQ_SEG_MAX_LEN  = 30     # individual dash segment must be <= this px long
LLOQ_SEG_MIN_LEN  = 3      # individual dash segment must be >= this px long
LLOQ_Y_BAND       = 18     # px band around detected LLOQ row to clear
LLOQ_COMP_MAX     = 400    # components larger than this are protected (curve/marker)
LLOQ_GAP_MIN      = 3      # minimum gap between dashes (px)
LLOQ_GAP_MAX      = 60     # maximum gap between dashes (px)
LLOQ_CV_MAX       = 0.45   # max coefficient of variation for dash lengths
SHAPE_LINE_SPAN   = 0.45   # component spanning > this fraction of image = line
SHAPE_LINE_ASPECT = 8.0    # and aspect ratio above this = axis/grid line
SHAPE_ERRBAR_MIN  = 25     # thin (<=2px) strokes longer than this = error bar
LEGEND_MIN_COMPS  = 6      # min connected components inside candidate legend box
LEGEND_MAX_FRAC   = 0.35   # legend box must be <= this fraction of image area
LEGEND_RIGHT_FRAC = 0.50   # legend is usually in the right half or bottom
LEGEND_BOTTOM_FRAC= 0.50   # legend is usually in the bottom half
PLOT_AREA_PAD     = 4      # extra pad inside axes when defining plot area


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _dark_binary(img_bgr: np.ndarray, thresh: int = 128) -> np.ndarray:
    """Binary mask: True where pixel is dark (foreground ink)."""
    gray = _to_gray(img_bgr)
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return (bw > 0).astype(np.uint8)


def _max_run(px_1d: np.ndarray) -> int:
    """Longest contiguous run of True values in a 1-D boolean array."""
    m = run = 0
    for p in px_1d:
        if p:
            run += 1
            m = max(m, run)
        else:
            run = 0
    return m


def _row_segments(xs: np.ndarray):
    """Split sorted x-coords into contiguous run lengths (gap>2 = new segment)."""
    segs = []
    if len(xs) == 0:
        return segs
    s, e = xs[0], xs[0]
    for x in xs[1:]:
        if x - e <= 2:
            e = x
        else:
            segs.append(e - s + 1)
            s, e = x, x
    segs.append(e - s + 1)
    return segs


# ── Step 1: Axis detection ────────────────────────────────────────────────────

def detect_axes(img_bgr: np.ndarray):
    """
    Detect the x-axis (horizontal rule) and y-axis (vertical rule) of a
    black-and-white chart.

    Strategy (adapted from run_A4_auto_v22._detect_axes):
      - Threshold to get dark pixels.
      - For each row/column, compute the longest continuous dark run.
      - A row with run >= MIN_AXIS_RUN_FRAC * W is a candidate x-axis.
      - A column with run >= MIN_AXIS_RUN_FRAC * H is a candidate y-axis.
      - Among candidates, pick the bottom-most row (x-axis) and left-most
        column (y-axis) -- the standard chart frame convention.

    Returns
    -------
    axis_row : int or None   -- y-pixel of the x-axis
    axis_col : int or None   -- x-pixel of the y-axis
    """
    H, W = img_bgr.shape[:2]
    gray = _to_gray(img_bgr)

    # Two passes: strict (very dark) then relaxed (medium dark)
    for thresh in (100, 160):
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        dark = (bw > 0).astype(np.uint8)

        # --- x-axis candidates (horizontal rows) ---
        row_runs = []
        for r in range(2, H - 2):
            run = _max_run(dark[r, :])
            if run >= MIN_AXIS_RUN_FRAC * W:
                row_runs.append((r, run))

        # --- y-axis candidates (vertical columns) ---
        col_runs = []
        for c in range(2, W - 2):
            run = _max_run(dark[:, c])
            if run >= MIN_AXIS_RUN_FRAC * H:
                col_runs.append((c, run))

        if row_runs or col_runs:
            break

    axis_row = None
    axis_col = None

    if row_runs:
        # x-axis: bottom-most strong candidate (below the data region)
        # Group nearby rows and pick the densest
        row_runs.sort(key=lambda x: x[0])
        # take the bottom-most group
        bottom_r = max(r for r, _ in row_runs)
        # cluster: all rows within 5px of bottom_r
        cluster = [r for r, _ in row_runs if abs(r - bottom_r) <= 5]
        axis_row = int(np.median(cluster))

    if col_runs:
        # y-axis: left-most strong candidate
        col_runs.sort(key=lambda x: x[0])
        left_c = min(c for c, _ in col_runs)
        cluster = [c for c, _ in col_runs if abs(c - left_c) <= 5]
        axis_col = int(np.median(cluster))

    # Fallback: if one axis is missing, try to infer from the other
    # and from dense dark bands at the image margins
    if axis_row is None:
        # look for a dense dark row in the bottom 40% of the image
        bottom_start = int(H * 0.55)
        col_sums = dark[bottom_start:, :].sum(axis=1).astype(float)
        if col_sums.max() > W * 0.3:
            best = int(np.argmax(col_sums)) + bottom_start
            axis_row = best

    if axis_col is None:
        # look for a dense dark column in the left 40% of the image
        right_end = int(W * 0.45)
        row_sums = dark[:, :right_end].sum(axis=0).astype(float)
        if row_sums.max() > H * 0.3:
            best = int(np.argmax(row_sums))
            axis_col = best

    return axis_row, axis_col


# ── Step 2: Plot area ─────────────────────────────────────────────────────────

def detect_plot_area(img_bgr: np.ndarray,
                     axis_row: int | None,
                     axis_col: int | None):
    """
    Define the plotting rectangle from the detected axes.

    Returns (x0, y0, x1, y1) or None if axes are missing.
    The rectangle is the region INSIDE the axes (where data lives).

    Strategy for right boundary:
      - Look for a right-side vertical axis line (if the chart has a frame).
      - Otherwise, estimate from the rightmost dense dark column in the
        data region (excluding margin text columns).
    """
    H, W = img_bgr.shape[:2]
    if axis_row is None and axis_col is None:
        return None

    pad = PLOT_AREA_PAD
    x0 = (axis_col + pad) if axis_col is not None else 0
    y0 = 0
    y1 = (axis_row - pad) if axis_row is not None else H - 1

    # ── Detect right boundary ────────────────────────────────────────────
    # Look for a right-side vertical axis line (strong vertical run)
    gray = _to_gray(img_bgr)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    dark = (bw > 0).astype(np.uint8)

    # Search for right-axis in the right half of the image
    right_col = None
    search_start = int(W * 0.5)
    for c in range(W - 2, search_start, -1):
        run = _max_run(dark[:y1 + 1, c] if axis_row else dark[:, c])
        if run >= MIN_AXIS_RUN_FRAC * (y1 if axis_row else H):
            right_col = c
            break

    if right_col is not None:
        x1 = right_col - pad
    else:
        # Fallback: estimate right boundary from data density.
        # Use the plot rows (y0..y1) and find the rightmost column with
        # meaningful dark pixel density, ignoring the far-right margin.
        # Exclude the rightmost 15% of the image (likely margin text).
        max_x_search = int(W * 0.85)
        data_region = dark[y0:y1 + 1, x0:max_x_search]
        col_sums = data_region.sum(axis=0).astype(float)
        if col_sums.max() > 0:
            # Find the rightmost column with >= 5% of peak density
            threshold = col_sums.max() * 0.05
            dense_cols = np.where(col_sums >= threshold)[0]
            if len(dense_cols) > 0:
                x1 = int(dense_cols.max()) + x0 + pad
            else:
                x1 = max_x_search
        else:
            x1 = max_x_search

    x1 = min(x1, W - 1)

    # Sanity: plot area must be at least 10% of image in each dimension
    if (x1 - x0) < W * 0.10 or (y1 - y0) < H * 0.10:
        return None

    return (int(x0), int(y0), int(x1), int(y1))


# ── Step 3: Legend detection ──────────────────────────────────────────────────

def detect_legend_box(img_bgr: np.ndarray,
                      plot_area: tuple | None,
                      axis_row: int | None):
    """
    Detect a legend panel in a black-and-white chart.

    The legend is typically:
      - Below the x-axis (below axis_row) OR in the lower-right of the plot.
      - A rectangular cluster of small connected components (marker glyphs +
        text labels) that is clearly separated from the main data region.

    Strategy:
      1. Look in the region below the x-axis (if detected).
      2. Find connected components; cluster them spatially.
      3. Accept as legend if the cluster has enough components and is compact.

    Returns (x0, y0, x1, y1) or None.
    """
    H, W = img_bgr.shape[:2]
    gray = _to_gray(img_bgr)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Define search region: below x-axis (if known) or bottom 35% of image
    if axis_row is not None:
        search_y0 = axis_row + AXIS_PAD + 2
    else:
        search_y0 = int(H * 0.65)
    search_y1 = H

    if search_y0 >= search_y1:
        return None

    region = bw[search_y0:search_y1, :]
    n, lbl, stats, centroids = cv2.connectedComponentsWithStats(region, 8)

    if n < LEGEND_MIN_COMPS + 1:
        return None

    # Collect all non-trivial components (area 3..500)
    comps = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 3 <= a <= 600:
            cx = float(centroids[i][0])
            cy = float(centroids[i][1]) + search_y0
            comps.append((cx, cy, stats[i, cv2.CC_STAT_LEFT],
                          stats[i, cv2.CC_STAT_TOP] + search_y0,
                          stats[i, cv2.CC_STAT_WIDTH],
                          stats[i, cv2.CC_STAT_HEIGHT]))

    if len(comps) < LEGEND_MIN_COMPS:
        return None

    # Cluster components by proximity (simple grid-based grouping)
    # Use a sliding window: find the densest rectangular cluster
    xs_arr = np.array([c[0] for c in comps])
    ys_arr = np.array([c[1] for c in comps])

    # Exclude components that are likely to be in the right-side margin
    # (e.g. patent text running vertically on the right side of the image).
    # Heuristic: if a component is in the rightmost 15% of the image AND
    # below the plot area, it is likely margin text, not legend.
    right_margin_x = W * 0.85
    comps_filtered = [c for c in comps
                      if not (c[0] > right_margin_x)]
    if len(comps_filtered) >= LEGEND_MIN_COMPS:
        comps = comps_filtered

    # Bounding box of all legend-region components
    bx0 = int(min(c[2] for c in comps))
    by0 = int(min(c[3] for c in comps))
    bx1 = int(max(c[2] + c[4] for c in comps))
    by1 = int(max(c[3] + c[5] for c in comps))

    # Sanity checks
    box_w = bx1 - bx0
    box_h = by1 - by0
    box_area_frac = (box_w * box_h) / float(H * W)

    if box_area_frac > LEGEND_MAX_FRAC:
        # Too large -- probably the whole bottom margin, not a legend box
        # Try to tighten: find the densest sub-region
        # Use row/col density to find the actual legend extent
        sub = bw[by0:by1+1, bx0:bx1+1]
        if sub.size == 0:
            return None
        row_d = sub.sum(axis=1).astype(float)
        col_d = sub.sum(axis=0).astype(float)
        # Keep rows/cols with at least 5% of max density
        thr_r = max(3, row_d.max() * 0.05)
        thr_c = max(3, col_d.max() * 0.05)
        dense_rows = np.where(row_d >= thr_r)[0]
        dense_cols = np.where(col_d >= thr_c)[0]
        if len(dense_rows) == 0 or len(dense_cols) == 0:
            return None
        by0 = by0 + int(dense_rows.min())
        by1 = by0 + int(dense_rows.max()) + 1
        bx0 = bx0 + int(dense_cols.min())
        bx1 = bx0 + int(dense_cols.max()) + 1
        box_area_frac = ((bx1-bx0)*(by1-by0)) / float(H*W)
        if box_area_frac > LEGEND_MAX_FRAC:
            return None

    # Add a small padding
    pad = 6
    return (max(0, bx0 - pad), max(0, by0 - pad),
            min(W - 1, bx1 + pad), min(H - 1, by1 + pad))


# ── Step 4: LLOQ line detection ───────────────────────────────────────────────

def detect_lloq_line(binary_mask: np.ndarray,
                     plot_area: tuple | None):
    """
    Detect a horizontal dashed reference line (e.g. LLOQ = ...) inside the
    plot area.

    A dashed line has:
      - Multiple short dash segments of similar length
      - Gaps between segments of similar size
      - Spans >= LLOQ_SPAN_FRAC of the plot width

    Returns row index (int) or None.
    """
    if plot_area is None:
        H, W = binary_mask.shape[:2]
        px0, py0, px1, py1 = 0, 0, W - 1, H - 1
    else:
        px0, py0, px1, py1 = plot_area

    plot_w = px1 - px0
    if plot_w < 20:
        return None

    for y in range(py0, py1 + 1):
        row = binary_mask[y, px0:px1 + 1]
        if row.sum() == 0:
            continue
        xs = np.where(row > 0)[0]
        x_span = int(xs.max()) - int(xs.min()) + 1
        if x_span < plot_w * LLOQ_SPAN_FRAC:
            continue
        seg_lens = np.array(_row_segments(xs), dtype=float)
        if len(seg_lens) < LLOQ_DASH_MIN:
            continue
        if seg_lens.max() > LLOQ_SEG_MAX_LEN:
            continue
        if seg_lens.min() < LLOQ_SEG_MIN_LEN:
            continue
        # Segment lengths must be similar (low CV)
        if seg_lens.mean() > 0 and seg_lens.std() / seg_lens.mean() > LLOQ_CV_MAX:
            continue
        # Gaps between segments
        seg_starts, seg_ends = [], []
        s, e = xs[0], xs[0]
        for x in xs[1:]:
            if x - e <= 2:
                e = x
            else:
                seg_ends.append(e)
                seg_starts.append(x)
                s, e = x, x
        seg_ends.append(e)
        if len(seg_starts) >= 2:
            gaps = np.array([seg_starts[i] - seg_ends[i - 1]
                             for i in range(1, len(seg_starts))], dtype=float)
            # Gaps must be within expected range for a dashed line
            if gaps.min() < LLOQ_GAP_MIN or gaps.max() > LLOQ_GAP_MAX:
                continue
            if gaps.mean() > 0 and gaps.std() / gaps.mean() > LLOQ_CV_MAX:
                continue
        else:
            # Need at least 2 gaps to confirm dashed pattern
            continue
        return int(y)

    return None



# ── Axis-line removal with marker preservation ──────────────────────────────────

def _remove_axis_lines_preserve_markers(
        mask,
        plot_area,
        axis_row,
        axis_col,
        H, W,
        border_px=6,
        marker_min_area=15,
        marker_max_aspect=3.5,
):
    # Remove x/y axis lines inside plot area while preserving data markers.
    # Strategy: row/col erase then restore compact blobs (markers).
    if plot_area is None:
        return mask

    m = mask.copy().astype('uint8')
    orig = mask.copy().astype('uint8')
    px0, py0, px1, py1 = plot_area
    px0 = max(0, px0); py0 = max(0, py0)
    px1 = min(W - 1, px1); py1 = min(H - 1, py1)
    plot_w = max(1, px1 - px0)
    plot_h = max(1, py1 - py0)

    erased_rows = set()
    erased_cols = set()

    def _longest_continuous_run(arr):
        """Return the length of the longest continuous run of non-zero values."""
        max_run = cur = 0
        for v in arr:
            if v > 0:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        return max_run

    # ── Find and erase x-axis row ─────────────────────────────────────────────
    # Search the bottom 40% of plot area (axis can be anywhere near bottom).
    # Only erase a row if it has a CONTINUOUS run >= 60% of plot_w
    # (i.e. an unbroken line -- not a row that just happens to have many markers).
    search_h = max(border_px, int(plot_h * 0.40))
    band_y0 = max(py0, py1 - search_h)
    band_y1 = py1
    best_run, best_row = 0, None
    for r in range(band_y0, band_y1 + 1):
        run = _longest_continuous_run(m[r, px0:px1 + 1])
        if run > best_run:
            best_run, best_row = run, r
    # Threshold: continuous run must be >= 60% of plot width
    # (a row with scattered markers will NOT reach this threshold)
    if best_row is not None and best_run >= plot_w * 0.60:
        for r in range(max(0, best_row - 1), min(H, best_row + 2)):
            m[r, px0:px1 + 1] = 0
            erased_rows.add(r)

    # ── Find and erase y-axis col ─────────────────────────────────────────────
    # Search the left 20% of plot area.
    # A y-axis line is a near-vertical line along the left edge.
    # Markers interrupt it, so the longest CONTINUOUS run may be < 60%.
    # Use a combined criterion:
    #   - longest continuous run >= 40% of plot_h, OR
    #   - total filled pixels >= 55% of plot_h
    # AND the column must be in the leftmost 20% of the plot area.
    search_w = max(border_px, int(plot_w * 0.20))
    band_x0 = px0
    band_x1 = min(px1, px0 + search_w)
    best_score, best_col = 0, None
    for c in range(band_x0, band_x1 + 1):
        col_arr = m[py0:py1 + 1, c]
        longest = _longest_continuous_run(col_arr)
        total   = int(col_arr.sum())
        # Score: max of (longest/plot_h) and (total/plot_h), weighted
        score = max(longest / plot_h, total / plot_h * 0.8)
        if score > best_score:
            best_score, best_col = score, c
    # Threshold: score >= 0.40 (i.e. longest run >= 40% OR total >= 50%)
    if best_col is not None and best_score >= 0.40:
        for c in range(max(0, best_col - 1), min(W, best_col + 2)):
            m[py0:py1 + 1, c] = 0
            erased_cols.add(c)

    if not erased_rows and not erased_cols:
        return m

    # Restore marker fragments cut by the erase.
    # Strategy: scan the ORIGINAL mask in a wide band around the erased rows/cols.
    # For each connected blob in the original, check if it is compact (marker-like).
    # If compact AND its centroid does NOT lie on an erased row/col, restore it.
    # This prevents restoring the axis line itself while recovering cut markers.
    def _restore_markers(bx0, by0, bx1, by1):
        band_orig = orig[by0:by1 + 1, bx0:bx1 + 1].copy()
        n, lbl, stats, centroids = cv2.connectedComponentsWithStats(band_orig, 8)
        for i in range(1, n):
            a  = int(stats[i, cv2.CC_STAT_AREA])
            bw_s = int(stats[i, cv2.CC_STAT_WIDTH])
            bh_s = int(stats[i, cv2.CC_STAT_HEIGHT])
            if a < marker_min_area:
                continue
            long_side  = max(bw_s, bh_s)
            short_side = max(1, min(bw_s, bh_s))
            aspect     = long_side / max(short_side, 1)
            if aspect > marker_max_aspect:
                continue
            # Get pixel coordinates of this blob in image space
            ys, xs = np.where(lbl == i)
            ys_img = ys + by0
            xs_img = xs + bx0
            # Count how many pixels fall on erased rows or cols
            on_erased = np.sum(
                np.isin(ys_img, list(erased_rows)) |
                np.isin(xs_img, list(erased_cols))
            )
            # If > 50% of the blob's pixels are on erased lines, it IS the axis line
            # (or a fragment of it) -- do NOT restore.
            if on_erased / max(1, a) > 0.50:
                continue
            m[ys_img, xs_img] = orig[ys_img, xs_img]

    if erased_rows:
        min_er = max(py0, min(erased_rows) - 20)
        max_er = min(py1, max(erased_rows) + 20)
        _restore_markers(px0, min_er, px1, max_er)

    if erased_cols:
        min_ec = max(px0, min(erased_cols) - 20)
        max_ec = min(px1, max(erased_cols) + 20)
        _restore_markers(min_ec, py0, max_ec, py1)

    return m



# ── Step 5: Noise removal from binary mask ────────────────────────────────────

def remove_noise_from_binary(binary_mask: np.ndarray,
                              img_bgr: np.ndarray,
                              axis_row: int | None,
                              axis_col: int | None,
                              plot_area: tuple | None,
                              legend_box: tuple | None,
                              lloq_row: int | None) -> np.ndarray:
    """
    Remove structural noise from a binary (dark-on-white) mask:

    Pass 0: Remove axis lines and the region outside the plot area.
    Pass 1: Remove long thin strokes (axis rules, grid lines, error bars)
            by connected-component shape analysis.
    Pass 2: Remove LLOQ dashed reference line and its inline text label.
    Pass 3: Remove legend box contents (marker glyphs + text labels).
    Pass 4: Remove text characters (small clustered components outside
            the plot area -- axis labels, title, annotations).
    Pass 5: Remove axis-adjacent tick marks and small noise fragments.

    The mask is modified in-place and returned.
    """
    H, W = binary_mask.shape[:2]
    out = binary_mask.copy().astype(np.uint8)

    # ── Pass 0: blank everything outside the plot area ────────────────────────
    if plot_area is not None:
        px0, py0, px1, py1 = plot_area
        outside = np.ones((H, W), dtype=np.uint8)
        outside[py0:py1 + 1, px0:px1 + 1] = 0
        out[outside > 0] = 0

    # ── Pass 1: shape-based axis/grid/error-bar removal ───────────────────────
    out = _drop_long_thin_strokes(out, plot_area, H, W)

    # ── Pass 1b: remove axis border lines inside plot area (preserve markers) ──
    out = _remove_axis_lines_preserve_markers(
        out, plot_area, axis_row, axis_col, H, W
    )

    # ── Pass 2: LLOQ dashed line removal ─────────────────────────────────────
    if lloq_row is not None:
        y_lo = max(0, lloq_row - LLOQ_Y_BAND)
        y_hi = min(H, lloq_row + LLOQ_Y_BAND + 1)
        # Only remove small components centred near the LLOQ row
        tmp = out.copy()
        tmp[:y_lo, :] = 0
        tmp[y_hi:, :] = 0
        n, lbl, stats, cen = cv2.connectedComponentsWithStats(tmp, 8)
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a <= LLOQ_COMP_MAX:
                out[lbl == i] = 0

    # ── Pass 3: legend box removal ────────────────────────────────────────────
    if legend_box is not None:
        lx0, ly0, lx1, ly1 = legend_box
        box_area_frac = ((lx1 - lx0 + 1) * (ly1 - ly0 + 1)) / float(H * W)
        if box_area_frac < LEGEND_MAX_FRAC:
            # Remove all components whose bounding box lies inside the legend
            n, lbl, stats, cen = cv2.connectedComponentsWithStats(out, 8)
            for i in range(1, n):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                inside = (x >= lx0 - 3 and y >= ly0 - 3 and
                          x + w <= lx1 + 3 and y + h <= ly1 + 3)
                if inside:
                    out[lbl == i] = 0

    # ── Pass 4: text character removal (outside plot area) ───────────────────
    out = _remove_text_components(out, plot_area, H, W)

    # ── Pass 4b: aggressive margin text removal ───────────────────────────────
    # Remove ALL components outside the plot area that are small (text-sized).
    # This catches rotated patent text, axis labels, title, etc.
    out = _remove_all_margin_components(out, plot_area, H, W)

    # ── Pass 5: axis-adjacent tick / noise removal ────────────────────────────
    out = _remove_axis_ticks(out, axis_row, axis_col, H, W)

    return out


def _drop_long_thin_strokes(mask: np.ndarray,
                             plot_area: tuple | None,
                             H: int, W: int) -> np.ndarray:
    """Remove connected components that are long, thin, straight rules
    (axis lines, grid lines, error bars). Curve/marker blobs are compact
    and are left untouched."""
    m = mask.copy()
    if m.sum() == 0:
        return m
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ref_w = (plot_area[2] - plot_area[0]) if plot_area else W
    ref_h = (plot_area[3] - plot_area[1]) if plot_area else H
    for i in range(1, n):
        a    = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 3:
            continue
        long_side  = max(w, h)
        short_side = max(1, min(w, h))
        aspect     = long_side / short_side
        # Spans a large fraction of the plot in one direction
        spans = (w >= ref_w * SHAPE_LINE_SPAN) or (h >= ref_h * SHAPE_LINE_SPAN)
        # Drop: extremely elongated AND spans most of the plot
        if aspect >= SHAPE_LINE_ASPECT and short_side <= 6 and spans:
            m[lbl == i] = 0
        # Drop thin vertical/horizontal error-bar strokes
        elif short_side <= 2 and long_side > SHAPE_ERRBAR_MIN:
            m[lbl == i] = 0
    return m


def _remove_text_components(mask: np.ndarray,
                             plot_area: tuple | None,
                             H: int, W: int) -> np.ndarray:
    """Remove small clustered components that look like text characters.
    Text characters are:
      - Small area (< CHAR_MAX_AREA)
      - Aspect ratio 0.2 .. 4.0 (not extreme vertical ticks)
      - Clustered near other similar components (words have multiple chars)
    Only removes components OUTSIDE the plot area to protect data markers.
    """
    m = mask.copy()
    if m.sum() == 0:
        return m

    # Build an "outside plot" mask for restricting text removal
    outside = np.ones((H, W), dtype=bool)
    if plot_area is not None:
        px0, py0, px1, py1 = plot_area
        outside[py0:py1 + 1, px0:px1 + 1] = False

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    small_comps = []
    largest_area = 0
    largest_idx  = -1

    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > largest_area:
            largest_area = a
            largest_idx  = i
        if a < CHAR_MAX_AREA:
            wi = int(stats[i, cv2.CC_STAT_WIDTH])
            hi = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = stats[i, cv2.CC_STAT_LEFT] + wi / 2.0
            cy = stats[i, cv2.CC_STAT_TOP]  + hi / 2.0
            aspect = wi / max(hi, 1)
            if 0.2 <= aspect <= 4.0:
                # Only consider components outside the plot area
                if outside[int(cy), int(cx)]:
                    small_comps.append((i, cx, cy, a))

    if len(small_comps) < 2:
        return m

    sc_arr = np.array([[cx, cy] for _, cx, cy, _ in small_comps], dtype=float)
    diffs  = sc_arr[:, np.newaxis, :] - sc_arr[np.newaxis, :, :]
    dists  = np.sqrt((diffs ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    neighbour_count = (dists < TEXT_CLUSTER_R).sum(axis=1)

    for k, (i, cx_i, cy_i, area) in enumerate(small_comps):
        if neighbour_count[k] >= 2 and area < largest_area:
            m[lbl == i] = 0

    return m


def _remove_all_margin_components(mask: np.ndarray,
                                   plot_area: tuple | None,
                                   H: int, W: int) -> np.ndarray:
    """Remove ALL connected components that lie entirely outside the plot area.
    This is a catch-all for axis labels, title text, patent margin text, etc.
    Components that overlap with the plot area are preserved."""
    if plot_area is None:
        return mask
    m = mask.copy()
    if m.sum() == 0:
        return m
    px0, py0, px1, py1 = plot_area
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        cx = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Component bounding box
        cx1 = cx + cw
        cy1 = cy + ch
        # Check if it overlaps with the plot area
        overlap = not (cx1 < px0 or cx > px1 or cy1 < py0 or cy > py1)
        if not overlap:
            m[lbl == i] = 0
    return m


def _remove_axis_ticks(mask: np.ndarray,
                       axis_row: int | None,
                       axis_col: int | None,
                       H: int, W: int) -> np.ndarray:
    """Remove small thin components adjacent to the detected axes
    (tick marks, axis-line fragments, small noise)."""
    m = mask.copy()
    if m.sum() == 0:
        return m
    axis_rows = [axis_row] if axis_row is not None else []
    axis_cols = [axis_col] if axis_col is not None else []
    n, lbl, stats, cen = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        a   = int(stats[i, cv2.CC_STAT_AREA])
        if a >= TICK_AREA_MAX:
            continue
        wi  = int(stats[i, cv2.CC_STAT_WIDTH])
        hi  = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx_i = float(cen[i][0])
        cy_i = float(cen[i][1])
        near = (any(abs(cy_i - ay) <= TICK_MARGIN for ay in axis_rows) or
                any(abs(cx_i - ac) <= TICK_MARGIN for ac in axis_cols))
        thin = min(wi, hi) <= 3
        if near and thin:
            m[lbl == i] = 0
    return m


# ── Step 6: Convenience wrapper ───────────────────────────────────────────────

# Margin added around user-supplied plot area for preprocessing
# (to include axis lines that sit just outside the dragged rectangle)
AXIS_MARGIN = 15


def preprocess(img_bgr: np.ndarray,
               user_plot_area: tuple | None = None,
               verbose: bool = True) -> dict:
    """
    Run the full preprocessing pipeline on a BGR chart image.

    Parameters
    ----------
    img_bgr        : BGR numpy array (the chart image)
    user_plot_area : optional (x0,y0,x1,y1) override from GUI drag
                     A margin of AXIS_MARGIN px is added around this box
                     for axis detection/removal, but coordinate conversion
                     always uses the original dragged box.
    verbose        : print progress messages

    Returns
    -------
    dict with keys:
        'axis_row'       : int or None
        'axis_col'       : int or None
        'plot_area'      : (x0,y0,x1,y1) expanded area used for preprocessing
        'user_plot_area' : (x0,y0,x1,y1) original dragged area (coord conversion)
        'legend_box'     : (x0,y0,x1,y1) or None
        'lloq_row'       : int or None
        'clean_fn'       : callable(binary_mask) -> cleaned_mask
        'debug_img'      : annotated BGR image
    """
    H, W = img_bgr.shape[:2]

    # 1. Axes
    axis_row, axis_col = detect_axes(img_bgr)
    if verbose:
        print(f"[preprocess] axis_row={axis_row}, axis_col={axis_col}")

    # 2. Plot area
    # If user supplied a drag box, expand it by AXIS_MARGIN for preprocessing
    # so that axis lines just outside the drag box are also captured.
    user_orig = user_plot_area  # original drag box (for coord conversion)
    if user_plot_area is not None:
        ux0, uy0, ux1, uy1 = user_plot_area
        plot_area = (
            max(0, ux0 - AXIS_MARGIN),
            max(0, uy0 - AXIS_MARGIN),
            min(W - 1, ux1 + AXIS_MARGIN),
            min(H - 1, uy1 + AXIS_MARGIN),
        )
    else:
        plot_area = detect_plot_area(img_bgr, axis_row, axis_col)
        user_orig = plot_area
    if verbose:
        print(f"[preprocess] user_plot_area={user_orig}")
        print(f"[preprocess] expanded plot_area={plot_area}")

    # 3. Legend
    legend_box = detect_legend_box(img_bgr, plot_area, axis_row)
    if verbose:
        print(f"[preprocess] legend_box={legend_box}")

    # 4. LLOQ line (needs a binary mask of the plot area)
    gray = _to_gray(img_bgr)
    _, bw_full = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bw_full = (bw_full > 0).astype(np.uint8)
    lloq_row = detect_lloq_line(bw_full, plot_area)
    if verbose:
        print(f"[preprocess] lloq_row={lloq_row}")

    # 5. Build clean_fn closure
    def clean_fn(binary_mask: np.ndarray) -> np.ndarray:
        return remove_noise_from_binary(
            binary_mask, img_bgr,
            axis_row, axis_col, plot_area,
            legend_box, lloq_row
        )

    # 6. Debug visualisation
    debug_img = img_bgr.copy()
    if axis_row is not None:
        cv2.line(debug_img, (0, axis_row), (W - 1, axis_row), (0, 0, 255), 2)
    if axis_col is not None:
        cv2.line(debug_img, (axis_col, 0), (axis_col, H - 1), (255, 0, 0), 2)
    if plot_area is not None:
        px0, py0, px1, py1 = plot_area
        cv2.rectangle(debug_img, (px0, py0), (px1, py1), (0, 200, 0), 2)
    if legend_box is not None:
        lx0, ly0, lx1, ly1 = legend_box
        cv2.rectangle(debug_img, (lx0, ly0), (lx1, ly1), (200, 0, 200), 2)
    if lloq_row is not None:
        cv2.line(debug_img, (0, lloq_row), (W - 1, lloq_row), (0, 200, 200), 2)

    return {
        'axis_row':       axis_row,
        'axis_col':       axis_col,
        'plot_area':      plot_area,      # expanded (for noise removal)
        'user_plot_area': user_orig,      # original drag box (for coord conversion)
        'legend_box':     legend_box,
        'lloq_row':       lloq_row,
        'clean_fn':       clean_fn,
        'debug_img':      debug_img,
    }


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    if len(sys.argv) < 2:
        print("Usage: python chart_preprocessing.py <image_path> [out_dir]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(img_path)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: cannot read {img_path}")
        sys.exit(1)

    info = preprocess(img, verbose=True)

    # Save debug image
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(out_dir, f"{stem}_preprocess_debug.png"),
                info['debug_img'])

    # Show cleaned binary mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bw = (bw > 0).astype(np.uint8)
    cleaned = info['clean_fn'](bw)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_cleaned_mask.png"),
                (cleaned * 255).astype(np.uint8))

    print(f"\nSaved debug image and cleaned mask to: {out_dir}")
