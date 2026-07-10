"""
run_gui.py
==========
Local GUI front-end for the chartocode2 chart-digitisation pipeline.

Run from the src/ directory:
    python run_gui.py

Requirements (install once):
    pip install opencv-python-headless scipy matplotlib pillow

Workflow
--------
1. Load a **cropped plotting-area image** (PNG / JPG / TIFF).
   → The entire image is automatically treated as the plotting area.
2. Optionally drag a rectangle to define the **Legend Area** (if legend
   is inside the cropped image).
3. Check which of the 12 marker classes are present in the chart.
4. Enter X-axis range (min / max) and Y-axis range (min / max).
5. Choose X-axis scale (Linear / Log10) and Y-axis scale (Linear / Log10).
6. Click **Run Detection**.
7. Results are saved next to the input image:
     <stem>_detected.png   – original image with detected markers overlaid
     <stem>_data.csv       – detected data points (class, x_data, y_data)

Notes
-----
- The pipeline (1_point_detection_v3.py … 5_correction.py) is loaded
  dynamically from the same directory as this script.
- If the trained model (../models/chart_marker_net_v3.pth) is not found,
  the ViT-based detector is skipped and only the segment detector runs.
- The preprocessing module (chart_preprocessing.py) is used to suppress
  axis lines, legend, LLOQ lines, and text noise before detection.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import cv2
import numpy as np

# ── Resolve paths ─────────────────────────────────────────────────────────────
SRC_DIR      = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "chart_marker_net_v3.pth"

# ── Marker class definitions (from 1_point_detection_v3.py) ──────────────────
ALL_MARKERS = [
    ("filled_circle",       "●  Filled Circle"),
    ("open_circle",         "○  Open Circle"),
    ("filled_square",       "■  Filled Square"),
    ("open_square",         "□  Open Square"),
    ("open_triangle",       "△  Open Triangle (up)"),
    ("open_inv_triangle",   "▽  Open Triangle (down)"),
    ("filled_triangle",     "▲  Filled Triangle (up)"),
    ("filled_inv_triangle", "▼  Filled Triangle (down)"),
    ("open_rhombus",        "◇  Open Rhombus"),
    ("filled_rhombus",      "◆  Filled Rhombus"),
    ("x_marker",            "✕  X Marker"),
    ("plus_marker",         "+  Plus Marker"),
]

# Colour for each marker class overlay
MARKER_COLORS = {
    "filled_circle":       (220,  30,  30),   # vivid red
    "open_circle":         (230, 100,   0),   # deep orange
    "filled_square":       ( 20, 160,  20),   # vivid green
    "open_square":         (  0, 130, 200),   # sky blue
    "open_triangle":       ( 80,  30, 220),   # deep violet
    "open_inv_triangle":   (200,   0, 200),   # magenta
    "filled_triangle":     (  0, 190, 190),   # cyan
    "filled_inv_triangle": (180,  60, 180),   # purple
    "open_rhombus":        (220, 180,   0),   # gold/yellow
    "filled_rhombus":      (160,  80,   0),   # brown
    "x_marker":            ( 30, 180, 100),   # teal-green
    "plus_marker":         (255,  20, 120),   # hot pink
}

# ── Dynamic module loader ─────────────────────────────────────────────────────
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# ── Coordinate conversion helpers ────────────────────────────────────────────
def px_to_data(px: float, py: float,
               plot_area_px: tuple,
               x_range: tuple, y_range: tuple,
               x_log: bool, y_log: bool) -> tuple[float, float]:
    """Convert pixel coordinates inside the plot area to data coordinates."""
    ax0, ay0, ax1, ay1 = plot_area_px
    # Normalise to [0,1]
    fx = (px - ax0) / max(ax1 - ax0, 1)
    fy = (py - ay0) / max(ay1 - ay0, 1)
    fy = 1.0 - fy   # y-axis is inverted in image coords

    x_min, x_max = x_range
    y_min, y_max = y_range

    if x_log:
        lx0 = math.log10(max(x_min, 1e-300))
        lx1 = math.log10(max(x_max, 1e-300))
        x_data = 10 ** (lx0 + fx * (lx1 - lx0))
    else:
        x_data = x_min + fx * (x_max - x_min)

    if y_log:
        ly0 = math.log10(max(y_min, 1e-300))
        ly1 = math.log10(max(y_max, 1e-300))
        y_data = 10 ** (ly0 + fy * (ly1 - ly0))
    else:
        y_data = y_min + fy * (y_max - y_min)

    return x_data, y_data


# ── Dual Y-axis boundary detection ──────────────────────────────────────────
def detect_right_yaxis(img_bgr: np.ndarray,
                       plot_area_px: tuple,
                       min_dark_frac: float = 0.25) -> int | None:
    """
    Detect the x-coordinate of a right-side Y-axis line inside the plot area.
    Scans the right half of the plot area for a column with high dark-pixel
    fraction (i.e. a continuous vertical black line).

    Returns the x-coordinate of the right Y-axis, or None if not found.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ax0, ay0, ax1, ay1 = plot_area_px
    mid_x = (ax0 + ax1) // 2

    best_x, best_frac = None, 0.0
    for x in range(mid_x, min(ax1 + 30, gray.shape[1])):
        col = gray[ay0:ay1, x]
        dark_frac = float(np.sum(col < 50)) / max(len(col), 1)
        if dark_frac > best_frac:
            best_frac = dark_frac
            best_x = x

    # Only accept if clearly a solid vertical line (>25% dark)
    if best_frac >= min_dark_frac:
        return best_x
    return None


# ── Detection logic ───────────────────────────────────────────────────────────
def run_detection(img_bgr: np.ndarray,
                  plot_area_px: tuple,
                  legend_area_px: tuple | None,
                  known_classes: list[str],
                  x_range: tuple,
                  y_range: tuple,
                  x_log: bool,
                  y_log: bool,
                  has_errorbars: bool | None = None,
                  upscale: float = 1.0,
                  conf_thresh: float | None = None,
                  stride: int | None = None,
                  has_lines: bool = True,
                  log_fn=print) -> dict:
    """
    Run the chartocode2 pipeline restricted to the user-specified areas.

    Returns
    -------
    dict with keys:
        'detections'  : list of {class_name, cx_px, cy_px, x_data, y_data}
        'overlay_img' : BGR image with detections drawn
    """
    H, W = img_bgr.shape[:2]
    ax0, ay0, ax1, ay1 = plot_area_px
    _orig_img_bgr      = img_bgr          # keep original for overlay
    _orig_plot_area_px = plot_area_px     # keep original for overlay
    _orig_legend_px    = legend_area_px   # keep original for overlay

    # ── Upscale: auto-detect from legend if upscale=None ──────────────────────
    if upscale is None and legend_area_px is not None:
        try:
            from chart_preprocessing import estimate_optimal_scale as _eos
            _scale_result = _eos(img_bgr, legend_box=legend_area_px)
            # estimate_optimal_scale returns (scale, info_dict) or just scale
            if isinstance(_scale_result, tuple):
                upscale = float(_scale_result[0])
            else:
                upscale = float(_scale_result)
            log_fn(f"[Step 0a] Auto upscale from legend: {upscale}x")
        except Exception as _e:
            log_fn(f"[Step 0a] Auto upscale failed ({_e}); using 1.0")
            upscale = 1.0
    # Support both upscale (>1) and downscale (<1); 1.0 = no resize.
    # Formula: diameter * 2 * scale = 19px  →  scale = 9.5 / diameter
    _upscale = float(upscale) if upscale is not None else 1.0
    if _upscale == 0.0:
        _upscale = 1.0

    if _upscale != 1.0:
        new_w = int(round(W * _upscale))
        new_h = int(round(H * _upscale))
        interp = cv2.INTER_CUBIC if _upscale > 1.0 else cv2.INTER_AREA
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
        plot_area_px   = tuple(int(v * _upscale) for v in plot_area_px)
        if legend_area_px is not None:
            legend_area_px = tuple(int(v * _upscale) for v in legend_area_px)
        ax0, ay0, ax1, ay1 = plot_area_px
        H, W = img_bgr.shape[:2]
        log_fn(f"[Step 0a] Scaled x{_upscale}: {W}x{H}  plot_area={plot_area_px}")

    # ── Build preprocessing info from user-supplied areas ─────────────────
    log_fn("[Step 0] Building preprocessing info from user areas …")
    try:
        from chart_preprocessing import preprocess as _cp
        # preprocess() will expand plot_area_px by AXIS_MARGIN for noise removal
        # but keeps user_plot_area for coordinate conversion.
        prep_info = _cp(img_bgr,
                         user_plot_area=plot_area_px,
                         user_legend_box=legend_area_px,
                         verbose=False)
        log_fn(f"  expanded plot_area = {prep_info['plot_area']}")
        log_fn(f"  user plot_area     = {prep_info.get('user_plot_area', plot_area_px)}")
        log_fn(f"  legend_box = {prep_info['legend_box']}")
        log_fn(f"  lloq_row   = {prep_info['lloq_row']}")
    except ImportError:
        log_fn("  chart_preprocessing not found; skipping noise removal.")
        prep_info = None

    # ── Segment detection + error-bar removal (lines mode only) ─────────────
    segs = []
    _img_for_vit = img_bgr
    _prep_info_for_vit = prep_info
    _eb_info_list = []

    if not has_lines:
        log_fn("[Step 1] No-lines mode: segment detection skipped.")
        log_fn("[Step 1b] No-lines mode: error-bar removal skipped.")
    else:
        # ── Error-bar gate (Function 1 auto-detect or GUI override) ──────────
        # has_errorbars=True  → always run stem removal (Function 2)
        # has_errorbars=False → always skip
        # has_errorbars=None  → auto-detect via detect_has_errorbars() (Function 1)
        if has_errorbars is None:
            try:
                from chart_preprocessing import detect_has_errorbars as _deb
                has_errorbars = _deb(img_bgr, prep_info=prep_info)
                log_fn(f"  [Function 1] detect_has_errorbars → {has_errorbars}")
            except Exception:
                has_errorbars = False
                log_fn("  [Function 1] auto-detect failed; assuming no error bars.")

        # ── Stage 2: segment detection ────────────────────────────────────────
        log_fn("[Step 1] Segment detection …")
        seg_v2   = SRC_DIR / "3_segment_detection_v2.py"
        seg_orig = SRC_DIR / "3_segment_detection.py"
        seg_path = seg_v2 if seg_v2.exists() else seg_orig
        log_fn(f"  Loading: {seg_path.name}")
        mod3 = _load("segment_detector", seg_path)
        import inspect as _insp
        _seg_sig = _insp.signature(mod3.detect)
        if 'prep_info' in _seg_sig.parameters:
            segs = mod3.detect(img_bgr, prep_info=prep_info)
        else:
            segs = mod3.detect(img_bgr)
        log_fn(f"  {len(segs)} segments detected")

        # ── Function 2: Error-bar stem + T-cap removal (if has_errorbars) ────
        if has_errorbars and prep_info is not None:
            log_fn("[Step 1b] Error-bar stem + T-cap removal (Function 2) …")
            try:
                from chart_preprocessing import remove_errorbars_from_mask as _rem_eb
                import math as _math_eb

                def _seg_to_dict_eb(s):
                    x1, y1, x2, y2 = s
                    dx = abs(x2 - x1); dy = abs(y2 - y1)
                    angle = _math_eb.degrees(_math_eb.atan2(dy, dx + 1e-9))
                    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle}

                segs_dicts = [_seg_to_dict_eb(s) for s in segs]
                vert_segs  = [s for s in segs_dicts if abs(s['angle'] - 90) <= 20]
                if vert_segs:
                    _gray_eb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, _bw_eb = cv2.threshold(_gray_eb, 128, 255, cv2.THRESH_BINARY_INV)
                    _bw_eb = (_bw_eb > 0).astype('uint8')
                    _mask_eb = prep_info['clean_fn'](_bw_eb)
                    _mask_no_stem, _eb_info_list = _rem_eb(_mask_eb, vert_segs, segs)
                    log_fn(f"  Removed {len(vert_segs)} stem(s); "
                           f"{len(_eb_info_list)} stem(s) processed.")
                    # Build stem-erased BGR image for ViT
                    _img_no_stem = img_bgr.copy()
                    _stem_px = (_mask_eb.astype('uint8') - _mask_no_stem.astype('uint8')).clip(0, 1)
                    _img_no_stem[_stem_px == 1] = 255
                    _img_for_vit = _img_no_stem
                    # Wrap stem-free mask into prep_info for ViT
                    _orig_clean_fn = prep_info['clean_fn']
                    def _make_sfn(sfm, ofn):
                        def _sfn(bw): return np.minimum(ofn(bw), sfm)
                        return _sfn
                    _prep_info_for_vit = dict(prep_info)
                    _prep_info_for_vit['clean_fn'] = _make_sfn(_mask_no_stem, _orig_clean_fn)
                else:
                    log_fn("  No vertical stems found; skipping removal.")
            except Exception as _e_eb:
                import traceback as _tb_eb
                log_fn(f"  Error-bar removal failed: {_e_eb}")
                log_fn(_tb_eb.format_exc())
        elif not has_errorbars:
            log_fn("[Step 1b] Error-bar removal skipped (has_errorbars=False).")

    # ── Stage 1 (ViT point detection) if model exists ────────────────────
    kept = []
    if MODEL_PATH.exists() and known_classes:
        log_fn("[Step 2] ViT point detection (adaptive NMS) …")
        try:
            # Prefer _v2 version (supports prep_info); fall back to original
            nms_v2   = SRC_DIR / "2_point_detection_adaptive_nms_v2.py"
            nms_orig = SRC_DIR / "2_point_detection_adaptive_nms.py"
            nms_path = nms_v2 if nms_v2.exists() else nms_orig
            log_fn(f"  Loading: {nms_path.name}")
            mod2 = _load("adaptive_nms", nms_path)

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                tmp_path = tf.name
            cv2.imwrite(tmp_path, _img_for_vit)

            import inspect
            sig = inspect.signature(mod2.detect_with_adaptive_nms)
            supports_prep = 'prep_info' in sig.parameters

            call_kwargs = dict(
                img_path         = tmp_path,
                model_path       = str(MODEL_PATH),
                known_classes    = known_classes,
                detector_py_path = str(SRC_DIR / "1_point_detection_v3.py"),
            )
            if conf_thresh is not None:
                call_kwargs['conf_thresh'] = conf_thresh
            if stride is not None:
                call_kwargs['stride'] = stride
            # When upscaled, KDE-based d_est can be unstable (more raw detections
            # cause finer mode splitting). Instead, estimate d on the ORIGINAL image
            # and scale it by _upscale to get the correct NMS bin width.
            import math as _math
            _sig_nms = inspect.signature(mod2.detect_with_adaptive_nms)
            if _upscale > 1.0 and 'd_override' in _sig_nms.parameters:
                try:
                    import tempfile as _tf2, os as _os2
                    with _tf2.NamedTemporaryFile(suffix='.png', delete=False) as _tf2f:
                        _tmp2 = _tf2f.name
                    cv2.imwrite(_tmp2, _orig_img_bgr)
                    _r_orig = mod2.detect_with_adaptive_nms(
                        img_path         = _tmp2,
                        model_path       = str(MODEL_PATH),
                        known_classes    = known_classes,
                        detector_py_path = str(SRC_DIR / '1_point_detection_v3.py'),
                    )
                    _os2.unlink(_tmp2)
                    _d_orig = _r_orig.get('d_est', None)
                    if _d_orig is not None:
                        _d_scaled = _d_orig * _upscale
                        call_kwargs['d_override'] = _d_scaled
                        log_fn(f"  d_override={_d_scaled:.1f}px (orig d_est={_d_orig:.1f} x {_upscale})")
                except Exception as _de:
                    log_fn(f"  d_override estimation failed: {_de}")
            if supports_prep:
                # Compute tight scan boundary:
                # - Left  boundary: axis_col (left Y-axis x) from prep_info
                # - Right boundary: right Y-axis x (detected) or user plot_area x1
                # This prevents scanning axis tick/label areas on either side.
                _vit_prep = dict(_prep_info_for_vit)
                _scan_x0, _scan_y0, _scan_x1, _scan_y1 = plot_area_px

                # Y-axis boundary detection with margin
                # Margin keeps ViT windows clear of the axis lines themselves.
                _AXIS_MARGIN_PX = 10

                # Left Y-axis: use axis_col from prep_info
                _axis_col = prep_info.get('axis_col', None) if prep_info else None
                if _axis_col is not None:
                    _cand_x0 = _axis_col + _AXIS_MARGIN_PX
                    if _cand_x0 > _scan_x0:
                        _scan_x0 = _cand_x0
                        log_fn(f"  Left Y-axis at x={_axis_col}; scan x0 \u2192 {_scan_x0} (+{_AXIS_MARGIN_PX}px margin)")

                # Right Y-axis: detect vertical line in right half of plot area
                _right_yaxis = detect_right_yaxis(img_bgr, plot_area_px)
                if _right_yaxis is not None:
                    _cand_x1 = _right_yaxis - _AXIS_MARGIN_PX
                    if _cand_x1 < _scan_x1:
                        _scan_x1 = _cand_x1
                        log_fn(f"  Right Y-axis at x={_right_yaxis}; scan x1 \u2192 {_scan_x1} (-{_AXIS_MARGIN_PX}px margin)")

                _vit_prep['plot_area'] = (_scan_x0, _scan_y0, _scan_x1, _scan_y1)
                call_kwargs['prep_info'] = _vit_prep
                log_fn(f"  ViT scan range: x=[{_scan_x0},{_scan_x1}] y=[{_scan_y0},{_scan_y1}]")
            else:
                log_fn("  (prep_info not supported by this version – skipping noise filter)")

            result2 = mod2.detect_with_adaptive_nms(**call_kwargs)
            os.unlink(tmp_path)
            kept = result2["kept"]
            log_fn(f"  {len(kept)} markers detected by ViT")
        except Exception as e:
            import traceback
            log_fn(f"  ViT detection failed: {e}")
            log_fn(traceback.format_exc())
            kept = []
    else:
        if not MODEL_PATH.exists():
            log_fn("[Step 2] Model not found – ViT detection skipped.")
            log_fn(f"  (expected: {MODEL_PATH})")
        else:
            log_fn("[Step 2] No marker classes selected – ViT detection skipped.")

    # ── Filter detections to plot area (excluding legend area) ─────────────
    log_fn("[Step 3] Filtering detections to plot area …")
    def _in_plot_not_legend(d):
        cx = d.get('cx', d.get('cx_px', -1))
        cy = d.get('cy', d.get('cy_px', -1))
        # Must be inside plot area
        if not (ax0 <= cx <= ax1 and ay0 <= cy <= ay1):
            return False
        # Must NOT be inside legend area (if specified)
        if legend_area_px is not None:
            lx0, ly0, lx1, ly1 = legend_area_px
            if lx0 <= cx <= lx1 and ly0 <= cy <= ly1:
                return False
        return True
    kept_in = [d for d in kept if _in_plot_not_legend(d)]
    log_fn(f"  {len(kept_in)} markers inside plot area (legend excluded)")

    # ── Scale coordinates back to original image space if scaled ───────────
    if _upscale != 1.0:
        for d in kept_in:
            if 'cx' in d: d['cx'] = d['cx'] / _upscale
            if 'cy' in d: d['cy'] = d['cy'] / _upscale
        # Restore original-scale plot_area_px for coordinate conversion
        ax0, ay0, ax1, ay1 = plot_area_px
        ax0 = int(ax0 / _upscale); ay0 = int(ay0 / _upscale)
        ax1 = int(ax1 / _upscale); ay1 = int(ay1 / _upscale)
        plot_area_px = (ax0, ay0, ax1, ay1)
        if legend_area_px is not None:
            legend_area_px = tuple(int(v / _upscale) for v in legend_area_px)

    # ── Convert pixel → data coordinates ────────────────────────────────────────────────────────────────────
    log_fn("[Step 4] Converting pixel → data coordinates …")
    detections = []
    for d in kept_in:
        cx_px = d.get('cx', d.get('cx_px', 0))
        cy_px = d.get('cy', d.get('cy_px', 0))
        xd, yd = px_to_data(cx_px, cy_px, plot_area_px,
                             x_range, y_range, x_log, y_log)
        detections.append({
            'class_name': d['class_name'],
            'cx_px':      cx_px,
            'cy_px':      cy_px,
            'x_data':     xd,
            'y_data':     yd,
            'confidence': d.get('confidence', 1.0),
        })

    # Sort by x_data
    detections.sort(key=lambda d: d['x_data'])

    # ── Build overlay image (always on original-scale image) ────────────
    log_fn("[Step 5] Building overlay image …")
    overlay = _orig_img_bgr.copy()
    _oax0, _oay0, _oax1, _oay1 = _orig_plot_area_px

    # Draw plot area rectangle
    cv2.rectangle(overlay, (_oax0, _oay0), (_oax1, _oay1), (0, 200, 0), 2)

    # Draw legend area rectangle
    if _orig_legend_px is not None:
        lx0, ly0, lx1, ly1 = _orig_legend_px
        cv2.rectangle(overlay, (lx0, ly0), (lx1, ly1), (200, 0, 200), 2)

    # Dynamic overlay sizes proportional to image dimensions
    _oh, _ow = overlay.shape[:2]
    _ref_dim  = max(_oh, _ow)                        # longest side
    _r_outer  = max(4, int(_ref_dim * 0.013))        # outer circle radius
    _r_inner  = max(2, int(_r_outer * 0.25))         # inner dot radius
    _font_sc  = max(0.30, _ref_dim * 0.00065)        # font scale
    _thickness = max(1, int(_ref_dim * 0.002))       # line thickness

    # Draw detected markers (cx_px/cy_px already in original-scale coords)
    for d in detections:
        cx = int(round(float(d['cx_px'])))
        cy = int(round(float(d['cy_px'])))
        color = MARKER_COLORS.get(d['class_name'], (255, 0, 0))
        # BGR order
        color_bgr = (color[2], color[1], color[0])
        cv2.circle(overlay, (cx, cy), _r_outer, color_bgr, _thickness)
        cv2.circle(overlay, (cx, cy), _r_inner, color_bgr, -1)
        # Label
        short = d['class_name'].replace('_marker', '').replace('_', ' ')
        cv2.putText(overlay, short, (cx + _r_outer + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, _font_sc, color_bgr, _thickness, cv2.LINE_AA)

    # ── Extract legend text labels ───────────────────────────────────────────────
    legend_labels: dict[str, str] = {}
    if _orig_legend_px is not None:
        try:
            from chart_preprocessing import extract_legend_labels as _ell
            # Use the original-scale legend box and image
            _detected_classes = list(dict.fromkeys(d['class_name'] for d in detections))
            legend_labels = _ell(
                _orig_img_bgr,
                _orig_legend_px,
                known_classes=_detected_classes,
                verbose=False,
            )
            log_fn(f"  Legend labels: {legend_labels}")
        except Exception as _le:
            log_fn(f"  Legend label extraction failed: {_le}")

    # Attach series_label to each detection
    for d in detections:
        d['series_label'] = legend_labels.get(d['class_name'], '')

    log_fn(f"[Done] {len(detections)} data points found.")
    return {
        'detections':  detections,
        'overlay_img': overlay,
        'segs':        segs,
        'legend_labels': legend_labels,
    }


# ── GUI Application ───────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────

class CorrectionWindow(tk.Toplevel):
    """
    Interactive correction editor opened after detection.

    Features
    --------
    - Left panel : original image with detected markers drawn as coloured circles.
                   Mouse interactions:
                     • Left-click + drag on a marker  → move it
                     • Double-click on empty area      → add new marker
                     • Right-click on a marker         → delete it
    - Right panel: reconstructed plot (matplotlib) built from current point set,
                   updated live after every edit.
    - Bottom bar  : class selector for new markers, "Export Data" button.
    """

    # Display canvas size for the original image panel
    CANVAS_W = 820
    CANVAS_H = 620

    # Marker hit radius (canvas pixels)
    HIT_R = 10

    def __init__(self, parent: tk.Tk,
                 img_bgr: np.ndarray,
                 detections: list,
                 plot_area_px: tuple,
                 x_range: tuple,
                 y_range: tuple,
                 x_log: bool,
                 y_log: bool,
                 img_path: str | None = None,
                 target_classes: list | None = None):
        super().__init__(parent)
        self.title("✏️  Correction Editor")
        self.resizable(True, True)

        # ── State ──────────────────────────────────────────────────────────
        self._img_bgr      = img_bgr.copy()
        self._plot_area_px = plot_area_px          # (x0,y0,x1,y1) in image px
        self._x_range      = x_range
        self._y_range      = y_range
        self._x_log        = x_log
        self._y_log        = y_log
        self._img_path     = img_path

        # Working copy of detections (list of dicts with cx_px, cy_px, class_name, …)
        import copy
        self._points: list[dict] = copy.deepcopy(detections)

        # Classes available in the combobox:
        # 1) classes actually detected (preserving first-appearance order)
        # 2) + any target_classes checked in GUI but not detected (appended at end)
        # 3) fallback to ALL_MARKERS if both are empty
        _seen = []
        for d in detections:
            if d['class_name'] not in _seen:
                _seen.append(d['class_name'])
        # Merge with target_classes so user can always add any checked symbol
        _available = list(_seen)
        if target_classes:
            for cls in target_classes:
                if cls not in _available:
                    _available.append(cls)
        self._detected_classes: list[str] = _available if _available else [k for k, _ in ALL_MARKERS]

        # Canvas display scale (fit-to-canvas)
        self._scale_x = 1.0
        self._scale_y = 1.0

        # Zoom / pan state
        self._zoom: float = 1.0          # extra zoom multiplier on top of fit-scale
        self._pan_x: int  = 0            # canvas offset in pixels
        self._pan_y: int  = 0
        self._pan_drag_start: tuple | None = None  # (event.x, event.y, pan_x, pan_y)
        self._pan_mode: bool = False     # True = hand/pan mode, False = edit mode

        # Drag state (marker move)
        self._drag_idx: int | None = None
        self._drag_offset: tuple   = (0, 0)

        self._build_ui()
        # Delay first render until the window is fully laid out
        # so winfo_width/height return real values
        self.after(150, self._refresh_canvas)
        self.after(200, self._refresh_recon)

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top toolbar ───────────────────────────────────────────────────
        tb = tk.Frame(self, bd=1, relief=tk.RAISED)
        tb.pack(side=tk.TOP, fill=tk.X)

        tk.Label(tb, text="✏️  Correction Editor",
                 font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, padx=8, pady=4)

        tk.Label(tb, text="Add class:",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(20, 2))
        self._add_class_var = tk.StringVar(
            value=self._detected_classes[0] if self._detected_classes else ALL_MARKERS[0][0])
        cls_menu = ttk.Combobox(tb, textvariable=self._add_class_var,
                                values=self._detected_classes,
                                width=22, state="readonly",
                                font=("Helvetica", 9))
        cls_menu.pack(side=tk.LEFT, padx=2, pady=4)

        tk.Label(tb,
                 text="  LMB drag=move  |  Dbl-click=add  |  RMB=delete",
                 font=("Helvetica", 9), fg="#555555").pack(side=tk.LEFT, padx=10)

        tk.Button(tb, text="💾  Export Data",
                  command=self._export,
                  font=("Helvetica", 10, "bold"),
                  bg="#28a745", fg="white", padx=8).pack(side=tk.RIGHT, padx=8, pady=4)

        tk.Button(tb, text="🔄  Refresh Recon",
                  command=self._refresh_recon,
                  font=("Helvetica", 10), padx=6).pack(side=tk.RIGHT, padx=4, pady=4)

        # Pan mode toggle
        self._pan_btn = tk.Button(tb, text="✋ Pan", command=self._toggle_pan_mode,
                                  font=("Helvetica", 10), padx=6,
                                  relief=tk.RAISED, bg="SystemButtonFace")
        self._pan_btn.pack(side=tk.RIGHT, padx=4, pady=4)

        # Zoom controls
        tk.Label(tb, text="Zoom:",
                 font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=(12, 2))
        tk.Button(tb, text="➕", command=self._zoom_in,
                  font=("Helvetica", 11, "bold"), width=2).pack(side=tk.RIGHT, padx=1, pady=4)
        tk.Button(tb, text="➖", command=self._zoom_out,
                  font=("Helvetica", 11, "bold"), width=2).pack(side=tk.RIGHT, padx=1, pady=4)
        tk.Button(tb, text="1:1", command=self._zoom_reset,
                  font=("Helvetica", 9), width=3).pack(side=tk.RIGHT, padx=1, pady=4)
        self._zoom_label = tk.Label(tb, text="100%",
                                    font=("Helvetica", 9), width=5)
        self._zoom_label.pack(side=tk.RIGHT, padx=2)

        # ── Main area: left canvas + right recon ──────────────────────────
        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        # Left: original image with editable markers
        left_frame = tk.LabelFrame(main, text="Original Image  (edit markers)",
                                   font=("Helvetica", 10, "bold"))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._canvas = tk.Canvas(left_frame,
                                 width=self.CANVAS_W, height=self.CANVAS_H,
                                 bg="#aaaaaa", cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)

        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Double-Button-1>", self._on_dblclick)
        self._canvas.bind("<Button-3>",        self._on_rclick)
        # Zoom with mouse wheel
        self._canvas.bind("<MouseWheel>",      self._on_mousewheel)       # Windows/macOS
        self._canvas.bind("<Button-4>",        self._on_mousewheel)       # Linux scroll up
        self._canvas.bind("<Button-5>",        self._on_mousewheel)       # Linux scroll down
        # Pan with middle-button drag
        self._canvas.bind("<ButtonPress-2>",   self._on_pan_start)
        self._canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # Right: reconstructed plot
        right_frame = tk.LabelFrame(main, text="Reconstructed Plot",
                                    font=("Helvetica", 10, "bold"),
                                    width=480)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=4, pady=4)
        right_frame.pack_propagate(False)

        self._recon_canvas = tk.Canvas(right_frame, bg="#ffffff",
                                       width=460, height=self.CANVAS_H)
        self._recon_canvas.pack(fill=tk.BOTH, expand=True)

        # ── Status bar ────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self._status_var,
                 font=("Helvetica", 9), anchor="w",
                 relief=tk.SUNKEN, bd=1).pack(fill=tk.X, side=tk.BOTTOM)

    # ── Canvas rendering ───────────────────────────────────────────────────
    # Fixed marker sizes on canvas (pixels) — independent of zoom
    _MARKER_R  = 7    # outer ring radius
    _MARKER_R2 = 3    # inner dot radius
    _MARKER_LW = 2    # ring line width

    def _refresh_canvas(self):
        """Redraw the original image, then overlay markers as fixed-size canvas items."""
        img = self._img_bgr.copy()
        H, W = img.shape[:2]

        # Draw plot area boundary on the image (scales with zoom, that's fine)
        ax0, ay0, ax1, ay1 = self._plot_area_px
        cv2.rectangle(img, (ax0, ay0), (ax1, ay1), (0, 200, 0), 2)

        # Scale to fit canvas, then apply zoom
        cw = self._canvas.winfo_width()  or self.CANVAS_W
        ch = self._canvas.winfo_height() or self.CANVAS_H
        fit_scale = min(cw / W, ch / H)
        total_scale = fit_scale * self._zoom
        self._scale_x = total_scale
        self._scale_y = total_scale
        nw = max(1, int(W * total_scale))
        nh = max(1, int(H * total_scale))

        # Clamp pan so image doesn't go fully off-screen
        self._pan_x = max(-(nw - 20), min(cw - 20, self._pan_x))
        self._pan_y = max(-(nh - 20), min(ch - 20, self._pan_y))

        from PIL import Image as _PILImage, ImageTk as _PILImageTk
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = _PILImage.fromarray(img_rgb).resize((nw, nh), _PILImage.LANCZOS)
        self._tk_img = _PILImageTk.PhotoImage(pil)
        self._canvas.delete("all")
        self._canvas.create_image(self._pan_x, self._pan_y,
                                  anchor=tk.NW, image=self._tk_img)

        # Draw markers as fixed-size tkinter canvas items (not affected by zoom)
        R  = self._MARKER_R
        R2 = self._MARKER_R2
        LW = self._MARKER_LW
        for d in self._points:
            ccx, ccy = self._img_to_canvas(d['cx_px'], d['cy_px'])
            ccx, ccy = int(round(ccx)), int(round(ccy))
            rgb = MARKER_COLORS.get(d['class_name'], (255, 0, 0))
            hex_col = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            # Outer ring
            self._canvas.create_oval(ccx - R, ccy - R, ccx + R, ccy + R,
                                     outline=hex_col, width=LW, fill="")
            # Inner filled dot
            self._canvas.create_oval(ccx - R2, ccy - R2, ccx + R2, ccy + R2,
                                     outline=hex_col, fill=hex_col)
            # Label text
            short = d['class_name'].replace('_marker', '').replace('_', ' ')
            self._canvas.create_text(ccx + R + 3, ccy - 1,
                                     text=short, anchor=tk.W,
                                     fill=hex_col,
                                     font=("Helvetica", 8, "bold"))

        # Update zoom label
        if hasattr(self, '_zoom_label'):
            self._zoom_label.config(text=f"{int(self._zoom * 100)}%")

        self._status_var.set(
            f"{len(self._points)} markers  |  "
            f"LMB-drag=move  Dbl-click=add [{self._add_class_var.get()}]  RMB=delete"
        )

    # ── Reconstructed plot ─────────────────────────────────────────────────
    def _refresh_recon(self):
        """Render a reconstructed matplotlib plot from current points."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io
        from PIL import Image as _PILImage, ImageTk as _PILImageTk

        ax0, ay0, ax1, ay1 = self._plot_area_px
        x_range = self._x_range
        y_range = self._y_range
        x_log   = self._x_log
        y_log   = self._y_log

        # Canvas size
        cw = self._recon_canvas.winfo_width()  or 460
        ch = self._recon_canvas.winfo_height() or self.CANVAS_H

        fig, ax = plt.subplots(figsize=(cw / 96, ch / 96), dpi=96)
        fig.patch.set_facecolor("white")

        # Group by class
        from collections import defaultdict
        by_class: dict[str, list] = defaultdict(list)
        for d in self._points:
            by_class[d['class_name']].append(d)

        SYM_MPL_LOCAL = {
            'filled_circle':       ('o', True),
            'open_circle':         ('o', False),
            'filled_square':       ('s', True),
            'open_square':         ('s', False),
            'open_triangle':       ('^', False),
            'open_inv_triangle':   ('v', False),
            'filled_triangle':     ('^', True),
            'filled_inv_triangle': ('v', True),
            'open_rhombus':        ('D', False),
            'filled_rhombus':      ('D', True),
            'x_marker':            ('X', True),
            'plus_marker':         ('P', True),
        }
        COLOR_LIST = [
            '#0077BB', '#EE7733', '#009944', '#CC3311',
            '#33BBEE', '#44BB99', '#EE3377', '#999933',
            '#DDCC77', '#884400', '#BBBBBB', '#AA3377',
        ]
        class_order = list(by_class.keys())

        for i, cls in enumerate(class_order):
            pts = sorted(by_class[cls], key=lambda d: d['x_data'])
            xs  = [d['x_data'] for d in pts]
            ys  = [d['y_data'] for d in pts]
            mcode, filled = SYM_MPL_LOCAL.get(cls, ('o', True))
            color = COLOR_LIST[i % len(COLOR_LIST)]
            fc    = color if filled else 'white'
            _lbl = (pts[0].get('series_label', '') or '').strip() if pts else ''
            _lbl = _lbl if _lbl else cls.replace('_', ' ')
            ax.plot(xs, ys,
                    marker=mcode, color=color,
                    markerfacecolor=fc, markeredgecolor=color,
                    markeredgewidth=1.2, markersize=7,
                    linewidth=1.2, label=_lbl)

        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, linestyle='--', alpha=0.4)
        if len(class_order) <= 8:
            ax.legend(fontsize=7, loc='best')
        ax.set_title("Reconstructed", fontsize=9)
        fig.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=96)
        plt.close(fig)
        buf.seek(0)
        pil = _PILImage.open(buf)
        # Resize to canvas
        pil = pil.resize((cw, ch), _PILImage.LANCZOS)
        self._tk_recon_img = _PILImageTk.PhotoImage(pil)
        self._recon_canvas.delete("all")
        self._recon_canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_recon_img)

    # ── Coordinate helpers ─────────────────────────────────────────────────
    def _canvas_to_img(self, cx, cy):
        """Convert canvas pixel → original image pixel (accounting for pan)."""
        return (cx - self._pan_x) / self._scale_x, (cy - self._pan_y) / self._scale_y

    def _img_to_canvas(self, ix, iy):
        """Convert original image pixel → canvas pixel (accounting for pan)."""
        return ix * self._scale_x + self._pan_x, iy * self._scale_y + self._pan_y

    def _find_marker_at(self, cx, cy) -> int | None:
        """Return index of the nearest marker within HIT_R canvas pixels, or None."""
        best_i, best_d = None, self.HIT_R + 1
        for i, d in enumerate(self._points):
            mcx, mcy = self._img_to_canvas(d['cx_px'], d['cy_px'])
            dist = math.hypot(cx - mcx, cy - mcy)
            if dist < best_d:
                best_d = dist
                best_i = i
        return best_i

    def _update_data_coords(self, d: dict):
        """Recompute x_data / y_data from current cx_px / cy_px."""
        xd, yd = px_to_data(
            d['cx_px'], d['cy_px'],
            self._plot_area_px,
            self._x_range, self._y_range,
            self._x_log, self._y_log,
        )
        d['x_data'] = xd
        d['y_data'] = yd

    # ── Mouse events ───────────────────────────────────────────────────────
    def _on_press(self, event):
        if self._pan_mode:
            # Pan mode: start panning
            self._pan_drag_start = (event.x, event.y, self._pan_x, self._pan_y)
            return
        idx = self._find_marker_at(event.x, event.y)
        if idx is not None:
            self._drag_idx = idx
            mcx, mcy = self._img_to_canvas(
                self._points[idx]['cx_px'],
                self._points[idx]['cy_px'])
            self._drag_offset = (event.x - mcx, event.y - mcy)
        else:
            self._drag_idx = None

    def _on_drag(self, event):
        if self._pan_mode:
            # Pan mode: move image
            if self._pan_drag_start is None:
                return
            sx, sy, px0, py0 = self._pan_drag_start
            self._pan_x = px0 + (event.x - sx)
            self._pan_y = py0 + (event.y - sy)
            self._refresh_canvas()
            return
        if self._drag_idx is None:
            return
        # New canvas position (corrected for grab offset)
        ncx = event.x - self._drag_offset[0]
        ncy = event.y - self._drag_offset[1]
        # Clamp to plot area in canvas coords
        ax0, ay0, ax1, ay1 = self._plot_area_px
        cax0, cay0 = self._img_to_canvas(ax0, ay0)
        cax1, cay1 = self._img_to_canvas(ax1, ay1)
        ncx = max(cax0, min(cax1, ncx))
        ncy = max(cay0, min(cay1, ncy))
        # Convert back to image coords
        nix, niy = self._canvas_to_img(ncx, ncy)
        d = self._points[self._drag_idx]
        d['cx_px'] = nix
        d['cy_px'] = niy
        self._update_data_coords(d)
        self._refresh_canvas()

    def _on_release(self, event):
        if self._pan_mode:
            self._pan_drag_start = None
            return
        if self._drag_idx is not None:
            self._drag_idx = None
            self._refresh_recon()

    def _on_dblclick(self, event):
        """Add a new marker at the double-clicked position."""
        # Ignore if clicking on an existing marker
        if self._find_marker_at(event.x, event.y) is not None:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        # Clamp to plot area
        ax0, ay0, ax1, ay1 = self._plot_area_px
        ix = max(ax0, min(ax1, ix))
        iy = max(ay0, min(ay1, iy))
        cls = self._add_class_var.get()
        new_d = {
            'class_name': cls,
            'cx_px':      ix,
            'cy_px':      iy,
            'confidence': 1.0,
        }
        self._update_data_coords(new_d)
        self._points.append(new_d)
        self._refresh_canvas()
        self._refresh_recon()
        self._status_var.set(
            f"Added {cls} at ({ix:.0f}, {iy:.0f}) px  →  "
            f"({new_d['x_data']:.4g}, {new_d['y_data']:.4g})"
        )

    def _on_rclick(self, event):
        """Delete the marker nearest to the right-click position."""
        idx = self._find_marker_at(event.x, event.y)
        if idx is None:
            return
        d = self._points.pop(idx)
        self._refresh_canvas()
        self._refresh_recon()
        self._status_var.set(
            f"Deleted {d['class_name']} at "
            f"({d['cx_px']:.0f}, {d['cy_px']:.0f}) px"
        )


    # ── Zoom / Pan ────────────────────────────────────────────────────────────
    _ZOOM_STEP = 1.25

    def _toggle_pan_mode(self):
        """Toggle between pan (hand) mode and edit mode."""
        self._pan_mode = not self._pan_mode
        if self._pan_mode:
            self._canvas.config(cursor="fleur")
            self._pan_btn.config(relief="sunken", bg="#aaddff")
        else:
            self._canvas.config(cursor="crosshair")
            self._pan_btn.config(relief="raised", bg="SystemButtonFace")

    _ZOOM_MIN  = 0.2
    _ZOOM_MAX  = 10.0

    def _apply_zoom(self, factor: float, pivot_cx=None, pivot_cy=None):
        """Multiply zoom by factor, keeping canvas point (pivot_cx, pivot_cy) fixed."""
        new_zoom = max(self._ZOOM_MIN, min(self._ZOOM_MAX, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        if pivot_cx is not None and pivot_cy is not None:
            img_px = (pivot_cx - self._pan_x) / self._scale_x
            img_py = (pivot_cy - self._pan_y) / self._scale_y
            self._zoom = new_zoom
            self._refresh_canvas()
            self._pan_x = int(pivot_cx - img_px * self._scale_x)
            self._pan_y = int(pivot_cy - img_py * self._scale_y)
        else:
            self._zoom = new_zoom
        self._refresh_canvas()

    def _zoom_in(self):
        cw = self._canvas.winfo_width()  or self.CANVAS_W
        ch = self._canvas.winfo_height() or self.CANVAS_H
        self._apply_zoom(self._ZOOM_STEP, cw // 2, ch // 2)

    def _zoom_out(self):
        cw = self._canvas.winfo_width()  or self.CANVAS_W
        ch = self._canvas.winfo_height() or self.CANVAS_H
        self._apply_zoom(1.0 / self._ZOOM_STEP, cw // 2, ch // 2)

    def _zoom_reset(self):
        self._zoom  = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._refresh_canvas()

    def _on_mousewheel(self, event):
        """Zoom in/out with mouse wheel, centred on cursor position."""
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            factor = self._ZOOM_STEP
        else:
            factor = 1.0 / self._ZOOM_STEP
        self._apply_zoom(factor, event.x, event.y)

    def _on_pan_start(self, event):
        self._pan_drag_start = (event.x, event.y, self._pan_x, self._pan_y)

    def _on_pan_drag(self, event):
        if self._pan_drag_start is None:
            return
        sx, sy, px0, py0 = self._pan_drag_start
        self._pan_x = px0 + (event.x - sx)
        self._pan_y = py0 + (event.y - sy)
        self._refresh_canvas()

    def _on_pan_end(self, event):
        self._pan_drag_start = None

    # ── Export ─────────────────────────────────────────────────────────────
    def _export(self):
        """Save corrected CSV and side-by-side comparison PNG."""
        from tkinter import filedialog as _fd
        import csv

        if not self._points:
            messagebox.showinfo("Nothing to export", "No markers to export.")
            return

        # Choose output file (CSV)
        default_stem = Path(self._img_path).stem if self._img_path else "chart"
        csv_path_str = _fd.asksaveasfilename(
            title="Save CSV as",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{default_stem}_corrected.csv",
        )
        if not csv_path_str:
            return
        csv_path = Path(csv_path_str)
        out_dir = csv_path.parent
        stem = csv_path.stem  # user-chosen name (without .csv)

        # ── CSV ────────────────────────────────────────────────────────────
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["series_label", "class_name", "x_data", "y_data",
                               "confidence", "cx_px", "cy_px"])
            writer.writeheader()
            for d in sorted(self._points, key=lambda r: (r['class_name'], r['x_data'])):
                writer.writerow({
                    "series_label": d.get('series_label', ''),
                    "class_name":  d['class_name'],
                    "x_data":      f"{d['x_data']:.8g}",
                    "y_data":      f"{d['y_data']:.8g}",
                    "confidence":  f"{d.get('confidence', 1.0):.4f}",
                    "cx_px":       f"{d['cx_px']:.1f}",
                    "cy_px":       f"{d['cy_px']:.1f}",
                })

        # ── Side-by-side PNG: original overlay | reconstructed plot ────────
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import io

        # Left: original with markers
        orig_vis = self._img_bgr.copy()
        ax0, ay0, ax1, ay1 = self._plot_area_px
        cv2.rectangle(orig_vis, (ax0, ay0), (ax1, ay1), (0, 200, 0), 2)
        H, W = orig_vis.shape[:2]
        ref   = max(H, W)
        r_out = max(5, int(ref * 0.013))
        r_in  = max(2, int(r_out * 0.3))
        fsc   = max(0.30, ref * 0.00065)
        thk   = max(1, int(ref * 0.002))
        for d in self._points:
            cx = int(round(float(d['cx_px'])))
            cy = int(round(float(d['cy_px'])))
            rgb = MARKER_COLORS.get(d['class_name'], (255, 0, 0))
            bgr = (rgb[2], rgb[1], rgb[0])
            cv2.circle(orig_vis, (cx, cy), r_out, bgr, thk)
            cv2.circle(orig_vis, (cx, cy), r_in, bgr, -1)
        orig_rgb = cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB)

        # Right: reconstructed matplotlib
        from collections import defaultdict
        SYM_MPL_L = {
            'filled_circle': ('o', True), 'open_circle': ('o', False),
            'filled_square': ('s', True), 'open_square': ('s', False),
            'open_triangle': ('^', False), 'open_inv_triangle': ('v', False),
            'filled_triangle': ('^', True), 'filled_inv_triangle': ('v', True),
            'open_rhombus': ('D', False), 'filled_rhombus': ('D', True),
            'x_marker': ('X', True), 'plus_marker': ('P', True),
        }
        COLORS = ['#0077BB','#EE7733','#009944','#CC3311',
                  '#33BBEE','#44BB99','#EE3377','#999933',
                  '#DDCC77','#884400','#BBBBBB','#AA3377']

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
        ax_l.imshow(orig_rgb)
        ax_l.set_title("Original + Corrected Markers", fontsize=11)
        ax_l.axis('off')

        by_cls: dict = defaultdict(list)
        for d in self._points:
            by_cls[d['class_name']].append(d)
        for i, (cls, pts) in enumerate(sorted(by_cls.items())):
            pts_s = sorted(pts, key=lambda d: d['x_data'])
            xs = [d['x_data'] for d in pts_s]
            ys = [d['y_data'] for d in pts_s]
            mcode, filled = SYM_MPL_L.get(cls, ('o', True))
            color = COLORS[i % len(COLORS)]
            fc    = color if filled else 'white'
            # Use series_label if available, else fall back to class_name
            _lbl = (pts_s[0].get('series_label', '') or '').strip() if pts_s else ''
            _lbl = _lbl if _lbl else cls.replace('_', ' ')
            ax_r.plot(xs, ys, marker=mcode, color=color,
                      markerfacecolor=fc, markeredgecolor=color,
                      markeredgewidth=1.2, markersize=7,
                      linewidth=1.2, label=_lbl)
        if self._x_log: ax_r.set_xscale('log')
        if self._y_log: ax_r.set_yscale('log')
        ax_r.set_xlim(self._x_range)
        ax_r.set_ylim(self._y_range)
        ax_r.grid(True, linestyle='--', alpha=0.4)
        ax_r.legend(fontsize=8, loc='best')
        ax_r.set_title("Reconstructed Plot", fontsize=11)
        fig.tight_layout()

        png_path = out_dir / f"{stem}_corrected_comparison.png"
        fig.savefig(str(png_path), dpi=120, bbox_inches='tight')
        plt.close(fig)

        messagebox.showinfo(
            "Exported",
            f"Corrected data saved:\n  {csv_path}\n  {png_path}"
        )




# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    # Canvas display size (image is scaled to fit)
    CANVAS_W = 900
    CANVAS_H = 650

    def __init__(self):
        super().__init__()
        self.title("chartocode2 — Chart Digitiser")
        self.resizable(True, True)

        # State
        self.img_path:    str | None = None
        self.img_bgr:     np.ndarray | None = None   # original full-res
        self.img_display: np.ndarray | None = None   # scaled for canvas
        self.scale_x:     float = 1.0
        self.scale_y:     float = 1.0

        self.plot_rect:   tuple | None = None   # (x0,y0,x1,y1) in display px
        self.legend_rect: tuple | None = None
        self._drag_start: tuple | None = None
        self._drag_mode:  str = "plot"   # "plot" or "legend"
        self._rect_id:    int | None = None
        self._plot_rid:   int | None = None
        self._legend_rid: int | None = None

        self.marker_vars: dict[str, tk.BooleanVar] = {}
        self.result_detections: list = []

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top toolbar ──────────────────────────────────────────────────
        toolbar = tk.Frame(self, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="📂  Load Image", command=self._load_image,
                  font=("Helvetica", 11, "bold"), padx=8).pack(side=tk.LEFT, padx=4, pady=3)

        # Drag mode: plot area (green) or legend area (purple)
        self._mode_var = tk.StringVar(value="plot")
        tk.Label(toolbar, text="Drag to set:",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(10, 2))
        tk.Radiobutton(toolbar, text="Plot Area",
                       variable=self._mode_var, value="plot",
                       font=("Helvetica", 10, "bold"), fg="#007700"
                       ).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(toolbar, text="Legend Area",
                       variable=self._mode_var, value="legend",
                       font=("Helvetica", 10, "bold"), fg="#cc00cc"
                       ).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="🗑  Clear Rects",
                  command=self._clear_rects,
                  font=("Helvetica", 10), padx=6).pack(side=tk.LEFT, padx=4)

        tk.Button(toolbar, text="▶  Run Detection",
                  command=self._run,
                  font=("Helvetica", 11, "bold"),
                  bg="#2a7ae2", fg="white", padx=10).pack(side=tk.RIGHT, padx=8, pady=3)

        tk.Button(toolbar, text="✏️  Correct",
                  command=self._open_correction,
                  font=("Helvetica", 11, "bold"),
                  bg="#e67e22", fg="white", padx=8).pack(side=tk.RIGHT, padx=4, pady=3)

        # ── Main pane: canvas (left) + controls (right) ───────────────────
        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas
        canvas_frame = tk.Frame(main, bd=2, relief=tk.SUNKEN)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.canvas = tk.Canvas(canvas_frame,
                                width=self.CANVAS_W, height=self.CANVAS_H,
                                bg="#cccccc", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Right panel — scrollable
        right_outer = tk.Frame(main, width=295)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        right_outer.pack_propagate(False)

        _rscroll = tk.Scrollbar(right_outer, orient=tk.VERTICAL)
        _rscroll.pack(side=tk.RIGHT, fill=tk.Y)

        _rcanvas = tk.Canvas(right_outer, yscrollcommand=_rscroll.set,
                             highlightthickness=0)
        _rcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _rscroll.config(command=_rcanvas.yview)

        right = tk.Frame(_rcanvas)
        _rcanvas_win = _rcanvas.create_window((0, 0), window=right, anchor='nw')

        def _on_right_configure(event):
            _rcanvas.configure(scrollregion=_rcanvas.bbox('all'))
            _rcanvas.itemconfig(_rcanvas_win, width=_rcanvas.winfo_width())
        right.bind('<Configure>', _on_right_configure)

        def _on_rcanvas_resize(event):
            _rcanvas.itemconfig(_rcanvas_win, width=event.width)
        _rcanvas.bind('<Configure>', _on_rcanvas_resize)

        # Mouse-wheel scrolling on the right panel
        def _on_mousewheel(event):
            _rcanvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        right_outer.bind_all('<MouseWheel>', _on_mousewheel)

        self._build_right_panel(right)

        # ── Status / log bar ──────────────────────────────────────────────
        log_frame = tk.LabelFrame(self, text="Log", font=("Helvetica", 9))
        log_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.log_text = tk.Text(log_frame, height=5, font=("Courier", 9),
                                state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4")
        sb = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.X)

    def _build_right_panel(self, parent):
        # ── Marker checkboxes ─────────────────────────────────────────────
        mf = tk.LabelFrame(parent, text="Marker Classes",
                           font=("Helvetica", 10, "bold"), padx=4, pady=4)
        mf.pack(fill=tk.X, padx=2, pady=4)

        for key, label in ALL_MARKERS:
            var = tk.BooleanVar(value=False)
            self.marker_vars[key] = var
            cb = tk.Checkbutton(mf, text=label, variable=var,
                                font=("Helvetica", 9), anchor="w")
            cb.pack(fill=tk.X)

        btn_row = tk.Frame(mf)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        tk.Button(btn_row, text="All", command=self._select_all_markers,
                  width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="None", command=self._deselect_all_markers,
                  width=6).pack(side=tk.LEFT, padx=2)

        # ── Axis ranges ───────────────────────────────────────────────────
        af = tk.LabelFrame(parent, text="Axis Ranges",
                           font=("Helvetica", 10, "bold"), padx=6, pady=6)
        af.pack(fill=tk.X, padx=2, pady=4)

        def _row(frame, label, default_min, default_max):
            r = tk.Frame(frame)
            r.pack(fill=tk.X, pady=1)
            tk.Label(r, text=label, width=5, anchor="w",
                     font=("Helvetica", 9)).pack(side=tk.LEFT)
            tk.Label(r, text="min:", font=("Helvetica", 9)).pack(side=tk.LEFT)
            e_min = tk.Entry(r, width=8, font=("Helvetica", 9))
            e_min.insert(0, default_min)
            e_min.pack(side=tk.LEFT, padx=2)
            tk.Label(r, text="max:", font=("Helvetica", 9)).pack(side=tk.LEFT)
            e_max = tk.Entry(r, width=8, font=("Helvetica", 9))
            e_max.insert(0, default_max)
            e_max.pack(side=tk.LEFT, padx=2)
            return e_min, e_max

        self.x_min_e, self.x_max_e = _row(af, "X:", "0", "1")
        self.y_min_e, self.y_max_e = _row(af, "Y:", "0", "1")

        # ── Scale selectors ───────────────────────────────────────────────
        sf = tk.LabelFrame(parent, text="Scale",
                           font=("Helvetica", 10, "bold"), padx=6, pady=6)
        sf.pack(fill=tk.X, padx=2, pady=4)

        self.x_scale_var = tk.StringVar(value="linear")
        self.y_scale_var = tk.StringVar(value="linear")

        for axis, var in [("X-axis", self.x_scale_var),
                          ("Y-axis", self.y_scale_var)]:
            r = tk.Frame(sf)
            r.pack(fill=tk.X, pady=1)
            tk.Label(r, text=axis + ":", width=7, anchor="w",
                     font=("Helvetica", 9)).pack(side=tk.LEFT)
            tk.Radiobutton(r, text="Linear", variable=var, value="linear",
                           font=("Helvetica", 9)).pack(side=tk.LEFT)
            tk.Radiobutton(r, text="Log₁₀", variable=var, value="log10",
                           font=("Helvetica", 9)).pack(side=tk.LEFT)


        # ── Error Bars ────────────────────────────────────────────────────────
        self.upscale_var   = tk.StringVar(value="1.0")
        self.conf_thresh_e = None   # use default
        self.stride_e      = None   # use default

        ebf = tk.LabelFrame(parent, text="Error Bars",
                            font=("Helvetica", 10, "bold"), padx=6, pady=6)
        ebf.pack(fill=tk.X, padx=2, pady=4)

        # Three-state: Auto / Yes / No
        # Auto  → has_errorbars=None  (Function 1 auto-detect)
        # Yes   → has_errorbars=True  (always run Function 2)
        # No    → has_errorbars=False (always skip Function 2)
        self.errorbar_var = tk.StringVar(value="auto")
        eb_row = tk.Frame(ebf)
        eb_row.pack(fill=tk.X)
        tk.Label(eb_row, text="Error bars:",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        for _val, _txt in [("auto", "Auto"), ("yes", "Yes"), ("no", "No")]:
            tk.Radiobutton(eb_row, text=_txt, variable=self.errorbar_var,
                           value=_val,
                           font=("Helvetica", 9)).pack(side=tk.LEFT, padx=2)
        tk.Label(ebf,
                 text="Auto: detect automatically\n"
                      "Yes:  always remove stems/T-caps\n"
                      "No:   skip error-bar removal",
                 font=("Helvetica", 8), fg="#555555", justify=tk.LEFT
                 ).pack(fill=tk.X)

        # ── Lines toggle ────────────────────────────────────────────────────────────────────
        lf = tk.LabelFrame(parent, text="Plot Type",
                           font=("Helvetica", 10, "bold"), padx=6, pady=6)
        lf.pack(fill=tk.X, padx=2, pady=4)
        self.has_lines_var = tk.BooleanVar(value=True)
        tk.Checkbutton(lf,
                       text="Lines present  (uncheck for scatter/dot plots)",
                       variable=self.has_lines_var,
                       font=("Helvetica", 9)).pack(anchor="w")
        tk.Label(lf,
                 text="Checked:   segment detection + error-bar removal enabled\n"
                      "Unchecked: ViT marker detection only (faster)",
                 font=("Helvetica", 8), fg="#555555", justify=tk.LEFT
                 ).pack(fill=tk.X)

        # ── Scale (upscale) control ───────────────────────────────────────────
        sf = tk.LabelFrame(parent, text="Scale (Upscale)",
                           font=("Helvetica", 10, "bold"), padx=6, pady=6)
        sf.pack(fill=tk.X, padx=2, pady=4)

        # Auto / Manual toggle
        self.scale_mode_var = tk.StringVar(value="auto")
        sm_row = tk.Frame(sf)
        sm_row.pack(fill=tk.X)
        tk.Radiobutton(sm_row, text="Auto (from legend)",
                       variable=self.scale_mode_var, value="auto",
                       command=self._on_scale_mode_change,
                       font=("Helvetica", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(sm_row, text="Manual",
                       variable=self.scale_mode_var, value="manual",
                       command=self._on_scale_mode_change,
                       font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(8, 0))

        # Manual entry row
        self._scale_manual_frame = tk.Frame(sf)
        self._scale_manual_frame.pack(fill=tk.X, pady=(4, 0))
        tk.Label(self._scale_manual_frame, text="Scale value:",
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        self._scale_entry = tk.Entry(self._scale_manual_frame,
                                     textvariable=self.upscale_var,
                                     width=8, font=("Helvetica", 9))
        self._scale_entry.pack(side=tk.LEFT, padx=4)
        tk.Label(self._scale_manual_frame,
                 text="(e.g. 0.5, 1.0, 2.0)",
                 font=("Helvetica", 8), fg="#555555").pack(side=tk.LEFT)
        self._scale_entry.config(state=tk.DISABLED)  # start disabled (auto mode)

        tk.Label(sf,
                 text="Auto: computed from legend swatch size\n"
                      "Manual: override directly (useful for colour plots)",
                 font=("Helvetica", 8), fg="#555555", justify=tk.LEFT
                 ).pack(fill=tk.X)

        # ── Area info display ─────────────────────────────────────────────
        info_f = tk.LabelFrame(parent, text="Selected Areas",
                               font=("Helvetica", 10, "bold"), padx=4, pady=4)
        info_f.pack(fill=tk.X, padx=2, pady=4)

        self.plot_area_lbl  = tk.Label(info_f, text="Plot area:   (not set)",
                                       font=("Courier", 8), anchor="w")
        self.plot_area_lbl.pack(fill=tk.X)
        self.legend_area_lbl = tk.Label(info_f, text="Legend area: (not set)",
                                        font=("Courier", 8), anchor="w")
        self.legend_area_lbl.pack(fill=tk.X)

        # ── Save button ───────────────────────────────────────────────────
        tk.Button(parent, text="💾  Save Results",
                  command=self._save_results,
                  font=("Helvetica", 10, "bold"),
                  bg="#28a745", fg="white").pack(fill=tk.X, padx=2, pady=6)

    # ── Helpers ───────────────────────────────────────────────────────────
    def _log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def _select_all_markers(self):
        for v in self.marker_vars.values():
            v.set(True)

    def _deselect_all_markers(self):
        for v in self.marker_vars.values():
            v.set(False)

    # ── Image loading ─────────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select chart image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("All files", "*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{path}")
            return
        self.img_path = path
        self.img_bgr  = img
        self.plot_rect   = None
        self.legend_rect = None
        self.result_detections = []
        # Clear any previous plot/legend rects
        if hasattr(self, '_plot_rect_img'):   del self._plot_rect_img
        if hasattr(self, '_legend_rect_img'): del self._legend_rect_img
        H_img, W_img = img.shape[:2]
        self._show_image(img)
        self._log(f"Loaded: {path}  ({W_img}×{H_img})")
        self._log("Select 'Plot Area' in toolbar and drag to set the plot area.")
        self.plot_area_lbl.config(text="Plot area:   (not set — drag to define)")
        self.legend_area_lbl.config(text="Legend area: (not set)")

    def _show_image(self, img_bgr: np.ndarray):
        """Scale img_bgr to fit the canvas and display it."""
        H, W = img_bgr.shape[:2]
        cw = self.canvas.winfo_width()  or self.CANVAS_W
        ch = self.canvas.winfo_height() or self.CANVAS_H
        scale = min(cw / W, ch / H, 1.0)
        self.scale_x = scale
        self.scale_y = scale
        nw, nh = max(1, int(W * scale)), max(1, int(H * scale))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil = Image.fromarray(img_rgb).resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        # canvas.delete("all") invalidates all item IDs — reset them
        self._plot_rid   = None
        self._legend_rid = None
        self._rect_id    = None
        # Redraw existing rects
        if self.plot_rect:
            self._draw_rect_on_canvas(self.plot_rect, "plot")
        if self.legend_rect:
            self._draw_rect_on_canvas(self.legend_rect, "legend")

    def _img_to_canvas(self, x, y):
        return x * self.scale_x, y * self.scale_y

    def _canvas_to_img(self, cx, cy):
        return cx / self.scale_x, cy / self.scale_y

    # ── Drag interactions ─────────────────────────────────────────────────
    def _on_press(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_mode  = self._mode_var.get()
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        color = "#00cc00" if self._drag_mode == "plot" else "#cc00cc"
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, x1, y1, outline=color, width=2, dash=(4, 2))

    def _on_release(self, event):
        if not self._drag_start:
            return
        cx0, cy0 = self._drag_start
        cx1, cy1 = event.x, event.y
        self._drag_start = None
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

        # Normalise
        cx0, cx1 = min(cx0, cx1), max(cx0, cx1)
        cy0, cy1 = min(cy0, cy1), max(cy0, cy1)
        if cx1 - cx0 < 5 or cy1 - cy0 < 5:
            return

        # Convert to image coordinates
        ix0, iy0 = self._canvas_to_img(cx0, cy0)
        ix1, iy1 = self._canvas_to_img(cx1, cy1)
        rect_img = (int(ix0), int(iy0), int(ix1), int(iy1))

        # Route to plot or legend based on drag mode
        if self._drag_mode == "plot":
            self.plot_rect = (cx0, cy0, cx1, cy1)
            self._plot_rect_img = rect_img
            self._draw_rect_on_canvas(self.plot_rect, "plot")
            self.plot_area_lbl.config(
                text=f"Plot area:   {rect_img[0]},{rect_img[1]} → {rect_img[2]},{rect_img[3]}")
            self._log(f"Plot area set: {rect_img}")
        else:
            self.legend_rect = (cx0, cy0, cx1, cy1)
            self._legend_rect_img = rect_img
            self._draw_rect_on_canvas(self.legend_rect, "legend")
            self.legend_area_lbl.config(
                text=f"Legend area: {rect_img[0]},{rect_img[1]} → {rect_img[2]},{rect_img[3]}")
            self._log(f"Legend area set: {rect_img}")
            # (Auto-scale computed at Run time, not here)

    def _draw_rect_on_canvas(self, rect_canvas, mode):
        x0, y0, x1, y1 = rect_canvas
        if mode == "plot":
            if self._plot_rid:
                self.canvas.delete(self._plot_rid)
            self._plot_rid = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#00cc00", width=2)
        else:
            if self._legend_rid:
                self.canvas.delete(self._legend_rid)
            self._legend_rid = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="#cc00cc", width=2)

    def _on_scale_mode_change(self):
        """Enable/disable the manual scale entry based on mode selection."""
        if self.scale_mode_var.get() == "manual":
            self._scale_entry.config(state=tk.NORMAL)
        else:
            self._scale_entry.config(state=tk.DISABLED)

    def _auto_scale(self):
        """Estimate optimal upscale from legend marker size; update upscale_var."""
        if self.img_bgr is None:
            return
        legend_box = getattr(self, '_legend_rect_img', None)
        if legend_box is None:
            return
        try:
            from chart_preprocessing import estimate_optimal_scale as _eos
            import importlib.util as _ilu, sys as _sys
            # Get ViT window size P from the detector module
            _det_path = str(SRC_DIR / "1_point_detection_v3.py")
            if 'det_mod_vit' not in _sys.modules:
                _spec = _ilu.spec_from_file_location('det_mod_vit', _det_path)
                _mod  = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _sys.modules['det_mod_vit'] = _mod
            _P = getattr(_sys.modules['det_mod_vit'], 'P', 19)
            scale, info = _eos(
                self.img_bgr,
                legend_box=legend_box,
                vit_window_px=_P,
                verbose=True,
            )
            self.upscale_var.set(str(scale))
            _md = info['median_diameter']
            _md_str = f"{_md:.1f}" if _md is not None else "N/A"
            self._log(
                f"[Auto scale] legend glyphs={info['n_glyphs_found']}  "
                f"median_diam={_md_str}px  "
                f"raw={info['raw_scale']:.3f}  "
                f"→ scale={scale}  (ViT P={_P}px)"
            )
        except Exception as _e:
            import traceback as _tb
            self._log(f"[Auto scale] failed: {_e}\n{_tb.format_exc()}")

    def _clear_rects(self):
        # Clear both plot and legend rects
        self.plot_rect   = None
        self.legend_rect = None
        if hasattr(self, '_plot_rect_img'):
            del self._plot_rect_img
        if hasattr(self, '_legend_rect_img'):
            del self._legend_rect_img
        self._plot_rid   = None
        self._legend_rid = None
        self._rect_id    = None
        # Redraw canvas from scratch to guarantee rects are gone
        if self.img_bgr is not None:
            self._show_image(self.img_bgr)
        self.plot_area_lbl.config(text="Plot area:   (not set)")
        self.legend_area_lbl.config(text="Legend area: (not set)")
        self._log("Rects cleared.")

    # ── Run detection ─────────────────────────────────────────────────────
    def _run(self):
        if self.img_bgr is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        if not hasattr(self, '_plot_rect_img'):
            messagebox.showwarning("No plot area",
                                   "Please drag to set the Plot Area first.\n"
                                   "(Select 'Plot Area' in toolbar, then drag on the image.)")
            return

        # Parse axis ranges
        try:
            x_min = float(self.x_min_e.get())
            x_max = float(self.x_max_e.get())
            y_min = float(self.y_min_e.get())
            y_max = float(self.y_max_e.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Axis range values must be numbers.")
            return
        if x_min >= x_max or y_min >= y_max:
            messagebox.showerror("Invalid range",
                                 "min must be less than max for both axes.")
            return

        x_log = self.x_scale_var.get() == "log10"
        y_log = self.y_scale_var.get() == "log10"

        known_classes = [k for k, v in self.marker_vars.items() if v.get()]

        legend_area = getattr(self, '_legend_rect_img', None)

        # Resolve error-bar toggle: "auto" → None, "yes" → True, "no" → False
        _eb_str = getattr(self, 'errorbar_var', None)
        _eb_str = _eb_str.get() if _eb_str else 'auto'
        has_errorbars = None if _eb_str == 'auto' else (_eb_str == 'yes')

        # Lines present toggle
        _has_lines = getattr(self, 'has_lines_var', None)
        _has_lines = bool(_has_lines.get()) if _has_lines else True

        # Parse detection options (upscale / conf_thresh / stride)
        try:
            _upscale = float(self.upscale_var.get())
        except Exception:
            _upscale = 1.0
        # conf_thresh / stride: use defaults (None = let run_detection decide)
        _conf   = None
        _stride = None

        # Auto-compute upscale from legend (skip if manual mode)
        if self.scale_mode_var.get() == "auto":
            self._auto_scale()

        self._log("=" * 50)
        self._log(f"Running detection …")
        self._log(f"  Plot area   : {self._plot_rect_img}")
        self._log(f"  Legend      : {legend_area}")
        self._log(f"  Markers     : {known_classes or '(all, ViT decides)'}")
        self._log(f"  X range     : [{x_min}, {x_max}]  {'log10' if x_log else 'linear'}")
        self._log(f"  Y range     : [{y_min}, {y_max}]  {'log10' if y_log else 'linear'}")
        self._log(f"  Error bars  : {_eb_str} (→ has_errorbars={has_errorbars})")

        # Run in background thread to keep GUI responsive
        def _worker():
            try:
                result = run_detection(
                    img_bgr        = self.img_bgr,
                    plot_area_px   = self._plot_rect_img,
                    legend_area_px = legend_area,
                    known_classes  = known_classes,
                    x_range        = (x_min, x_max),
                    y_range        = (y_min, y_max),
                    x_log          = x_log,
                    y_log          = y_log,
                    has_errorbars  = has_errorbars,
                    upscale        = _upscale,
                    conf_thresh    = _conf,
                    stride         = _stride,
                    has_lines      = _has_lines,
                    log_fn         = self._log,
                )
                self.result_detections = result['detections']
                self._overlay_img = result['overlay_img']
                self.after(0, self._show_overlay)
            except Exception as e:
                import traceback
                self.after(0, lambda: self._log(f"ERROR: {e}\n{traceback.format_exc()}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_overlay(self):
        """Display the overlay image on the canvas, then open CorrectionWindow."""
        self._show_image(self._overlay_img)
        n = len(self.result_detections)
        self._log(f"Overlay displayed. {n} data points found.")
        if n > 0:
            self._log("  class                  x_data          y_data")
            for d in self.result_detections[:20]:
                self._log(f"  {d['class_name']:<22} {d['x_data']:>14.6g}  {d['y_data']:>14.6g}")
            if n > 20:
                self._log(f"  … ({n - 20} more rows in saved CSV)")

        # Auto-open correction editor (even if 0 detections — user can add manually)
        plot_area = getattr(self, '_plot_rect_img', None)
        if plot_area is not None:
            try:
                x_min = float(self.x_min_e.get())
                x_max = float(self.x_max_e.get())
                y_min = float(self.y_min_e.get())
                y_max = float(self.y_max_e.get())
            except ValueError:
                x_min, x_max, y_min, y_max = 0, 1, 0, 1
            x_log = self.x_scale_var.get() == "log10"
            y_log = self.y_scale_var.get() == "log10"
            CorrectionWindow(
                parent        = self,
                img_bgr       = self.img_bgr,
                detections    = self.result_detections,
                plot_area_px  = plot_area,
                x_range       = (x_min, x_max),
                y_range       = (y_min, y_max),
                x_log         = x_log,
                y_log         = y_log,
                img_path      = self.img_path,
                target_classes= [k for k, v in self.marker_vars.items() if v.get()],
            )

    # ── Open correction editor ───────────────────────────────────────────
    def _open_correction(self):
        """Open the CorrectionWindow with current detection results."""
        if self.img_bgr is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        plot_area = getattr(self, '_plot_rect_img', None)
        if plot_area is None:
            messagebox.showwarning("No plot area", "Plot area not set.")
            return
        try:
            x_min = float(self.x_min_e.get())
            x_max = float(self.x_max_e.get())
            y_min = float(self.y_min_e.get())
            y_max = float(self.y_max_e.get())
        except ValueError:
            x_min, x_max, y_min, y_max = 0, 1, 0, 1
        x_log = self.x_scale_var.get() == "log10"
        y_log = self.y_scale_var.get() == "log10"
        CorrectionWindow(
            parent        = self,
            img_bgr       = self.img_bgr,
            detections    = self.result_detections,
            plot_area_px  = plot_area,
            x_range       = (x_min, x_max),
            y_range       = (y_min, y_max),
            x_log         = x_log,
            y_log         = y_log,
            img_path      = self.img_path,
            target_classes= [k for k, v in self.marker_vars.items() if v.get()],
        )

    # ── Save results ──────────────────────────────────────────────────────
    def _save_results(self):
        if not self.result_detections and not hasattr(self, '_overlay_img'):
            messagebox.showinfo("Nothing to save",
                                "Run detection first.")
            return

        if self.img_path:
            stem = Path(self.img_path).stem
            out_dir = Path(self.img_path).parent
        else:
            stem = "chart"
            out_dir = Path.cwd()

        # Save overlay image
        if hasattr(self, '_overlay_img'):
            img_out = out_dir / f"{stem}_detected.png"
            cv2.imwrite(str(img_out), self._overlay_img)
            self._log(f"Saved overlay → {img_out}")

        # Save CSV
        if self.result_detections:
            csv_out = out_dir / f"{stem}_data.csv"
            with open(csv_out, "w") as f:
                f.write("class_name,x_data,y_data,confidence,cx_px,cy_px\n")
                for d in self.result_detections:
                    f.write(f"{d['class_name']},"
                            f"{d['x_data']:.8g},"
                            f"{d['y_data']:.8g},"
                            f"{d.get('confidence', ''):.4f},"
                            f"{d['cx_px']:.1f},"
                            f"{d['cy_px']:.1f}\n")
            self._log(f"Saved data   → {csv_out}")
            messagebox.showinfo("Saved",
                                f"Results saved to:\n{img_out}\n{csv_out}")
        else:
            messagebox.showinfo("Saved",
                                f"Overlay saved to:\n{img_out}\n(No detections to export as CSV)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Make sure src/ is on sys.path so sibling modules are importable
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    try:
        from PIL import Image, ImageTk  # noqa: F401
    except ImportError:
        print("ERROR: Pillow is required.  Install with:  pip install pillow")
        sys.exit(1)

    app = App()
    app.mainloop()

# ═══════════════════════════════════════════════════════════════════════════════
# Human-Assisted Correction Window
# ═══════════════════════════════════════════════════════════════════════════════

