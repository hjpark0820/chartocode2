"""
adaptive_nms_detection.py
=========================
Adaptive x-column NMS with sliding window and data-driven window size.

Pipeline
--------
1. Run the V3 ViT sliding-window detector to collect ALL raw detections
   (no NMS applied yet).
2. Pool all raw cx values from symbol-class detections (all classes together).
3. Estimate the inter-mode distance *d* from the multimodal x-distribution:
     - Build a 1-D KDE over all raw cx values.
     - Find local maxima (modes) of the KDE.
     - Estimate d = median of consecutive mode spacings.
4. Set  window_width = 0.75 * d.
5. Apply per-class sliding-window x-column NMS with the adaptive window_width.
   The window slides 1 px at a time across the full image width.  Within each
   window position, the highest-confidence detection per class is kept; all
   others are suppressed.  A detection is only suppressed once (the first
   window in which it is the non-maximum).
6. Return kept detections, suppressed detections, and the KDE mode positions.

Public API
----------
estimate_mode_distance(cx_values, img_width, bandwidth_factor=0.3)
    -> (d, x_grid, density, peaks, mode_xs, spacings)  or  (None, x_grid, density)

xcol_nms_with_suppressed(dets, window_width)
    -> (kept, suppressed)

detect_with_adaptive_nms(img_path, model_path, known_classes, ...)
    -> dict with keys:
         'kept'         : list of kept detection dicts
         'suppressed'   : list of suppressed detection dicts
         'mode_xs'      : 1-D array of KDE mode x-positions
         'd_est'        : estimated inter-mode distance (px)
         'bin_width'    : adaptive window width used for NMS
         'img_bgr'      : original image (BGR numpy array)
"""

import os
import sys
import importlib.util
from collections import defaultdict

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_detector_module(detector_py_path: str):
    """Dynamically load chart_marker_detector_v3.py from an arbitrary path."""
    spec = importlib.util.spec_from_file_location(
        'chart_marker_detector_v3', detector_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules['chart_marker_detector_v3'] = mod
    return mod


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def estimate_mode_distance(cx_values, img_width, bandwidth_factor=0.3):
    """
    Fit a KDE to all raw cx values, find local maxima (modes), and return
    the median consecutive mode spacing as the estimated inter-mode distance *d*.

    Parameters
    ----------
    cx_values : array-like
        Raw x-coordinates of all symbol detections (before NMS).
    img_width : int
        Width of the source image in pixels (used for the evaluation grid).
    bandwidth_factor : float
        KDE bandwidth expressed as a multiple of the patch size P (default 0.3).
        bandwidth_px = bandwidth_factor * P  (P = 19 px).

    Returns
    -------
    On success : (d, x_grid, density, peaks, mode_xs, spacings)
    On failure : (None, x_grid, density)
    """
    cx = np.array(cx_values, dtype=float)
    if len(cx) < 2:
        x_grid = np.linspace(0, img_width, img_width * 2)
        return None, x_grid, np.zeros_like(x_grid)

    P_est = 19.0                          # marker patch size (px)
    bw_px = bandwidth_factor * P_est      # e.g. 0.3 * 19 = 5.7 px
    std_cx = np.std(cx)
    bw_method = bw_px / std_cx if std_cx > 0 else 0.1

    kde = gaussian_kde(cx, bw_method=bw_method)
    x_grid = np.linspace(0, img_width, img_width * 2)
    density = kde(x_grid)

    min_dist_samples = int(0.4 * P_est * 2)   # ~15 samples ~ 7.5 px
    peaks, _ = find_peaks(
        density,
        distance=min_dist_samples,
        prominence=density.max() * 0.02,
    )

    if len(peaks) < 2:
        return None, x_grid, density

    mode_xs = x_grid[peaks]
    spacings = np.diff(mode_xs)
    d = float(np.median(spacings))
    return d, x_grid, density, peaks, mode_xs, spacings


def xcol_nms_with_suppressed(dets, window_width):
    """
    Per-class sliding-window x-column NMS.

    A window of width ``window_width`` slides 1 px at a time from x=0 to
    x=img_max_cx.  At each position, within each class, if more than one
    detection falls inside the window, all but the highest-confidence one are
    marked for suppression.  A detection is suppressed at most once (the first
    window position where it is the non-maximum).

    This is equivalent to: for every pair of same-class detections whose
    cx values are within ``window_width`` px of each other, suppress the
    lower-confidence one.

    Parameters
    ----------
    dets : list of dict
        Each dict must contain at least: 'cx', 'class_idx', 'class_name',
        'confidence'.
    window_width : float
        Sliding window width in pixels.

    Returns
    -------
    kept : list of dict
    suppressed : list of dict
        Suppressed dicts gain two extra keys:
        'original_class_name' and 'original_class_idx'.
    """
    by_class = defaultdict(list)
    for d in dets:
        by_class[d['class_idx']].append(d)

    suppressed_ids = set()   # ids of detections marked for suppression

    # Assign a random tiebreak value to each detection once, so that when
    # multiple detections share the same confidence score the winner is chosen
    # uniformly at random rather than by insertion order.
    import random
    tiebreak = {id(d): random.random() for d in dets}

    for cls_dets in by_class.values():
        # Sort by cx so we can use a two-pointer sweep.
        # Strategy: scan left-to-right; whenever a detection has NOT yet been
        # suppressed, it becomes the "anchor".  All detections within
        # window_width of the anchor that have a lower (confidence, tiebreak)
        # score are suppressed immediately.  This guarantees at most one kept
        # detection per window-width band.
        sorted_dets = sorted(cls_dets, key=lambda x: x['cx'])
        n = len(sorted_dets)
        for i in range(n):
            if id(sorted_dets[i]) in suppressed_ids:
                continue   # already suppressed by an earlier anchor
            # sorted_dets[i] is the current anchor (not yet suppressed)
            anchor = sorted_dets[i]
            anchor_key = (anchor['confidence'], tiebreak[id(anchor)])
            # Suppress all neighbours within window_width that are weaker
            for j in range(i + 1, n):
                if sorted_dets[j]['cx'] - anchor['cx'] > window_width:
                    break
                neighbour = sorted_dets[j]
                if id(neighbour) in suppressed_ids:
                    continue
                neighbour_key = (neighbour['confidence'], tiebreak[id(neighbour)])
                if neighbour_key <= anchor_key:
                    suppressed_ids.add(id(neighbour))
                else:
                    # Neighbour is stronger — suppress the anchor instead and
                    # promote the neighbour as the new anchor
                    suppressed_ids.add(id(anchor))
                    anchor = neighbour
                    anchor_key = neighbour_key

    kept, suppressed = [], []
    for d in dets:
        if id(d) in suppressed_ids:
            sup = dict(d)
            sup['original_class_name'] = d['class_name']
            sup['original_class_idx']  = d['class_idx']
            sup['class_name']          = 'suppressed'
            sup['class_idx']           = -2
            suppressed.append(sup)
        else:
            kept.append(d)
    return kept, suppressed


def detect_with_adaptive_nms(
        img_path,
        model_path,
        known_classes,
        detector_py_path = None,
        out_dir          = None,
        conf_thresh      = None,
        stride           = None,
        bandwidth_factor = 0.3,
        bin_width_factor = 1.0,
        batch_size       = 512,
        prep_info        = None,
        d_override       = None,
):
    """
    Full pipeline: sliding-window detection -> adaptive NMS.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    model_path : str
        Path to the ``chart_marker_net_v3.pth`` weights file.
    known_classes : list of str
        Symbol class names to retain in the final output
        (e.g. ['filled_circle', 'open_circle', 'filled_square', 'open_square']).
    detector_py_path : str, optional
        Path to ``chart_marker_detector_v3.py``.  If None, the module must
        already be importable as ``chart_marker_detector_v3``.
    out_dir : str, optional
        If provided, diagnostic figures are saved here as JPEG files.
    conf_thresh : float, optional
        Override the detector's default confidence threshold.
    stride : int, optional
        Override the detector's default sliding-window stride.
    bandwidth_factor : float
        KDE bandwidth = bandwidth_factor * P  (default 0.3).
    bin_width_factor : float
        adaptive bin_width = bin_width_factor * d  (default 0.75).
    batch_size : int
        Number of patches per GPU/CPU forward pass (default 512).
    prep_info : dict or None
        Preprocessing result from chart_preprocessing.preprocess().
        If provided, the sliding-window scan is restricted to the detected
        plot area (plot_area) and the clean_fn is applied to filter out
        axis/legend/text noise from the binary dark-pixel check.

    Returns
    -------
    dict with keys:
        'kept'       - list of kept detection dicts (filtered to known_classes)
        'suppressed' - list of suppressed detection dicts
        'mode_xs'    - 1-D numpy array of KDE mode x-positions
        'd_est'      - estimated inter-mode distance (px), or None
        'bin_width'  - adaptive bin width used for NMS
        'img_bgr'    - original image as a BGR numpy array
    """
    import torch

    # load detector module
    if detector_py_path is not None:
        mod = _load_detector_module(detector_py_path)
    else:
        import chart_marker_detector_v3 as mod

    # detector constants
    _conf_thresh    = conf_thresh if conf_thresh is not None else mod.CONF_THRESH
    _stride         = stride      if stride      is not None else mod.STRIDE
    unknown_thresh  = mod.UNKNOWN_THRESH
    min_dark_frac   = mod.MIN_DARK_FRAC
    P               = mod.P
    fixed_bin_width = P * getattr(mod, 'XCOL_NMS_WIDTH_FACTOR', 2.5)

    # load model & image
    model, device = mod._load_model(model_path)
    img_bgr  = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W     = img_gray.shape

    min_dark_pixels = int(P * P * min_dark_frac)

    # ── Preprocessing: restrict scan to plot area, build noise mask ──────
    # If prep_info is provided, build a cleaned binary mask so that
    # axis/legend/text pixels are excluded from the dark-pixel check.
    # The sliding-window scan is also restricted to the detected plot area.
    _clean_mask = None
    _scan_x0 = 0; _scan_y0 = 0; _scan_x1 = W; _scan_y1 = H
    if prep_info is not None:
        # Build cleaned binary mask
        _gray_tmp = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, _bw_tmp = cv2.threshold(_gray_tmp, 128, 255, cv2.THRESH_BINARY_INV)
        _bw_tmp = (_bw_tmp > 0).astype('uint8')
        _clean_mask = prep_info['clean_fn'](_bw_tmp)
        # Restrict scan to plot area
        pa = prep_info.get('plot_area', None)
        if pa is not None:
            _scan_x0, _scan_y0, _scan_x1, _scan_y1 = pa

    # sliding-window pass (collect all raw detections)
    raw_dets = []
    batch_coords, batch_tensors = [], []

    def _flush():
        if not batch_tensors:
            return
        t = torch.tensor(np.stack(batch_tensors), dtype=torch.float32).to(device)
        with torch.no_grad():
            from torch.amp import autocast
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                probs = torch.softmax(model(t), dim=1).cpu().numpy()
        for (bx, by), prob in zip(batch_coords, probs):
            max_prob = float(prob.max())
            ci       = int(prob.argmax())
            if ci == mod.N_SYMBOLS:
                continue
            elif max_prob < unknown_thresh:
                patch = mod.extract_patch_padded(img_gray, bx, by, P)
                ecx, ecy = mod._estimate_center_in_patch(bx, by, patch, P)
                raw_dets.append({
                    'cx': ecx, 'cy': ecy,
                    'class_idx': -1, 'class_name': 'unknown',
                    'confidence': round(max_prob, 4),
                })
            elif max_prob >= _conf_thresh:
                patch = mod.extract_patch_padded(img_gray, bx, by, P)
                ecx, ecy = mod._estimate_center_in_patch(bx, by, patch, P)
                raw_dets.append({
                    'cx': ecx, 'cy': ecy,
                    'class_idx': ci,
                    'class_name': mod.CLASS_NAMES[ci],
                    'confidence': round(max_prob, 4),
                })
        batch_coords.clear()
        batch_tensors.clear()

    for cy_w in range(_scan_y0, _scan_y1, _stride):
        for cx_w in range(_scan_x0, _scan_x1, _stride):
            patch = mod.extract_patch_padded(img_gray, cx_w, cy_w, P)
            # Use cleaned mask if available: count dark pixels only in clean regions
            if _clean_mask is not None:
                # Extract the corresponding patch from the clean mask
                _half = P // 2
                _y0c = max(0, cy_w - _half); _y1c = min(H, cy_w + _half + 1)
                _x0c = max(0, cx_w - _half); _x1c = min(W, cx_w + _half + 1)
                _cm_patch = _clean_mask[_y0c:_y1c, _x0c:_x1c]
                if _cm_patch.sum() < min_dark_pixels:
                    continue
            else:
                _, bw_mask = cv2.threshold(patch, 200, 255, cv2.THRESH_BINARY_INV)
                if np.count_nonzero(bw_mask) < min_dark_pixels:
                    continue
            batch_coords.append((cx_w, cy_w))
            batch_tensors.append(mod.patch_to_tensor(patch))
            if len(batch_tensors) == batch_size:
                _flush()
    _flush()

    # estimate adaptive bin width
    symbol_dets = [d for d in raw_dets if d['class_idx'] >= 0]
    all_cx      = [d['cx'] for d in symbol_dets]

    result = estimate_mode_distance(all_cx, W, bandwidth_factor=bandwidth_factor)

    if result[0] is None:
        d_est              = None
        adaptive_bin_width = fixed_bin_width
        mode_xs            = np.array([])
    else:
        d_est, _x_grid, _density, _peaks, mode_xs, _spacings = result
        # d_override: caller can supply a pre-computed d_est (e.g. estimated on
        # the original-scale image before upscaling) to avoid KDE instability
        # on the upscaled image.
        if d_override is not None:
            d_est = float(d_override)
        adaptive_bin_width = bin_width_factor * d_est

    # apply adaptive NMS
    kept_all, suppressed = xcol_nms_with_suppressed(symbol_dets, adaptive_bin_width)
    kept = [d for d in kept_all if d['class_name'] in known_classes]

    # optional diagnostic figures
    if out_dir is not None:
        _save_diagnostics(
            out_dir, img_bgr, all_cx, W, P,
            fixed_bin_width, adaptive_bin_width, d_est,
            result, kept, suppressed,
        )

    return {
        'kept':       kept,
        'suppressed': suppressed,
        'mode_xs':    mode_xs,
        'd_est':      d_est,
        'bin_width':  adaptive_bin_width,
        'img_bgr':    img_bgr,
    }


# ---------------------------------------------------------------------------
# Diagnostic visualisation (called only when out_dir is given)
# ---------------------------------------------------------------------------

def _save_diagnostics(
    out_dir, img_bgr, all_cx, W, P,
    fixed_bin_width, adaptive_bin_width, d_est,
    kde_result, kept, suppressed,
):
    """Save two diagnostic JPEG figures to *out_dir*."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image as PILImage

    os.makedirs(out_dir, exist_ok=True)

    def _save_jpg(fig, path):
        tmp = path.replace('.jpg', '_tmp.png')
        fig.savefig(tmp, dpi=150, bbox_inches='tight')
        plt.close(fig)
        PILImage.open(tmp).convert('RGB').save(path, 'JPEG', quality=92)
        os.remove(tmp)

    # Figure 1: x-distribution + KDE + modes
    if kde_result[0] is not None:
        _d, x_grid, density, peaks, mode_xs, spacings = kde_result
    else:
        x_grid  = kde_result[1]
        density = kde_result[2]
        peaks   = np.array([], dtype=int)
        mode_xs = np.array([])

    fig1, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    title_d   = f'{d_est:.1f}px' if d_est is not None else 'N/A'
    title_bin = f'{adaptive_bin_width:.1f}px'
    ax.set_title(
        f'X-coordinate distribution of all raw symbol detections\n'
        f'(n={len(all_cx)})  KDE bandwidth=0.3xP={0.3*P:.0f}px  |  '
        f'd={title_d}  |  adaptive bin=0.75xd={title_bin}',
        fontsize=11,
    )
    ax.hist(all_cx, bins=np.arange(0, W + 1, 2), color='steelblue',
            alpha=0.4, density=True, label='Raw cx histogram (2 px bins)')
    ax.plot(x_grid, density, 'k-', lw=1.5, label='KDE')
    if len(peaks) > 0:
        ax.plot(mode_xs, density[peaks], 'rv', ms=8, label=f'Modes (n={len(peaks)})')
        for i, mx in enumerate(mode_xs):
            ax.axvline(mx, color='red', lw=0.7, ls='--', alpha=0.6)
            ax.text(mx, density[peaks[i]] * 1.05, f'{mx:.0f}',
                    ha='center', fontsize=7, color='red')
    for b in range(int(np.ceil(W / fixed_bin_width)) + 1):
        ax.axvline(b * fixed_bin_width, color='orange', lw=0.8, ls='-', alpha=0.5,
                   label='Fixed bins' if b == 0 else None)
    for b in range(int(np.ceil(W / adaptive_bin_width)) + 1):
        ax.axvline(b * adaptive_bin_width, color='green', lw=0.8, ls='-', alpha=0.5,
                   label='Adaptive bins' if b == 0 else None)
    ax.set_xlabel('cx (px)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, W)
    plt.tight_layout()
    _save_jpg(fig1, os.path.join(out_dir, 'adaptive_nms_xdist.jpg'))

    # Figure 2: detection overlay
    RADIUS, THICK = 5, 2
    RED  = (0,   0, 255)
    BLUE = (255, 80,  0)

    def _draw(img_in, kept_k, sup_k):
        img = img_in.copy()
        for d in sup_k:
            cv2.circle(img, (int(round(d['cx'])), int(round(d['cy']))), 3, BLUE, -1)
        for d in kept_k:
            cx, cy, nm = int(round(d['cx'])), int(round(d['cy'])), d['class_name']
            if nm == 'filled_circle':
                cv2.circle(img, (cx, cy), RADIUS, RED, -1)
            elif nm == 'open_circle':
                cv2.circle(img, (cx, cy), RADIUS, RED, THICK)
            elif nm == 'filled_square':
                cv2.rectangle(img, (cx-RADIUS, cy-RADIUS), (cx+RADIUS, cy+RADIUS), RED, -1)
            elif nm == 'open_square':
                cv2.rectangle(img, (cx-RADIUS, cy-RADIUS), (cx+RADIUS, cy+RADIUS), RED, THICK)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_adapt = _draw(img_bgr, kept, suppressed)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
    fig2.suptitle('Adaptive x-column NMS', fontsize=13, fontweight='bold')
    axes[0].imshow(img_rgb);   axes[0].set_title('Original', fontsize=11); axes[0].axis('off')
    axes[1].imshow(img_adapt)
    axes[1].set_title(
        f'Adaptive NMS  (bin=0.75xd={adaptive_bin_width:.1f}px, d={title_d})\n'
        f'kept={len(kept)}, suppressed={len(suppressed)}',
        fontsize=10,
    )
    axes[1].axis('off')
    for b in range(int(np.ceil(W / adaptive_bin_width)) + 1):
        axes[1].axvline(b * adaptive_bin_width, color='green', lw=0.8, ls='--', alpha=0.6)
    legend_handles = [
        mpatches.Patch(color='red',     label='Kept (known symbols)'),
        mpatches.Patch(color='#0050FF', label='Suppressed'),
    ]
    axes[1].legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    _save_jpg(fig2, os.path.join(out_dir, 'adaptive_nms_detections.jpg'))


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run adaptive-NMS point detection on a chart image.')
    parser.add_argument('img_path',   help='Path to the input image')
    parser.add_argument('model_path', help='Path to chart_marker_net_v3.pth')
    parser.add_argument('--detector', default=None,
                        help='Path to chart_marker_detector_v3.py '
                             '(if not already on sys.path)')
    parser.add_argument('--out_dir',  default='/tmp/adaptive_nms_out',
                        help='Directory for diagnostic figures')
    parser.add_argument('--classes',  nargs='+',
                        default=['filled_circle', 'open_circle',
                                 'filled_square', 'open_square'],
                        help='Known symbol class names to retain')
    args = parser.parse_args()

    detector_py = args.detector
    if detector_py is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '1_point_detection_v3.py')
        if os.path.isfile(candidate):
            detector_py = candidate

    result = detect_with_adaptive_nms(
        img_path         = args.img_path,
        model_path       = args.model_path,
        known_classes    = args.classes,
        detector_py_path = detector_py,
        out_dir          = args.out_dir,
    )

    print(f"\nAdaptive NMS results:")
    print(f"  d_est      = {result['d_est']}")
    print(f"  bin_width  = {result['bin_width']:.1f} px")
    print(f"  mode_xs    = {np.round(result['mode_xs'], 1).tolist()}")
    print(f"  kept       = {len(result['kept'])}")
    print(f"  suppressed = {len(result['suppressed'])}")
    for cls in args.classes:
        n = sum(1 for d in result['kept'] if d['class_name'] == cls)
        print(f"    {cls}: {n}")
    print(f"\nDiagnostic figures saved to: {args.out_dir}")
