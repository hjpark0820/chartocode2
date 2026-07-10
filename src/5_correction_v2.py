"""
5_correction.py
===============
Stage 4: Greedy SSIM-based correction pipeline with suppressed-point activation.

This module is the final stage of the chart-digitisation pipeline:

  Stage 1a  1_point_detection_v3.py           -- V3 ViT sliding-window detector
  Stage 1b  2_point_detection_adaptive_nms.py -- adaptive x-column NMS
  Stage 2   3_segment_detection.py            -- directional-probe segment detector
  Stage 3   4_segment_refinement.py           -- mode-grid segment refinement
  Stage 4   5_correction.py  (this file)      -- greedy SSIM correction

Algorithm (per iteration)
-------------------------
1.  Render a synthetic image I_t from the current active point set P.
2.  Detect segments L_t from I_t.
3.  Select l* via two-stage strategy:
    - Current strategy: l* = argmax_{l in L_refined} delta-SSIM(I_t, I_t union l)
      (l* drawn FIRST on white canvas, render on top -- z-order last).
      If gain < L_STAR_GAIN_THRESH, skip to old strategy.
    - Old strategy: exhaustively iterate through ALL L_t segments ranked by
      SSIM gain (l* drawn FIRST on white canvas, I_0 composited on top).
      Try each l* until an improvement is found or all exhausted.
4.  Build candidate locations from the two endpoints of l*.
5.  Generate perturbations: ADD / REPLACE / DELETE at each candidate,
    plus ACTIVATE (promote the single closest suppressed point per endpoint).
    DELETE/REPLACE candidates restricted to 2D Euclidean distance <= PT_TOL.
6.  Filter out perturbations that create NEW same-class close-pair conflicts
    (2D Euclidean distance <= X_APPROX) relative to the baseline P_in.
7.  Apply post-perturbation sliding-window NMS to each candidate that adds
    a point (ADD/ACTIVATE/REPLACE), so the SSIM score reflects the clean set.
8.  Accept the perturbation that minimises 1 - SSIM(I_0, I(P_tilde)).
9.  Apply final post-perturbation NMS on the accepted point set.
10. If no improvement -> stop.

Key design choices
------------------
* l* overlay thickness = 1 px; rendered line segments use linewidth = 3
  for l* identification (Rule i) and linewidth = 1 for SSIM comparison (Rule ii).
* l* is drawn FIRST (z-order last): draw l* on white canvas, then render
  segments + symbols on top.  l* only shows in true gaps of the render.
* ACTIVATE considers only the closest suppressed point per l* endpoint
  (at most 2 suppressed candidates per iteration).
* Conflict check uses 2D Euclidean distance, not x-column buckets.
* Post-perturbation NMS uses a sliding window (0.75 * grid_step).
* The mode_xs array from Stage 1b is used directly as the x-grid for
  segment refinement (replaces the old k-means infer_x_grid).

Public API
----------
run_pipeline(plot_name, img_path, out_dir,
             model_path, detector_py_path,
             known_classes=KNOWN,
             mode_xs=None)
    -> history list  [(iter, action, n_pts, 1-SSIM, improved), ...]

run_correction(img_path, model_path, detector_py_path,
               known_classes=KNOWN, out_dir=None, mode_xs=None)
    -> dict with keys: 'history', 'out_dir'
"""

import os
import sys
import math
import random
import importlib.util
from collections import defaultdict, Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity as _ssim_fn
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN = [
    'filled_circle', 'open_circle',
    'filled_square',  'open_square',
]

T_MAX    = 25      # maximum correction iterations
PT_TOL           = 18.0   # px radius for candidate / suppressed search
X_APPROX         = 8.0    # px tolerance for same-class close-pair conflict check
                           # (overridden at runtime to grid_step * 0.5)
RENDER_MARKER_PT = 5.0    # marker size (pt) for rendered plots
L_STAR_GAIN_THRESH = 1e-4 # threshold for current-strategy l* gain; below this
                           # skip directly to old strategy

SYM_COLORS = {
    'filled_circle':       '#0077BB',
    'open_circle':         '#EE7733',
    'filled_square':       '#009944',
    'open_square':         '#CC3311',
    'open_triangle':       '#33BBEE',
    'open_inv_triangle':   '#44BB99',
    'filled_triangle':     '#EE3377',
    'filled_inv_triangle': '#999933',
    'plus_marker':         '#AA3377',
    'x_marker':            '#BBBBBB',
    'open_rhombus':        '#DDCC77',
    'filled_rhombus':      '#884400',
    'suppressed':          '#AAAAAA',
}

SYM_MPL = {
    'filled_circle':       ('o', True,  1.0),
    'open_circle':         ('o', False, 1.0),
    'filled_square':       ('s', True,  0.9),
    'open_square':         ('s', False, 0.9),
    'open_triangle':       ('^', False, 1.0),
    'open_inv_triangle':   ('v', False, 1.0),
    'filled_triangle':     ('^', True,  1.0),
    'filled_inv_triangle': ('v', True,  1.0),
    'plus_marker':         ('P', True,  1.1),
    'x_marker':            ('X', True,  1.1),
    'open_rhombus':        ('D', False, 0.9),
    'filled_rhombus':      ('D', True,  0.9),
    'suppressed':          ('x', False, 0.8),
}

BASE_MS   = 5
ALPHA_DET = 0.70

ACTION_COLORS = {
    'ADD':      '#009944',
    'DELETE':   '#CC3311',
    'REPLACE':  '#FF8800',
    'ACTIVATE': '#9900CC',
    'NONE':     '#888888',
}


# ---------------------------------------------------------------------------
# Module loading helper
# ---------------------------------------------------------------------------

def _load_module(name, path):
    """Dynamically load a Python module from an arbitrary file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# SSIM distance
# ---------------------------------------------------------------------------

def ssim_dist(A, B):
    """1 - SSIM between two BGR images (lower = more similar)."""
    a = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    b = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    return 1.0 - float(_ssim_fn(a, b, data_range=1.0))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_from_points(points, pa_shape, linewidth=1.0):
    """
    Render a synthetic chart image from a list of active point dicts.

    Parameters
    ----------
    points : list of dict
        Each dict must have 'cx', 'cy', 'class_name', 'class_idx'.
        Points with class_name='suppressed' are excluded automatically.
    pa_shape : (H, W)
        Pixel dimensions of the plotting area.
    linewidth : float
        Width of the connecting lines (default 1.0).

    Returns
    -------
    BGR numpy array of shape (H, W, 3).
    """
    from chart_marker_detector_v3 import (
        CLASS_NAMES as _CN, N_SYMBOLS,
        DPI, MARKER_PT, _MPL_MARKERS,
    )

    H_pa, W_pa = pa_shape
    active = [p for p in points if p.get('class_name') != 'suppressed']

    by_class = defaultdict(list)
    for p in active:
        ci = p.get('class_idx', _CN.index(p['class_name']))
        by_class[ci].append((p['cx'], p['cy']))

    fig_w_in = W_pa / DPI
    fig_h_in = H_pa / DPI
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, W_pa)
    ax.set_ylim(0, H_pa)
    ax.axis('off')

    for ci, pts in by_class.items():
        if len(pts) < 2:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [H_pa - p[1] for p in pts_sorted]   # flip y (matplotlib origin = bottom)
        ax.plot(xs, ys, color='black', linewidth=linewidth, zorder=ci)

    ms_pt = RENDER_MARKER_PT
    for ci, pts in by_class.items():
        mcode, filled = _MPL_MARKERS[ci]
        fc = 'black' if filled else 'white'
        for (px, py) in pts:
            py_mpl = H_pa - py
            ax.plot(px, py_mpl,
                    marker=mcode, markersize=ms_pt,
                    color='black', markerfacecolor=fc,
                    markeredgecolor='black', markeredgewidth=0.8,
                    linestyle='none', zorder=N_SYMBOLS + ci * 3 + 1)

    fig.canvas.draw()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf  = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgba = buf.astype(np.float32) / 255.0
    alpha    = rgba[:, :, 3:4]
    white_bg = np.ones_like(rgba[:, :, :3])
    rgb_comp = rgba[:, :, :3] * alpha + white_bg * (1.0 - alpha)
    img_bgr  = cv2.cvtColor((rgb_comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    plt.close(fig)

    if img_bgr.shape[:2] != (H_pa, W_pa):
        img_bgr = cv2.resize(img_bgr, (W_pa, H_pa))
    return img_bgr


def _render_on_canvas(points, pa_shape, linewidth=3.0, underlay_bgr=None):
    """
    Render segments and symbols onto a canvas with transparent background.
    If underlay_bgr is provided (H x W x 3 uint8), it is used as the
    background instead of white -- so l* drawn beforehand is underneath
    (z-order last for l*).
    """
    from chart_marker_detector_v3 import (
        CLASS_NAMES as _CN, N_SYMBOLS,
        DPI, MARKER_PT, _MPL_MARKERS,
    )
    H_pa, W_pa = pa_shape
    active = [p for p in points if p.get('class_name') != 'suppressed']
    by_class = defaultdict(list)
    for p in active:
        ci = p.get('class_idx', _CN.index(p['class_name']))
        by_class[ci].append((p['cx'], p['cy']))

    fig_w_in = W_pa / DPI
    fig_h_in = H_pa / DPI
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Use transparent background so gaps let the underlay show through
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.set_xlim(0, W_pa)
    ax.set_ylim(0, H_pa)
    ax.axis('off')

    for ci, pts in by_class.items():
        if len(pts) < 2:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [H_pa - p[1] for p in pts_sorted]
        ax.plot(xs, ys, color='black', linewidth=linewidth, zorder=ci)

    ms_pt = RENDER_MARKER_PT
    for ci, pts in by_class.items():
        mcode, filled = _MPL_MARKERS[ci]
        fc = 'black' if filled else 'white'
        for (px, py) in pts:
            py_mpl = H_pa - py
            ax.plot(px, py_mpl,
                    marker=mcode, markersize=ms_pt,
                    color='black', markerfacecolor=fc,
                    markeredgecolor='black', markeredgewidth=0.8,
                    linestyle='none', zorder=N_SYMBOLS + ci * 3 + 1)

    fig.canvas.draw()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf  = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    rgba = buf.astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:4]
    buf_h, buf_w = rgba.shape[:2]

    if underlay_bgr is not None:
        # Composite: drawn content on top, underlay shows through transparent gaps
        underlay_resized = cv2.resize(underlay_bgr, (buf_w, buf_h))
        underlay_rgb = cv2.cvtColor(underlay_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb_comp = rgba[:, :, :3] * alpha + underlay_rgb * (1.0 - alpha)
    else:
        white_bg = np.ones_like(rgba[:, :, :3])
        rgb_comp = rgba[:, :, :3] * alpha + white_bg * (1.0 - alpha)

    img_bgr = cv2.cvtColor((rgb_comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    plt.close(fig)
    if img_bgr.shape[:2] != (H_pa, W_pa):
        img_bgr = cv2.resize(img_bgr, (W_pa, H_pa))
    return img_bgr


# ---------------------------------------------------------------------------
# Conflict check (2D Euclidean, new conflicts only)
# ---------------------------------------------------------------------------

def count_new_conflicts(P_baseline, P_new, x_approx):
    """Return number of NEW same-class close-pair conflicts in P_new vs P_baseline.

    A conflict is a pair of same-class active points whose 2D Euclidean distance
    is <= x_approx.  Only conflicts that are NOT already present in P_baseline
    are counted, so pre-existing near-duplicates do not block all perturbations.
    """
    def _conflict_pairs(P):
        active = [p for p in P if p.get('class_name') != 'suppressed']
        pairs  = set()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if active[i]['class_name'] == active[j]['class_name']:
                    d = math.hypot(active[i]['cx'] - active[j]['cx'],
                                   active[i]['cy'] - active[j]['cy'])
                    if d <= x_approx:
                        key = (active[i]['class_name'],
                               round(active[i]['cx']), round(active[i]['cy']),
                               round(active[j]['cx']), round(active[j]['cy']))
                        pairs.add(key)
        return pairs

    new_conflicts = _conflict_pairs(P_new) - _conflict_pairs(P_baseline)
    return len(new_conflicts)


# ---------------------------------------------------------------------------
# Post-perturbation sliding-window NMS
# ---------------------------------------------------------------------------

def post_perturbation_nms(P, S, window):
    """Apply sliding-window NMS to P after a perturbation.

    For each class, sort active points by cx.  Sweep with a sliding window
    of width `window`.  When two same-class points fall within the window,
    keep the one with higher confidence (ties broken randomly); move the
    other to S.

    Returns (P_clean, S_updated).
    """
    active  = [p for p in P if p.get('class_name') != 'suppressed']
    by_cls  = defaultdict(list)
    for p in active:
        by_cls[p['class_name']].append(p)

    keep_ids = set()
    for cls, pts in by_cls.items():
        pts_sorted = sorted(pts, key=lambda p: p['cx'])
        suppressed_ids = set()
        for i, anchor in enumerate(pts_sorted):
            if id(anchor) in suppressed_ids:
                continue
            for j in range(i + 1, len(pts_sorted)):
                other = pts_sorted[j]
                if other['cx'] - anchor['cx'] > window:
                    break
                if id(other) in suppressed_ids:
                    continue
                # Within window -- suppress the weaker one
                a_conf = anchor.get('confidence', 0.0)
                o_conf = other.get('confidence', 0.0)
                if o_conf > a_conf or (o_conf == a_conf and random.random() < 0.5):
                    suppressed_ids.add(id(anchor))
                    anchor = other   # promote other as new anchor
                else:
                    suppressed_ids.add(id(other))
            if id(anchor) not in suppressed_ids:
                keep_ids.add(id(anchor))

    P_clean   = [p for p in active if id(p) in keep_ids]
    newly_sup = [p for p in active if id(p) not in keep_ids]
    for p in newly_sup:
        p_copy = dict(p)
        p_copy['class_name'] = 'suppressed'
        S = S + [p_copy]
    return P_clean, S


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------

def generate_perturbations(P, sup_near, cand_locs, symbol_set):
    """
    Generate ADD / REPLACE / DELETE / ACTIVATE perturbations.

    Parameters
    ----------
    P          : current active point list
    sup_near   : suppressed points near l* endpoints (at most 2)
    cand_locs  : list of {'cx', 'cy'} dicts (l* endpoints)
    symbol_set : list of symbol class names to try

    Returns
    -------
    list of (action_str, P_tilde) tuples
    """
    from chart_marker_detector_v3 import CLASS_NAMES as _CN

    perturbations = []

    for c in cand_locs:
        cx, cy = c['cx'], c['cy']
        # Use 2D Euclidean distance (PT_TOL) to restrict DELETE/REPLACE candidates
        # to points that are genuinely near the anchor -- not just the same x-column.
        Q = [p for p in P
             if math.hypot(p['cx'] - cx, p['cy'] - cy) <= PT_TOL
             and p['class_name'] != 'suppressed']

        for s in symbol_set:
            ci     = _CN.index(s)
            new_pt = {'cx': cx, 'cy': cy, 'class_name': s, 'class_idx': ci}
            perturbations.append(('ADD', P + [new_pt]))
            for q in Q:
                P_new = [p for p in P if p is not q] + [new_pt]
                perturbations.append(('REPLACE', P_new))

        for q in Q:
            perturbations.append(('DELETE', [p for p in P if p is not q]))

    # ACTIVATE: promote a suppressed point into P with a real symbol type
    for sup in sup_near:
        for s in symbol_set:
            ci = _CN.index(s)
            activated = {'cx': sup['cx'], 'cy': sup['cy'],
                         'class_name': s, 'class_idx': ci,
                         '_activated_from_suppressed': True}
            perturbations.append(('ACTIVATE', P + [activated]))

    return perturbations


# ---------------------------------------------------------------------------
# l* selection strategies
# ---------------------------------------------------------------------------

def _select_l_star_current(L_refined, I_t, points, lw=1):
    """l* = argmax_{l in L_refined} delta-SSIM(I_t, I_t union l).

    l* is drawn FIRST onto a white canvas (thin, lw=1), then all segments
    and symbols are rendered ON TOP with their normal appearance.  This means
    l* is only visible in the true gaps of the render -- it is hidden beneath
    rendered lines and beneath the white interiors of open symbols (z-order last).
    """
    H, W = I_t.shape[:2]

    best_l, best_d = None, -float('inf')
    for seg in L_refined:
        # 1. Draw l* on a white canvas first
        underlay = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.line(underlay,
                 (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])),
                 (0, 0, 0), lw)
        # 2. Render segments + symbols on top (l* is z-order last)
        I_ov = _render_on_canvas(points, (H, W), linewidth=3.0, underlay_bgr=underlay)
        d = ssim_dist(I_t, I_ov)
        if d > best_d:
            best_d, best_l = d, seg
    return best_l, best_d


def _select_l_star_old(L_t, I0, lw=1):
    """l* = argmax_{l in L_t} delta-SSIM(I_0, I_0 union l).

    The overlay segment is drawn FIRST onto a white canvas (lw=1, black),
    then the original image I_0 is composited ON TOP using a dilated mask
    so that l* is hidden inside open-symbol interiors in I_0 (z-order last).
    """
    H, W = I0.shape[:2]
    I0_gray   = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    raw_mask  = (I0_gray < 250)
    # Dilate to fill open-symbol interiors in the original image
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    orig_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel).astype(bool)

    best_l, best_d = None, -float('inf')
    for seg in L_t:
        # 1. Draw l* on a white canvas
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.line(canvas,
                 (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])),
                 (0, 0, 0), lw)
        # 2. Composite: wherever I_0 has content (dilated), overwrite with I_0
        I_ov = canvas.copy()
        I_ov[orig_mask] = I0[orig_mask]
        d = ssim_dist(I0, I_ov)
        if d > best_d:
            best_d, best_l = d, seg
    return best_l, best_d


def _rank_all_l_star_old(L_t, I0, lw=1):
    """Rank ALL L_t segments by SSIM gain (descending) for exhaustive search.

    Same z-order-last compositing as _select_l_star_old, but returns the
    full ranked list instead of just the best.
    """
    H, W = I0.shape[:2]
    I0_gray   = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    raw_mask  = (I0_gray < 250)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    orig_mask = cv2.dilate(raw_mask.astype(np.uint8), kernel).astype(bool)

    scored = []
    for seg in L_t:
        canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        cv2.line(canvas,
                 (int(seg[0]), int(seg[1])),
                 (int(seg[2]), int(seg[3])),
                 (0, 0, 0), lw)
        I_ov = canvas.copy()
        I_ov[orig_mask] = I0[orig_mask]
        d = ssim_dist(I0, I_ov)
        scored.append((seg, d))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# One correction iteration (with all fixes)
# ---------------------------------------------------------------------------

def run_one_iteration(P_in, S_in, I0, L_refined, known_classes,
                      nms_window=None):
    """
    Execute one greedy correction step.

    Parameters
    ----------
    P_in          : list of active point dicts
    S_in          : list of suppressed point dicts
    I0            : original chart image (BGR)
    L_refined     : refined segments from Stage 3
    known_classes : list of symbol class names
    nms_window    : float or None
        Sliding-window width for post-perturbation NMS.
        If None, NMS is skipped.

    Returns a state dict with keys:
      converged, P_in, P_out, S_in, S_out,
      I_t, I_out, L_t, l_star, l_star_diff, l_star_strategy,
      anchor_left, anchor_right, candidates, sup_near,
      added, removed, action, best_dist, baseline, improved,
      n_valid_perturbs, n_sup_near
    """
    from segment_detector import detect_debug

    H, W = I0.shape[:2]
    # Rule i: render I_t with thick lines so the l* overlay stands out
    I_t  = render_from_points(P_in, (H, W), linewidth=3.0)

    seg_res = detect_debug(I_t)
    L_t     = seg_res['segments']

    # Stage 1: current l* strategy
    best_l_star, best_l_diff = _select_l_star_current(L_refined, I_t, P_in)
    l_star_strategy = 'current'
    use_old_strategy = (best_l_star is None) or (best_l_diff < L_STAR_GAIN_THRESH)

    if best_l_star is None and not L_t:
        return {'converged': True, 'P_out': P_in, 'S_out': S_in,
                'I_t': I_t, 'L_t': L_t}

    # Rule ii: thin render (lw=1, default) for SSIM scoring
    I_t_thin = render_from_points(P_in, (H, W))   # lw=1
    baseline = ssim_dist(I0, I_t_thin)
    best_P, best_d, best_action = P_in, baseline, 'NONE'
    improved = False
    candidates, sup_near, valid_perturbs = [], [], []
    anchor_left, anchor_right = (0.0, 0.0), (0.0, 0.0)

    # --- Current strategy: only run if gain is above threshold ---
    if not use_old_strategy:
        lx = float(min(best_l_star[0], best_l_star[2]))
        ly = float(best_l_star[1] if best_l_star[0] <= best_l_star[2] else best_l_star[3])
        rx = float(max(best_l_star[0], best_l_star[2]))
        ry = float(best_l_star[3] if best_l_star[0] <= best_l_star[2] else best_l_star[1])
        anchor_left  = (lx, ly)
        anchor_right = (rx, ry)
        cand_locs = [{'cx': anchor_left[0],  'cy': anchor_left[1]},
                     {'cx': anchor_right[0], 'cy': anchor_right[1]}]

        candidates, seen_ids = [], set()
        for anchor in [anchor_left, anchor_right]:
            for p in P_in:
                pid = id(p)
                if pid in seen_ids:
                    continue
                if math.hypot(p['cx'] - anchor[0], p['cy'] - anchor[1]) <= PT_TOL:
                    candidates.append(p)
                    seen_ids.add(pid)

        sup_near, seen_sup = [], set()
        for anchor in [anchor_left, anchor_right]:
            best_sup, best_d_sup = None, float('inf')
            for s in S_in:
                d = math.hypot(s['cx'] - anchor[0], s['cy'] - anchor[1])
                if d <= PT_TOL and d < best_d_sup:
                    best_d_sup, best_sup = d, s
            if best_sup is not None and id(best_sup) not in seen_sup:
                sup_near.append(best_sup)
                seen_sup.add(id(best_sup))

        all_perturbs   = generate_perturbations(P_in, sup_near, cand_locs, known_classes)
        valid_perturbs = [(a, pt) for a, pt in all_perturbs
                          if count_new_conflicts(P_in, pt, X_APPROX) == 0]

        for act, pt in valid_perturbs:
            # Apply post-perturbation NMS if the action adds a point
            if nms_window is not None and act in ('ADD', 'ACTIVATE', 'REPLACE'):
                pt, _ = post_perturbation_nms(pt, S_in, nms_window)
            d = ssim_dist(I0, render_from_points(pt, (H, W)))
            if d < best_d:
                best_d, best_P, best_action = d, pt, act

        improved = best_d < baseline - 1e-7

    # --- Old strategy: exhaustively iterate through ALL L_t segments ---
    if (use_old_strategy or not improved) and L_t:
        scored_lt = _rank_all_l_star_old(L_t, I0)
        print(f"  Old strategy: {len(scored_lt)} L_t segments to try")

        for seg_rank, (old_l_star, old_l_diff) in enumerate(scored_lt, 1):
            lx2 = float(min(old_l_star[0], old_l_star[2]))
            ly2 = float(old_l_star[1] if old_l_star[0] <= old_l_star[2] else old_l_star[3])
            rx2 = float(max(old_l_star[0], old_l_star[2]))
            ry2 = float(old_l_star[3] if old_l_star[0] <= old_l_star[2] else old_l_star[1])
            anchor_left2  = (lx2, ly2)
            anchor_right2 = (rx2, ry2)
            cand_locs2 = [{'cx': anchor_left2[0], 'cy': anchor_left2[1]},
                          {'cx': anchor_right2[0], 'cy': anchor_right2[1]}]

            candidates2, seen_ids2 = [], set()
            for anchor in [anchor_left2, anchor_right2]:
                for p in P_in:
                    pid = id(p)
                    if pid in seen_ids2:
                        continue
                    if math.hypot(p['cx'] - anchor[0], p['cy'] - anchor[1]) <= PT_TOL:
                        candidates2.append(p)
                        seen_ids2.add(pid)

            sup_near2, seen_sup2 = [], set()
            for anchor in [anchor_left2, anchor_right2]:
                best_sup2, best_d2 = None, float('inf')
                for s in S_in:
                    d2 = math.hypot(s['cx'] - anchor[0], s['cy'] - anchor[1])
                    if d2 <= PT_TOL and d2 < best_d2:
                        best_d2, best_sup2 = d2, s
                if best_sup2 is not None and id(best_sup2) not in seen_sup2:
                    sup_near2.append(best_sup2)
                    seen_sup2.add(id(best_sup2))

            all_perturbs2   = generate_perturbations(P_in, sup_near2, cand_locs2, known_classes)
            valid_perturbs2 = [(a, pt) for a, pt in all_perturbs2
                               if count_new_conflicts(P_in, pt, X_APPROX) == 0]

            local_best_d, local_best_P, local_best_action = best_d, best_P, best_action
            for act, pt in valid_perturbs2:
                if nms_window is not None and act in ('ADD', 'ACTIVATE', 'REPLACE'):
                    pt, _ = post_perturbation_nms(pt, S_in, nms_window)
                d = ssim_dist(I0, render_from_points(pt, (H, W)))
                if d < local_best_d:
                    local_best_d, local_best_P, local_best_action = d, pt, act

            if local_best_d < baseline - 1e-7:
                best_d, best_P, best_action = local_best_d, local_best_P, local_best_action
                improved        = True
                best_l_star     = old_l_star
                best_l_diff     = old_l_diff
                l_star_strategy = f'old[{seg_rank}]'
                anchor_left     = anchor_left2
                anchor_right    = anchor_right2
                candidates      = candidates2
                sup_near        = sup_near2
                valid_perturbs  = valid_perturbs2
                print(f"    Found improvement at L_t rank {seg_rank}  "
                      f"gain={old_l_diff:.5f}  action={best_action}")
                break
            else:
                print(f"    L_t rank {seg_rank}: gain={old_l_diff:.5f}  no improvement")

        if not improved:
            print(f"  Old strategy: all {len(scored_lt)} L_t segments exhausted, "
                  f"no improvement found")

    if not improved:
        best_action = 'NONE'
        best_P      = P_in

    # Update S_out: remove activated suppressed point from S
    S_out = S_in
    if improved and best_action == 'ACTIVATE':
        new_pts = [p for p in best_P if p.get('_activated_from_suppressed')]
        if new_pts:
            act_pt = new_pts[0]
            S_out  = [s for s in S_in
                      if not (abs(s['cx'] - act_pt['cx']) < X_APPROX
                              and abs(s['cy'] - act_pt['cy']) < X_APPROX)]
            for p in best_P:
                p.pop('_activated_from_suppressed', None)

    # Final post-perturbation NMS on the accepted point set
    if improved and nms_window is not None and best_action in ('ADD', 'ACTIVATE', 'REPLACE'):
        best_P, S_out = post_perturbation_nms(best_P, S_out, nms_window)

    p_in_ids  = {id(p) for p in P_in}
    p_out_ids = {id(p) for p in best_P}
    added   = [p for p in best_P if id(p) not in p_in_ids]
    removed = [p for p in P_in   if id(p) not in p_out_ids]

    # Rule ii: output render also uses thin lines for SSIM reporting.
    I_out = render_from_points(best_P, (H, W))   # lw=1

    return {
        'converged':        False,
        'P_in':             P_in,
        'P_out':            best_P,
        'S_in':             S_in,
        'S_out':            S_out,
        'sup_near':         sup_near,
        'I_t':              I_t,
        'I_out':            I_out,
        'L_t':              L_t,
        'l_star':           best_l_star,
        'l_star_diff':      best_l_diff,
        'l_star_strategy':  l_star_strategy,
        'anchor_left':      anchor_left,
        'anchor_right':     anchor_right,
        'candidates':       candidates,
        'added':            added,
        'removed':          removed,
        'action':           best_action,
        'best_dist':        best_d,
        'baseline':         baseline,
        'improved':         improved,
        'n_valid_perturbs': len(valid_perturbs),
        'n_sup_near':       len(sup_near),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _plot_det(ax, d, alpha=ALPHA_DET, ms_mult=1.0, zorder=5,
              override_color=None, override_mfc=None):
    cn    = d['class_name']
    color = override_color or SYM_COLORS.get(cn, 'gray')
    mcode, filled, ms_scale = SYM_MPL.get(cn, ('o', True, 1.0))
    ms  = BASE_MS * ms_scale * ms_mult
    mfc = (override_mfc if override_mfc is not None
           else (color if filled else 'white'))
    mec = override_color or ('#111111' if not filled else color)
    ax.plot(d['cx'], d['cy'], marker=mcode, color='white', mfc='white',
            mew=2.0, ms=ms + 2, linestyle='none', alpha=0.45, zorder=zorder - 1)
    ax.plot(d['cx'], d['cy'], marker=mcode, color=color, mfc=mfc,
            mew=1.2, ms=ms, linestyle='none', alpha=alpha, zorder=zorder)


def _draw_seg(ax, seg, color, lw=1.5, alpha=0.85, linestyle='-', zorder=3):
    ax.plot([seg[0], seg[2]], [seg[1], seg[3]],
            color=color, lw=lw, alpha=alpha, linestyle=linestyle,
            solid_capstyle='round', zorder=zorder)


def _legend_handles(P, S=None):
    seen = {}
    for d in list(P) + (list(S) if S else []):
        cn = d['class_name']
        if cn not in seen:
            color = SYM_COLORS.get(cn, 'gray')
            mcode, filled, _ = SYM_MPL.get(cn, ('o', True, 1.0))
            mfc = color if filled else 'white'
            lbl = cn if cn != 'suppressed' else 'suppressed (NMS)'
            seen[cn] = Line2D([0], [0], marker=mcode, color=color, mfc=mfc,
                              mew=1.2, ms=7, linestyle='none', label=lbl)
    return list(seen.values())


def _save_jpg(fig, path):
    tmp = path.replace('.jpg', '_tmp.png')
    fig.savefig(tmp, dpi=130, bbox_inches='tight', facecolor='#F8F8F8')
    plt.close(fig)
    PILImage.open(tmp).convert('RGB').save(path, 'JPEG', quality=92)
    os.remove(tmp)


# ---------------------------------------------------------------------------
# Per-iteration figure (5-panel)
# ---------------------------------------------------------------------------

def build_iter_figure(iter_n, P_in, S_in, state,
                      I0, I0_rgb, L_refined, out_path):
    """Build and save the 5-panel per-iteration diagnostic figure."""
    I_t       = state['I_t']
    I_out     = state['I_out']
    l_star    = state['l_star']
    action    = state['action']
    best_dist = state['best_dist']

    I_t_rgb   = cv2.cvtColor(I_t,   cv2.COLOR_BGR2RGB)
    I_out_rgb = cv2.cvtColor(I_out, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(28, 6.5), facecolor='#F8F8F8')
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.88, bottom=0.10)

    for ax, lbl in zip(axes, ['A', 'B', 'C', 'D', 'E']):
        ax.text(-0.02, 1.02, lbl, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='right',
                color='#333333')
        ax.axis('off')

    # A: I0 + P_in + S_in + L_refined
    ax = axes[0]
    ax.imshow(I0_rgb)
    for seg in L_refined:
        _draw_seg(ax, seg, '#00AA44', lw=1.0, alpha=0.50, zorder=2)
    for d in S_in:
        _plot_det(ax, d, alpha=0.35, ms_mult=0.8, zorder=3)
    for d in P_in:
        _plot_det(ax, d, zorder=5)
    ax.set_title(f'A -- I_0 + P_in + S_in + L_ref\n'
                 f'{len(P_in)} active  {len(S_in)} suppressed  {len(L_refined)} segs',
                 fontsize=9, pad=4)

    # B: I_t + L_t + l*
    ax = axes[1]
    ax.imshow(I_t_rgb)
    for seg in state['L_t']:
        _draw_seg(ax, seg, '#4488CC', lw=0.9, alpha=0.55, zorder=2)
    lx = float(min(l_star[0], l_star[2]))
    ly = float(l_star[1] if l_star[0] <= l_star[2] else l_star[3])
    rx = float(max(l_star[0], l_star[2]))
    ry = float(l_star[3] if l_star[0] <= l_star[2] else l_star[1])
    ax.plot([lx, rx], [ly, ry], color='#CC2200', lw=2.5, alpha=0.95, zorder=6,
            solid_capstyle='round')
    ax.plot([lx, rx], [ly, ry], color='#CC2200', lw=7, alpha=0.12, zorder=5)
    strat = state.get('l_star_strategy', 'current')
    strat_str = '(current: L_0->I_t)' if strat == 'current' else f'(fallback: L_t->I_0 {strat})'
    ax.set_title(f'B -- I_t + L_t + l* {strat_str}\ndelta-SSIM={state["l_star_diff"]:.5f}',
                 fontsize=9, pad=4)

    # C: Candidates + suppressed near anchors
    ax = axes[2]
    ax.imshow(I_t_rgb)
    for anchor in [state['anchor_left'], state['anchor_right']]:
        circ = plt.Circle(anchor, PT_TOL, color='#FF8800', fill=False,
                          lw=1.5, alpha=0.75, zorder=4)
        ax.add_patch(circ)
        ax.plot(*anchor, 'D', ms=5, color='#FF8800', mew=0, alpha=0.9, zorder=5)
    for d in state['candidates']:
        _plot_det(ax, d, override_color='#CC2200', ms_mult=1.4, zorder=6)
    for d in state['sup_near']:
        ax.plot(d['cx'], d['cy'], 'D', ms=7, color='#9900CC',
                mew=1.5, mfc='none', alpha=0.85, zorder=7)
        ax.annotate(d.get('original_class_name', '?')[:4],
                    (d['cx'], d['cy']), fontsize=5.5, color='#9900CC',
                    ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', zorder=8)
    ax.set_title(f'C -- Candidates\n{len(state["candidates"])} active  '
                 f'{state["n_sup_near"]} suppressed near anchors  '
                 f'{state["n_valid_perturbs"]} valid perturbs',
                 fontsize=9, pad=4)

    # D: Action taken
    ax = axes[3]
    ax.imshow(I_t_rgb)
    for d in P_in:
        _plot_det(ax, d, alpha=0.25, zorder=3)
    for d in state['removed']:
        ax.plot(d['cx'], d['cy'], 'x', ms=12, mew=2.5, color='#CC3311',
                alpha=0.95, zorder=7)
    for d in state['added']:
        ax.plot(d['cx'], d['cy'], '*', ms=13,
                color=ACTION_COLORS.get(action, '#009944'),
                mew=0, alpha=0.95, zorder=7)
        _plot_det(ax, d, ms_mult=1.5, zorder=8)
    act_col = ACTION_COLORS.get(action, '#888888')
    ax.set_title(f'D -- Action: {action}\n'
                 f'added={len(state["added"])}  removed={len(state["removed"])}',
                 fontsize=9, pad=4, color=act_col, fontweight='bold')

    # E: After-state
    ax = axes[4]
    ax.imshow(I_out_rgb)
    for seg in L_refined:
        _draw_seg(ax, seg, '#00AA44', lw=1.0, alpha=0.45, zorder=2)
    for d in state['S_out']:
        _plot_det(ax, d, alpha=0.30, ms_mult=0.8, zorder=3)
    for d in state['P_out']:
        _plot_det(ax, d, zorder=5)
    ax.set_title(f'E -- I_out + P_out + S_out + L_ref\n'
                 f'{len(state["P_out"])} active  {len(state["S_out"])} suppressed  '
                 f'1-SSIM={best_dist:.5f}',
                 fontsize=9, pad=4)

    handles = _legend_handles(state['P_out'], state['S_out'])
    fig.legend(handles=handles, ncol=min(len(handles), 9),
               fontsize=7.5, loc='lower center',
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, columnspacing=0.8)

    strat_label = state.get('l_star_strategy', 'current')
    strat_str2  = ('l*: L_0->I_t' if strat_label == 'current'
                   else f'l*: L_t->I_0 ({strat_label})')
    plt.suptitle(
        f'Greedy Correction -- Iteration {iter_n}  |  '
        f'Action: {action}  |  '
        f'{strat_str2}  delta-SSIM={state["l_star_diff"]:.5f}  |  '
        f'{len(state["candidates"])} active cands  '
        f'{state["n_sup_near"]} suppressed cands  |  '
        f'{state["n_valid_perturbs"]} valid perturbs  |  '
        f'1-SSIM: {best_dist:.5f}',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _save_jpg(fig, out_path)
    print(f'    Saved: {out_path}')


# ---------------------------------------------------------------------------
# Convergence curve
# ---------------------------------------------------------------------------

def build_convergence_figure(history, plot_name, out_path):
    """Plot and save the 1-SSIM convergence curve."""
    ts    = [h[0] for h in history]
    dists = [h[3] for h in history]
    acts  = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor('#F8F8F8')
    ax.set_facecolor('#FAFAFA')
    ax.plot(ts, dists, 'o-', color='#0077BB', lw=2.2, ms=8, zorder=4)
    for t_i, d_i, a_i in zip(ts, dists, acts):
        c = ACTION_COLORS.get(a_i, '#888888')
        ax.plot(t_i, d_i, 'o', ms=10, color=c, zorder=5)
        ax.annotate(f'{a_i}\n{d_i:.4f}',
                    xy=(t_i, d_i), xytext=(0, 14), textcoords='offset points',
                    ha='center', fontsize=7.5, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='#AAAAAA', alpha=0.88))
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('1 - SSIM  (lower = better)', fontsize=12)
    ax.set_title(f'{plot_name} -- SSIM convergence curve',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(ts)
    ax.grid(True, alpha=0.3)
    leg = [Line2D([0], [0], marker='o', color=c, ms=9, mew=0,
                  linestyle='none', label=a)
           for a, c in ACTION_COLORS.items()]
    ax.legend(handles=leg, fontsize=9, loc='upper right', framealpha=0.9)
    plt.tight_layout()
    _save_jpg(fig, out_path)
    print(f'    Convergence curve: {out_path}')


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def build_combined_figure(iter_paths, out_path):
    """Stack all per-iteration figures vertically into one JPEG."""
    imgs = [cv2.imread(p) for p in iter_paths if os.path.exists(p)]
    if not imgs:
        return
    max_w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[1] < max_w:
            pad = np.full((im.shape[0], max_w - im.shape[1], 3), 245, dtype=np.uint8)
            im  = np.hstack([im, pad])
        padded.append(im)
    combined = np.vstack(padded)
    PILImage.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)).save(
        out_path, 'JPEG', quality=92)
    print(f'    Combined figure: {out_path}')


# ---------------------------------------------------------------------------
# Full pipeline for one plot
# ---------------------------------------------------------------------------

def run_pipeline(plot_name, img_path, out_dir,
                 model_path, detector_py_path,
                 known_classes=None,
                 mode_xs=None,
                 prep_info=None):
    """
    Run the full 4-stage pipeline for a single chart image.

    Parameters
    ----------
    plot_name        : str
        Human-readable name for the plot (used in figure titles).
    img_path         : str
        Path to the input image.
    out_dir          : str
        Directory for output figures.
    model_path       : str
        Path to chart_marker_net_v3.pth.
    detector_py_path : str
        Path to 1_point_detection_v3.py.
    known_classes    : list of str, optional
        Symbol class names to keep (default: KNOWN).
    mode_xs          : array-like or None
        KDE mode x-positions from Stage 1b.  If provided, used directly as
        the x-grid for segment refinement (skips k-means infer_x_grid).
        If None, infer_x_grid() is called on the detected points.

    Returns
    -------
    history : list of (iter, action, n_pts, 1-SSIM, improved)
    """
    global X_APPROX   # override at runtime based on grid step

    if known_classes is None:
        known_classes = KNOWN

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  Plot: {plot_name}")
    print(f"{'='*65}")

    # Load detector module
    _load_module('chart_marker_detector_v3', detector_py_path)

    # Add deliverable directory to path so sibling modules are importable
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    from segment_detector import detect_debug
    from refine_segments  import infer_x_grid, refine, grid_step

    # Stage 1b: adaptive NMS detection
    _adnms_path = os.path.join(_this_dir, '2_point_detection_adaptive_nms.py')
    _adnms_mod  = _load_module('adaptive_nms_detection', _adnms_path)
    detect_with_adaptive_nms = _adnms_mod.detect_with_adaptive_nms

    # ── Preprocessing: run if not already done externally ─────────────────
    # If prep_info was passed from pipeline.py, reuse it.
    # Otherwise, try to run preprocessing here (for standalone Stage 4 use).
    if prep_info is None:
        try:
            from chart_preprocessing import preprocess as _cp
            _img_for_prep = cv2.imread(img_path)
            prep_info = _cp(_img_for_prep, verbose=False)
            print('  Preprocessing (axes/legend/text) applied internally.')
        except ImportError:
            pass  # chart_preprocessing not available; proceed without it

    print('  Detecting markers (adaptive NMS) ...')
    det_result = detect_with_adaptive_nms(
        img_path         = img_path,
        model_path       = model_path,
        known_classes    = known_classes,
        detector_py_path = detector_py_path,
        prep_info        = prep_info,
    )
    P0 = det_result['kept']
    S0 = det_result['suppressed']
    if mode_xs is None:
        mode_xs = det_result['mode_xs']
    else:
        mode_xs = np.asarray(mode_xs)

    cnt = {k: sum(1 for d in P0 if d['class_name'] == k) for k in known_classes}
    print(f"  P0 (active):     {len(P0)} pts  "
          + '  '.join(f'{k}={v}' for k, v in cnt.items() if v > 0))
    print(f"  S0 (suppressed): {len(S0)} pts")

    # Stage 2+3: segment detection & refinement
    I0     = cv2.imread(img_path)
    H, W   = I0.shape[:2]
    I0_rgb = cv2.cvtColor(I0, cv2.COLOR_BGR2RGB)

    print('  Detecting & refining segments ...')
    # Pass prep_info so Stage 2 removes axis/legend/text noise from I0.
    # Note: detect_debug is called WITHOUT prep_info inside run_one_iteration
    # (for synthetic renders I_t), which is correct -- they are already clean.
    seg_result = detect_debug(I0, prep_info=prep_info)
    L0_raw     = seg_result['segments']

    if mode_xs is not None and len(mode_xs) >= 2:
        grid_xs = list(mode_xs)
    else:
        grid_xs = infer_x_grid(P0)

    step         = grid_step(grid_xs)
    L_refined, _ = refine(L0_raw, grid_xs)
    print(f"  L_refined: {len(L_refined)} segs  (grid step={step:.1f}px)")

    # Override X_APPROX and compute NMS window based on actual grid step
    X_APPROX   = step * 0.5
    nms_window = step * 0.75
    print(f"  X_APPROX={X_APPROX:.1f}px  NMS_WINDOW={nms_window:.1f}px")

    # Stage 4: greedy SSIM correction
    # Rule ii: initial baseline render uses thin lines (lw=1) for fair SSIM.
    I_init       = render_from_points(P0, (H, W))   # lw=1
    initial_dist = ssim_dist(I0, I_init)
    print(f"  Initial 1-SSIM: {initial_dist:.5f}")

    P_current  = P0
    S_current  = S0
    history    = [(0, '--', len(P0), initial_dist, False)]
    iter_paths = []

    for t in range(1, T_MAX + 1):
        print(f"\n  -- Iteration {t} " + '-' * 40)
        state = run_one_iteration(P_current, S_current, I0, L_refined,
                                  known_classes, nms_window=nms_window)

        if state['converged']:
            print(f"  Converged (no l* found) at iteration {t}.")
            break

        strat = state.get('l_star_strategy', 'current')
        print(f"  l* [{strat}] delta-SSIM={state['l_star_diff']:.5f}  "
              f"active_cands={len(state['candidates'])}  "
              f"sup_near={state['n_sup_near']}  "
              f"valid_perturbs={state['n_valid_perturbs']}")
        print(f"  Action: {state['action']}  "
              f"added={len(state['added'])}  removed={len(state['removed'])}  "
              f"1-SSIM={state['best_dist']:.5f}  "
              f"({'improved' if state['improved'] else 'no improvement'})")

        history.append((t, state['action'], len(state['P_out']),
                        state['best_dist'], state['improved']))

        out_path = os.path.join(out_dir, f'iter{t:02d}.jpg')
        build_iter_figure(t, P_current, S_current, state,
                          I0, I0_rgb, L_refined, out_path)
        iter_paths.append(out_path)

        P_current = state['P_out']
        S_current = state['S_out']

        if not state['improved']:
            print('  No improvement -- stopping.')
            break

    final_dist = history[-1][3]
    print(f"\n  Final: {len(P_current)} active pts  {len(S_current)} suppressed  "
          f"|  1-SSIM: {final_dist:.5f}")

    build_convergence_figure(history, plot_name,
                             os.path.join(out_dir, 'convergence.jpg'))
    build_combined_figure(iter_paths,
                          os.path.join(out_dir, 'all_iterations_combined.jpg'))

    print(f"\n  Iteration log:")
    print(f"  {'Iter':>4}  {'Action':>10}  {'N pts':>6}  {'1-SSIM':>10}  {'Improved':>8}")
    for t_i, act, npts, dist, imp in history:
        print(f"  {t_i:>4}  {act:>10}  {npts:>6}  {dist:>10.5f}  {str(imp):>8}")

    return history


# ---------------------------------------------------------------------------
# Convenience wrapper (single-image, no plot_name required)
# ---------------------------------------------------------------------------

def run_correction(img_path, model_path, detector_py_path,
                   known_classes=None, out_dir=None, mode_xs=None,
                   prep_info=None):
    """
    Convenience wrapper that runs the full pipeline on a single image.

    Parameters
    ----------
    prep_info : dict or None
        Preprocessing result from chart_preprocessing.preprocess().
        If provided, axis/legend/text/LLOQ noise removal is applied
        to Stage 1b and Stage 2 detections.

    Returns
    -------
    dict with keys:
        'history'  -- convergence history list
        'out_dir'  -- path to the output directory
    """
    if known_classes is None:
        known_classes = KNOWN
    if out_dir is None:
        out_dir = os.path.join('/tmp', 'correction_out',
                               os.path.splitext(os.path.basename(img_path))[0])

    plot_name = os.path.splitext(os.path.basename(img_path))[0]
    history   = run_pipeline(
        plot_name        = plot_name,
        img_path         = img_path,
        out_dir          = out_dir,
        model_path       = model_path,
        detector_py_path = detector_py_path,
        known_classes    = known_classes,
        mode_xs          = mode_xs,
        prep_info        = prep_info,
    )
    return {'history': history, 'out_dir': out_dir}


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the full chart-digitisation pipeline (Stages 1-4).')
    parser.add_argument('img_path',   help='Path to the input chart image')
    parser.add_argument('model_path', help='Path to chart_marker_net_v3.pth')
    parser.add_argument('--detector', default=None,
                        help='Path to 1_point_detection_v3.py '
                             '(auto-detected if in same directory)')
    parser.add_argument('--out_dir',  default=None,
                        help='Output directory  (default: /tmp/correction_out/<name>)')
    parser.add_argument('--classes',  nargs='+', default=None,
                        help='Known symbol class names  '
                             '(default: filled_circle open_circle '
                             'filled_square open_square)')
    args = parser.parse_args()

    # Auto-locate detector script in the same directory as this file
    detector_py = args.detector
    if detector_py is None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '1_point_detection_v3.py')
        if os.path.isfile(candidate):
            detector_py = candidate

    result = run_correction(
        img_path         = args.img_path,
        model_path       = args.model_path,
        detector_py_path = detector_py,
        known_classes    = args.classes,
        out_dir          = args.out_dir,
    )
    print(f"\nOutput saved to: {result['out_dir']}")
