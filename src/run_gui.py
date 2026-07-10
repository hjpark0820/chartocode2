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
1. Load a chart image (PNG / JPG / TIFF).
2. Drag a rectangle on the canvas to define the **Plotting Area**.
3. Drag a second rectangle to define the **Legend Area** (optional).
4. Check which of the 12 marker classes are present in the chart.
5. Enter X-axis range (min / max) and Y-axis range (min / max).
6. Choose X-axis scale (Linear / Log10) and Y-axis scale (Linear / Log10).
7. Click **Run Detection**.
8. Results are saved next to the input image:
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
    "filled_circle":       (255,  60,  60),
    "open_circle":         (255, 140,   0),
    "filled_square":       ( 60, 180,  60),
    "open_square":         (  0, 200, 100),
    "open_triangle":       ( 60,  60, 255),
    "open_inv_triangle":   (120,  60, 255),
    "filled_triangle":     (  0, 180, 255),
    "filled_inv_triangle": (180,   0, 255),
    "open_rhombus":        (255,   0, 180),
    "filled_rhombus":      (200,   0,  80),
    "x_marker":            (180, 120,   0),
    "plus_marker":         (  0, 160, 160),
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


# ── Detection logic ───────────────────────────────────────────────────────────
def run_detection(img_bgr: np.ndarray,
                  plot_area_px: tuple,
                  legend_area_px: tuple | None,
                  known_classes: list[str],
                  x_range: tuple,
                  y_range: tuple,
                  x_log: bool,
                  y_log: bool,
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

    # ── Build preprocessing info from user-supplied areas ─────────────────
    log_fn("[Step 0] Building preprocessing info from user areas …")
    try:
        from chart_preprocessing import preprocess as _cp
        # Override plot_area and legend_box with user-supplied values
        prep_info = _cp(img_bgr, user_plot_area=plot_area_px, verbose=False)
        # Override legend_box if user drew one
        if legend_area_px is not None:
            # Rebuild clean_fn with user legend box
            from chart_preprocessing import (
                remove_noise_from_binary,
                detect_axes, detect_lloq_line
            )
            axis_row = prep_info['axis_row']
            axis_col = prep_info['axis_col']
            lloq_row = prep_info['lloq_row']
            legend_box = legend_area_px

            def clean_fn(binary_mask: np.ndarray) -> np.ndarray:
                return remove_noise_from_binary(
                    binary_mask, img_bgr,
                    axis_row, axis_col, plot_area_px,
                    legend_box, lloq_row
                )
            prep_info['legend_box'] = legend_box
            prep_info['clean_fn']   = clean_fn
        log_fn(f"  plot_area  = {prep_info['plot_area']}")
        log_fn(f"  legend_box = {prep_info['legend_box']}")
        log_fn(f"  lloq_row   = {prep_info['lloq_row']}")
    except ImportError:
        log_fn("  chart_preprocessing not found; skipping noise removal.")
        prep_info = None

    # ── Stage 2: segment detection (no ViT model needed) ─────────────────
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

    # ── Stage 1 (ViT point detection) if model exists ─────────────────────
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
            cv2.imwrite(tmp_path, img_bgr)

            import inspect
            sig = inspect.signature(mod2.detect_with_adaptive_nms)
            supports_prep = 'prep_info' in sig.parameters

            call_kwargs = dict(
                img_path         = tmp_path,
                model_path       = str(MODEL_PATH),
                known_classes    = known_classes,
                detector_py_path = str(SRC_DIR / "1_point_detection_v3.py"),
            )
            if supports_prep:
                call_kwargs['prep_info'] = prep_info
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

    # ── Convert pixel → data coordinates ─────────────────────────────────
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

    # ── Build overlay image ───────────────────────────────────────────────
    log_fn("[Step 5] Building overlay image …")
    overlay = img_bgr.copy()

    # Draw plot area rectangle
    cv2.rectangle(overlay, (ax0, ay0), (ax1, ay1), (0, 200, 0), 2)

    # Draw legend area rectangle
    if legend_area_px is not None:
        lx0, ly0, lx1, ly1 = legend_area_px
        cv2.rectangle(overlay, (lx0, ly0), (lx1, ly1), (200, 0, 200), 2)

    # Draw detected markers
    for d in detections:
        cx = int(round(float(d['cx_px'])))
        cy = int(round(float(d['cy_px'])))
        color = MARKER_COLORS.get(d['class_name'], (255, 0, 0))
        # BGR order
        color_bgr = (color[2], color[1], color[0])
        cv2.circle(overlay, (cx, cy), 8, color_bgr, 2)
        cv2.circle(overlay, (cx, cy), 2, color_bgr, -1)
        # Label
        short = d['class_name'].replace('_marker', '').replace('_', ' ')
        cv2.putText(overlay, short, (cx + 10, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_bgr, 1, cv2.LINE_AA)

    log_fn(f"[Done] {len(detections)} data points found.")
    return {
        'detections':  detections,
        'overlay_img': overlay,
        'segs':        segs,
    }


# ── GUI Application ───────────────────────────────────────────────────────────
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

        self._mode_var = tk.StringVar(value="plot")
        tk.Radiobutton(toolbar, text="Draw Plotting Area",
                       variable=self._mode_var, value="plot",
                       font=("Helvetica", 10)).pack(side=tk.LEFT, padx=6)
        tk.Radiobutton(toolbar, text="Draw Legend Area",
                       variable=self._mode_var, value="legend",
                       font=("Helvetica", 10)).pack(side=tk.LEFT, padx=6)

        tk.Button(toolbar, text="🗑  Clear Rects",
                  command=self._clear_rects,
                  font=("Helvetica", 10), padx=6).pack(side=tk.LEFT, padx=4)

        tk.Button(toolbar, text="▶  Run Detection",
                  command=self._run,
                  font=("Helvetica", 11, "bold"),
                  bg="#2a7ae2", fg="white", padx=10).pack(side=tk.RIGHT, padx=8, pady=3)

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

        # Right panel
        right = tk.Frame(main, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        right.pack_propagate(False)

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
        self._show_image(img)
        self._log(f"Loaded: {path}  ({img.shape[1]}×{img.shape[0]})")
        self.plot_area_lbl.config(text="Plot area:   (not set)")
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
        self._plot_rid   = None
        self._legend_rid = None
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

        if self._drag_mode == "plot":
            self.plot_rect = (cx0, cy0, cx1, cy1)   # canvas coords for display
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

    def _clear_rects(self):
        self.plot_rect   = None
        self.legend_rect = None
        if hasattr(self, '_plot_rect_img'):
            del self._plot_rect_img
        if hasattr(self, '_legend_rect_img'):
            del self._legend_rect_img
        if self._plot_rid:
            self.canvas.delete(self._plot_rid)
            self._plot_rid = None
        if self._legend_rid:
            self.canvas.delete(self._legend_rid)
            self._legend_rid = None
        self.plot_area_lbl.config(text="Plot area:   (not set)")
        self.legend_area_lbl.config(text="Legend area: (not set)")
        self._log("Rectangles cleared.")

    # ── Run detection ─────────────────────────────────────────────────────
    def _run(self):
        if self.img_bgr is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return
        if not hasattr(self, '_plot_rect_img'):
            messagebox.showwarning("No plot area",
                                   "Please draw the Plotting Area first.")
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

        self._log("=" * 50)
        self._log(f"Running detection …")
        self._log(f"  Plot area : {self._plot_rect_img}")
        self._log(f"  Legend    : {legend_area}")
        self._log(f"  Markers   : {known_classes or '(all, ViT decides)'}")
        self._log(f"  X range   : [{x_min}, {x_max}]  {'log10' if x_log else 'linear'}")
        self._log(f"  Y range   : [{y_min}, {y_max}]  {'log10' if y_log else 'linear'}")

        # Run in background thread to keep GUI responsive
        def _worker():
            try:
                result = run_detection(
                    img_bgr       = self.img_bgr,
                    plot_area_px  = self._plot_rect_img,
                    legend_area_px= legend_area,
                    known_classes = known_classes,
                    x_range       = (x_min, x_max),
                    y_range       = (y_min, y_max),
                    x_log         = x_log,
                    y_log         = y_log,
                    log_fn        = self._log,
                )
                self.result_detections = result['detections']
                self._overlay_img = result['overlay_img']
                self.after(0, self._show_overlay)
            except Exception as e:
                import traceback
                self.after(0, lambda: self._log(f"ERROR: {e}\n{traceback.format_exc()}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_overlay(self):
        """Display the overlay image on the canvas."""
        self._show_image(self._overlay_img)
        n = len(self.result_detections)
        self._log(f"Overlay displayed. {n} data points found.")
        if n > 0:
            self._log("  class                  x_data          y_data")
            for d in self.result_detections[:20]:
                self._log(f"  {d['class_name']:<22} {d['x_data']:>14.6g}  {d['y_data']:>14.6g}")
            if n > 20:
                self._log(f"  … ({n - 20} more rows in saved CSV)")

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
