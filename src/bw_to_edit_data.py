"""
Adapter: convert the B&W (chartocode2) run_detection() output into the SAME
edit_data.json schema the colour pipeline (run_A4_auto_v39.py) emits, so the
unified web GUI reuses ONE point editor / live reconstruction / CSV path.

Colour edit_data.json (target):
{
  "image":       {"width", "height"},
  "plot_area":   [x0, y0, x1, y1],
  "calibration": {"x": {p0,p1,v0,v1,log,kind}, "y": {...}},
  "curves":      [{"name","label","rgb":[r,g,b],"points":[{"x","y"}, ...]}, ...]
}

B&W run_detection() (source) returns:
  {'detections': [{'class_name','cx_px','cy_px','x_data','y_data','confidence'}, ...],
   'legend_labels': {...}, 'overlay_img', ...}
Series are grouped by marker SHAPE class (class_name); each shape = one curve.
The source plot is greyscale, so distinct colours are ASSIGNED from a palette.
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional, Tuple

_SERIES_PALETTE = [
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207],
]
_CLASS_LABEL = {
    "filled_circle": "Filled circle", "open_circle": "Open circle",
    "filled_square": "Filled square", "open_square": "Open square",
    "filled_triangle": "Filled triangle", "open_triangle": "Open triangle",
    "filled_inv_triangle": "Filled inv triangle", "open_inv_triangle": "Open inv triangle",
    "filled_rhombus": "Filled diamond", "open_rhombus": "Open diamond",
    "x_marker": "X mark", "plus_marker": "Plus",
}


def _calib(plot_edge_lo: float, plot_edge_hi: float,
           v_lo: float, v_hi: float, is_log: bool) -> Dict[str, Any]:
    """2-anchor calibration. run_detection maps the plot-area edges linearly onto
    (v_lo, v_hi) (in log10 space if is_log), exactly matching px_to_data(), so the
    two anchors are simply the plot-area edges paired with the axis range ends."""
    return {"p0": float(plot_edge_lo), "p1": float(plot_edge_hi),
            "v0": float(v_lo), "v1": float(v_hi),
            "log": bool(is_log), "kind": "log" if is_log else "linear"}


def bw_to_edit_data(
    result: Dict[str, Any],
    plot_area_px: Tuple[float, float, float, float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    x_log: bool,
    y_log: bool,
    image_size: Tuple[int, int],
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build the common edit_data dict from a B&W run_detection() result.

    plot_area_px = (ax0, ay0, ax1, ay1); x_range/y_range = (min, max) axis values.
    """
    ax0, ay0, ax1, ay1 = [float(v) for v in plot_area_px]
    W, H = int(image_size[0]), int(image_size[1])
    labels = labels or result.get("legend_labels") or {}

    # group detections by marker class -> one curve each
    series: Dict[str, List[dict]] = {}
    for d in result.get("detections", []):
        series.setdefault(d["class_name"], []).append(d)

    curves = []
    for i, (cls, dets) in enumerate(series.items()):
        pts = sorted(({"x": int(round(d["cx_px"])), "y": int(round(d["cy_px"]))}
                      for d in dets), key=lambda p: p["x"])
        curves.append({
            "name": cls,
            "label": labels.get(cls) or _CLASS_LABEL.get(cls, cls),
            "rgb": _SERIES_PALETTE[i % len(_SERIES_PALETTE)],
            "points": pts,
        })

    return {
        "image": {"width": W, "height": H},
        "plot_area": [int(ax0), int(ay0), int(ax1), int(ay1)],
        "calibration": {
            # x: left edge -> x_min, right edge -> x_max
            "x": _calib(ax0, ax1, x_range[0], x_range[1], x_log),
            # y: bottom edge (ay1) -> y_min, top edge (ay0) -> y_max (image coords)
            "y": _calib(ay1, ay0, y_range[0], y_range[1], y_log),
        },
        "curves": curves,
    }


def write_bw_edit_data(out_dir: str, *args, **kwargs) -> str:
    data = bw_to_edit_data(*args, **kwargs)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "edit_data.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
