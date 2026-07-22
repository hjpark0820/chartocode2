"""
color_correct_cli.py -- Step-5 SSIM correction for the COLOUR pipeline.

Runs as a post-process on a finished colour job, so run_A4_auto_v<N>.py itself is
not touched:

    python color_correct_cli.py <input.png> <out_dir> --plot-area x0,y0,x1,y1
           [--legend-area x0,y0,x1,y1] [--correct-iters N] [--prev-state state.json]

It reads <out_dir>/edit_data.json (curves with their display RGB + points),
corrects each colour curve independently via color_step5, writes the corrected
edit_data.json back, redraws the overlay, and saves a state file so a following
run can continue from this result (same behaviour as the B&W Step 5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Windows consoles default to cp949/cp1252, which cannot encode characters the
# correction prints (e.g. an arrow).  Force UTF-8 with replacement so a stray
# glyph can never abort a curve's correction.
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from color_step5 import correct_colour_curves, swatch_ink_mask  # noqa: E402

STATE_NAME = "color_correction_state.json"


def _p4(s):
    if not s or not s.strip():
        return None
    v = [int(round(float(t))) for t in s.split(",")]
    return tuple(v) if len(v) == 4 else None


def _draw_overlay(img_bgr, curves):
    """Original image with the corrected points marked in each curve's colour."""
    ov = img_bgr.copy()
    ref = max(ov.shape[:2])
    ro = max(4, int(ref * 0.011))
    ri = max(2, int(ro * 0.30))
    th = max(1, int(ref * 0.002))
    for c in curves:
        rgb = c.get("rgb") or [200, 0, 0]
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        for p in c.get("points", []):
            x, y = int(round(p["x"])), int(round(p["y"]))
            cv2.circle(ov, (x, y), ro, bgr, th)
            cv2.circle(ov, (x, y), ri, bgr, -1)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--plot-area", default="")
    ap.add_argument("--legend-area", default="")
    ap.add_argument("--correct-iters", default="")
    ap.add_argument("--prev-state", default="")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        print("[color-correct] ERROR: cannot read", a.image)
        sys.exit(2)
    H, W = img.shape[:2]

    ed_path = os.path.join(a.out_dir, "edit_data.json")
    if not os.path.exists(ed_path):
        print("[color-correct] ERROR: edit_data.json not found in", a.out_dir)
        sys.exit(2)
    with open(ed_path, "r", encoding="utf-8") as f:
        ed = json.load(f)

    pa = _p4(a.plot_area) or (0, 0, W - 1, H - 1)
    lg = _p4(a.legend_area)
    iters = int(a.correct_iters) if a.correct_iters.strip() else None

    # Continue from a previous correction when one is supplied, so repeated
    # presses compose instead of restarting from the raw detection.
    prev = None
    if a.prev_state.strip() and os.path.exists(a.prev_state.strip()):
        try:
            with open(a.prev_state.strip(), "r", encoding="utf-8") as f:
                prev = json.load(f)
            print(f"[color-correct] continuing from previous state "
                  f"({len(prev.get('curves', {}))} curves)")
        except Exception as e:
            print(f"[color-correct] prev-state load failed ({e}); starting fresh")
            prev = None

    curves_in = []
    for c in ed.get("curves", []):
        name = c.get("name") or c.get("label") or f"curve{len(curves_in)}"
        pts = [(float(p["x"]), float(p["y"])) for p in c.get("points", [])]
        if prev and name in (prev.get("curves") or {}):
            pts = [tuple(v) for v in prev["curves"][name]]
        curves_in.append({
            "name": name,
            "swatch_rgb": tuple(c.get("rgb") or (0, 0, 0)),
            "points": pts,
            "ink_mask": None,          # built from the swatch colour
        })

    # Shared x-column grid: data points across curves line up on the same x
    # positions, so cluster every curve's x values into columns.  These columns
    # are what the grid x path suppressed candidates are anchored to.
    _allx = sorted(x for c in curves_in for (x, _y) in c["points"])
    grid_xs = []
    if _allx:
        _tolx = max(4.0, (pa[2] - pa[0]) * 0.012)
        _grp = [_allx[0]]
        for v in _allx[1:]:
            if v - _grp[-1] <= _tolx:
                _grp.append(v)
            else:
                grid_xs.append(sum(_grp) / len(_grp)); _grp = [v]
        grid_xs.append(sum(_grp) / len(_grp))

    print(f"[color-correct] {len(curves_in)} curves, "
          f"{sum(len(c['points']) for c in curves_in)} points, "
          f"plot_area={pa}, iters={iters or 'default'}, "
          f"grid={len(grid_xs)} x-columns")

    results = correct_colour_curves(
        img, curves_in, pa, legend_box=lg,
        max_iters=iters, out_dir=os.path.join(a.out_dir, "step5"),
        grid_xs=grid_xs,
    )
    by_name = {r["name"]: r for r in results}

    # write corrected points back into edit_data.json
    for c in ed.get("curves", []):
        name = c.get("name") or c.get("label")
        r = by_name.get(name)
        if not r:
            continue
        c["points"] = [{"x": int(round(x)), "y": int(round(y))}
                       for (x, y) in sorted(r["points"])]
    with open(ed_path, "w", encoding="utf-8") as f:
        json.dump(ed, f)

    cv2.imwrite(os.path.join(a.out_dir, "data_points_overlay.png"),
                _draw_overlay(img, ed.get("curves", [])))

    with open(os.path.join(a.out_dir, STATE_NAME), "w", encoding="utf-8") as f:
        json.dump({"curves": {r["name"]: [[float(x), float(y)] for x, y in r["points"]]
                              for r in results}}, f)

    for r in results:
        if r.get("ssim_before") is None:
            print(f"[color-correct]   {r['name']}: {r['n_before']} -> {r['n_after']} pts")
        else:
            print(f"[color-correct]   {r['name']}: {r['n_before']} -> {r['n_after']} pts, "
                  f"1-SSIM {r['ssim_before']:.5f} -> {r['ssim_after']:.5f}")
    print("[color-correct] done ->", a.out_dir)


if __name__ == "__main__":
    main()
