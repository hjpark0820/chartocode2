"""
Command-line wrapper around chartocode2's run_detection(), used by the unified
server so B&W detection runs in its OWN process (main thread) — exactly like the
standalone GUI. This avoids the "partially initialised module 'torch' (circular
import)" error that happens when torch is first imported inside a uvicorn worker
thread. It writes the common edit_data.json (+ overlay) into <out_dir>.

Usage:
  python bw_detect_cli.py <image> <out_dir>
      --plot-area x0,y0,x1,y1
      [--legend-area x0,y0,x1,y1]
      [--x-min V --x-max V --y-min V --y-max V]
      [--x-log] [--y-log]
      [--classes filled_circle,open_square,...]   # empty => all 12
      [--conf 0.3] [--errorbars 0|1]
"""
import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BW_ALL_CLASSES = [
    "filled_circle", "open_circle", "filled_square", "open_square",
    "filled_triangle", "open_triangle", "filled_inv_triangle", "open_inv_triangle",
    "filled_rhombus", "open_rhombus", "x_marker", "plus_marker",
]


def _p4(s):
    return tuple(int(float(v)) for v in s.split(",")) if s else None


def main():
    # Force UTF-8 output. run_detection logs contain unicode (arrows, ellipsis);
    # on Windows the default console/pipe encoding is cp1252 and printing those
    # raises UnicodeEncodeError, killing the process AFTER detection succeeds.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    def _safe_log(*args):
        msg = " ".join(str(a) for a in args)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))

    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--plot-area", required=True)
    ap.add_argument("--legend-area", default="")
    ap.add_argument("--x-min", default="0"); ap.add_argument("--x-max", default="1")
    ap.add_argument("--y-min", default="0"); ap.add_argument("--y-max", default="1")
    ap.add_argument("--x-log", action="store_true")
    ap.add_argument("--y-log", action="store_true")
    ap.add_argument("--classes", default="")
    ap.add_argument("--conf", default="")
    ap.add_argument("--errorbars", default="")
    ap.add_argument("--scale", default="",
                    help="manual upscale factor; empty = auto-detect from legend")
    ap.add_argument("--correct", action="store_true",
                    help="run Step-5 SSIM greedy correction to refine points")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # import torch FIRST, on this process's main thread, so its submodules
    # initialise cleanly before run_detection triggers `import torch` internally.
    try:
        import torch  # noqa: F401
        print(f"[bw-cli] torch {torch.__version__} loaded")
    except Exception as e:
        print(f"[bw-cli] ERROR: torch not available: {e}")
        sys.exit(3)

    import bw_pipeline           # tkinter stub + run_detection
    import bw_to_edit_data

    img = cv2.imread(a.image)
    if img is None:
        print("[bw-cli] ERROR: could not read image"); sys.exit(2)
    H, W = img.shape[:2]
    pa = _p4(a.plot_area)
    xr = (float(a.x_min), float(a.x_max))
    yr = (float(a.y_min), float(a.y_max))
    kc = [c.strip() for c in a.classes.split(",") if c.strip()] or list(BW_ALL_CLASSES)
    eb = None if a.errorbars == "" else (a.errorbars.strip() in ("1", "true", "yes", "on"))

    result = bw_pipeline.run_detection(
        img_bgr=img, plot_area_px=pa, legend_area_px=_p4(a.legend_area),
        known_classes=kc, x_range=xr, y_range=yr,
        x_log=a.x_log, y_log=a.y_log, has_errorbars=eb,
        upscale=(float(a.scale) if a.scale.strip() else None),
        conf_thresh=(float(a.conf) if a.conf else None),
        stride=None, has_lines=True, log_fn=_safe_log,
    )

    # ── Optional Step-5 SSIM greedy correction: refine the point set ───────────
    if a.correct:
        try:
            import importlib.util, tempfile
            gui = sys.modules.get("run_gui_v2")          # imported via bw_pipeline
            model_path = str(gui.MODEL_PATH)
            detector_py = str(gui.SRC_DIR / "1_point_detection_v3.py")
            _spec = importlib.util.spec_from_file_location(
                "correction_v2", os.path.join(HERE, "5_correction_v2.py"))
            mod5 = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(mod5)

            up = float(result.get("upscale", 1.0) or 1.0)
            scaled = result.get("scaled_img_bgr")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as _tf:
                _tmp = _tf.name
            cv2.imwrite(_tmp, scaled if scaled is not None else img)
            _safe_log("[bw-cli] running Step-5 SSIM correction ...")
            r5 = mod5.run_correction(
                img_path=_tmp, model_path=model_path, detector_py_path=detector_py,
                known_classes=kc, mode_xs=result.get("mode_xs"),
                prep_info=result.get("prep_info"), return_diag_imgs=False,
            )
            try:
                os.unlink(_tmp)
            except Exception:
                pass
            inv = 1.0 / up if up else 1.0
            corrected = []
            for p in r5.get("P_current", []):
                if p.get("class_name") == "suppressed":
                    continue
                cx = float(p.get("cx", p.get("cx_px", 0))) * inv
                cy = float(p.get("cy", p.get("cy_px", 0))) * inv
                corrected.append({"class_name": p.get("class_name", "filled_circle"),
                                  "cx_px": cx, "cy_px": cy})
            if corrected:
                result["detections"] = corrected
                _safe_log(f"[bw-cli] Step-5 correction: {len(corrected)} active points")
            else:
                _safe_log("[bw-cli] Step-5 produced no points; keeping raw detections")
        except Exception as _ce:
            import traceback
            _safe_log("[bw-cli] Step-5 correction failed (keeping raw detections): "
                      + str(_ce))
            _safe_log(traceback.format_exc())

    ed = bw_to_edit_data.bw_to_edit_data(result, pa, xr, yr, a.x_log, a.y_log, (W, H))
    with open(os.path.join(a.out_dir, "edit_data.json"), "w") as f:
        json.dump(ed, f, indent=2)
    ov = result.get("overlay_img")
    if ov is not None:
        cv2.imwrite(os.path.join(a.out_dir, "data_points_overlay.png"), ov)

    n = sum(len(c["points"]) for c in ed["curves"])
    print(f"[bw-cli] done: {len(ed['curves'])} series, {n} points -> {a.out_dir}")


if __name__ == "__main__":
    main()
