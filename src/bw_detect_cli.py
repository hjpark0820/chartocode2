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
    ap.add_argument("--correct-iters", default="",
                    help="max number of Step-5 correction iterations (empty = default cap)")
    ap.add_argument("--prev-state", default="",
                    help="path to a previous correction_state.json; when given, Step-5 "
                         "continues from that point set instead of re-detecting")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # import torch FIRST, on this process's main thread, so its submodules
    # initialise cleanly before run_detection triggers `import torch` internally.
    import time as _tmod
    _t_imp0 = _tmod.perf_counter()
    try:
        import torch  # noqa: F401
        _cuda = torch.cuda.is_available()
        _dev = torch.cuda.get_device_name(0) if _cuda else "CPU"
        print(f"[bw-cli] torch {torch.__version__} loaded  "
              f"(cuda_available={_cuda}, device={_dev})")
        if not _cuda and "+cpu" in torch.__version__:
            print("[bw-cli] NOTE: this is a CPU-ONLY torch build (+cpu) — inference "
                  "runs on CPU (slow). Install a CUDA build to use the GPU.")
    except Exception as e:
        print(f"[bw-cli] ERROR: torch not available: {e}")
        sys.exit(3)

    import bw_pipeline           # tkinter stub + run_detection
    import bw_to_edit_data
    print(f"[bw-cli] [timing] imports (torch+pipeline): {_tmod.perf_counter()-_t_imp0:.2f}s")

    # BUILD SIGNATURE — print key module paths + mtime/size so the ACTUAL deployed
    # files can be verified (if a fix "does nothing", check these match your edits).
    try:
        import time as _tsig
        _srcdir = os.path.dirname(os.path.abspath(__file__))
        print("[bw-cli] BUILD SIGNATURE (src=%s):" % _srcdir)
        for _sf in ('run_gui_v2.py', '2_point_detection_adaptive_nms_v2.py',
                    '5_correction_v2.py', 'chart_marker_detector_v3.py'):
            _p = os.path.join(_srcdir, _sf)
            if os.path.exists(_p):
                _mt = _tsig.strftime('%Y-%m-%d %H:%M', _tsig.localtime(os.path.getmtime(_p)))
                print("[bw-cli]   %-38s mtime=%s size=%d" % (_sf, _mt, os.path.getsize(_p)))
            else:
                print("[bw-cli]   %-38s NOT FOUND" % _sf)
    except Exception as _sige:
        print("[bw-cli] BUILD SIGNATURE failed: %s" % _sige)

    img = cv2.imread(a.image)
    if img is None:
        print("[bw-cli] ERROR: could not read image"); sys.exit(2)
    H, W = img.shape[:2]
    pa = _p4(a.plot_area)
    xr = (float(a.x_min), float(a.x_max))
    yr = (float(a.y_min), float(a.y_max))
    kc = [c.strip() for c in a.classes.split(",") if c.strip()] or list(BW_ALL_CLASSES)
    eb = None if a.errorbars == "" else (a.errorbars.strip() in ("1", "true", "yes", "on"))

    # Load a previous Step-5 state up front: when present the ViT detection is
    # redundant (its points are replaced by the saved ones), so we skip it.
    _prev_state = None
    if a.correct and a.prev_state.strip() and os.path.exists(a.prev_state.strip()):
        try:
            with open(a.prev_state.strip(), "r", encoding="utf-8") as _pf:
                _prev_state = json.load(_pf)
            print(f"[bw-cli] prev-state loaded: "
                  f"{len(_prev_state.get('P_current') or [])} active, "
                  f"{len(_prev_state.get('S_current') or [])} suppressed")
        except Exception as _pe:
            print(f"[bw-cli] prev-state load failed ({_pe}); starting fresh")
            _prev_state = None

    _t_det0 = _tmod.perf_counter()
    result = bw_pipeline.run_detection(
        skip_point_detection=bool(_prev_state),
        img_bgr=img, plot_area_px=pa, legend_area_px=_p4(a.legend_area),
        known_classes=kc, x_range=xr, y_range=yr,
        x_log=a.x_log, y_log=a.y_log, has_errorbars=eb,
        upscale=(float(a.scale) if a.scale.strip() else None),
        conf_thresh=(float(a.conf) if a.conf else None),
        stride=None, has_lines=True, log_fn=_safe_log,
    )
    print(f"[bw-cli] [timing] run_detection total: {_tmod.perf_counter()-_t_det0:.2f}s")

    def _write_state(_P, _S, _what, _mode_xs=None):
        """Persist the point state (SCALED coords) so the next Step-5 continues
        from exactly these points instead of re-detecting on its own."""
        try:
            def _plain(_pts):
                out = []
                for _p in (_pts or []):
                    out.append({"cx": float(_p.get("cx", _p.get("cx_px", 0.0))),
                                "cy": float(_p.get("cy", _p.get("cy_px", 0.0))),
                                "class_name": str(_p.get("class_name", "")),
                                "class_idx": int(_p.get("class_idx", 0))})
                return out
            _mx = _mode_xs
            if _mx is None and _prev_state is not None:
                _mx = _prev_state.get("mode_xs")
            try:
                _mx = [float(v) for v in (_mx if _mx is not None else [])]
            except Exception:
                _mx = []
            with open(os.path.join(a.out_dir, "correction_state.json"), "w",
                      encoding="utf-8") as _sf:
                json.dump({"P_current": _plain(_P), "S_current": _plain(_S),
                           "mode_xs": _mx}, _sf)
            _safe_log(f"[bw-cli] saved correction_state.json from {_what} "
                      f"({len(_P or [])} active, {len(_S or [])} suppressed)")
        except Exception as _se:
            _safe_log(f"[bw-cli] state save failed: {_se}")

    # Save the DETECTION state too, so the first Step-5 press continues from the
    # points the user just saw rather than starting from a fresh detection.
    if not _prev_state:
        _write_state(result.get("kept_scaled"), result.get("suppressed_scaled"),
                     "detection", result.get("mode_xs"))

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
            _t_corr0 = _tmod.perf_counter()
            # Continue from a previous Step-5 result when one is supplied, so
            # successive corrections compose instead of re-detecting from scratch.
            _init_P = _prev_state.get("P_current") if _prev_state else None
            _init_S = _prev_state.get("S_current") if _prev_state else None
            _mode_xs = (_prev_state.get("mode_xs") if _prev_state else None) \
                       or result.get("mode_xs")
            if _prev_state:
                _safe_log(f"[bw-cli] continuing from previous state: "
                          f"{len(_init_P or [])} active, {len(_init_S or [])} suppressed")
            r5 = mod5.run_correction(
                img_path=_tmp, model_path=model_path, detector_py_path=detector_py,
                known_classes=kc, mode_xs=_mode_xs,
                prep_info=result.get("prep_info"), return_diag_imgs=False,
                max_iters=(int(a.correct_iters) if a.correct_iters.strip() else None),
                d_override=result.get("d_override"),
                init_points=_init_P, init_suppressed=_init_S,
            )
            _safe_log(f"[bw-cli] [timing] correction total: "
                      f"{_tmod.perf_counter() - _t_corr0:.2f}s")
            # Persist the corrected state so the NEXT Step-5 can continue from it.
            _write_state(r5.get("P_current"), r5.get("S_current"), "correction", _mode_xs)
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
                # Redraw the overlay from the CORRECTED points so the overlay image
                # (plot 2) matches edit_data.json / the live reconstruction (plot 3).
                # Mirrors run_detection's Step-5 overlay style (same colours/sizes).
                try:
                    _MC = getattr(gui, "MARKER_COLORS", {})
                    _ov2 = img.copy()
                    cv2.rectangle(_ov2, (pa[0], pa[1]), (pa[2], pa[3]), (0, 200, 0), 2)
                    _lg = _p4(a.legend_area)
                    if _lg is not None:
                        cv2.rectangle(_ov2, (_lg[0], _lg[1]), (_lg[2], _lg[3]), (200, 0, 200), 2)
                    _ref = max(_ov2.shape[:2])
                    _ro = max(4, int(_ref * 0.013)); _ri = max(2, int(_ro * 0.25))
                    _fs = max(0.30, _ref * 0.00065); _th = max(1, int(_ref * 0.002))
                    for _d in corrected:
                        _cx = int(round(float(_d['cx_px']))); _cy = int(round(float(_d['cy_px'])))
                        _c = _MC.get(_d['class_name'], (255, 0, 0)); _cbgr = (_c[2], _c[1], _c[0])
                        cv2.circle(_ov2, (_cx, _cy), _ro, _cbgr, _th)
                        cv2.circle(_ov2, (_cx, _cy), _ri, _cbgr, -1)
                        _sh = _d['class_name'].replace('_marker', '').replace('_', ' ')
                        cv2.putText(_ov2, _sh, (_cx + _ro + 2, _cy - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, _fs, _cbgr, _th, cv2.LINE_AA)
                    result["overlay_img"] = _ov2
                    _safe_log("[bw-cli] overlay redrawn from corrected points (plot 2 = plot 3)")
                except Exception as _oe:
                    _safe_log("[bw-cli] overlay redraw failed: " + str(_oe))
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
