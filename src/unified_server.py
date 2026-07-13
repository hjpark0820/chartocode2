"""
Unified digitizer server — one FastAPI app, two modes.

  mode=color : colour PK/PD plots -> run_A4_auto_v39.py (subprocess CLI)
  mode=bw    : greyscale / patent plots -> chartocode2 run_detection() (in-process)

Both modes end by writing the SAME edit_data.json (+ overlay + input.png) into the
job's out/ dir, so the front-end editor / live reconstruction / CSV are shared.

Heavy B&W deps (torch, ultralytics, timm, tkinter) are imported LAZILY, only when
a bw request arrives — a colour-only deployment never needs them.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

HERE = Path(__file__).resolve().parent
JOBS_ROOT = Path(tempfile.gettempdir()) / "unified_digitizer_jobs"
JOBS_ROOT.mkdir(exist_ok=True)

app = FastAPI(title="Unified Chart Digitizer")

# lazy singletons for the B&W path
_BW = {"run_detection": None, "adapter": None}


def _pick_color_pipeline() -> Path | None:
    """Highest run_A4_auto_v<N>.py (env PLOT_PIPELINE overrides)."""
    env = os.environ.get("PLOT_PIPELINE")
    if env and Path(env).exists():
        return Path(env)
    cands = []
    for d in (HERE, HERE.parent, HERE.parent / "webapp"):
        cands += list(d.glob("run_A4_auto_v*.py"))
    if not cands:
        return None

    def ver(p):
        try:
            return int("".join(ch for ch in p.stem.split("_v")[-1] if ch.isdigit()))
        except Exception:
            return -1
    return sorted(cands, key=ver)[-1]


def _load_bw():
    """Import the B&W entry point + adapter on first use (heavy deps)."""
    if _BW["run_detection"] is None:
        sys.path.insert(0, str(HERE))
        import bw_pipeline               # installs tkinter stub, re-exports run_detection
        import bw_to_edit_data
        _BW["run_detection"] = bw_pipeline.run_detection
        _BW["adapter"] = bw_to_edit_data.bw_to_edit_data
    return _BW["run_detection"], _BW["adapter"]


def _parse4(s):
    return tuple(int(float(v)) for v in s.split(",")) if s.strip() else None


def _truthy(s):
    return s.strip().lower() in ("1", "true", "on", "yes")


@app.get("/", response_class=HTMLResponse)
def index():
    # Serve index.html from the SAME folder as this server (src/) first, so you can
    # keep everything in one place. Falls back to ../webapp/index.html if present.
    idx = HERE / "index.html"
    if not idx.exists():
        idx = HERE.parent / "webapp" / "index.html"
    return idx.read_text(encoding="utf-8") if idx.exists() else "<h1>index.html not found</h1>"


@app.post("/digitize")
async def digitize(
    image: UploadFile = File(...),
    mode: str = Form("color"),                 # "color" | "bw"
    # shared
    plot_area: str = Form(""),
    x_min: str = Form(""), x_max: str = Form(""),
    y_min: str = Form(""), y_max: str = Form(""),
    x_log: str = Form(""), y_log: str = Form(""),
    # colour-only
    legend_box: str = Form(""),
    # bw-only
    legend_area: str = Form(""),
    known_classes: str = Form(""),             # comma list, or "" = all
    has_errorbars: str = Form(""),             # "", "1", "0"
    conf: str = Form(""),
    scale: str = Form(""),                     # "" = auto (from legend); else manual factor
    correct: str = Form(""),                   # "", "1" = run Step-5 SSIM correction
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / "input.png"
    with in_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    # input copy the front-end can display
    shutil.copyfile(in_path, out_dir / "input.png")

    if mode == "bw":
        return _run_bw(job_id, in_path, out_dir, plot_area, legend_area,
                       known_classes, x_min, x_max, y_min, y_max, x_log, y_log,
                       has_errorbars, conf, correct, scale)
    return _run_color(job_id, in_path, out_dir, plot_area, legend_box,
                      x_min, x_max, y_min, y_max, x_log, y_log)


def _run_color(job_id, in_path, out_dir, plot_area, legend_box,
               x_min, x_max, y_min, y_max, x_log, y_log):
    pipeline = _pick_color_pipeline()
    if pipeline is None:
        raise HTTPException(500, "no run_A4_auto_v<N>.py found (set PLOT_PIPELINE)")
    cmd = [sys.executable, str(pipeline), str(in_path), str(out_dir)]
    if legend_box.strip(): cmd += ["--legend-box", legend_box.strip()]
    if plot_area.strip():  cmd += ["--plot-area", plot_area.strip()]
    if x_min.strip():      cmd += ["--x-min", x_min.strip()]
    if x_max.strip():      cmd += ["--x-max", x_max.strip()]
    if y_min.strip():      cmd += ["--y-min", y_min.strip()]
    if y_max.strip():      cmd += ["--y-max", y_max.strip()]
    if _truthy(x_log):     cmd += ["--x-log"]
    if _truthy(y_log):     cmd += ["--y-log"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ok = (out_dir / "edit_data.json").exists()
    return _response(job_id, out_dir, "color", ok,
                     (proc.stdout or "") + (proc.stderr or ""))


# the 12 marker classes the B&W model knows (background excluded). When the user
# ticks nothing we fall back to ALL of them, because run_detection() SKIPS ViT
# detection entirely on an empty known_classes list (it filters kept-in by class).
BW_ALL_CLASSES = [
    "filled_circle", "open_circle", "filled_square", "open_square",
    "filled_triangle", "open_triangle", "filled_inv_triangle", "open_inv_triangle",
    "filled_rhombus", "open_rhombus", "x_marker", "plus_marker",
]


def _run_bw(job_id, in_path, out_dir, plot_area, legend_area, known_classes,
            x_min, x_max, y_min, y_max, x_log, y_log, has_errorbars, conf,
            correct="", scale=""):
    """Run B&W detection in a SEPARATE process (main thread) via bw_detect_cli.py,
    mirroring the colour path. This is what fixes the torch circular-import: torch
    is imported on that process's main thread, exactly like the standalone GUI,
    instead of inside a uvicorn worker thread."""
    if _parse4(plot_area) is None:
        raise HTTPException(400, "bw mode requires plot_area = x0,y0,x1,y1")
    cli = HERE / "bw_detect_cli.py"
    if not cli.exists():
        raise HTTPException(500, "bw_detect_cli.py not found next to unified_server.py")
    cmd = [sys.executable, str(cli), str(in_path), str(out_dir),
           "--plot-area", plot_area.strip(),
           "--x-min", (x_min or "0"), "--x-max", (x_max or "1"),
           "--y-min", (y_min or "0"), "--y-max", (y_max or "1")]
    if legend_area.strip():   cmd += ["--legend-area", legend_area.strip()]
    if _truthy(x_log):        cmd += ["--x-log"]
    if _truthy(y_log):        cmd += ["--y-log"]
    if known_classes.strip(): cmd += ["--classes", known_classes.strip()]
    if conf.strip():          cmd += ["--conf", conf.strip()]
    if has_errorbars != "":   cmd += ["--errorbars", "1" if _truthy(has_errorbars) else "0"]
    if scale.strip():         cmd += ["--scale", scale.strip()]
    if _truthy(correct):      cmd += ["--correct"]

    print("[unified] BW subprocess:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    # Echo the child's output to the server console so failures are visible.
    if proc.stdout:
        print("---- bw_detect_cli stdout ----\n" + proc.stdout)
    if proc.stderr:
        print("---- bw_detect_cli stderr ----\n" + proc.stderr)
    print(f"---- bw_detect_cli exit code: {proc.returncode} ----")
    ok = (out_dir / "edit_data.json").exists()
    return _response(job_id, out_dir, "bw", ok,
                     (proc.stdout or "") + "\n" + (proc.stderr or ""))


def _response(job_id, out_dir, mode, ok, log):
    def url(name):
        p = out_dir / name
        return f"/result/{job_id}/{name}" if p.exists() else None
    return JSONResponse({
        "job_id": job_id, "mode": mode, "ok": ok,
        "edit_data_url": url("edit_data.json"),
        "input_url": url("input.png"),
        "overlay_url": url("data_points_overlay.png"),
        "xlsx_url": url("data_points.xlsx"),
        "log": log[-4000:],
    })


@app.get("/result/{job_id}/{filename}")
def result(job_id: str, filename: str):
    path = JOBS_ROOT / job_id / "out" / filename
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(path))


if __name__ == "__main__":
    import uvicorn
    print("colour pipeline:", _pick_color_pipeline())
    uvicorn.run(app, host="0.0.0.0", port=8000)
