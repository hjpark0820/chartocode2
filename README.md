# Plot Digitizer (Colour + B&W)

A local web app that extracts the underlying data points from PK/PD
concentration–time plots (and similar scientific figures). It has two modes:

- **Colour** — separates curves by colour (classic computer-vision pipeline).
- **B&W** — detects markers by shape with a ViT/CNN model (deep-learning pipeline).

Both modes feed the same in-browser editor, so you can review, correct, and
download the extracted points as a spreadsheet. Everything runs on your own
machine; nothing is uploaded.

---

## Required files

Only the files below are needed to run the app. Everything else in the repository
is training / evaluation / experimental code and can be removed.

```
chartocode2/
├── README.md                              # this file
├── requirements.txt                       # Python dependencies
├── models/
│   └── chart_marker_net_v3.pth            # B&W marker model (the only model used)
└── src/
    ├── unified_server.py                  # web server — start this
    ├── index.html                         # web front-end
    │
    ├── run_A4_auto_v39.py                 # COLOUR pipeline (self-contained)
    │
    ├── bw_detect_cli.py                   # B&W detection runner (subprocess)
    ├── bw_pipeline.py                     # loads run_detection without the desktop GUI
    ├── bw_to_edit_data.py                 # converts B&W output to the shared format
    │
    ├── run_gui_v2.py                      # B&W core: run_detection()
    ├── chart_preprocessing.py             # B&W core: preprocessing / scale / legend
    ├── 1_point_detection_v3.py            # B&W core: ViT marker detector
    ├── 2_point_detection_adaptive_nms_v2.py  # B&W core: adaptive-NMS detection
    ├── 3_segment_detection_v2.py          # B&W core: segment detector (primary)
    ├── 3_segment_detection.py             # B&W core: segment detector (used by preprocessing)
    ├── 4_segment_refinement.py            # B&W core: segment refinement (Step 5)
    └── 5_correction_v2.py                 # B&W core: Step-5 SSIM correction
```

That is **15 files in `src/`**, plus `models/chart_marker_net_v3.pth`,
`requirements.txt`, and this README.

---

## Requirements

- **Python 3.12** (3.10+ works).
- **Tesseract OCR** — used by the Colour mode to read axis numbers automatically.
  Optional if you always type axis values by hand.
  - Windows: install from the UB Mannheim Tesseract build (default location
    `C:\Program Files\Tesseract-OCR`).
  - macOS: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`

### Python packages

```bash
# Web layer
pip install fastapi "uvicorn[standard]" python-multipart

# Colour pipeline
pip install opencv-python numpy scipy matplotlib openpyxl pytesseract

# B&W pipeline (deep learning — larger download)
pip install -r requirements.txt
```

On macOS/Linux use `python3`/`pip3`. On Windows, if `pip` is not found, use
`python -m pip ...`.

> The Colour mode does **not** need PyTorch. If you only use Colour mode, you can
> skip `pip install -r requirements.txt` — the heavy deep-learning packages are
> imported only when B&W mode is used.

---

## Running the app

```bash
cd src
python unified_server.py          # macOS/Linux: python3 unified_server.py
```

Then open **http://localhost:8000** in your browser.
(Use `localhost`, not `0.0.0.0`.) Keep the terminal window open while you use the
app; press `Ctrl+C` to stop it.

---

## Using it

1. **Choose PNG** — upload your plot image.
2. **Mode** — pick Colour or B&W.
3. **Draw** the plot area (and the legend) by dragging a rectangle.
4. **Axis values** — enter x/y min & max; tick *log scale* for log axes.
   Scientific notation is accepted (`1e-15`, `10^-15`).
5. **B&W only** — tick the marker shapes present; choose Auto or Manual scale.
6. **Run** — review the overlay.
7. **Correct** — open the editor to drag / add / delete points. In B&W mode you
   can also **Run Step 5 correction** to refine the points automatically and
   compare before/after.
8. **Download** the data as a spreadsheet.

---

## Notes

- The B&W model file is `models/chart_marker_net_v3.pth`. The server passes this
  to both detection and Step-5 correction.
- The first B&W run in a session is slower because the model loads once.
- For best B&W accuracy, tick **only** the marker shapes actually present in your
  plot rather than leaving all of them selected.
