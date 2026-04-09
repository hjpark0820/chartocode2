# Chart Data Extraction Pipeline

A computer vision pipeline that automatically extracts data point coordinates from scientific paper / patent chart images.

## Architecture Overview

```
Input Image (chart PNG)
        │
        ▼
┌───────────────┐     ┌───────────────────┐
│ YOLOv8n       │────▶│ CNN Classifier     │
│ (Detection)   │     │ (Classification)   │
│ 3M params     │     │ 94K params, 9 cls  │
└───────────────┘     └────────┬──────────┘
                               │
                    detected symbols with classes
                               │
                               ▼
                 ┌─────────────────────────┐
                 │ Coordinate Extractor    │
                 │ - Plotting area detect  │
                 │ - Axis tick mapping     │
                 │ - Pixel → real coords   │
                 │ - RANSAC outlier filter │
                 └────────────┬────────────┘
                              │
                              ▼
                   CSV / reconstructed chart
```

**Two-stage detection:**
1. **YOLOv8n** (nano) detects symbol bounding boxes
2. **Lightweight CNN** classifies each 32x32 crop into 9 classes:
   `filled_circle`, `open_circle`, `filled_square`, `open_square`,
   `filled_triangle`, `open_triangle`, `filled_diamond`, `open_diamond`, `background`

**Coordinate extraction:** pixel positions are mapped to real-world values via axis tick interpolation (supports both linear and log scales).

## Project Structure

```
.
├── src/                          # Source code
│   ├── run_detection_pipeline.py # Main detection pipeline (entry point)
│   ├── coordinate_extractor.py   # Pixel→coordinate mapping
│   ├── symbol_classifier.py      # CNN model definition & training
│   ├── synthetic_chart_generator.py    # v1 synthetic data generator
│   ├── synthetic_chart_generator_v2.py # v2 generator (patent-style artifacts)
│   ├── prepare_yolo_dataset.py   # COCO→YOLO format converter
│   ├── train_yolo_detector.py    # YOLO v1 training script
│   ├── train_yolo_detector_v2.py # YOLO v2 training script
│   ├── train_classifier_v2.py    # CNN v2 training script
│   ├── test_real_chart.py        # Real chart test (v1)
│   ├── test_real_chart_v2.py     # Real chart test (v2, improved)
│   ├── evaluate_pipeline.py      # Evaluation metrics
│   ├── evaluate_v2.py            # v2 model evaluation
│   └── test_coordinate_extraction.py # Coordinate extraction tests
├── data/
│   ├── raw/                      # Original chart images
│   ├── synthetic/                # v1 synthetic training data
│   ├── synthetic_v2/             # v2 synthetic data (patent-style)
│   └── annotations/             # Manual annotations
├── models/
│   ├── yolo_detector_v1/        # YOLO v1 weights
│   ├── yolo_detector_v2/        # YOLO v2 weights
│   ├── classifier_v1/           # CNN v1 weights
│   └── classifier_v2/           # CNN v2 weights
├── experiments/                  # Experiment outputs & logs
├── requirements.txt
└── README.md
```

## 1. Environment Setup

### Prerequisites

- Python 3.9+
- macOS / Linux (tested on Mac mini 16GB, CPU-only training)

### Install Dependencies

```bash
# Clone or download the project
cd /path/to/Hongju

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
python -c "from sahi import AutoDetectionModel; print('SAHI OK')"
```

## 2. Full Pipeline: From Training to Inference

If you want to reproduce the entire pipeline from scratch (generate data → train models → evaluate), follow these steps in order.

### Step 1: Generate Synthetic Training Data

```bash
cd src/

# Generate v1 synthetic data (basic charts)
python synthetic_chart_generator.py

# Generate v2 synthetic data (with patent-style scan artifacts)
python synthetic_chart_generator_v2.py
```

This creates chart images and COCO-format annotations under `data/synthetic/` and `data/synthetic_v2/`.

### Step 2: Convert Annotations to YOLO Format

```bash
python prepare_yolo_dataset.py
```

Converts COCO annotations to YOLO format and creates `dataset.yaml` under `data/synthetic/yolo/`.

### Step 3: Train YOLO Detector

```bash
# v1: Train from pretrained YOLOv8n
python train_yolo_detector.py

# v2: Fine-tune from v1 best weights on v2 data
python train_yolo_detector_v2.py
```

Training runs on CPU by default (safe for 16GB machines). Outputs are saved under `models/yolo_detector_v1/` or `models/yolo_detector_v2/`.

### Step 4: Train CNN Classifier

```bash
# v2 classifier (extracts patches and trains automatically)
python train_classifier_v2.py
```

Trains a lightweight CNN on 32x32 patches extracted from synthetic data. Model saved to `models/classifier_v2/best_model.pth`.

### Step 5: Evaluate

```bash
# Evaluate v2 models on synthetic validation + real chart
python evaluate_v2.py
```

## 3. Run Inference on New Images

### Quick Detection (Symbol Detection Only)

Detect symbols in a chart image and output a visualization + JSON:

```bash
cd src/

python run_detection_pipeline.py <IMAGE_PATH> [--output OUTPUT_DIR] [--conf THRESHOLD]
```

**Example:**

```bash
python run_detection_pipeline.py ../data/raw/original_plot_fig3.png \
    --output ../experiments/my_results \
    --conf 0.3
```

**Output:**
- `<name>_detections.png` — annotated image with detected symbols
- `<name>_detections.json` — detection results (bbox, class, confidence)

### Full Extraction (Detection + Coordinate Extraction)

For end-to-end extraction from a real chart with known axis information, use `test_real_chart_v2.py` as a template:

```bash
python test_real_chart_v2.py
```

This script demonstrates the complete workflow:
1. Load YOLO detector + CNN classifier
2. Define axis tick positions and plotting area bounds
3. Run standard + SAHI detection, merge and deduplicate
4. Filter by plotting area, axis proximity, and confidence
5. Extract real-world coordinates via axis interpolation
6. Apply RANSAC outlier filtering
7. Output CSV data + reconstructed chart

**Output (under `experiments/real_chart_test_v2_improved/`):**
- `fig3_filtered_detections.png` — visualization of filtered detections
- `fig3_reconstructed_v2_improved.png` — reconstructed chart from extracted data
- `fig3_extracted_data_v2_improved.csv` — extracted coordinates (series, x, y, pixel positions, confidence)

### Adapting to a New Chart

To extract data from a new chart image, you need to:

1. **Determine the plotting area** (pixel bounds of the data region, excluding legend/axis labels)
2. **Identify axis tick positions** (pixel positions + real values for each tick)
3. **Specify axis scale type** (linear or logarithmic)

Then modify `test_real_chart_v2.py` or write a similar script:

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from run_detection_pipeline import load_models, detect_symbols, detect_symbols_sahi, visualize_detections
from coordinate_extractor import parse_tick_labels_manual, extract_coordinates, group_by_series, ransac_filter, PlottingArea

# 1. Load models
yolo_path = "../models/yolo_detector_v2/train/weights/best.pt"
classifier_path = "../models/classifier_v2/best_model.pth"
detector, classifier = load_models(yolo_path, classifier_path, device="cpu")

# 2. Define plot area (pixel coordinates of the data region)
plot_area = PlottingArea(
    x_min_px=100,   # left edge of plot area
    x_max_px=800,   # right edge
    y_min_px=50,    # top edge (image coordinates)
    y_max_px=500,   # bottom edge
)

# 3. Define axis ticks: list of (pixel_position, "value_string")
x_ticks = [
    (100, "1e-10"), (200, "1e-9"), (300, "1e-8"),  # log scale example
]
y_ticks = [
    (500, "0.0"), (400, "0.1"), (300, "0.2"),  # linear scale example
]

# 4. Parse axis info
x_axis, y_axis = parse_tick_labels_manual(x_ticks, y_ticks)

# 5. Detect symbols
image_path = "your_chart.png"
detections = detect_symbols(image_path, detector, classifier, conf_threshold=0.3, device="cpu")

# 6. Extract coordinates
coords = extract_coordinates(detections, x_axis, y_axis, plot_area)
series = group_by_series(coords)

# 7. RANSAC filter
for cls, points in series.items():
    inliers, outliers = ransac_filter(points)
    print(f"{cls}: {len(inliers)} valid points")
    for p in inliers:
        print(f"  X={p['real_x']:.3e}, Y={p['real_y']:.4f}")
```

### Using the Python API

You can also import functions directly:

```python
from run_detection_pipeline import load_models, detect_symbols, visualize_detections

detector, classifier = load_models(
    yolo_path="models/yolo_detector_v2/train/weights/best.pt",
    classifier_path="models/classifier_v2/best_model.pth",
)

detections = detect_symbols("chart.png", detector, classifier, conf_threshold=0.3)

for det in detections:
    print(det["final_class"], det["center_px"], det["det_conf"])
```

## Notes

- **CPU-only recommended**: MPS (Apple Silicon GPU) has known stability issues with PyTorch. All scripts default to `device="cpu"`.
- **Memory**: The pipeline is designed for 16GB RAM machines. Avoid running training and inference simultaneously.
- **Axis tick positions**: Currently must be specified manually per chart. Automatic axis OCR is a planned improvement.
- **Supported symbol types**: 8 types (filled/open variants of circle, square, triangle, diamond).

## Related Work and References

For a curated list of open-source projects, tools, and research in the chart data extraction space, see [references.md](references.md).
