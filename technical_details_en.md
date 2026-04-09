# Technical Details

## 1. Overall Approach: Two-Stage Detection + Coordinate Mapping

Rather than using a single end-to-end model, we split the pipeline into three independent modules connected in series:

```
Input Image
  │
  ├─ Stage 1: YOLOv8n ──→ Localize symbol positions (bounding boxes)
  │
  ├─ Stage 2: Lightweight CNN ──→ Fine-grained classification of each detection
  │
  └─ Stage 3: Coordinate Extraction ──→ Pixel positions → Real-value coordinates
```

The rationale: YOLO excels at localization but has limited classification accuracy (many of the 8 symbol classes look similar). A dedicated CNN for classification yields better results. Each model focuses on what it does best.

## 2. What Each Step Uses

### Step 1: Symbol Localization — YOLOv8n (Nano)

- Model: Ultralytics YOLOv8n, approximately **3M parameters**
- We use the smallest nano variant because our machine has only 16GB of RAM
- Input: Full chart image (resized to 640×640)
- Output: Bounding boxes + confidence scores for all detected symbols
- Training data: Synthetic scatter/line plots generated with matplotlib — 200 images for v1, 200 for v2 (with scan noise simulation)
- Training config: batch=8, epochs=30–50, CPU training
- Performance: mAP50 = 0.949

### Step 2: Symbol Classification — Custom Lightweight CNN

- Model: Custom 3-layer convolutional network, approximately **94K parameters**
- Architecture: `Conv(1→16, 3×3) → Conv(16→32, 3×3) → Conv(32→64, 3×3) → FC → 9 classes`
- Input: **32×32 grayscale patch** cropped from the center of each YOLO detection box
- Output: Classification into 9 classes — 8 symbol types (filled/open × circle/square/triangle/diamond) + background
- Classification decision logic:
  - If CNN and YOLO agree on the class → use that class directly
  - If CNN predicts background → trust YOLO (CNN likely couldn't see the symbol clearly)
  - If CNN confidence > 0.8 and disagrees with YOLO → trust CNN
  - Otherwise → trust YOLO

### Step 3 (Optional): SAHI Sliced Inference

- Purpose: Improve recall for small symbols that standard detection misses
- Method: Use the SAHI framework to slice the image into overlapping 320×320 patches, run YOLO on each slice independently, then stitch results back together
- Results are merged with standard detections using **10-pixel radius deduplication** (two detections with centers < 10px apart are treated as duplicates)

### Step 4: Post-Processing Filters

Four layers of filtering applied sequentially:
1. **Plot area filter**: Keep only detections within the data region; exclude false positives from legend, axis labels, and margins
2. **Axis proximity filter**: Remove detections too close to axis lines (typically tick marks misdetected as symbols)
3. **Confidence filter**: Remove low-confidence detections (threshold: 0.20)
4. **Small series filter**: If a class has only 1–3 detected points, it is likely a false positive and is discarded entirely

### Step 5: Coordinate Extraction

- Requires axis tick positions to be known in advance (pixel coordinates + corresponding real values)
- Uses `numpy.interp` for linear interpolation to map each symbol's pixel position to real-world values
- Logarithmic axis handling: Interpolation is performed in log space, then converted back to real values
- Automatic scale detection: If max_value / min_value > 100, the axis is treated as logarithmic

### Step 6: RANSAC Outlier Filtering

- Uses scikit-learn's RANSACRegressor
- Fits a degree-2 polynomial in log-x space (dose-response curves are typically sigmoidal)
- Adaptive threshold: 15% of the y-value range
- Points that deviate from the fitted curve are flagged as outliers and removed

## 3. API / Pipeline Integration

The entire pipeline runs as pure Python code with no external APIs or cloud services. The key function call chain:

```python
# 1. Load both models
detector, classifier = load_models(yolo_path, classifier_path)

# 2. Standard detection: YOLO localization → CNN classification
std_dets = detect_symbols(image_path, detector, classifier, conf_threshold=0.30)

# 3. SAHI detection: sliced inference → CNN classification
sahi_dets = detect_symbols_sahi(image_path, yolo_path, classifier, conf_threshold=0.25)

# 4. Merge and deduplicate
merged = merge_and_dedup(std_dets, sahi_dets, dedup_radius=10)

# 5. Filter (plot area, axis proximity, confidence)
filtered = filter_by_area(merged, plot_area)
filtered = filter_near_axes(filtered, plot_area)
filtered = filter_by_confidence(filtered, min_conf=0.20)

# 6. Coordinate mapping: pixel → real values
coords = extract_coordinates(filtered, x_axis, y_axis, plot_area)

# 7. Group by symbol type
series = group_by_series(coords)

# 8. RANSAC outlier filtering
for cls, points in series.items():
    inliers, outliers = ransac_filter_improved(points)
```

All functions are located in the `src/` directory. Model weights are stored in the `models/` directory. Everything runs locally with no external service dependencies.

## 4. Training Data Generation

Since no pre-existing labeled dataset was available, we programmatically generated synthetic training images:

- Matplotlib is used to render randomized scatter/line plots
- Symbol types, quantities, positions, and axis ranges are all randomized
- COCO-format annotations (bounding boxes + class labels) are generated automatically
- The v2 version adds **scan artifact simulation**: Gaussian noise, contrast reduction, and slight blur to better mimic the appearance of scanned patent documents

## 5. Current Limitations

| Aspect | Current Status | Root Cause |
|--------|---------------|------------|
| Real chart recall ~33% | Only 16 out of ~48 data points extracted | Domain gap between synthetic training data and real scanned documents |
| Missing one series | open_circle series not detected at all | Hollow symbols are hard to distinguish from background noise in scans |
| Axis config is manual | Tick pixel positions must be entered by hand for each chart | Automatic axis OCR and tick detection not yet implemented |
| Validated on one chart only | FIG.3 | Lack of annotated real images for systematic evaluation |

**Most critical next step**: Annotate 5–10 real patent charts and use them to fine-tune both YOLO and the CNN. Based on common experience, even a small amount of real-data fine-tuning can boost recall from the ~30% range to 70–80%.
