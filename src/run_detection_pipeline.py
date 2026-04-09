"""
End-to-end detection pipeline:
1. Load trained YOLOv8 detector
2. Run on input image (with optional SAHI slicing)
3. Run CNN classifier on detected crops
4. Visualize results
5. Extract coordinates
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Import our classifier
import sys
sys.path.insert(0, os.path.dirname(__file__))
from symbol_classifier import SymbolClassifierCNN, CLASS_NAMES


def load_models(
    yolo_path: str = "models/yolo_detector_v1/train/weights/best.pt",
    classifier_path: str = "models/classifier_v1/best_model.pth",
    device: str = "cpu",  # Use CPU for inference to avoid MPS bugs
):
    """Load both detection and classification models."""
    # YOLO detector
    detector = YOLO(yolo_path)

    # CNN classifier
    classifier = SymbolClassifierCNN(num_classes=len(CLASS_NAMES))
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=device, weights_only=True)
    )
    classifier.to(device)
    classifier.eval()

    return detector, classifier


def detect_symbols(
    image_path: str,
    detector: YOLO,
    classifier: SymbolClassifierCNN,
    conf_threshold: float = 0.3,
    device: str = "cpu",
) -> list:
    """
    Detect symbols in a chart image.

    Returns list of detections: [
        {
            "bbox": [x1, y1, x2, y2],
            "center_px": (cx, cy),
            "det_class": str,       # from YOLO
            "det_conf": float,      # YOLO confidence
            "cls_class": str,       # from CNN classifier
            "cls_conf": float,      # CNN confidence
            "final_class": str,     # combined result
        }
    ]
    """
    # Run YOLO detection on CPU to avoid MPS bugs
    results = detector.predict(
        image_path,
        conf=conf_threshold,
        iou=0.4,            # Lower IoU for overlapping symbols
        device="cpu",
        verbose=False,
        max_det=500,
    )

    detections = []
    img = Image.open(image_path).convert('RGB')
    img_gray = img.convert('L')
    img_arr = np.array(img_gray)

    if len(results) == 0 or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
        conf = float(boxes.conf[i].cpu())
        cls_id = int(boxes.cls[i].cpu())

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # YOLO class name
        det_class_names = [
            "filled_circle", "open_circle",
            "filled_square", "open_square",
            "filled_triangle", "open_triangle",
            "filled_diamond", "open_diamond",
        ]
        det_class = det_class_names[cls_id] if cls_id < len(det_class_names) else "unknown"

        # Extract 32x32 patch for CNN classification
        half = 16
        px_cx, px_cy = int(round(cx)), int(round(cy))
        h, w = img_arr.shape

        y_start = max(0, px_cy - half)
        y_end = min(h, px_cy + half)
        x_start = max(0, px_cx - half)
        x_end = min(w, px_cx + half)

        patch = img_arr[y_start:y_end, x_start:x_end]

        # Pad if needed
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            padded = np.full((32, 32), 255, dtype=np.uint8)
            py = (32 - patch.shape[0]) // 2
            px = (32 - patch.shape[1]) // 2
            padded[py:py+patch.shape[0], px:px+patch.shape[1]] = patch
            patch = padded

        # CNN classification
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0) / 255.0
        patch_tensor = patch_tensor.to(device)

        with torch.no_grad():
            logits = classifier(patch_tensor)
            probs = F.softmax(logits, dim=1)
            cls_conf, cls_pred = probs.max(1)
            cls_class = CLASS_NAMES[cls_pred.item()]
            cls_conf = cls_conf.item()

        # Combine: prefer YOLO class if CNN agrees or CNN says background
        if cls_class == "background" or cls_class == det_class:
            final_class = det_class
        else:
            # If CNN is very confident and disagrees, trust CNN
            if cls_conf > 0.8:
                final_class = cls_class
            else:
                final_class = det_class

        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "center_px": (float(cx), float(cy)),
            "det_class": det_class,
            "det_conf": conf,
            "cls_class": cls_class,
            "cls_conf": cls_conf,
            "final_class": final_class,
        })

    return detections


def detect_symbols_sahi(
    image_path: str,
    detector_path: str,
    classifier: "SymbolClassifierCNN",
    conf_threshold: float = 0.25,
    slice_size: int = 320,
    overlap_ratio: float = 0.2,
    device: str = "cpu",
) -> list:
    """
    Detect symbols using SAHI (Sliced Aided Hyper Inference).
    Slices the image into overlapping patches for better small-object detection.
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=detector_path,
        confidence_threshold=conf_threshold,
        device=device,
    )

    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0,
    )

    img = Image.open(image_path).convert('RGB')
    img_gray = img.convert('L')
    img_arr = np.array(img_gray)
    h, w = img_arr.shape

    det_class_names = [
        "filled_circle", "open_circle",
        "filled_square", "open_square",
        "filled_triangle", "open_triangle",
        "filled_diamond", "open_diamond",
    ]

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        conf = pred.score.value
        cls_id = pred.category.id

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        det_class = det_class_names[cls_id] if cls_id < len(det_class_names) else "unknown"

        # CNN classification on 32x32 patch
        half = 16
        px_cx, px_cy = int(round(cx)), int(round(cy))
        y_start = max(0, px_cy - half)
        y_end = min(h, px_cy + half)
        x_start = max(0, px_cx - half)
        x_end = min(w, px_cx + half)

        patch = img_arr[y_start:y_end, x_start:x_end]
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            padded = np.full((32, 32), 255, dtype=np.uint8)
            py = (32 - patch.shape[0]) // 2
            px = (32 - patch.shape[1]) // 2
            padded[py:py+patch.shape[0], px:px+patch.shape[1]] = patch
            patch = padded

        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0) / 255.0
        patch_tensor = patch_tensor.to(device)

        with torch.no_grad():
            logits = classifier(patch_tensor)
            probs = F.softmax(logits, dim=1)
            cls_conf, cls_pred = probs.max(1)
            cls_class = CLASS_NAMES[cls_pred.item()]
            cls_conf_val = cls_conf.item()

        if cls_class == "background" or cls_class == det_class:
            final_class = det_class
        else:
            if cls_conf_val > 0.8:
                final_class = cls_class
            else:
                final_class = det_class

        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "center_px": (float(cx), float(cy)),
            "det_class": det_class,
            "det_conf": float(conf),
            "cls_class": cls_class,
            "cls_conf": float(cls_conf_val),
            "final_class": final_class,
        })

    return detections


def visualize_detections(
    image_path: str,
    detections: list,
    output_path: str,
    show_labels: bool = True,
):
    """Visualize detected symbols on the image."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Color map for classes
    colors = {
        "filled_circle": "red",
        "open_circle": "orangered",
        "filled_square": "blue",
        "open_square": "dodgerblue",
        "filled_triangle": "green",
        "open_triangle": "limegreen",
        "filled_diamond": "purple",
        "open_diamond": "orchid",
        "background": "gray",
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["final_class"]
        conf = det["det_conf"]
        color = colors.get(cls, "white")

        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # Draw label
        if show_labels:
            label = f"{cls[:3]}:{conf:.2f}"
            draw.text((x1, y1 - 10), label, fill=color)

        # Draw center point
        cx, cy = det["center_px"]
        draw.ellipse([cx-2, cy-2, cx+2, cy+2], fill=color)

    img.save(output_path)
    print(f"Visualization saved to: {output_path}")
    return img


def detect_symbols_best(
    image_path: str,
    detector: YOLO,
    detector_path: str,
    classifier: "SymbolClassifierCNN",
    conf_threshold: float = 0.25,
    device: str = "cpu",
) -> list:
    """
    Best detection strategy: combine standard + SAHI, deduplicate.
    Uses standard detection as primary (high precision) and SAHI to fill gaps (high recall).
    """
    # Standard detection (high precision)
    std_dets = detect_symbols(
        image_path, detector, classifier,
        conf_threshold=conf_threshold + 0.05, device=device,
    )

    # SAHI detection (high recall)
    sahi_dets = detect_symbols_sahi(
        image_path, detector_path, classifier,
        conf_threshold=conf_threshold, slice_size=320,
        overlap_ratio=0.2, device=device,
    )

    # Merge: start with standard, add SAHI detections that don't overlap
    merged = list(std_dets)
    for sahi_det in sahi_dets:
        cx, cy = sahi_det["center_px"]
        is_duplicate = False
        for existing in merged:
            ex, ey = existing["center_px"]
            if ((cx - ex)**2 + (cy - ey)**2) ** 0.5 < 8:
                is_duplicate = True
                break
        if not is_duplicate:
            merged.append(sahi_det)

    return merged


def run_pipeline(
    image_path: str,
    output_dir: str = "experiments/detection_results",
    conf_threshold: float = 0.3,
):
    """Run the full detection pipeline on an image."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading models...")
    detector, classifier = load_models(device="cpu")

    print(f"Detecting symbols in: {image_path}")
    detections = detect_symbols(
        image_path, detector, classifier,
        conf_threshold=conf_threshold,
        device="cpu",
    )

    print(f"\nDetected {len(detections)} symbols:")
    class_counts = {}
    for det in detections:
        cls = det["final_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    # Visualize
    fname = Path(image_path).stem
    vis_path = os.path.join(output_dir, f"{fname}_detections.png")
    visualize_detections(image_path, detections, vis_path)

    # Save detections JSON
    json_path = os.path.join(output_dir, f"{fname}_detections.json")
    with open(json_path, 'w') as f:
        json.dump(detections, f, indent=2)
    print(f"Detections saved to: {json_path}")

    return detections


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to chart image")
    parser.add_argument("--output", default="experiments/detection_results")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    run_pipeline(args.image, args.output, args.conf)
