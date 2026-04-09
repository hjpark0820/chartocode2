"""
Train YOLOv8 symbol detector with memory-safe settings.
Designed for 16GB RAM Mac mini (MPS backend).
"""

from ultralytics import YOLO
import os
import gc
import torch
from pathlib import Path

def train():
    # ... garbage collection code ...

    # Use Path(__file__) to find exactly where your script is sitting
    # This automatically handles 'OneDrive', '문서', and 'Documents'
    current_script_path = Path(__file__).parent.absolute()
    
    # Point to the dataset.yaml we just generated
    data_yaml = current_script_path / "data" / "synthetic_v2" / "yolo" / "dataset.yaml"
    
    # Set the output directory
    output_dir = current_script_path.parent / "models" / "yolo_detector_v1"

    # Convert to string for the YOLO model (it expects strings, not Path objects)
    data_yaml = str(data_yaml)
    output_dir = str(output_dir)
    # Use YOLOv8n (nano) - smallest model, ~3.2M params
    # Memory-safe settings for 16GB Mac:
    #   - batch=8 (very conservative)
    #   - imgsz=640
    #   - workers=0 (no multiprocessing to save memory)
    #   - device=mps
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,           # Small batch to save memory
        workers=0,          # No extra workers
        device="mps",
        project=output_dir,
        name="train",
        patience=15,        # Early stopping
        save_period=10,     # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        # Memory optimization
        cache=False,        # Don't cache images in RAM
        amp=True,           # Mixed precision
        cos_lr=True,        # Cosine LR scheduler
        lr0=0.01,
        lrf=0.01,
        # Augmentation (conservative)
        hsv_h=0.0,         # No hue shift (grayscale-like charts)
        hsv_s=0.0,         # No saturation shift
        hsv_v=0.2,         # Slight brightness variation
        degrees=5.0,        # Small rotation
        translate=0.1,
        scale=0.3,
        flipud=0.0,         # No vertical flip for charts
        fliplr=0.0,         # No horizontal flip for charts
        mosaic=0.5,         # Reduced mosaic
        mixup=0.0,          # No mixup
    )

    print(f"\nTraining complete!")
    print(f"Best model: {output_dir}/train/weights/best.pt")

    return results


if __name__ == "__main__":
    train()
