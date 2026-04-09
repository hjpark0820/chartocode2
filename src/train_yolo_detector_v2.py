"""
Train YOLOv8 symbol detector v2 on improved synthetic data.
Memory-safe settings for Windows/Mac.

Key changes from v1:
- Uses v2 synthetic data (more realistic, patent-style)
- Trains on CPU to avoid MPS bugs (slower but stable)
- Frequent checkpoint saving
"""

from ultralytics import YOLO
import os
import gc
import torch
from pathlib import Path

def train():
    gc.collect()
    # Check for MPS (Mac) or CUDA (NVIDIA) and clear cache if needed
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Setup dynamic paths based on script location
    # Assuming this script is in .../project/src/
    SRC_DIR = Path(__file__).parent.absolute()
    PROJECT_ROOT = SRC_DIR.parent

    # Define paths dynamically
    data_yaml = SRC_DIR / "data" / "synthetic_v2" / "yolo" / "dataset.yaml"
    output_dir = PROJECT_ROOT / "models" / "yolo_detector_v2"
    
    # Path to v1 weights for transfer learning
    v1_weights = PROJECT_ROOT / "models" / "yolo_detector_v1" / "train" / "weights" / "best.pt"

    # Convert paths to strings for Ultralytics/OS functions
    data_yaml_str = str(data_yaml)
    output_dir_str = str(output_dir)
    v1_weights_str = str(v1_weights)

    # Start from v1 best weights (transfer learning) if available
    if v1_weights.exists():
        print(f"Fine-tuning from v1 weights: {v1_weights_str}")
        model = YOLO(v1_weights_str)
    else:
        print("v1 weights not found. Starting from YOLOv8n pretrained weights.")
        model = YOLO("yolov8n.pt")

    # Train on CPU for 100% stability across platforms
    results = model.train(
        data=data_yaml_str,
        epochs=30,           # Fewer epochs since fine-tuning
        imgsz=640,
        batch=8,
        workers=0,
        device="cpu",        # CPU for stability
        project=output_dir_str,
        name="train",
        patience=10,
        save_period=5,       # Save every 5 epochs
        plots=True,
        verbose=True,
        cache=False,
        amp=False,           # No AMP on CPU
        cos_lr=True,
        lr0=0.005,           # Lower LR for fine-tuning
        lrf=0.01,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved in: {output_dir_str}/train/weights/best.pt")
    return results

if __name__ == "__main__":
    train()