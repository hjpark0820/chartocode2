"""
Extract patches from v2 synthetic data and train CNN classifier v2.
Reuses the patch extraction from v1 generator and the classifier architecture.

Memory-safe: runs on CPU to avoid MPS issues.
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

# PROJECT_DIR = "/Users/jason/MyProject/RA/Hongju"
CURRENT_DIR = Path(__file__).parent.absolute()

# Symbol definitions matching v2 generator
SYMBOL_NAMES = [
    "filled_circle", "open_circle",
    "filled_square", "open_square",
    "filled_triangle", "open_triangle",
    "filled_diamond", "open_diamond",
]
NUM_SYMBOL_CLASSES = 8


def extract_patches_v2(
    data_dir: str,
    output_dir: str,
    patch_size: int = 32,
    n_negative_per_image: int = 10,
    n_offset_per_symbol: int = 2,
    seed: int = 123,
):
    """Extract classification patches from v2 synthetic data."""
    random.seed(seed)
    np.random.seed(seed)

    ann_path = os.path.join(data_dir, "annotations.json")
    images_dir = os.path.join(data_dir, "images")

    with open(ann_path) as f:
        coco = json.load(f)

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)

    for name in SYMBOL_NAMES:
        os.makedirs(os.path.join(patches_dir, name), exist_ok=True)
    os.makedirs(os.path.join(patches_dir, "background"), exist_ok=True)

    patch_metadata = []
    patch_id = 0
    half = patch_size // 2

    # Group annotations by image
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    for img_info in tqdm(coco["images"], desc="Extracting v2 patches"):
        img_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        img = Image.open(img_path).convert('L')
        img_arr = np.array(img)
        h, w = img_arr.shape

        anns = img_anns.get(img_id, [])
        symbol_centers = []

        for ann in anns:
            cx, cy = ann["center_px"]
            cx, cy = int(round(cx)), int(round(cy))
            symbol_centers.append((cx, cy))

            # 1. Positive patch (centered on symbol)
            if half <= cx < w - half and half <= cy < h - half:
                patch = img_arr[cy - half:cy + half, cx - half:cx + half]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    cat_name = SYMBOL_NAMES[ann["category_id"]]
                    fname = f"patch_{patch_id:06d}.png"
                    Image.fromarray(patch).save(
                        os.path.join(patches_dir, cat_name, fname)
                    )
                    patch_metadata.append({
                        "id": patch_id,
                        "file": f"{cat_name}/{fname}",
                        "class": cat_name,
                        "class_id": ann["category_id"],
                        "source_image": img_info["file_name"],
                    })
                    patch_id += 1

            # 2. Offset-negative patches (near-miss background)
            for _ in range(n_offset_per_symbol):
                off_x = cx + random.choice([-1, 1]) * random.randint(3, 5)
                off_y = cy + random.choice([-1, 1]) * random.randint(3, 5)
                if half <= off_x < w - half and half <= off_y < h - half:
                    patch = img_arr[off_y - half:off_y + half, off_x - half:off_x + half]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        fname = f"patch_{patch_id:06d}.png"
                        Image.fromarray(patch).save(
                            os.path.join(patches_dir, "background", fname)
                        )
                        patch_metadata.append({
                            "id": patch_id,
                            "file": f"background/{fname}",
                            "class": "background",
                            "class_id": NUM_SYMBOL_CLASSES,
                            "source_image": img_info["file_name"],
                        })
                        patch_id += 1

        # 3. Random background patches
        for _ in range(n_negative_per_image):
            rx = random.randint(half, w - half - 1)
            ry = random.randint(half, h - half - 1)
            too_close = any(
                abs(rx - sx) < patch_size and abs(ry - sy) < patch_size
                for sx, sy in symbol_centers
            )
            if not too_close:
                patch = img_arr[ry - half:ry + half, rx - half:rx + half]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    fname = f"patch_{patch_id:06d}.png"
                    Image.fromarray(patch).save(
                        os.path.join(patches_dir, "background", fname)
                    )
                    patch_metadata.append({
                        "id": patch_id,
                        "file": f"background/{fname}",
                        "class": "background",
                        "class_id": NUM_SYMBOL_CLASSES,
                        "source_image": img_info["file_name"],
                    })
                    patch_id += 1

    meta_path = os.path.join(output_dir, "patch_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(patch_metadata, f, indent=2)

    class_counts = {}
    for p in patch_metadata:
        c = p["class"]
        class_counts[c] = class_counts.get(c, 0) + 1

    print(f"\nv2 patches extracted: {patch_id}")
    for cls, cnt in sorted(class_counts.items()):
        print(f"  {cls}: {cnt}")

    return patch_metadata


def train_classifier_v2():
    """Train CNN classifier v2 on v2 synthetic patches."""
    from symbol_classifier import train_classifier

    # Define paths relative to the script location
    # This automatically handles OneDrive/문서/Documents naming issues
    train_dir = CURRENT_DIR / "data" / "synthetic_v2" / "train"
    val_dir   = CURRENT_DIR / "data" / "synthetic_v2" / "val"
    output_dir = CURRENT_DIR.parent / "models" / "classifier_v2"

    print("=" * 60)
    print("STEP 1: Extract patches from v2 data")
    print("=" * 60)

    # Convert to string because the extraction function uses os.path.join
    train_dir_str = str(train_dir)
    val_dir_str   = str(val_dir)

    # Extract train patches
    print("\n--- Train set ---")
    extract_patches_v2(train_dir_str, train_dir_str, seed=123)

    # Extract val patches
    print("\n--- Val set ---")
    extract_patches_v2(val_dir_str, val_dir_str, seed=456)

    print("\n" + "=" * 60)
    print("STEP 2: Train classifier v2")
    print("=" * 60)

    # Train on CPU for stability as per your comments
    train_classifier(
        train_dir=train_dir_str,
        val_dir=val_dir_str,
        output_dir=str(output_dir),
        epochs=30,
        batch_size=128,
        lr=1e-3,
        device="cpu",
    )

    print("\nClassifier v2 training complete!")
    print(f"Model saved to: {output_dir}/best_model.pth")


if __name__ == "__main__":
    train_classifier_v2()
