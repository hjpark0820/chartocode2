"""
Convert COCO annotations to YOLO format for training.
Also creates the dataset YAML config for ultralytics.
"""

import json
import os
import shutil
from pathlib import Path


def coco_to_yolo(coco_path: str, images_dir: str, output_dir: str):
    """Convert COCO annotations to YOLO format."""
    with open(coco_path) as f:
        coco = json.load(f)

    labels_dir = os.path.join(output_dir, "labels")
    yolo_images_dir = os.path.join(output_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(yolo_images_dir, exist_ok=True)

    # Build image id -> info map
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    n_images = 0
    n_labels = 0

    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        fname = img_info["file_name"]

        # Symlink image
        src = os.path.join(images_dir, fname)
        dst = os.path.join(yolo_images_dir, fname)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)

        # Write YOLO label file
        label_fname = Path(fname).stem + ".txt"
        label_path = os.path.join(labels_dir, label_fname)

        anns = img_anns.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # YOLO format: class cx cy w h (normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clip to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))

            cat_id = ann["category_id"]
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_labels += 1

        with open(label_path, 'w') as f:
            f.write("\n".join(lines))

        n_images += 1

    print(f"Converted {n_images} images, {n_labels} labels to YOLO format")
    print(f"  Images: {yolo_images_dir}")
    print(f"  Labels: {labels_dir}")


def create_dataset_yaml(output_path: str, train_dir: str, val_dir: str):
    """Create YOLO dataset YAML config."""
    class_names = [
        "filled_circle", "open_circle",
        "filled_square", "open_square",
        "filled_triangle", "open_triangle",
        "filled_diamond", "open_diamond",
    ]

    yaml_content = f"""# Chart Symbol Detection Dataset
path: {os.path.abspath(os.path.dirname(output_path))}
train: {os.path.abspath(os.path.join(train_dir, 'images'))}
val: {os.path.abspath(os.path.join(val_dir, 'images'))}

nc: {len(class_names)}
names: {class_names}
"""

    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"Dataset YAML saved to: {output_path}")

if __name__ == "__main__":
    base = "data/synthetic_v2" 

    # 1. Convert train
    print("Converting train set...")
    coco_to_yolo(
        f"{base}/train/annotations.json", 
        f"{base}/train/images",           
        f"{base}/yolo/train",
    )

    # 2. Convert val
    print("\nConverting val set...")
    coco_to_yolo(
        f"{base}/val/annotations.json", 
        f"{base}/val/images",           
        f"{base}/yolo/val",
    )

    # 3. CREATE THE YAML FILE (This was missing!)
    print("\nCreating dataset.yaml...")
    create_dataset_yaml(
        output_path=f"{base}/yolo/dataset.yaml",
        train_dir=f"{base}/yolo/train",
        val_dir=f"{base}/yolo/val"
    )