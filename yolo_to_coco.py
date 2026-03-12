"""
yolo_to_coco.py
---------------
Convert augmented YOLO-seg dataset to COCO segmentation format
required by SAM3 fine-tuning.

Input:  augmented_dataset/  (YOLO format: images/train + labels/train)
Output: augmented_coco/     (COCO format: train/_annotations.coco.json + images)

Usage:
    python3.12 yolo_to_coco.py
    python3.12 yolo_to_coco.py --src augmented_dataset --out augmented_coco
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    """Parse a YOLO-seg label file and return list of COCO-style annotations."""
    annotations = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class + at least 3 xy pairs
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Convert normalised coords to absolute pixel coords
        abs_coords = []
        for i in range(0, len(coords) - 1, 2):
            px = coords[i] * img_w
            py = coords[i + 1] * img_h
            abs_coords.extend([round(px, 2), round(py, 2)])

        if len(abs_coords) < 6:
            continue

        # Compute bounding box from polygon points
        xs = abs_coords[0::2]
        ys = abs_coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w < 5 or bbox_h < 5:
            continue

        # Compute area from polygon
        pts = np.array(abs_coords, dtype=np.float32).reshape(-1, 2)
        area = float(cv2.contourArea(pts))
        if area < 10:
            continue

        annotations.append({
            "category_id": class_id,
            "segmentation": [abs_coords],
            "bbox": [round(x_min, 2), round(y_min, 2),
                     round(bbox_w, 2), round(bbox_h, 2)],
            "area": round(area, 2),
            "iscrowd": 0,
        })

    return annotations


def convert(src: Path, out: Path, train_ratio: float = 0.85):
    """Convert YOLO dataset to COCO format with train/valid split."""

    img_src = src / "images" / "train"
    lbl_src = src / "labels" / "train"

    image_files = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    print(f"Found {len(image_files)} images in {img_src}")

    # Shuffle and split into train / valid
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(image_files))
    split_idx = int(len(image_files) * train_ratio)
    train_indices = set(indices[:split_idx].tolist())

    splits = {"train": [], "valid": []}
    for i, img_path in enumerate(image_files):
        split = "train" if i in train_indices else "valid"
        splits[split].append(img_path)

    print(f"Train: {len(splits['train'])}  |  Valid: {len(splits['valid'])}")

    for split_name, split_files in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "wall", "supercategory": "structure"}
            ],
        }

        ann_id = 0
        skipped = 0

        for img_id, img_path in enumerate(split_files):
            lbl_path = lbl_src / (img_path.stem + ".txt")
            if not lbl_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]

            # Copy image to split dir
            dst = split_dir / img_path.name
            shutil.copy2(img_path, dst)

            coco["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })

            anns = parse_yolo_label(lbl_path, w, h)
            for ann in anns:
                ann["id"] = ann_id
                ann["image_id"] = img_id
                coco["annotations"].append(ann)
                ann_id += 1

        # Write COCO JSON
        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w") as f:
            json.dump(coco, f)

        print(f"  [{split_name}] {len(coco['images'])} images, "
              f"{len(coco['annotations'])} annotations → {json_path.name}")
        if skipped:
            print(f"    (skipped {skipped} without labels)")

    print(f"\nDone — COCO dataset saved to {out}/")
    print(f"Structure:")
    print(f"  {out}/train/_annotations.coco.json + images")
    print(f"  {out}/valid/_annotations.coco.json + images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="augmented_dataset",
                    help="Input YOLO dataset directory")
    ap.add_argument("--out", default="augmented_coco",
                    help="Output COCO dataset directory")
    ap.add_argument("--split", type=float, default=0.85,
                    help="Train ratio (default 0.85)")
    args = ap.parse_args()

    convert(Path(args.src), Path(args.out), args.split)


if __name__ == "__main__":
    main()
