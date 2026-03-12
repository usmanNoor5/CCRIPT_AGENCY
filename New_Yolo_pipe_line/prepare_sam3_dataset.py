#!/usr/bin/env python3
"""
prepare_sam3_dataset.py — Prepare binary images + COCO split for SAM3 training.

1. Read COCO annotations from train/
2. Binarize all images (adaptive threshold + invert)
3. Split into 80/20 train/valid (image-level)
4. Write separate COCO JSONs for train and valid
5. Copy binarized images to sam3_dataset/train/ and sam3_dataset/valid/

Usage:
    python3.12 prepare_sam3_dataset.py
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def binarize_image(img: np.ndarray, block_size: int = 51, C: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    inverted = cv2.bitwise_not(binary)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def main():
    random.seed(42)
    input_dir = Path("train")
    output_dir = Path("sam3_dataset")
    coco_json = input_dir / "_annotations.coco.json"

    if not coco_json.exists():
        raise SystemExit(f"COCO annotations not found: {coco_json}")

    with open(coco_json) as f:
        coco = json.load(f)

    # Build image ID to annotations mapping
    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    # Shuffle and split images 80/20
    images = coco["images"]
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    valid_images = images[split_idx:]

    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)}, Valid: {len(valid_images)}")

    for split_name, split_images in [("train", train_images), ("valid", valid_images)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Collect annotations for this split
        split_ann_id = 1
        split_annotations = []
        split_img_entries = []
        img_id_map = {}

        for new_id, img_info in enumerate(split_images, start=1):
            old_id = img_info["id"]
            img_id_map[old_id] = new_id
            fname = img_info["file_name"]
            src = input_dir / fname

            if not src.exists():
                print(f"  [skip] {fname} not found")
                continue

            # Binarize and save
            img = cv2.imread(str(src))
            if img is None:
                print(f"  [skip] Cannot read {fname}")
                continue

            bin_img = binarize_image(img)
            cv2.imwrite(str(split_dir / fname), bin_img)

            # Update image entry with new ID
            new_img = dict(img_info)
            new_img["id"] = new_id
            split_img_entries.append(new_img)

            # Copy annotations with updated IDs
            for ann in img_anns.get(old_id, []):
                new_ann = dict(ann)
                new_ann["id"] = split_ann_id
                new_ann["image_id"] = new_id
                split_annotations.append(new_ann)
                split_ann_id += 1

        # Write COCO JSON
        split_coco = {
            "images": split_img_entries,
            "annotations": split_annotations,
            "categories": coco["categories"],
        }
        # Copy any extra top-level keys (info, licenses)
        for key in coco:
            if key not in ("images", "annotations", "categories"):
                split_coco[key] = coco[key]

        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w") as f:
            json.dump(split_coco, f)

        total_anns = len(split_annotations)
        print(f"  {split_name}: {len(split_img_entries)} images, {total_anns} annotations → {split_dir}")

    print(f"\nDataset ready: {output_dir}")
    print(f"  {output_dir}/train/ — training images + _annotations.coco.json")
    print(f"  {output_dir}/valid/ — validation images + _annotations.coco.json")


if __name__ == "__main__":
    main()
