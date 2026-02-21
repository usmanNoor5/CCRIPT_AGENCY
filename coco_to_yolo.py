"""
coco_to_yolo.py
---------------
Convert the COCO segmentation dataset to YOLO-seg format,
keeping only images that have 'Walls' annotations.

Output:
    yolo_dataset/
    ├── images/train/   <- images with at least 1 wall annotation
    ├── labels/train/   <- YOLO polygon labels (walls only, class 0)
    └── dataset.yaml

Usage:
    python3.12 coco_to_yolo.py
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

COCO_DIR   = Path("coco_dataset/train")
COCO_JSON  = COCO_DIR / "_annotations.coco.json"
OUT_DIR    = Path("yolo_dataset")
WALL_CLASS_NAME = "Walls"

images_out = OUT_DIR / "images" / "train"
labels_out = OUT_DIR / "labels" / "train"
images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)


def main():
    with open(COCO_JSON) as f:
        coco = json.load(f)

    # Find the Walls category id
    wall_cat_id = None
    for cat in coco["categories"]:
        if cat["name"] == WALL_CLASS_NAME:
            wall_cat_id = cat["id"]
            break

    if wall_cat_id is None:
        raise SystemExit(f"Category '{WALL_CLASS_NAME}' not found. "
                         f"Available: {[c['name'] for c in coco['categories']]}")

    print(f"Wall category id: {wall_cat_id}")

    # Build image_id → image info map
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group wall annotations by image
    wall_anns = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != wall_cat_id:
            continue
        if not ann.get("segmentation"):
            continue
        iid = ann["image_id"]
        wall_anns.setdefault(iid, []).append(ann)

    print(f"Images with wall annotations: {len(wall_anns)} / {len(id_to_img)}")
    print(f"Images skipped (no walls):    {len(id_to_img) - len(wall_anns)}\n")

    kept = 0
    for img_id, anns in wall_anns.items():
        img_info = id_to_img[img_id]
        src = COCO_DIR / img_info["file_name"]
        if not src.exists():
            print(f"  [skip] file not found: {src.name}")
            continue

        w, h = img_info["width"], img_info["height"]
        label_lines = []

        for ann in anns:
            # COCO segmentation can be a list of polygons (multiple parts)
            for seg in ann["segmentation"]:
                if len(seg) < 6:   # need at least 3 points
                    continue
                pts = np.array(seg, dtype=float).reshape(-1, 2)
                # Normalise to [0,1]
                pts[:, 0] /= w
                pts[:, 1] /= h
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
                label_lines.append(f"0 {coords}")   # class 0 = wall

        if not label_lines:
            continue

        # Copy image + write label
        dst_img = images_out / src.name
        dst_lbl = labels_out / (src.stem + ".txt")
        shutil.copy2(src, dst_img)
        dst_lbl.write_text("\n".join(label_lines))

        print(f"  [ok] {src.name}  →  {len(label_lines)} wall polygons")
        kept += 1

    # Write dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/train\n"
        f"nc: 1\n"
        f"names: ['wall']\n"
    )

    print(f"\nDone — {kept} images kept, dataset saved to {OUT_DIR}/")
    print(f"\nTo fine-tune your model run:")
    print(f"  yolo segment train \\")
    print(f"    model=runs/segment/runs_api/yoloseg_wall5/weights/best.pt \\")
    print(f"    data={yaml_path} \\")
    print(f"    epochs=50 imgsz=640 batch=8")


if __name__ == "__main__":
    main()
