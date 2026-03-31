"""
Draw COCO segmentation labels on random tiles from sam3_tiled_dataset.
"""

import json
import random
from pathlib import Path
import cv2
import numpy as np

NUM_SAMPLES = 50
SEED = 42

BASE = Path(__file__).resolve().parent / "sam3_tiled_seg_dataset"
OUT = Path(__file__).resolve().parent / "sam3_tiled_seg_dataset" / "labeled_samples"
OUT.mkdir(exist_ok=True)

pairs = []
for split in ("train", "valid"):
    split_dir = BASE / split
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        continue

    with open(ann_file) as f:
        coco = json.load(f)

    # Build image_id -> annotations mapping
    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco["images"]:
        img_id = img_info["id"]
        anns = img_anns.get(img_id, [])
        if not anns:
            continue
        img_path = split_dir / img_info["file_name"]
        if img_path.exists():
            pairs.append((img_path, anns, split))

print(f"Found {len(pairs)} images with annotations")

rng = random.Random(SEED)
if len(pairs) > NUM_SAMPLES:
    pairs = rng.sample(pairs, NUM_SAMPLES)

print(f"Drawing labels on {len(pairs)} random tiles\n")

for img_path, anns, split in pairs:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    result = img.copy()
    poly_count = 0
    bbox_only = 0

    for ann in anns:
        segs = ann.get("segmentation", [])
        has_poly = False
        for seg in segs:
            if len(seg) >= 6:
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                # Draw as open polyline — no closing edge
                cv2.polylines(result, [pts], False, (0, 255, 0), 2)
                has_poly = True
                poly_count += 1

        if not has_poly:
            bx, by, bw, bh = [int(v) for v in ann["bbox"]]
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            bbox_only += 1

    out_path = OUT / f"{img_path.stem}_labeled.png"
    cv2.imwrite(str(out_path), result)
    print(f"[{split}] {img_path.name}: {poly_count} polygons, {bbox_only} bbox-only")

print(f"\nDone! {len(pairs)} labeled images saved to: {OUT}")
