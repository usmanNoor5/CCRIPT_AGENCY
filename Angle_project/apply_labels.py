"""
Pick 100 random images from full_dataset, draw YOLO-seg labels on them, and save.
"""

import random
from pathlib import Path
import cv2
import numpy as np

NUM_SAMPLES = 100
SEED = 42

BASE = Path(__file__).resolve().parent / "full_dataset"
OUT = BASE / "labeled_samples"
OUT.mkdir(exist_ok=True)

# Collect all image/label pairs from train + val
pairs = []
for split in ("train", "val"):
    img_dir = BASE / "images" / split
    lbl_dir = BASE / "labels" / split
    if not img_dir.exists():
        continue
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists() and lbl_path.read_text().strip():
            pairs.append((img_path, lbl_path, split))

print(f"Found {len(pairs)} image/label pairs")

# Pick 100 random samples
rng = random.Random(SEED)
if len(pairs) > NUM_SAMPLES:
    pairs = rng.sample(pairs, NUM_SAMPLES)

print(f"Drawing labels on {len(pairs)} random images\n")

for img_path, lbl_path, split in pairs:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"SKIP: cannot read {img_path.name}")
        continue
    h, w = img.shape[:2]

    # Parse YOLO-seg label: each line = class_id x1 y1 x2 y2 ...
    polygons = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = parts[1:]
        pts = []
        for i in range(0, len(coords) - 1, 2):
            px = int(float(coords[i]) * w)
            py = int(float(coords[i + 1]) * h)
            pts.append([px, py])
        if len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.int32))

    # Draw: green filled (semi-transparent) + red outline
    overlay = img.copy()
    for poly in polygons:
        cv2.fillPoly(overlay, [poly], (0, 200, 0))
    result = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    for poly in polygons:
        cv2.polylines(result, [poly], True, (0, 0, 255), 2)

    # Save
    out_path = OUT / f"{img_path.stem}_labeled.png"
    cv2.imwrite(str(out_path), result)
    print(f"[{split}] {img_path.name}")
    print(f"  -> {len(polygons)} wall polygons drawn")
    print(f"  -> saved: {out_path.name}")

print(f"\nDone! {len(pairs)} labeled images saved to: {OUT}")
