#!/usr/bin/env python3
"""Visualize YOLO-seg labels on tiles to verify correctness."""

import sys
from pathlib import Path
import cv2
import numpy as np

def draw_labels(img_path: Path, lbl_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    overlay = img.copy()

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    count = 0

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, len(coords), 2):
                px = int(coords[i] * w)
                py = int(coords[i + 1] * h)
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            color = colors[count % len(colors)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(img, [pts], True, color, 2)
            count += 1

    # Blend overlay
    result = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    cv2.imwrite(str(out_path), result)
    return count


def main():
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("yolo_bin_dataset")
    out_dir = dataset / "label_preview"
    out_dir.mkdir(exist_ok=True)

    for split in ("train", "val", "test"):
        img_dir = dataset / "images" / split
        lbl_dir = dataset / "labels" / split
        if not img_dir.exists():
            continue

        # Pick up to 5 samples per split (skip augmented for clarity)
        images = sorted([f for f in img_dir.glob("*.png")
                        if "_hflip" not in f.stem and "_vflip" not in f.stem
                        and "_rot" not in f.stem and "_noise" not in f.stem])[:5]

        print(f"\n=== {split.upper()} (showing {len(images)} samples) ===")
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            out_path = out_dir / f"{split}_{img_path.stem}.png"
            count = draw_labels(img_path, lbl_path, out_path)
            print(f"  {img_path.name} → {count} polygons → {out_path.name}")

    print(f"\nPreviews saved to: {out_dir}")


if __name__ == "__main__":
    main()
