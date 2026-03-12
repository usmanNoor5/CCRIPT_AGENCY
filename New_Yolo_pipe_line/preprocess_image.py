#!/usr/bin/env python3
"""
preprocess_image.py — Preprocess a floor plan image:
  1. Tile into 2x2 grid
  2. Resize each tile to 1008x1008
  3. Auto-adjust contrast (contrast stretching)
  4. Convert to grayscale

Usage:
    python3.12 preprocess_image.py <image_path>
    python3.12 preprocess_image.py <image_path> --output preprocessed_output
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def contrast_stretch(img: np.ndarray) -> np.ndarray:
    """Apply contrast stretching (min-max normalization) to an image."""
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return img
    stretched = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return stretched


def main():
    parser = argparse.ArgumentParser(description="Tile, resize, contrast stretch, grayscale")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default="preprocessed_output",
                        help="Output folder (default: preprocessed_output)")
    parser.add_argument("--tile-rows", type=int, default=2)
    parser.add_argument("--tile-cols", type=int, default=2)
    parser.add_argument("--resize", type=int, default=1008, help="Resize each tile to this size")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    base_dir = Path(__file__).parent
    out_dir = base_dir / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    stem = img_path.stem
    tile_h = h // args.tile_rows
    tile_w = w // args.tile_cols

    print(f"Image: {img_path.name} ({w}x{h})")
    print(f"Tiling: {args.tile_rows}x{args.tile_cols} → tile size {tile_w}x{tile_h}")
    print(f"Resize: {args.resize}x{args.resize}")
    print(f"Output: {out_dir}")
    print()

    count = 0
    for r in range(args.tile_rows):
        for c in range(args.tile_cols):
            y1 = r * tile_h
            y2 = y1 + tile_h if r < args.tile_rows - 1 else h
            x1 = c * tile_w
            x2 = x1 + tile_w if c < args.tile_cols - 1 else w

            tile = img[y1:y2, x1:x2]

            # Resize to target size
            tile = cv2.resize(tile, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

            # Grayscale
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            # Contrast stretching
            tile = contrast_stretch(tile)

            # Save
            out_name = f"{stem}_r{r}_c{c}.png"
            cv2.imwrite(str(out_dir / out_name), tile)
            print(f"  Saved: {out_name}")
            count += 1

    print(f"\nDone. {count} tiles saved to {out_dir}")


if __name__ == "__main__":
    main()
