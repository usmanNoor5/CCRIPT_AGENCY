#!/usr/bin/env python3
"""
Convert all training images to adaptive binary inverted format.
Output: grayscale, adaptive threshold, inverted (white lines on black bg).

Usage:
    python3.12 convert_to_bin_inv.py
    python3.12 convert_to_bin_inv.py --block-size 51 --C 10
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

TRAIN_DIR = Path(__file__).parent / "Testing"
OUTPUT_DIR = Path(__file__).parent / "Testing/binary"


def convert_to_bin_inv(image_path: Path, block_size: int = 51, C: int = 10) -> np.ndarray:
    """
    Convert an image to adaptive binary inverted format.
    1. Convert to grayscale
    2. Apply adaptive threshold (complex adaptive)
    3. Invert (white lines on black background)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  WARNING: Could not read {image_path.name}, skipping")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold (Gaussian method for complex adaptive)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )

    # Invert: white lines on black background
    inverted = cv2.bitwise_not(binary)

    return inverted


def main():
    parser = argparse.ArgumentParser(description="Convert training images to binary inverted format")
    parser.add_argument("--block-size", type=int, default=51,
                        help="Adaptive threshold block size (must be odd, default: 51)")
    parser.add_argument("--C", type=int, default=10,
                        help="Adaptive threshold constant (default: 10)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory (default: train/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: train_bin_inv/)")
    args = parser.parse_args()

    input_dir = Path(args.input) if args.input else TRAIN_DIR
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find all PNG images (skip json)
    images = sorted([f for f in input_dir.iterdir()
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg")])

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Images: {len(images)}")
    print(f"Params: block_size={args.block_size}, C={args.C}")
    print()

    converted = 0
    for img_path in images:
        result = convert_to_bin_inv(img_path, args.block_size, args.C)
        if result is None:
            continue

        out_name = f"{img_path.stem}_bin_inv.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), result)
        converted += 1
        print(f"  [{converted}/{len(images)}] {img_path.name} → {out_name}  ({result.shape[1]}x{result.shape[0]})")

    print(f"\nDone! Converted {converted}/{len(images)} images to {output_dir}")


if __name__ == "__main__":
    main()
