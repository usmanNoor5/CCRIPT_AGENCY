"""
Convert a full-page floor plan image into 512x512 tiles for inference.

Usage:
    python3 testing/prepare_for_inference.py --input testing/images/page_007.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def auto_crop(img: np.ndarray, pad: int = 20) -> np.ndarray:
    """Crop to the relevant drawing area, removing white borders and title blocks."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    row_ink = np.sum(binary, axis=1)
    col_ink = np.sum(binary, axis=0)

    h, w = img.shape[:2]
    row_thresh = w * 0.01 * 255
    col_thresh = h * 0.01 * 255

    rows_with_ink = np.where(row_ink > row_thresh)[0]
    cols_with_ink = np.where(col_ink > col_thresh)[0]

    if len(rows_with_ink) == 0 or len(cols_with_ink) == 0:
        return img

    y1 = max(0, rows_with_ink[0] - pad)
    y2 = min(h, rows_with_ink[-1] + pad)
    x1 = max(0, cols_with_ink[0] - pad)
    x2 = min(w, cols_with_ink[-1] + pad)

    return img[y1:y2, x1:x2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare floor plan image for YOLO-seg inference.")
    ap.add_argument("--input", type=str, required=True, help="Path to full-page floor plan image")
    ap.add_argument("--tile-size", type=int, default=512, help="Tile size (default 512)")
    ap.add_argument("--overlap", type=int, default=64, help="Overlap between tiles in pixels")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: testing/tiles/)")
    ap.add_argument("--no-crop", action="store_true", help="Skip auto-cropping")
    args = ap.parse_args()

    img_path = Path(args.input)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    h_orig, w_orig = img.shape[:2]
    print(f"Input:     {img_path.name} ({w_orig}x{h_orig})")

    # Auto-crop to relevant drawing area
    if not args.no_crop:
        img = auto_crop(img)
        h_crop, w_crop = img.shape[:2]
        print(f"Cropped:   {w_crop}x{h_crop} (removed {w_orig - w_crop}px width, {h_orig - h_crop}px height)")

        out_dir_parent = Path(args.out) if args.out else img_path.parent.parent / "tiles"
        out_dir_parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir_parent / f"{img_path.stem}_cropped.png"), img)

    h, w = img.shape[:2]
    tile_sz = args.tile_size
    overlap = args.overlap
    stride = tile_sz - overlap

    out_dir = Path(args.out) if args.out else img_path.parent.parent / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tile size: {tile_sz}x{tile_sz}, overlap: {overlap}px, stride: {stride}px")

    count = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_sz, h)
            x2 = min(x + tile_sz, w)
            tile = img[y:y2, x:x2]

            # Pad if tile is smaller than tile_size
            if tile.shape[0] < tile_sz or tile.shape[1] < tile_sz:
                padded = np.ones((tile_sz, tile_sz, 3), dtype=np.uint8) * 255
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            # Skip mostly blank tiles (>95% white)
            gray_check = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            white_ratio = np.sum(gray_check > 240) / (tile_sz * tile_sz)
            if white_ratio > 0.95:
                continue

            # Save tile as-is (no post-processing)
            name = f"{img_path.stem}_tile_r{y:04d}_c{x:04d}.png"
            cv2.imwrite(str(out_dir / name), tile)
            count += 1

    print(f"\nSaved {count} tiles to {out_dir}/")
    print(f"\nRun inference with:")
    print(f"  python3 infer_walls.py --source {out_dir}/")


if __name__ == "__main__":
    main()
