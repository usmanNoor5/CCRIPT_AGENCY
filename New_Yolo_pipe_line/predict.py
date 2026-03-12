#!/usr/bin/env python3
"""
predict.py — Run trained YOLO-seg wall model on a floor plan image.

Steps:
  1. Load image, optionally binarize
  2. Tile into 640x640 patches
  3. Run YOLO inference on each tile
  4. Stitch masks back onto full image
  5. Save result with wall overlay

Usage:
    python3.12 predict.py <image_path>
    python3.12 predict.py <image_path> --no-bin
    python3.12 predict.py <image_path> --model runs/segment/wall_seg_raw_v1/weights/best.pt --no-bin
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def binarize_image(img: np.ndarray, block_size: int = 51, C: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    inverted = cv2.bitwise_not(binary)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def main():
    parser = argparse.ArgumentParser(description="Predict walls on a floor plan image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str,
                        default="runs/segment/wall_seg_v1/weights/best.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--no-bin", action="store_true", help="Skip binarization")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: <input>_predicted.png)")
    parser.add_argument("--save-tiles", action="store_true", help="Save individual tiles and their predictions")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    # Load image
    orig = cv2.imread(str(img_path))
    if orig is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    h, w = orig.shape[:2]
    print(f"Image: {img_path.name} ({w}x{h})")

    # Binarize for model input
    if args.no_bin:
        model_input = orig.copy()
        print("Binarize: No (raw image)")
    else:
        model_input = binarize_image(orig)
        print("Binarize: Yes")

    # Load model
    print(f"Model: {model_path}")
    model = YOLO(str(model_path))

    # Tile and predict
    tile_sz = args.tile_size
    stride = tile_sz - args.overlap
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    count_mask = np.zeros((h, w), dtype=np.float32)  # for averaging overlaps
    total_detections = 0
    tile_count = 0

    # Setup tile saving directory
    if args.save_tiles:
        tiles_dir = Path(f"{img_path.stem}_tiles")
        tiles_dir.mkdir(exist_ok=True)
        print(f"Saving tiles to: {tiles_dir}/")

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_sz, h)
            x2 = min(x + tile_sz, w)
            tile = model_input[y:y2, x:x2]

            th, tw = tile.shape[:2]
            if th < tile_sz or tw < tile_sz:
                padded = np.zeros((tile_sz, tile_sz, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            results = model.predict(tile, conf=args.conf, verbose=False, imgsz=tile_sz)
            tile_count += 1

            tile_mask = np.zeros((tile_sz, tile_sz), dtype=np.uint8)
            tile_dets = 0

            for result in results:
                if result.masks is not None:
                    for mask_data in result.masks.data:
                        mask_np = mask_data.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (tile_sz, tile_sz))
                        # Crop to actual tile size (remove padding)
                        mask_crop = mask_resized[:th, :tw]
                        wall_mask[y:y2, x:x2] = np.maximum(
                            wall_mask[y:y2, x:x2],
                            (mask_crop > 0.5).astype(np.uint8) * 255
                        )
                        tile_mask = np.maximum(
                            tile_mask,
                            (mask_resized > 0.5).astype(np.uint8) * 255
                        )
                        total_detections += 1
                        tile_dets += 1

            # Save individual tiles
            if args.save_tiles:
                tile_name = f"tile_r{y:05d}_c{x:05d}"
                cv2.imwrite(str(tiles_dir / f"{tile_name}_input.png"), tile)
                cv2.imwrite(str(tiles_dir / f"{tile_name}_mask.png"), tile_mask)
                # Overlay on tile
                tile_overlay = tile.copy()
                tile_colored = np.zeros_like(tile)
                tile_colored[:, :, 2] = 255
                tm_bool = tile_mask > 0
                if tm_bool.any():
                    tile_overlay[tm_bool] = cv2.addWeighted(tile, 0.5, tile_colored, 0.5, 0)[tm_bool]
                cv2.imwrite(str(tiles_dir / f"{tile_name}_overlay.png"), tile_overlay)

            count_mask[y:y2, x:x2] += 1.0

    print(f"Tiles processed: {tile_count}")
    print(f"Total wall detections: {total_detections}")

    # Create overlay on original image
    overlay = orig.copy()
    wall_colored = np.zeros_like(orig)
    wall_colored[:, :, 2] = 255  # Red channel
    mask_bool = wall_mask > 0
    overlay[mask_bool] = cv2.addWeighted(orig, 0.5, wall_colored, 0.5, 0)[mask_bool]

    # Draw contours
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    # Save
    out_path = args.output or str(img_path.stem) + "_predicted.png"
    cv2.imwrite(out_path, overlay)
    print(f"Output saved: {out_path}")

    # Also save raw mask
    mask_path = str(Path(out_path).stem) + "_mask.png"
    cv2.imwrite(mask_path, wall_mask)
    print(f"Mask saved: {mask_path}")


if __name__ == "__main__":
    main()
