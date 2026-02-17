"""
Run trained YOLO-seg model on images and visualize wall segmentation results.

Usage:
    # Run on a folder of tiles and stitch into one combined image
    python3 infer_walls.py --source testing/tiles_003/

    # Run on specific image(s)
    python3 infer_walls.py --source path/to/image.png

    # Adjust confidence
    python3 infer_walls.py --source testing/tiles/ --conf 0.3
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def infer_tile(model, img, imgsz, conf, device):
    """Run inference on a single tile, return binary mask and wall count."""
    h, w = img.shape[:2]
    results = model.predict(
        source=img,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )
    result = results[0]
    mask_combined = np.zeros((h, w), dtype=np.uint8)
    num_walls = 0

    if result.masks is not None:
        for mask_tensor in result.masks.data:
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (mask > 0.5).astype(np.uint8)
            mask_combined = np.maximum(mask_combined, binary)
            num_walls += 1

    return mask_combined, num_walls


def stitch_tiles(tile_dir, model, args):
    """Run inference on all tiles and stitch results into one full image."""
    tile_dir = Path(tile_dir)

    # Find the cropped base image
    cropped_files = sorted(tile_dir.glob("*_cropped.png"))
    if not cropped_files:
        return None
    cropped_path = cropped_files[0]
    base_img = cv2.imread(str(cropped_path))
    if base_img is None:
        return None

    stem = cropped_path.stem
    if stem.endswith("_cropped"):
        stem = stem[:-8]
    h_full, w_full = base_img.shape[:2]

    # Find all tiles with position pattern
    tile_pattern = re.compile(rf"{re.escape(stem)}_tile_r(\d+)_c(\d+)\.(png|jpg)")
    tiles = []
    for f in sorted(tile_dir.iterdir()):
        m = tile_pattern.match(f.name)
        if m:
            y, x = int(m.group(1)), int(m.group(2))
            tiles.append((f, y, x))

    if not tiles:
        return None

    print(f"Stitching {len(tiles)} tiles onto {w_full}x{h_full} canvas ({stem})")

    # Create full-size mask and confidence accumulator
    mask_full = np.zeros((h_full, w_full), dtype=np.float32)
    count_full = np.zeros((h_full, w_full), dtype=np.float32)
    total_walls = 0

    for tile_path, ty, tx in tiles:
        tile_img = cv2.imread(str(tile_path))
        if tile_img is None:
            continue

        tile_h, tile_w = tile_img.shape[:2]
        mask, n_walls = infer_tile(model, tile_img, args.imgsz, args.conf, args.device)
        total_walls += n_walls

        # Place mask onto full canvas (handle edge tiles)
        place_h = min(tile_h, h_full - ty)
        place_w = min(tile_w, w_full - tx)
        mask_full[ty:ty + place_h, tx:tx + place_w] += mask[:place_h, :place_w].astype(np.float32)
        count_full[ty:ty + place_h, tx:tx + place_w] += 1.0

        status = f"{n_walls} walls" if n_walls > 0 else "---"
        print(f"  {tile_path.name} -> {status}")

    # Average overlapping regions, threshold to binary
    count_full[count_full == 0] = 1
    mask_avg = mask_full / count_full
    mask_binary = (mask_avg > 0.3).astype(np.uint8)

    # Build the combined output image
    overlay = base_img.copy()

    # Draw red contours
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 3)

    # Green fill for wall regions
    wall_color = np.zeros_like(base_img)
    wall_color[mask_binary == 1] = (0, 200, 0)
    overlay = cv2.addWeighted(overlay, 1.0, wall_color, 0.4, 0)

    # Side by side: original | prediction
    combined = np.hstack([base_img, overlay])

    # Labels
    font_scale = max(0.8, min(h_full, w_full) / 2000)
    thickness = max(2, int(font_scale * 2.5))
    cv2.putText(combined, "Original", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.putText(combined, f"Prediction ({total_walls} walls)", (w_full + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 0), thickness)

    return combined, stem, total_walls, mask_binary


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer wall segmentation with trained YOLO-seg model.")
    ap.add_argument(
        "--model",
        type=str,
        default="runs/segment/runs_api/yoloseg_wall5/weights/best.pt",
        help="Path to trained .pt model",
    )
    ap.add_argument(
        "--source",
        type=str,
        default=None,
        help="Tile folder (with *_cropped.png + *_tile_*.png) or single image.",
    )
    ap.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    ap.add_argument("--device", type=str, default="0", help="Device: 0 for GPU, cpu for CPU")
    ap.add_argument(
        "--out",
        type=str,
        default="testing/inference_results",
        help="Output directory for results",
    )
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path), task="segment")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = Path(args.source) if args.source else Path("testing/tiles")

    print(f"Model:  {model_path}")
    print(f"Source: {src}")
    print(f"Output: {out_dir}/")
    print()

    if src.is_dir():
        # Try stitch mode: look for *_cropped.png in the folder
        result = stitch_tiles(src, model, args)
        if result is not None:
            combined, stem, total_walls, mask_binary = result
            out_path = out_dir / f"{stem}_walls_combined.png"
            cv2.imwrite(str(out_path), combined)
            print(f"\nSaved combined image: {out_path}")
            print(f"Total wall segments detected: {total_walls}")

            # Also save the mask alone
            mask_path = out_dir / f"{stem}_walls_mask.png"
            cv2.imwrite(str(mask_path), mask_binary * 255)
            print(f"Saved wall mask: {mask_path}")
            return

        # Fallback: run on individual images
        image_paths = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    elif src.is_file():
        image_paths = [src]
    else:
        raise SystemExit(f"Source not found: {src}")

    print(f"Images: {len(image_paths)}")
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask, num_walls = infer_tile(model, img, args.imgsz, args.conf, args.device)

        overlay = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

        wall_color = np.zeros_like(img)
        wall_color[mask == 1] = (0, 200, 0)
        overlay = cv2.addWeighted(overlay, 1.0, wall_color, 0.4, 0)

        combined = np.hstack([img, overlay])
        cv2.putText(combined, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Prediction ({num_walls} walls)", (w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        out_path = out_dir / f"result_{img_path.stem}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  {img_path.name} -> {num_walls} walls -> {out_path.name}")

    print(f"\nDone! Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
