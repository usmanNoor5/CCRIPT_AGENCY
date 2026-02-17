"""
Run trained YOLO-seg model on images, stitch tiles, then fit precise
rotated rectangles to each detected wall for crisp output lines.

Usage:
    python3 infer_walls.py --source testing/tiles/
    python3 infer_walls.py --source testing/tiles/ --conf 0.3
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def infer_tile(model, img, imgsz, conf, device):
    """Run inference on a single tile, return binary mask and wall count."""
    h, w = img.shape[:2]
    results = model.predict(source=img, imgsz=imgsz, conf=conf,
                            device=device, verbose=False)
    result = results[0]
    mask_combined = np.zeros((h, w), dtype=np.uint8)
    num_walls = 0
    if result.masks is not None:
        for mask_tensor in result.masks.data:
            mask = mask_tensor.cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_combined = np.maximum(mask_combined, (mask > 0.5).astype(np.uint8))
            num_walls += 1
    return mask_combined, num_walls


# ---------------------------------------------------------------------------
# Precise wall tracing via rotated rectangles
# ---------------------------------------------------------------------------

def trace_walls(mask_binary: np.ndarray, min_area: int = 200,
                epsilon_factor: float = 0.01) -> np.ndarray:
    """
    Retrace each YOLO wall blob as a clean precise outline.

    Takes the raw (jagged) YOLO contour and runs Douglas-Peucker simplification
    (approxPolyDP) to remove noise and produce crisp straight-segment lines.

    Args:
        mask_binary:    uint8 binary mask (0/1) from YOLO stitching.
        min_area:       Ignore blobs smaller than this (noise filter).
        epsilon_factor: Controls simplification. Higher = fewer, straighter segments.
                        0.005 = tight trace, 0.02 = very simplified.
    """
    h, w = mask_binary.shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Simplify the contour — removes jaggedness, keeps the wall shape
        epsilon = epsilon_factor * cv2.arcLength(cnt, closed=True)
        simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)
        cv2.drawContours(canvas, [simplified], 0, 255, 2)

    return canvas


# ---------------------------------------------------------------------------
# Stitch tiles
# ---------------------------------------------------------------------------

def stitch_tiles(tile_dir, model, args):
    tile_dir = Path(tile_dir)

    cropped_files = sorted(tile_dir.glob("*_cropped.png"))
    if not cropped_files:
        return None
    base_img = cv2.imread(str(cropped_files[0]))
    if base_img is None:
        return None

    stem = cropped_files[0].stem
    if stem.endswith("_cropped"):
        stem = stem[:-8]
    h_full, w_full = base_img.shape[:2]

    tile_pattern = re.compile(rf"{re.escape(stem)}_tile_r(\d+)_c(\d+)\.(png|jpg)")
    tiles = []
    for f in sorted(tile_dir.iterdir()):
        m = tile_pattern.match(f.name)
        if m:
            tiles.append((f, int(m.group(1)), int(m.group(2))))

    if not tiles:
        return None

    print(f"Stitching {len(tiles)} tiles onto {w_full}x{h_full} canvas ({stem})")

    mask_full  = np.zeros((h_full, w_full), dtype=np.float32)
    count_full = np.zeros((h_full, w_full), dtype=np.float32)
    total_walls = 0

    for tile_path, ty, tx in tiles:
        tile_img = cv2.imread(str(tile_path))
        if tile_img is None:
            continue
        tile_h, tile_w = tile_img.shape[:2]
        mask, n_walls = infer_tile(model, tile_img, args.imgsz, args.conf, args.device)
        total_walls += n_walls

        place_h = min(tile_h, h_full - ty)
        place_w = min(tile_w, w_full - tx)
        mask_full [ty:ty+place_h, tx:tx+place_w] += mask[:place_h, :place_w].astype(np.float32)
        count_full[ty:ty+place_h, tx:tx+place_w] += 1.0

        status = f"{n_walls} walls" if n_walls else "---"
        print(f"  {tile_path.name} -> {status}")

    count_full[count_full == 0] = 1
    mask_binary = (mask_full / count_full > 0.3).astype(np.uint8)

    # Fit precise rotated rectangles to each wall blob
    print("Tracing wall outlines ...")
    line_canvas = trace_walls(mask_binary, min_area=args.min_area,
                              epsilon_factor=args.epsilon)

    overlay = base_img.copy()
    overlay[line_canvas == 255] = (0, 0, 255)   # crisp red outlines

    combined = np.hstack([base_img, overlay])
    fs = max(0.8, min(h_full, w_full) / 2000)
    th = max(2, int(fs * 2.5))
    cv2.putText(combined, "Original", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    cv2.putText(combined, f"Walls ({total_walls})", (w_full + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th)

    return combined, stem, total_walls, mask_binary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="runs/segment/runs_api/yoloseg_wall5/weights/best.pt")
    ap.add_argument("--source",   default=None)
    ap.add_argument("--conf",     type=float, default=0.2)
    ap.add_argument("--imgsz",    type=int,   default=512)
    ap.add_argument("--device",   default="0")
    ap.add_argument("--out",      default="testing/inference_results")
    ap.add_argument("--min-area", type=int,   default=200,
                    help="Minimum blob area in px to trace (noise filter)")
    ap.add_argument("--epsilon",  type=float, default=0.01,
                    help="Douglas-Peucker simplification (0.005=tight, 0.02=smooth)")
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
        result = stitch_tiles(src, model, args)
        if result is not None:
            combined, stem, total_walls, mask_binary = result
            out_path = out_dir / f"{stem}_walls_combined.png"
            cv2.imwrite(str(out_path), combined)
            print(f"\nSaved: {out_path}")
            mask_path = out_dir / f"{stem}_walls_mask.png"
            cv2.imwrite(str(mask_path), mask_binary * 255)
            print(f"Saved: {mask_path}")
            return

        image_paths = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    elif src.is_file():
        image_paths = [src]
    else:
        raise SystemExit(f"Source not found: {src}")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask, num_walls = infer_tile(model, img, args.imgsz, args.conf, args.device)
        line_canvas = trace_walls(mask, min_area=args.min_area,
                                  epsilon_factor=args.epsilon)
        overlay = img.copy()
        overlay[line_canvas == 255] = (0, 0, 255)
        combined = np.hstack([img, overlay])
        cv2.putText(combined, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Walls ({num_walls})", (w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out_path = out_dir / f"result_{img_path.stem}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  {img_path.name} -> {num_walls} walls -> {out_path.name}")

    print(f"\nDone! Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
