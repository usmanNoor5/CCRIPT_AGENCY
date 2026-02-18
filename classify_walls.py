"""
Classify detected walls as outer (green outline) or inner (red outline).

Key insight: the YOLO mask has gaps — geometric approaches fail.
The ORIGINAL IMAGE has complete, unbroken wall lines drawn by the architect.

Approach:
  1. Threshold the original image to get all white/light regions
  2. Find the LARGEST connected white region = outside of the building
     (it won't leak through walls because the image lines are complete)
  3. Expand that outside region slightly to touch the YOLO mask pixels
  4. YOLO wall pixels touched by outside → outer (green outline)
     YOLO wall pixels not touched        → inner (red outline)

Usage:
    python3 classify_walls.py \
        --image  testing/tiles/page_004_cropped.png \
        --mask   testing/inference_results/page_004_walls_mask.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def classify_walls(orig_img: np.ndarray,
                   mask_binary: np.ndarray,
                   white_thresh: int = 200,
                   reach: int = 8,
                   border_pct: float = 0.05) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Combine two methods — union gives the most complete outer wall detection:

    Method 1 — Image exterior (largest white region in the original image):
      Finds the outside white space, expands it to touch wall pixels.
      Reliable where image walls are complete.

    Method 2 — Bbox border zone (border ring of the wall mask bounding box):
      Catches outer walls that the image method misses when the exterior
      white region doesn't reach all sides.

    outer = Method1 OR Method2
    inner = wall pixels not caught by either
    """
    h, w = orig_img.shape[:2]
    k = np.ones((reach * 2 + 1, reach * 2 + 1), np.uint8)

    # --- Method 1: image exterior ---
    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, white_thresh, 1, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(white, connectivity=8)
    outer1 = np.zeros_like(mask_binary)
    if num_labels > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        outside = (labels == int(np.argmax(sizes))).astype(np.uint8)
        outside_reach = cv2.dilate(outside, k, iterations=1)
        outer1 = cv2.bitwise_and(mask_binary, outside_reach)

    # --- Method 2: bbox border zone ---
    ys, xs = np.where(mask_binary == 1)
    outer2 = np.zeros_like(mask_binary)
    bbox = None
    if len(xs) > 0:
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox = (x1, y1, x2, y2)
        bw, bh = x2 - x1, y2 - y1
        border = int(min(bw, bh) * border_pct)
        zone = np.zeros((h, w), dtype=np.uint8)
        zone[y1:y2, x1:x2] = 1
        t = border
        zone[min(y1+t, y2):max(y2-t, y1),
             min(x1+t, x2):max(x2-t, x1)] = 0
        outer2 = cv2.bitwise_and(mask_binary, zone)

    # --- Union ---
    outer_mask = cv2.bitwise_or(outer1, outer2)
    inner_mask = np.clip(
        mask_binary.astype(np.int16) - outer_mask, 0, 1
    ).astype(np.uint8)

    return outer_mask, inner_mask, bbox


def draw_outlines(base_img: np.ndarray,
                  mask: np.ndarray,
                  outer_mask: np.ndarray,
                  inner_mask: np.ndarray,
                  bbox: tuple = None,
                  min_area: int = 200,
                  line_width: int = 2) -> np.ndarray:
    """Draw each wall contour outline coloured per-pixel by outer/inner zone."""
    result = base_img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        outline = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(outline, [cnt], 0, 1, line_width)

        result[outline & outer_mask == 1] = (0, 200, 0)   # green
        result[outline & inner_mask == 1] = (0, 0, 255)   # red

    # Draw cyan bounding box to show the border zone boundary
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",        required=True)
    ap.add_argument("--mask",         required=True)
    ap.add_argument("--out",          default="testing/inference_results")
    ap.add_argument("--white-thresh", type=int,   default=220,
                    help="Brightness threshold for open space (default 220)")
    ap.add_argument("--reach",        type=int,   default=20,
                    help="Px to expand exterior into wall pixels (default 20)")
    ap.add_argument("--border-pct",   type=float, default=0.05,
                    help="Border zone as fraction of min(bbox_w, bbox_h) (default 0.05)")
    ap.add_argument("--min-area",     type=int,   default=200)
    args = ap.parse_args()

    base_img = cv2.imread(args.image)
    mask_raw = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if base_img is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    if mask_raw is None:
        raise SystemExit(f"Cannot read mask: {args.mask}")

    _, mask = cv2.threshold(mask_raw, 127, 1, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    outer_mask, inner_mask, bbox = classify_walls(
        base_img, mask,
        white_thresh=args.white_thresh,
        reach=args.reach,
        border_pct=args.border_pct,
    )
    print(f"Outer wall pixels: {outer_mask.sum()}")
    print(f"Inner wall pixels: {inner_mask.sum()}")

    result = draw_outlines(base_img, mask, outer_mask, inner_mask,
                           bbox=bbox, min_area=args.min_area)

    h, w = result.shape[:2]
    fs = max(0.7, min(h, w) / 2500)
    th = max(1, int(fs * 2.5))
    cv2.putText(result, "Outer walls", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 200, 0), th)
    cv2.putText(result, "Inner walls", (20, 50 + int(fs * 45)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    cv2.imwrite(str(out_dir / f"{stem}_classified.png"), result)
    print(f"Saved: {out_dir}/{stem}_classified.png")


if __name__ == "__main__":
    main()
