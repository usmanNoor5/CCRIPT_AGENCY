"""
Convert CubiCasa5K dataset to YOLO-seg format for wall segmentation.

Reads SVG annotations, extracts wall polygons, tiles large images into
512x512 crops, and outputs YOLO-seg labels.

Usage:
    python3 datasets/cubicasa5k_to_yoloseg.py

Output:
    datasets/cubicasa5k_yoloseg/
        images/train/  images/val/
        labels/train/  labels/val/
        data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path
from xml.dom import minidom

import cv2
import numpy as np


def parse_wall_polygons(svg_path: str) -> list:
    """Parse SVG file and extract wall polygon vertices."""
    svg = minidom.parse(svg_path)
    walls = []

    for e in svg.getElementsByTagName("g"):
        elem_id = e.getAttribute("id")
        if elem_id not in ("Wall", "Railing"):
            continue

        # Find child <polygon> element
        pol = None
        for child in e.childNodes:
            if child.nodeName == "polygon":
                pol = child
                break
        if pol is None:
            continue

        points_str = pol.getAttribute("points").split(" ")
        points_str = [p for p in points_str if p.strip()]

        xs, ys = [], []
        for pt in points_str:
            parts = pt.split(",")
            if len(parts) != 2:
                continue
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))

        if len(xs) < 3:
            continue

        # Skip tiny walls (< 4px in either dimension)
        if abs(max(xs) - min(xs)) < 4 or abs(max(ys) - min(ys)) < 4:
            continue

        walls.append((xs, ys))

    return walls


def walls_to_yoloseg_lines(walls: list, img_w: int, img_h: int) -> list:
    """Convert wall polygons to YOLO-seg format lines."""
    lines = []
    for xs, ys in walls:
        coords = []
        for x, y in zip(xs, ys):
            xn = max(0.0, min(1.0, x / img_w))
            yn = max(0.0, min(1.0, y / img_h))
            coords.append(f"{xn:.6f}")
            coords.append(f"{yn:.6f}")
        lines.append("0 " + " ".join(coords))
    return lines


def tile_image_and_labels(
    img: np.ndarray,
    walls: list,
    tile_size: int,
    stride: int,
) -> list:
    """
    Tile a large image into tile_size x tile_size crops.
    Returns list of (tile_img, yolo_lines) for non-empty tiles.
    """
    h, w = img.shape[:2]
    results = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)

            tile = img[y:y2, x:x2]

            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded

            # Skip mostly blank tiles
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            if np.sum(gray > 240) / (tile_size * tile_size) > 0.97:
                continue

            # Clip wall polygons to this tile
            tile_walls = []
            for xs, ys in walls:
                # Check if polygon intersects this tile
                pxs = np.array(xs)
                pys = np.array(ys)

                if pxs.max() < x or pxs.min() > x2:
                    continue
                if pys.max() < y or pys.min() > y2:
                    continue

                # Shift and clip to tile coordinates
                txs = np.clip(pxs - x, 0, tile_size)
                tys = np.clip(pys - y, 0, tile_size)

                # Skip if clipped polygon is too small
                if abs(txs.max() - txs.min()) < 3 or abs(tys.max() - tys.min()) < 3:
                    continue

                tile_walls.append((txs.tolist(), tys.tolist()))

            if not tile_walls:
                continue

            # Convert to YOLO-seg format
            lines = walls_to_yoloseg_lines(tile_walls, tile_size, tile_size)
            results.append((tile, lines, y, x))

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert CubiCasa5K to YOLO-seg format.")
    ap.add_argument(
        "--src",
        type=str,
        default="datasets/cubicasa5k/cubicasa5k",
        help="Path to extracted CubiCasa5K root (contains train.txt, high_quality/, etc.)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="datasets/cubicasa5k_yoloseg",
        help="Output directory for YOLO-seg dataset",
    )
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--use-original", action="store_true", help="Use F1_original.png (higher res)")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    ap.add_argument("--max-samples", type=int, default=0, help="Max floor plans to process (0=all)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    tile_size = args.tile_size
    stride = tile_size - args.overlap

    # Collect all sample folders
    folders = []
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        fp = src / split_file
        if fp.exists():
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        folders.append(line)

    if not folders:
        # Fallback: scan for model.svg files
        for svg_path in sorted(src.rglob("model.svg")):
            folders.append(str(svg_path.parent.relative_to(src)))

    print(f"Found {len(folders)} floor plan samples")

    if args.max_samples > 0:
        random.shuffle(folders)
        folders = folders[: args.max_samples]
        print(f"Using {len(folders)} samples (--max-samples)")

    # Create output directories
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    skipped = 0

    for i, folder in enumerate(folders):
        sample_dir = src / folder

        svg_path = sample_dir / "model.svg"
        img_name = "F1_original.png" if args.use_original else "F1_scaled.png"
        img_path = sample_dir / img_name

        if not svg_path.exists() or not img_path.exists():
            skipped += 1
            continue

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        # Parse walls from SVG
        try:
            walls = parse_wall_polygons(str(svg_path))
        except Exception as e:
            print(f"  SKIP {folder}: SVG parse error: {e}")
            skipped += 1
            continue

        if not walls:
            skipped += 1
            continue

        # If using original image, scale SVG coords (they match F1_scaled.png)
        if args.use_original:
            scaled_path = sample_dir / "F1_scaled.png"
            if scaled_path.exists():
                scaled_img = cv2.imread(str(scaled_path))
                if scaled_img is not None:
                    sh, sw = scaled_img.shape[:2]
                    sx = img_w / sw
                    sy = img_h / sh
                    walls = [
                        ([x * sx for x in xs], [y * sy for y in ys])
                        for xs, ys in walls
                    ]

        # Tile image and assign walls to tiles
        tiles = tile_image_and_labels(img, walls, tile_size, stride)

        # Assign to train/val
        split = "val" if random.random() < args.val_ratio else "train"

        folder_id = folder.replace("/", "_")
        for tile_img, yolo_lines, ty, tx in tiles:
            name = f"cc5k_{folder_id}_r{ty:04d}_c{tx:04d}"

            cv2.imwrite(str(out / "images" / split / f"{name}.png"), tile_img)
            with open(out / "labels" / split / f"{name}.txt", "w") as f:
                f.write("\n".join(yolo_lines) + "\n")

            total_tiles += 1

        if (i + 1) % 50 == 0 or i == len(folders) - 1:
            print(f"  [{i+1}/{len(folders)}] {folder} -> {len(tiles)} tiles | Total: {total_tiles}")

    # Write data.yaml
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: wall\n"
    )

    print(f"\nDone!")
    print(f"  Samples processed: {len(folders) - skipped}")
    print(f"  Samples skipped: {skipped}")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Output: {out}/")
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
