"""
Convert my_dataset (COCO segmentation format) to YOLO-seg 512x512 tiles.

Reads _annotations.coco.json, tiles large images into 512x512 crops,
clips polygon annotations to each tile, and outputs YOLO-seg labels.

Usage:
    python3 datasets/my_dataset_to_yoloseg.py
    python3 datasets/my_dataset_to_yoloseg.py --walls-only
    python3 datasets/my_dataset_to_yoloseg.py --val-ratio 0.2
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def rotate_image_and_polygons(img, anns, angle, class_remap):
    """Rotate image and all polygon annotations by 90/180/270 degrees."""
    h, w = img.shape[:2]

    if angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # (x, y) -> (h - y - 1, x)
        def transform(x, y):
            return h - y, x
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        # (x, y) -> (w - x - 1, h - y - 1)
        def transform(x, y):
            return w - x, h - y
    elif angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # (x, y) -> (y, w - x - 1)
        def transform(x, y):
            return y, w - x
    else:
        return img, anns

    new_anns = []
    for ann in anns:
        cid = ann["category_id"]
        if cid not in class_remap:
            continue
        new_segs = []
        for seg in ann["segmentation"]:
            xs = seg[0::2]
            ys = seg[1::2]
            new_pts = []
            for x, y in zip(xs, ys):
                nx, ny = transform(x, y)
                new_pts.extend([nx, ny])
            new_segs.append(new_pts)
        new_ann = dict(ann)
        new_ann["segmentation"] = new_segs
        new_anns.append(new_ann)

    return rotated, new_anns


def polygon_intersects_tile(seg_pts, tx, ty, tile_size):
    """Check if a polygon intersects a tile region."""
    xs = seg_pts[0::2]
    ys = seg_pts[1::2]
    if max(xs) < tx or min(xs) > tx + tile_size:
        return False
    if max(ys) < ty or min(ys) > ty + tile_size:
        return False
    return True


def clip_polygon_to_tile(seg_pts, tx, ty, tile_size):
    """Clip polygon to tile and return normalized YOLO-seg coordinates."""
    xs = seg_pts[0::2]
    ys = seg_pts[1::2]

    # Shift to tile coords and clip
    txs = [max(0, min(tile_size, x - tx)) for x in xs]
    tys = [max(0, min(tile_size, y - ty)) for y in ys]

    # Check if clipped polygon is too small
    if max(txs) - min(txs) < 3 or max(tys) - min(tys) < 3:
        return None

    # Normalize to [0, 1]
    coords = []
    for x, y in zip(txs, tys):
        coords.append(f"{x / tile_size:.6f}")
        coords.append(f"{y / tile_size:.6f}")

    return coords


def main():
    ap = argparse.ArgumentParser(description="Convert my_dataset COCO to YOLO-seg 512x512 tiles.")
    ap.add_argument("--src", type=str, default="datasets/my_dataset",
                    help="Path to my_dataset folder")
    ap.add_argument("--out", type=str, default="datasets/my_dataset_yoloseg",
                    help="Output YOLO-seg dataset folder")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Fraction of images for validation")
    ap.add_argument("--walls-only", action="store_true",
                    help="Only include 'Walls' category (ignore other classes)")
    ap.add_argument("--rotations", type=str, default="0,90,180,270",
                    help="Comma-separated rotation angles (default: 0,90,180,270)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    tile_size = args.tile_size
    stride = tile_size - args.overlap

    # Load COCO annotations
    ann_path = src / "train" / "_annotations.coco.json"
    if not ann_path.exists():
        raise SystemExit(f"Annotation file not found: {ann_path}")

    with open(ann_path) as f:
        coco = json.load(f)

    # Build category mapping
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    print("Categories found:")
    for cid, name in sorted(cat_map.items()):
        print(f"  {cid}: {name}")

    if args.walls_only:
        # Find the Walls category ID
        walls_id = None
        for c in coco["categories"]:
            if c["name"] == "Walls":
                walls_id = c["id"]
                break
        if walls_id is None:
            raise SystemExit("'Walls' category not found in annotations")
        print(f"\n--walls-only: using only category {walls_id} (Walls)")
        # YOLO class mapping: Walls -> 0
        class_remap = {walls_id: 0}
        class_names = {0: "wall"}
    else:
        # Map all categories to sequential YOLO class IDs (skip id=0 'objects' parent)
        real_cats = [c for c in coco["categories"] if c["id"] > 0]
        class_remap = {}
        class_names = {}
        for i, c in enumerate(sorted(real_cats, key=lambda x: x["id"])):
            class_remap[c["id"]] = i
            class_names[i] = c["name"]
        print(f"\nYOLO class mapping: {class_remap}")

    # Build image id -> annotations lookup
    img_anns = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in class_remap:
            continue
        iid = ann["image_id"]
        if iid not in img_anns:
            img_anns[iid] = []
        img_anns[iid].append(ann)

    # Build image id -> info lookup
    img_info = {img["id"]: img for img in coco["images"]}

    # Create output dirs
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rotations = [int(r) for r in args.rotations.split(",")]
    print(f"Rotations: {rotations} ({len(rotations)}x multiplier)")

    total_tiles = 0
    total_labels = 0
    skipped_imgs = 0

    image_ids = list(img_info.keys())
    random.shuffle(image_ids)

    for idx, iid in enumerate(image_ids):
        info = img_info[iid]
        img_file = src / "train" / info["file_name"]

        if not img_file.exists():
            skipped_imgs += 1
            continue

        img_orig = cv2.imread(str(img_file))
        if img_orig is None:
            skipped_imgs += 1
            continue

        anns_orig = img_anns.get(iid, [])
        if not anns_orig:
            skipped_imgs += 1
            continue

        # Assign to train/val (same split for all rotations of this image)
        split = "val" if random.random() < args.val_ratio else "train"

        for angle in rotations:
            # Rotate image and annotations
            if angle == 0:
                img = img_orig
                anns = anns_orig
            else:
                img, anns = rotate_image_and_polygons(img_orig, anns_orig, angle, class_remap)

            h, w = img.shape[:2]

            # Tile the image
            for ty in range(0, h, stride):
                for tx in range(0, w, stride):
                    y2 = min(ty + tile_size, h)
                    x2 = min(tx + tile_size, w)

                    tile = img[ty:y2, tx:x2]

                    # Pad if needed
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded = np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255
                        padded[:tile.shape[0], :tile.shape[1]] = tile
                        tile = padded

                    # Skip mostly blank tiles
                    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                    if np.sum(gray > 240) / (tile_size * tile_size) > 0.95:
                        continue

                    # Collect labels for this tile
                    tile_lines = []
                    for ann in anns:
                        cid = ann["category_id"]
                        yolo_cls = class_remap.get(cid)
                        if yolo_cls is None:
                            continue

                        for seg in ann["segmentation"]:
                            if not polygon_intersects_tile(seg, tx, ty, tile_size):
                                continue

                            coords = clip_polygon_to_tile(seg, tx, ty, tile_size)
                            if coords is None:
                                continue

                            tile_lines.append(f"{yolo_cls} " + " ".join(coords))

                    if not tile_lines:
                        continue

                    # Save tile image and label
                    stem = Path(info["file_name"]).stem
                    name = f"{stem}_rot{angle}_r{ty:04d}_c{tx:04d}"

                    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), tile)
                    with open(out / "labels" / split / f"{name}.txt", "w") as f:
                        f.write("\n".join(tile_lines) + "\n")

                    total_tiles += 1
                    total_labels += len(tile_lines)

        if (idx + 1) % 10 == 0 or idx == len(image_ids) - 1:
            print(f"  [{idx+1}/{len(image_ids)}] tiles: {total_tiles}, labels: {total_labels}")

    # Write data.yaml
    nc = len(class_names)
    names_yaml = "\n".join(f"  {k}: {v}" for k, v in sorted(class_names.items()))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names:\n{names_yaml}\n"
    )

    # Count per split
    train_imgs = len(list((out / "images" / "train").glob("*")))
    val_imgs = len(list((out / "images" / "val").glob("*")))

    print(f"\nDone!")
    print(f"  Images processed: {len(image_ids) - skipped_imgs}")
    print(f"  Images skipped (missing file): {skipped_imgs}")
    print(f"  Total tiles: {total_tiles} (train: {train_imgs}, val: {val_imgs})")
    print(f"  Total labels: {total_labels}")
    print(f"  Classes: {nc} -> {class_names}")
    print(f"  Output: {out}/")
    print(f"  data.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
