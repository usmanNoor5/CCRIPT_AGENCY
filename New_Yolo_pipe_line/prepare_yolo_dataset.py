#!/usr/bin/env python3
"""
prepare_yolo_dataset.py — Full pipeline:
  1. Read COCO annotations
  2. Binarize images (adaptive threshold + invert)
  3. Tile into 640x640 with overlap, clip labels to each tile
  4. Shuffle ALL tiles, split into 70/20/10 train/val/test
  5. Augment train tiles: hflip, vflip, rot90, rot180, rot270, noise

Usage:
    python3.12 prepare_yolo_dataset.py --augment
    python3.12 prepare_yolo_dataset.py --augment --tile-size 640 --overlap 64
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


# ── Binary inversion ──────────────────────────────────────────────
def binarize_image(img: np.ndarray, block_size: int = 51, C: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    inverted = cv2.bitwise_not(binary)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


# ── COCO polygon → pixel coords ──────────────────────────────────
def coco_seg_to_pixel_polygons(annotation: dict) -> list[np.ndarray]:
    polys = []
    for seg in annotation.get("segmentation", []):
        if len(seg) < 6:
            continue
        polys.append(np.array(seg, dtype=float).reshape(-1, 2))
    return polys


# ── Clip polygon to tile and normalize ────────────────────────────
def clip_polygon_to_tile(poly: np.ndarray, tx: int, ty: int, tw: int, th: int,
                         min_area: float = 100.0) -> list[str]:
    local = poly.copy()
    local[:, 0] -= tx
    local[:, 1] -= ty

    clipped_x = np.clip(local[:, 0], 0, tw)
    clipped_y = np.clip(local[:, 1], 0, th)
    clipped = np.column_stack([clipped_x, clipped_y])

    x_range = clipped[:, 0].max() - clipped[:, 0].min()
    y_range = clipped[:, 1].max() - clipped[:, 1].min()
    if x_range < 3 or y_range < 3:
        return []

    area = x_range * y_range
    if area < min_area:
        return []

    norm = clipped.copy()
    norm[:, 0] /= tw
    norm[:, 1] /= th
    norm = np.clip(norm, 0.0, 1.0)

    keep = [0]
    for i in range(1, len(norm)):
        if not np.allclose(norm[i], norm[keep[-1]], atol=1e-5):
            keep.append(i)
    norm = norm[keep]

    if len(norm) < 3:
        return []

    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
    return [f"0 {coords}"]


# ── Augmentation helpers ──────────────────────────────────────────
def flip_horizontal(img, labels):
    flipped = cv2.flip(img, 1)
    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        nc = []
        for i in range(0, len(coords), 2):
            nc.append(1.0 - coords[i])
            nc.append(coords[i + 1])
        new_labels.append(cls + " " + " ".join(f"{c:.6f}" for c in nc))
    return flipped, new_labels


def flip_vertical(img, labels):
    flipped = cv2.flip(img, 0)
    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        nc = []
        for i in range(0, len(coords), 2):
            nc.append(coords[i])
            nc.append(1.0 - coords[i + 1])
        new_labels.append(cls + " " + " ".join(f"{c:.6f}" for c in nc))
    return flipped, new_labels


def rotate_90(img, labels):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        nc = []
        for i in range(0, len(coords), 2):
            nc.append(1.0 - coords[i + 1])
            nc.append(coords[i])
        new_labels.append(cls + " " + " ".join(f"{c:.6f}" for c in nc))
    return rotated, new_labels


def rotate_180(img, labels):
    rotated = cv2.rotate(img, cv2.ROTATE_180)
    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        nc = []
        for i in range(0, len(coords), 2):
            nc.append(1.0 - coords[i])
            nc.append(1.0 - coords[i + 1])
        new_labels.append(cls + " " + " ".join(f"{c:.6f}" for c in nc))
    return rotated, new_labels


def rotate_270(img, labels):
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_labels = []
    for line in labels:
        parts = line.split()
        cls = parts[0]
        coords = list(map(float, parts[1:]))
        nc = []
        for i in range(0, len(coords), 2):
            nc.append(coords[i + 1])
            nc.append(1.0 - coords[i])
        new_labels.append(cls + " " + " ".join(f"{c:.6f}" for c in nc))
    return rotated, new_labels


def add_noise(img, labels, intensity=15):
    noise = np.random.randint(-intensity, intensity + 1, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy, labels


AUGMENTATIONS = [
    ("hflip", flip_horizontal),
    ("vflip", flip_vertical),
    ("rot90", rotate_90),
    ("rot180", rotate_180),
    ("rot270", rotate_270),
    ("noise", add_noise),
]


# ── Main ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tile + augment COCO dataset for YOLO-seg")
    parser.add_argument("--input", type=str, default="train")
    parser.add_argument("--output", type=str, default="yolo_bin_dataset")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=64,
                        help="Overlap between tiles in pixels (default: 64)")
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Fraction for train (default: 0.7)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction for val (default: 0.2)")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Fraction for test (default: 0.1)")
    parser.add_argument("--no-bin", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--C", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    coco_json = input_dir / "_annotations.coco.json"

    if not coco_json.exists():
        raise SystemExit(f"COCO annotations not found: {coco_json}")

    with open(coco_json) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}

    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    tile_sz = args.tile_size
    stride = tile_sz - args.overlap

    print(f"Input:       {input_dir} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)")
    print(f"Output:      {output_dir}")
    print(f"Tile:        {tile_sz}x{tile_sz}, overlap={args.overlap}, stride={stride}")
    print(f"Binarize:    {'No' if args.no_bin else f'Yes (block={args.block_size}, C={args.C})'}")
    print(f"Augment:     {'Yes' if args.augment else 'No'}")
    print(f"Split:       {args.train_split:.0%} / {args.val_split:.0%} / {args.test_split:.0%} (by tiles)")
    print()

    # ── PHASE 1: Tile all images, collect in memory ───────────────
    print("Phase 1: Tiling all images...")
    all_tiles = []  # list of (tile_img, label_lines, tile_name, is_positive)

    for img_id in id_to_img:
        img_info = id_to_img[img_id]
        fname = img_info["file_name"]
        src_path = input_dir / fname

        if not src_path.exists():
            print(f"  [skip] {fname} not found")
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  [skip] Cannot read {fname}")
            continue

        h, w = img.shape[:2]

        if not args.no_bin:
            img = binarize_image(img, args.block_size, args.C)

        anns = img_anns.get(img_id, [])
        all_polys = []
        for ann in anns:
            all_polys.extend(coco_seg_to_pixel_polygons(ann))

        stem = src_path.stem
        img_pos = 0
        img_neg = 0

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + tile_sz, h)
                x2 = min(x + tile_sz, w)
                tile = img[y:y2, x:x2]

                th, tw = tile.shape[:2]
                if th < tile_sz or tw < tile_sz:
                    padded = np.zeros((tile_sz, tile_sz, 3), dtype=np.uint8)
                    padded[:th, :tw] = tile
                    tile = padded

                tile_labels = []
                for poly in all_polys:
                    labels = clip_polygon_to_tile(poly, x, y, tile_sz, tile_sz)
                    tile_labels.extend(labels)

                tile_name = f"{stem}_r{y:05d}_c{x:05d}"
                is_positive = len(tile_labels) > 0
                all_tiles.append((tile, tile_labels, tile_name, is_positive))

                if is_positive:
                    img_pos += 1
                else:
                    img_neg += 1

        print(f"  {fname} ({w}x{h}) → {img_pos} positive + {img_neg} negative = {img_pos + img_neg} tiles")

    total_pos = sum(1 for t in all_tiles if t[3])
    total_neg = sum(1 for t in all_tiles if not t[3])
    print(f"\nTotal tiles: {len(all_tiles)} ({total_pos} positive, {total_neg} negative)")

    # ── PHASE 2: Shuffle and split tiles 70/20/10 ────────────────
    print(f"\nPhase 2: Splitting tiles {args.train_split:.0%}/{args.val_split:.0%}/{args.test_split:.0%}...")
    random.shuffle(all_tiles)

    n = len(all_tiles)
    test_end = int(n * args.test_split)
    val_end = test_end + int(n * args.val_split)

    splits = {}
    splits["test"] = all_tiles[:test_end]
    splits["val"] = all_tiles[test_end:val_end]
    splits["train"] = all_tiles[val_end:]

    for s in ("train", "val", "test"):
        pos = sum(1 for t in splits[s] if t[3])
        neg = len(splits[s]) - pos
        print(f"  {s:>5}: {len(splits[s]):>5} tiles ({pos} positive, {neg} negative)")

    # ── PHASE 3: Save tiles + augment train ───────────────────────
    print(f"\nPhase 3: Saving tiles...")
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "test": 0, "train_aug": 0, "total_labels": 0}

    def save_tile(tile_img, label_lines, name, split):
        cv2.imwrite(str(output_dir / "images" / split / f"{name}.png"), tile_img)
        (output_dir / "labels" / split / f"{name}.txt").write_text("\n".join(label_lines))
        stats[split] += 1
        stats["total_labels"] += len(label_lines)

    for split_name, tile_list in splits.items():
        for tile_img, tile_labels, tile_name, is_positive in tile_list:
            save_tile(tile_img, tile_labels, tile_name, split_name)

            # Augment train tiles only
            if args.augment and split_name == "train":
                for aug_name, aug_fn in AUGMENTATIONS:
                    aug_img, aug_labels = aug_fn(tile_img, tile_labels)
                    save_tile(aug_img, aug_labels, f"{tile_name}_{aug_name}", "train")
                    stats["train_aug"] += 1

    # ── Write data.yaml ───────────────────────────────────────────
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: wall\n"
    )

    total_train = stats["train"] + stats["train_aug"]
    print(f"\n{'='*60}")
    print(f"Dataset ready: {output_dir}")
    print(f"  Train: {stats['train']} base + {stats['train_aug']} augmented = {total_train}")
    print(f"  Val:   {stats['val']}")
    print(f"  Test:  {stats['test']}")
    print(f"  Total labels: {stats['total_labels']}")
    print(f"  data.yaml: {yaml_path}")

    # Per-split positive/negative breakdown
    print(f"\nPositive/Negative breakdown:")
    for split_name, tile_list in splits.items():
        pos = sum(1 for t in tile_list if t[3])
        neg = len(tile_list) - pos
        total = len(tile_list)
        if split_name == "train" and args.augment:
            total_with_aug = total + total * len(AUGMENTATIONS)
            pos_with_aug = pos + pos * len(AUGMENTATIONS)
            neg_with_aug = neg + neg * len(AUGMENTATIONS)
            print(f"  {split_name:>5}: {pos_with_aug:>5} positive, {neg_with_aug:>5} negative (total {total_with_aug})")
        else:
            print(f"  {split_name:>5}: {pos:>5} positive, {neg:>5} negative (total {total})")

    print(f"\nTo train:")
    print(f"  yolo segment train data={yaml_path} model=yolo11n-seg.pt epochs=100 imgsz=640 batch=8")


if __name__ == "__main__":
    main()
