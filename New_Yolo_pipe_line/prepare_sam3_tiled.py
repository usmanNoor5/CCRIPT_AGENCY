#!/usr/bin/env python3
"""
prepare_sam3_tiled.py — Tile + augment COCO dataset for SAM3 training.

1. Read COCO annotations from train/
2. Binarize images
3. Tile into 1008x1008 patches with overlap, clip COCO annotations
4. Augment train tiles (hflip, vflip, rot90, rot180, rot270, noise)
5. Shuffle all tiles, split 80/20 train/valid
6. Save as COCO format for SAM3

Usage:
    python3.12 prepare_sam3_tiled.py
    python3.12 prepare_sam3_tiled.py --tile-size 1008 --overlap 100
"""

import argparse
import json
import random
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np


def binarize_image(img: np.ndarray, block_size: int = 51, C: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    inverted = cv2.bitwise_not(binary)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def clip_annotation_to_tile(ann, tx, ty, tw, th, min_area=100.0):
    """Clip a COCO annotation's bbox and segmentation to a tile region.
    Returns new annotation dict or None if clipped away."""
    # Clip bbox
    bx, by, bw, bh = ann["bbox"]
    # Convert to absolute coords
    x1 = max(bx - tx, 0)
    y1 = max(by - ty, 0)
    x2 = min(bx + bw - tx, tw)
    y2 = min(by + bh - ty, th)

    if x2 - x1 < 3 or y2 - y1 < 3:
        return None

    new_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    area = new_bbox[2] * new_bbox[3]
    if area < min_area:
        return None

    # Clip segmentation
    new_segs = []
    for seg in ann.get("segmentation", []):
        if len(seg) < 6:
            continue
        pts = np.array(seg, dtype=float).reshape(-1, 2)
        pts[:, 0] -= tx
        pts[:, 1] -= ty
        pts[:, 0] = np.clip(pts[:, 0], 0, tw)
        pts[:, 1] = np.clip(pts[:, 1], 0, th)

        # Check if clipped polygon is valid
        x_range = pts[:, 0].max() - pts[:, 0].min()
        y_range = pts[:, 1].max() - pts[:, 1].min()
        if x_range < 3 or y_range < 3:
            continue

        # Remove duplicate consecutive points
        keep = [0]
        for i in range(1, len(pts)):
            if not np.allclose(pts[i], pts[keep[-1]], atol=0.5):
                keep.append(i)
        pts = pts[keep]
        if len(pts) < 3:
            continue

        new_segs.append(pts.flatten().tolist())

    if not new_segs and not new_bbox:
        return None

    new_ann = dict(ann)
    new_ann["bbox"] = new_bbox
    new_ann["area"] = float(area)
    new_ann["segmentation"] = new_segs
    return new_ann


# ── Augmentation helpers (image + COCO annotations) ──────────────
def aug_hflip(img, anns, tw, th):
    flipped = cv2.flip(img, 1)
    new_anns = []
    for ann in anns:
        a = deepcopy(ann)
        bx, by, bw, bh = a["bbox"]
        a["bbox"] = [float(tw - bx - bw), float(by), float(bw), float(bh)]
        new_segs = []
        for seg in a.get("segmentation", []):
            pts = np.array(seg).reshape(-1, 2)
            pts[:, 0] = tw - pts[:, 0]
            new_segs.append(pts.flatten().tolist())
        a["segmentation"] = new_segs
        new_anns.append(a)
    return flipped, new_anns


def aug_vflip(img, anns, tw, th):
    flipped = cv2.flip(img, 0)
    new_anns = []
    for ann in anns:
        a = deepcopy(ann)
        bx, by, bw, bh = a["bbox"]
        a["bbox"] = [float(bx), float(th - by - bh), float(bw), float(bh)]
        new_segs = []
        for seg in a.get("segmentation", []):
            pts = np.array(seg).reshape(-1, 2)
            pts[:, 1] = th - pts[:, 1]
            new_segs.append(pts.flatten().tolist())
        a["segmentation"] = new_segs
        new_anns.append(a)
    return flipped, new_anns


def aug_rot90(img, anns, tw, th):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_anns = []
    for ann in anns:
        a = deepcopy(ann)
        bx, by, bw, bh = a["bbox"]
        a["bbox"] = [float(th - by - bh), float(bx), float(bh), float(bw)]
        new_segs = []
        for seg in a.get("segmentation", []):
            pts = np.array(seg).reshape(-1, 2)
            new_pts = np.column_stack([th - pts[:, 1], pts[:, 0]])
            new_segs.append(new_pts.flatten().tolist())
        a["segmentation"] = new_segs
        new_anns.append(a)
    return rotated, new_anns


def aug_rot180(img, anns, tw, th):
    rotated = cv2.rotate(img, cv2.ROTATE_180)
    new_anns = []
    for ann in anns:
        a = deepcopy(ann)
        bx, by, bw, bh = a["bbox"]
        a["bbox"] = [float(tw - bx - bw), float(th - by - bh), float(bw), float(bh)]
        new_segs = []
        for seg in a.get("segmentation", []):
            pts = np.array(seg).reshape(-1, 2)
            pts[:, 0] = tw - pts[:, 0]
            pts[:, 1] = th - pts[:, 1]
            new_segs.append(pts.flatten().tolist())
        a["segmentation"] = new_segs
        new_anns.append(a)
    return rotated, new_anns


def aug_rot270(img, anns, tw, th):
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_anns = []
    for ann in anns:
        a = deepcopy(ann)
        bx, by, bw, bh = a["bbox"]
        a["bbox"] = [float(by), float(tw - bx - bw), float(bh), float(bw)]
        new_segs = []
        for seg in a.get("segmentation", []):
            pts = np.array(seg).reshape(-1, 2)
            new_pts = np.column_stack([pts[:, 1], tw - pts[:, 0]])
            new_segs.append(new_pts.flatten().tolist())
        a["segmentation"] = new_segs
        new_anns.append(a)
    return rotated, new_anns


def aug_noise(img, anns, tw, th, intensity=15):
    noise = np.random.randint(-intensity, intensity + 1, img.shape, dtype=np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy, deepcopy(anns)


AUGMENTATIONS = [
    ("hflip", aug_hflip),
    ("vflip", aug_vflip),
    ("rot90", aug_rot90),
    ("rot180", aug_rot180),
    ("rot270", aug_rot270),
    ("noise", aug_noise),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="train")
    parser.add_argument("--output", default="sam3_tiled_dataset")
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    coco_json = input_dir / "_annotations.coco.json"

    with open(coco_json) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}
    img_anns = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    tile_sz = args.tile_size
    stride = tile_sz - args.overlap

    print(f"Input:   {input_dir} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)")
    print(f"Output:  {output_dir}")
    print(f"Tile:    {tile_sz}x{tile_sz}, overlap={args.overlap}, stride={stride}")
    print(f"Augment: {'Yes' if args.augment else 'No'}")
    print()

    # Phase 1: Tile all images
    # Each tile: (img, coco_anns, tile_name, is_positive)
    all_tiles = []

    for img_id in id_to_img:
        img_info = id_to_img[img_id]
        fname = img_info["file_name"]
        src = input_dir / fname
        if not src.exists():
            continue

        img = cv2.imread(str(src))
        if img is None:
            continue

        h, w = img.shape[:2]
        img = binarize_image(img)

        anns = img_anns.get(img_id, [])
        # Ensure bbox values are float
        for ann in anns:
            ann["bbox"] = [float(x) for x in ann["bbox"]]
            if "area" in ann:
                ann["area"] = float(ann["area"])

        stem = Path(fname).stem
        pos_count = 0
        neg_count = 0

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

                # Clip annotations to tile
                tile_anns = []
                for ann in anns:
                    clipped = clip_annotation_to_tile(ann, x, y, tile_sz, tile_sz)
                    if clipped:
                        tile_anns.append(clipped)

                tile_name = f"{stem}_r{y:05d}_c{x:05d}"
                is_positive = len(tile_anns) > 0
                all_tiles.append((tile, tile_anns, tile_name, is_positive))

                if is_positive:
                    pos_count += 1
                else:
                    neg_count += 1

        print(f"  {fname} ({w}x{h}) → {pos_count}+ {neg_count}- = {pos_count + neg_count} tiles")

    total_pos = sum(1 for t in all_tiles if t[3])
    total_neg = sum(1 for t in all_tiles if not t[3])
    print(f"\nTotal tiles: {len(all_tiles)} ({total_pos} positive, {total_neg} negative)")

    # Phase 2: Shuffle and split
    random.shuffle(all_tiles)
    split_idx = int(len(all_tiles) * args.train_split)
    splits = {
        "train": all_tiles[:split_idx],
        "valid": all_tiles[split_idx:],
    }

    for s in ("train", "valid"):
        pos = sum(1 for t in splits[s] if t[3])
        neg = len(splits[s]) - pos
        print(f"  {s:>5}: {len(splits[s])} tiles ({pos}+ {neg}-)")

    # Phase 3: Save tiles + augment train, output as COCO format
    print(f"\nPhase 3: Saving...")

    for split_name in ("train", "valid"):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        coco_images = []
        coco_annotations = []
        img_id_counter = 1
        ann_id_counter = 1

        tile_list = splits[split_name]

        def save_tile(tile_img, tile_anns, name):
            nonlocal img_id_counter, ann_id_counter
            fname = f"{name}.png"
            cv2.imwrite(str(split_dir / fname), tile_img)

            coco_images.append({
                "id": img_id_counter,
                "file_name": fname,
                "width": tile_sz,
                "height": tile_sz,
            })

            for ann in tile_anns:
                new_ann = deepcopy(ann)
                new_ann["id"] = ann_id_counter
                new_ann["image_id"] = img_id_counter
                new_ann["category_id"] = coco["categories"][0]["id"]
                new_ann["iscrowd"] = 0
                coco_annotations.append(new_ann)
                ann_id_counter += 1

            img_id_counter += 1

        for tile_img, tile_anns, tile_name, is_positive in tile_list:
            save_tile(tile_img, tile_anns, tile_name)

            # Augment train only
            if args.augment and split_name == "train":
                for aug_name, aug_fn in AUGMENTATIONS:
                    aug_img, aug_anns = aug_fn(tile_img, tile_anns, tile_sz, tile_sz)
                    save_tile(aug_img, aug_anns, f"{tile_name}_{aug_name}")

        # Write COCO JSON
        coco_out = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco["categories"],
        }
        json_path = split_dir / "_annotations.coco.json"
        with open(json_path, "w") as f:
            json.dump(coco_out, f)

        print(f"  {split_name}: {len(coco_images)} images, {len(coco_annotations)} annotations → {split_dir}")

    print(f"\nDataset ready: {output_dir}")


if __name__ == "__main__":
    main()
