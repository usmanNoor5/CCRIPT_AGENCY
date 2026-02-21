"""
sam2_relabel.py
---------------
Use existing YOLO wall masks (inference_results/*_walls_mask.png) to prompt
SAM2, get cleaner segmentation masks, and export a YOLO-seg training dataset
so you can fine-tune best.pt on SAM2-quality labels.

Pipeline:
    # 1. Relabel with SAM2
    python3.12 sam2_relabel.py \
        --masks-dir  testing/inference_results/ \
        --images-dir testing/tiles/ \
        --out-dir    sam2_dataset/

    # 2. Fine-tune YOLO on the new labels
    yolo segment train \
        model=runs/segment/runs_api/yoloseg_wall5/weights/best.pt \
        data=sam2_dataset/dataset.yaml \
        epochs=50 imgsz=640 batch=8

Requirements:
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install ultralytics
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

testing/images# ---------------------------------------------------------------------------
# SAM2 import
# ---------------------------------------------------------------------------
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_original_image(mask_path: Path, images_dir: Path) -> Path | None:
    """
    Mask name: page_003_walls_mask.png  → look for page_003*.png in images_dir
    """
    # Strip _walls_mask suffix to get the base page id (e.g. "page_003")
    stem = mask_path.stem.replace("_walls_mask", "")
    candidates = list(images_dir.glob(f"{stem}*.png")) + \
                 list(images_dir.glob(f"{stem}*.jpg"))
    # Prefer exact stem_cropped match, then any match
    exact = [c for c in candidates if c.stem == f"{stem}_cropped"]
    return exact[0] if exact else (candidates[0] if candidates else None)


def mask_to_bboxes(mask_binary: np.ndarray, min_area: int = 200):
    """
    Find connected wall blobs in binary mask → list of (x1,y1,x2,y2) bboxes.
    Each blob is one wall segment that will be sent to SAM2 as a prompt.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )
    bboxes = []
    for i in range(1, num):   # skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def mask_to_polygon(mask: np.ndarray, img_h: int, img_w: int,
                    min_area: int = 100) -> str | None:
    """Binary mask → normalised YOLO polygon string."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    epsilon = 0.005 * cv2.arcLength(cnt, closed=True)
    simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)
    pts = simplified.reshape(-1, 2).astype(float)
    pts[:, 0] /= img_w
    pts[:, 1] /= img_h
    return " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def relabel(masks_dir: Path, images_dir: Path, out_dir: Path,
            min_area: int = 200, device: str = "cuda") -> None:

    if not SAM2_AVAILABLE:
        raise RuntimeError(
            "SAM2 not installed.\n"
            "Run: pip install git+https://github.com/facebookresearch/sam2.git"
        )

    images_out = out_dir / "images" / "train"
    labels_out = out_dir / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(masks_dir.glob("*_walls_mask.png"))
    if not mask_paths:
        raise SystemExit(f"No *_walls_mask.png files found in {masks_dir}")

    print(f"Found {len(mask_paths)} mask(s).\n")
    print("Loading SAM2 (facebook/sam2-hiera-tiny) ...")
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
    print("SAM2 ready.\n")

    total_labels = 0

    for mask_path in mask_paths:
        orig_path = find_original_image(mask_path, images_dir)
        if orig_path is None:
            print(f"  [skip] no original image found for {mask_path.name}")
            continue

        img_bgr = cv2.imread(str(orig_path))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask_raw is None:
            print(f"  [skip] cannot read {mask_path.name} or {orig_path.name}")
            continue

        _, mask_bin = cv2.threshold(mask_raw, 127, 1, cv2.THRESH_BINARY)
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Extract individual wall blobs from the YOLO mask
        bboxes = mask_to_bboxes(mask_bin.astype(np.uint8), min_area=min_area)
        if not bboxes:
            print(f"  [skip] no wall blobs in {mask_path.name}")
            continue

        print(f"  {orig_path.name}  →  {len(bboxes)} wall blobs → prompting SAM2 ...")

        predictor.set_image(img_rgb)

        label_lines = []
        for (x1, y1, x2, y2) in bboxes:
            input_box = np.array([[x1, y1, x2, y2]], dtype=float)

            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=False,
                )

            best_mask = masks[0]   # (H, W) bool
            poly = mask_to_polygon(best_mask, h, w, min_area=min_area)
            if poly is None:
                continue

            label_lines.append(f"0 {poly}")   # class 0 = wall
            total_labels += 1

        if not label_lines:
            print(f"    → no valid SAM2 masks, skipping")
            continue

        # Write image + label
        shutil.copy2(orig_path, images_out / orig_path.name)
        lbl_path = labels_out / (orig_path.stem + ".txt")
        lbl_path.write_text("\n".join(label_lines))
        print(f"    → wrote {len(label_lines)} labels to {lbl_path.name}")

    # dataset.yaml
    dataset_yaml = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/train",
        "nc":    1,
        "names": ["wall"],
    }
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"\nDone — {total_labels} total SAM2 masks written.")
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nFine-tune command:")
    print(f"  yolo segment train \\")
    print(f"    model=runs/segment/runs_api/yoloseg_wall5/weights/best.pt \\")
    print(f"    data={yaml_path} \\")
    print(f"    epochs=50 imgsz=640 batch=8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir",  default="testing/inference_results/",
                    help="Folder containing *_walls_mask.png files")
    ap.add_argument("--images-dir", default="testing/tiles/",
                    help="Folder containing original floor plan images")
    ap.add_argument("--out-dir",    default="sam2_dataset/",
                    help="Output dataset directory")
    ap.add_argument("--min-area",   type=int, default=200,
                    help="Min blob area in pixels (default 200)")
    ap.add_argument("--device",     default="cuda")
    args = ap.parse_args()

    relabel(
        masks_dir=Path(args.masks_dir),
        images_dir=Path(args.images_dir),
        out_dir=Path(args.out_dir),
        min_area=args.min_area,
        device=args.device,
    )


if __name__ == "__main__":
    main()
