"""
augment_dataset.py
------------------
Data augmentation pipeline for YOLO wall segmentation training.

Steps:
  1. Load images + YOLO-seg labels from yolo_dataset/
  2. Convert to grayscale with CLAHE (preserves thin lines and dashes)
  3. Tile into 640x640 patches with overlap
  4. Keep only tiles that contain wall annotations
  5. Validate labels (normalised, ≥3 pts, non-zero area)
  6. Augment each kept tile:
       Geometric  : rotate 90 / 180 / 270  (labels transformed)
       Photometric: bright, dark, overexposed, underexposed,
                    gaussian noise, gaussian blur, motion blur, salt&pepper

Output:
    augmented_dataset/
    ├── images/train/
    ├── labels/train/
    └── dataset.yaml

Usage:
    python3.12 augment_dataset.py
    python3.12 augment_dataset.py --src yolo_dataset --out augmented_dataset --tile 640 --stride 512
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Grayscale + CLAHE  (preserves fine lines and dashes)
# ---------------------------------------------------------------------------

def to_grayscale_enhanced(bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale and apply CLAHE to boost contrast of thin lines.
    Returns a 3-channel BGR image (gray values in all channels) so it is
    compatible with the rest of the pipeline.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def parse_label_file(path: Path) -> list[np.ndarray]:
    """Return list of (N,2) float arrays in normalised [0,1] coords."""
    polys = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:          # class + ≥3 xy pairs
            continue
        coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
        polys.append(coords)
    return polys


def validate_polygon(pts: np.ndarray, tile_size: int,
                     min_area_px: int = 50) -> bool:
    """Return True if polygon is sane."""
    if len(pts) < 3:
        return False
    if not (np.all(pts >= 0) and np.all(pts <= 1)):
        return False
    abs_pts = (pts * tile_size).astype(np.float32)
    area = cv2.contourArea(abs_pts)
    return area >= min_area_px


def polys_to_label_lines(polys: list[np.ndarray]) -> str:
    lines = []
    for pts in polys:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"0 {coords}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def clip_polygon_to_tile(pts_norm: np.ndarray, tx: int, ty: int,
                          tile_size: int, img_w: int, img_h: int
                          ) -> np.ndarray | None:
    """
    Clip a normalised polygon to a tile and return tile-normalised coords.
    Returns None if the remaining polygon is too small.
    """
    # Absolute pixel coords
    abs_pts = pts_norm * np.array([img_w, img_h], dtype=float)

    # Clip to tile bounds
    clipped = abs_pts.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], tx, tx + tile_size)
    clipped[:, 1] = np.clip(clipped[:, 1], ty, ty + tile_size)

    # Shift to tile-relative
    clipped[:, 0] -= tx
    clipped[:, 1] -= ty

    # Drop duplicate consecutive points
    unique = [clipped[0]]
    for p in clipped[1:]:
        if not np.allclose(p, unique[-1], atol=0.5):
            unique.append(p)
    if len(unique) < 3:
        return None

    rel = np.array(unique, dtype=np.float32)
    area = cv2.contourArea(rel)
    if area < 50:
        return None

    # Normalise to [0,1] within tile
    norm = rel / tile_size
    norm = np.clip(norm, 0.0, 1.0)
    return norm


def tile_image_and_labels(img: np.ndarray, polys: list[np.ndarray],
                           tile_size: int, stride: int
                           ) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Slice image into tile_size×tile_size tiles (step=stride).
    Returns list of (tile_img, tile_polys) for tiles that have ≥1 wall polygon.
    """
    h, w = img.shape[:2]
    results = []

    for ty in range(0, max(1, h - tile_size + 1), stride):
        ty = min(ty, max(0, h - tile_size))
        for tx in range(0, max(1, w - tile_size + 1), stride):
            tx = min(tx, max(0, w - tile_size))

            tile_img = img[ty:ty + tile_size, tx:tx + tile_size]
            # Pad if the image is smaller than tile_size
            ph = tile_size - tile_img.shape[0]
            pw = tile_size - tile_img.shape[1]
            if ph > 0 or pw > 0:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, ph, 0, pw,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )

            tile_polys = []
            for poly in polys:
                clipped = clip_polygon_to_tile(poly, tx, ty, tile_size, w, h)
                if clipped is not None:
                    tile_polys.append(clipped)

            if tile_polys:
                results.append((tile_img, tile_polys))

    return results


# ---------------------------------------------------------------------------
# Geometric augmentations  (labels transform with image)
# ---------------------------------------------------------------------------

def rotate90(img: np.ndarray, polys: list[np.ndarray]
             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """90° clockwise."""
    rot_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot_polys = [np.column_stack([1 - p[:, 1], p[:, 0]]) for p in polys]
    return rot_img, rot_polys


def rotate180(img: np.ndarray, polys: list[np.ndarray]
              ) -> tuple[np.ndarray, list[np.ndarray]]:
    rot_img = cv2.rotate(img, cv2.ROTATE_180)
    rot_polys = [1.0 - p for p in polys]
    return rot_img, rot_polys


def rotate270(img: np.ndarray, polys: list[np.ndarray]
              ) -> tuple[np.ndarray, list[np.ndarray]]:
    """270° clockwise (= 90° counter-clockwise)."""
    rot_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rot_polys = [np.column_stack([p[:, 1], 1 - p[:, 0]]) for p in polys]
    return rot_img, rot_polys


# ---------------------------------------------------------------------------
# Photometric augmentations  (labels unchanged)
# ---------------------------------------------------------------------------

def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """factor > 1 = brighter, < 1 = darker."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """gamma < 1 = brighter (overexposed), gamma > 1 = darker (underexposed)."""
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                     dtype=np.uint8)
    return cv2.LUT(img, table)


def add_gaussian_noise(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def motion_blur(img: np.ndarray, ksize: int = 7) -> np.ndarray:
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, kernel)


def salt_and_pepper(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    out = img.copy()
    n = int(amount * img.size / img.shape[2] if len(img.shape) == 3
            else amount * img.size)
    # Salt
    coords = [np.random.randint(0, d, n) for d in img.shape[:2]]
    out[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, d, n) for d in img.shape[:2]]
    out[coords[0], coords[1]] = 0
    return out


def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return np.clip(cv2.filter2D(img, -1, kernel), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_sample(img: np.ndarray, polys: list[np.ndarray],
                img_out: Path, lbl_out: Path,
                stem: str, tile_size: int) -> int:
    """Validate polys, write image (JPEG) + label. Returns number written."""
    valid = [p for p in polys if validate_polygon(p, tile_size)]
    if not valid:
        return 0
    cv2.imwrite(str(img_out / f"{stem}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    (lbl_out / f"{stem}.txt").write_text(polys_to_label_lines(valid))
    return 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(src: Path, out: Path, tile_size: int, stride: int) -> None:
    img_src = src / "images" / "train"
    lbl_src = src / "labels" / "train"

    img_out = out / "images" / "train"
    lbl_out = out / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_src.glob("*.jpg")) + sorted(img_src.glob("*.png"))
    print(f"Source images : {len(img_paths)}")

    total = 0

    for img_path in img_paths:
        lbl_path = lbl_src / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue

        polys = parse_label_file(lbl_path)
        if not polys:
            continue

        # Step 1 — grayscale + CLAHE
        gray_bgr = to_grayscale_enhanced(bgr)

        # Step 2 — tile
        tiles = tile_image_and_labels(gray_bgr, polys, tile_size, stride)
        if not tiles:
            continue

        print(f"  {img_path.name}: {len(tiles)} wall tiles")

        for t_idx, (tile_img, tile_polys) in enumerate(tiles):
            base = f"{img_path.stem}_t{t_idx:03d}"

            # --- base tile ---
            total += save_sample(tile_img, tile_polys,
                                  img_out, lbl_out, base, tile_size)

            # --- rotations ---
            for angle, fn, suffix in [
                (90,  rotate90,  "_r90"),
                (180, rotate180, "_r180"),
                (270, rotate270, "_r270"),
            ]:
                ri, rp = fn(tile_img, tile_polys)
                total += save_sample(ri, rp, img_out, lbl_out,
                                      base + suffix, tile_size)

            # --- photometric (labels unchanged) ---
            photo_augs = [
                ("_bright",    adjust_brightness(tile_img, 1.4)),
                ("_dark",      adjust_brightness(tile_img, 0.6)),
                ("_overexp",   adjust_gamma(tile_img, 0.5)),
                ("_underexp",  adjust_gamma(tile_img, 2.0)),
                ("_gnoise",    add_gaussian_noise(tile_img, 12.0)),
                ("_gblur",     gaussian_blur(tile_img, 3)),
                ("_mblur",     motion_blur(tile_img, 7)),
                ("_sp",        salt_and_pepper(tile_img, 0.01)),
                ("_sharp",     sharpen(tile_img)),
            ]
            for suffix, aug_img in photo_augs:
                total += save_sample(aug_img, tile_polys,
                                      img_out, lbl_out,
                                      base + suffix, tile_size)

    # dataset.yaml
    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  str(out.resolve()),
            "train": "images/train",
            "val":   "images/train",
            "nc":    1,
            "names": ["wall"],
        }, f, default_flow_style=False)

    print(f"\nDone — {total} augmented tiles saved to {out}/")
    print(f"\nFine-tune command:")
    print(f"  yolo segment train \\")
    print(f"    model=runs/segment/runs_api/yoloseg_wall5/weights/best.pt \\")
    print(f"    data={yaml_path} \\")
    print(f"    epochs=100 imgsz={tile_size} batch=16")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",    default="yolo_dataset",
                    help="Input YOLO dataset directory (default: yolo_dataset)")
    ap.add_argument("--out",    default="augmented_dataset",
                    help="Output directory (default: augmented_dataset)")
    ap.add_argument("--tile",   type=int, default=640,
                    help="Tile size in pixels (default: 640)")
    ap.add_argument("--stride", type=int, default=512,
                    help="Tile stride/step in pixels (default: 512)")
    args = ap.parse_args()

    run(Path(args.src), Path(args.out), args.tile, args.stride)


if __name__ == "__main__":
    main()
