from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def preprocess_image(
    img_path: Path,
    *,
    mode: str,
    blur: int,
    block_size: int,
    c: int,
) -> "cv2.typing.MatLike":
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    work = norm
    if blur > 0:
        work = cv2.GaussianBlur(norm, (blur, blur), 0)

    if mode == "gray":
        out = gray
    elif mode == "norm":
        out = norm
    elif mode == "bin":
        out = cv2.adaptiveThreshold(
            work,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Keep 3 channels for compatibility with common training pipelines.
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        out = dst / rel
        if p.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess full YOLO-seg dataset images while preserving labels.")
    ap.add_argument("--in-dataset", type=Path, default=Path("dataset_api"), help="Input dataset root")
    ap.add_argument("--out-dataset", type=Path, default=Path("dataset_api_preprocessed"), help="Output dataset root")
    ap.add_argument("--mode", type=str, default="bin", choices=["gray", "norm", "bin"], help="Preprocess mode")
    ap.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel (odd, 0 disables)")
    ap.add_argument("--block-size", type=int, default=31, help="Adaptive threshold block size (odd)")
    ap.add_argument("--c", type=int, default=5, help="Adaptive threshold C value")
    args = ap.parse_args()

    in_ds = args.in_dataset
    out_ds = args.out_dataset
    if not in_ds.exists():
        raise SystemExit(f"Input dataset not found: {in_ds}")
    if args.block_size % 2 == 0 or args.block_size < 3:
        raise SystemExit("--block-size must be odd and >= 3")
    if args.blur < 0 or (args.blur % 2 == 0 and args.blur != 0):
        raise SystemExit("--blur must be 0 or an odd integer")

    # Copy labels and metadata first.
    copy_tree(in_ds / "labels", out_ds / "labels")
    if (in_ds / "manifest.json").exists():
        shutil.copy2(in_ds / "manifest.json", out_ds / "manifest.json")

    in_images = in_ds / "images"
    for split in ("train", "val"):
        src_dir = in_images / split
        dst_dir = out_ds / "images" / split
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in src_dir.glob("*.png"):
            out_img = preprocess_image(
                img_path,
                mode=args.mode,
                blur=args.blur,
                block_size=args.block_size,
                c=args.c,
            )
            cv2.imwrite(str(dst_dir / img_path.name), out_img)

    # Write a fresh data.yaml pointing to out dataset.
    data_yaml = out_ds / "data.yaml"
    data_yaml.write_text(
        f"path: {out_ds.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 2\n"
        "names:\n"
        "  0: wall\n"
        "  1: slab\n",
        encoding="utf-8",
    )

    # Optional metadata file for traceability.
    meta = {
        "source_dataset": str(in_ds.resolve()),
        "mode": args.mode,
        "blur": args.blur,
        "block_size": args.block_size,
        "c": args.c,
    }
    (out_ds / "preprocess_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Preprocessed dataset written to: {out_ds}")


if __name__ == "__main__":
    main()
