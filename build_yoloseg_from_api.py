from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from MLStructFP.db import DbLoader
from MLStructFP.db.image import RectBinaryImage, RectFloorPhoto, restore_plot_backend


@dataclass
class Sample:
    image: Path
    label: Path


class SlabBinaryImage:
    """
    Matplotlib-based slab mask generator (similar to RectBinaryImage but for slabs).
    """

    def __init__(self, image_size_px: int = 256) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        self._image_size = image_size_px
        self._plot = {}
        self._initialized = False
        self._plt = plt
        self._matplotlib = matplotlib

    def init(self) -> "SlabBinaryImage":
        if self._plt.get_backend() != "agg":
            self._plt.switch_backend("agg")
        self._initialized = True
        self._plot.clear()
        
        return self

    def _get_floor_plot(self, floor, store: bool) -> tuple["object", "object"]:
        floor_id = str(floor.id)
        if floor_id in self._plot:
            return self._plot[floor_id]

        fig = self._plt.figure(frameon=False)
        self._plt.style.use("default")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_aspect(aspect="equal")
        ax.grid(False)

        for s in floor.slab:
            s.plot_matplotlib(ax=ax, color="#000000")

        if store:
            if len(self._plot) >= 2:
                k1 = next(iter(self._plot.keys()))
                f1, _ = self._plot[k1]
                self._plt.close(f1)
                del self._plot[k1]
            self._plot[floor_id] = (fig, ax)

        return fig, ax

    def make_region(self, xmin: float, xmax: float, ymin: float, ymax: float, floor) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("SlabBinaryImage not initialized")

        fig, ax = self._get_floor_plot(floor, store=True)
        self._plt.figure(fig.number)
        ax.set_xlim(min(xmin, xmax), max(xmin, xmax))
        ax.set_ylim(min(ymin, ymax), max(ymin, ymax))

        buf = self._plt.ioff()
        from io import BytesIO

        ram = BytesIO()
        self._plt.savefig(ram, format="png", dpi=100, bbox_inches="tight", transparent=False)
        ram.seek(0)
        im = Image.open(ram).convert("RGB").resize((self._image_size, self._image_size))
        ram.close()
        if buf:
            self._plt.ion()

        return np.array(im, dtype=np.uint8)

    def close(self) -> None:
        if not self._initialized:
            return
        for f in self._plot.keys():
            fig, _ = self._plot[f]
            self._plt.close(fig)
        self._plot.clear()
        self._initialized = False


def _mask_to_polygons(mask: np.ndarray, *, min_area: float, approx_eps: float) -> list[list[tuple[float, float]]]:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    # RectBinaryImage produces 0/1 masks; treat any non-zero as foreground.
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[tuple[float, float]]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        eps = approx_eps * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 3:
            continue
        poly = [(float(p[0][0]), float(p[0][1])) for p in approx]
        polygons.append(poly)
    return polygons


def _write_yolo_label(label_path: Path, class_id: int, polygons: Iterable[list[tuple[float, float]]], w: int, h: int) -> None:
    lines: list[str] = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        coords: list[str] = []
        for x, y in poly:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _sample_name(prefix: str, floor_id: int, obj_id: int, angle: int, idx: int) -> str:
    return f"{prefix}_floor{floor_id}_id{obj_id}_rot{angle}_i{idx:04d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build YOLO‑seg dataset using MLStructFP API.")
    ap.add_argument("--fp-json", type=Path, required=True, help="Path to fp.json")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dataset directory")
    ap.add_argument("--image-size", type=int, default=512, help="Output crop size (px)")
    ap.add_argument("--crop-length", type=float, default=8.0, help="Crop length (m) from center")
    ap.add_argument("--train-split", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--rotations", type=str, default="0", help="Comma‑separated degrees, e.g. 0,90,180,270")
    ap.add_argument("--max-rects", type=int, default=0, help="Limit rect samples (0 = all)")
    ap.add_argument("--max-slabs", type=int, default=0, help="Limit slab samples (0 = all)")
    ap.add_argument("--no-rects", action="store_true", help="Disable rect (wall) samples")
    ap.add_argument("--no-slabs", action="store_true", help="Disable slab samples")
    ap.add_argument("--min-area", type=float, default=25.0, help="Min polygon area in px^2")
    ap.add_argument("--approx-eps", type=float, default=0.002, help="Polygon approx epsilon ratio")
    args = ap.parse_args()

    if not args.fp_json.exists():
        raise SystemExit(f"fp.json not found: {args.fp_json}")

    out_dir = args.out_dir
    img_train = out_dir / "images" / "train"
    img_val = out_dir / "images" / "val"
    lbl_train = out_dir / "labels" / "train"
    lbl_val = out_dir / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    rotations = [int(r.strip()) for r in args.rotations.split(",") if r.strip()]
    if not rotations:
        rotations = [0]

    db = DbLoader(str(args.fp_json))
    photo = RectFloorPhoto(image_size_px=int(args.image_size))
    rect_mask = RectBinaryImage(image_size_px=int(args.image_size)).init()
    slab_mask = SlabBinaryImage(image_size_px=int(args.image_size)).init()

    samples: list[Sample] = []
    rect_count = 0
    slab_count = 0
    include_rects = not args.no_rects
    include_slabs = not args.no_slabs

    for floor in db.floors:
        for angle in rotations:
            floor.mutate(angle=angle, sx=1, sy=1)

            if include_rects:
                for rect in floor.rect:
                    rect_count += 1
                    if args.max_rects > 0 and rect_count > args.max_rects:
                        break
                    try:
                        _, img = photo.make_rect(rect, crop_length=float(args.crop_length))
                        _, mask = rect_mask.make_rect(rect, crop_length=float(args.crop_length))
                    except Exception:
                        # Skip problematic crops (rare geometry/rotation edge cases)
                        continue
                    name = _sample_name("rect", floor.id, rect.id, angle, rect_count)
                    samples.append(_save_sample(name, img, mask, img_train, lbl_train, class_id=0,
                                                min_area=args.min_area, approx_eps=args.approx_eps))

            if include_slabs:
                for slab in floor.slab:
                    slab_count += 1
                    if args.max_slabs > 0 and slab_count > args.max_slabs:
                        break
                    xs = [p.x for p in slab.points]
                    ys = [p.y for p in slab.points]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    pad = float(args.crop_length)
                    try:
                        _, img = photo.make_region(xmin - pad, xmax + pad, ymin - pad, ymax + pad, floor)
                        mask = slab_mask.make_region(xmin - pad, xmax + pad, ymin - pad, ymax + pad, floor)
                    except Exception:
                        # Skip problematic crops (rare geometry/rotation edge cases)
                        continue
                    name = _sample_name("slab", floor.id, slab.id, angle, slab_count)
                    samples.append(_save_sample(name, img, mask, img_train, lbl_train, class_id=1,
                                                min_area=args.min_area, approx_eps=args.approx_eps))

            stop_rects = args.max_rects > 0 and rect_count >= args.max_rects
            stop_slabs = args.max_slabs > 0 and slab_count >= args.max_slabs
            if (stop_rects or not include_rects) and (stop_slabs or not include_slabs):
                break

    rect_mask.close()
    slab_mask.close()
    restore_plot_backend()

    # Split train/val
    rnd = random.Random(int(args.seed))
    rnd.shuffle(samples)
    n_train = int(len(samples) * float(args.train_split))
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    for s in val_samples:
        s.image.replace(img_val / s.image.name)
        s.label.replace(lbl_val / s.label.name)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: wall\n  1: slab\n",
        encoding="utf-8",
    )
    manifest = {
        "samples_total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "rect_samples": rect_count,
        "slab_samples": slab_count,
        "rotations": rotations,
        "image_size": args.image_size,
        "crop_length": args.crop_length,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_sample(
    name: str,
    img: np.ndarray,
    mask: np.ndarray,
    img_dir: Path,
    lbl_dir: Path,
    *,
    class_id: int,
    min_area: float,
    approx_eps: float,
) -> Sample:
    image_path = img_dir / f"{name}.png"
    label_path = lbl_dir / f"{name}.txt"
    Image.fromarray(img).save(image_path)

    polygons = _mask_to_polygons(mask, min_area=min_area, approx_eps=approx_eps)
    h, w = img.shape[:2]
    _write_yolo_label(label_path, class_id, polygons, w, h)
    return Sample(image=image_path, label=label_path)


if __name__ == "__main__":
    main()
