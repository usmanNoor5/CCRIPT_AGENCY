from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from MLStructFP.db import DbLoader
from MLStructFP.db.image import RectBinaryImage, RectFloorPhoto, restore_plot_backend

NUM_RECT_WORKERS = min(5, max(1, multiprocessing.cpu_count() - 1))
NUM_POSTPROCESS_WORKERS = min(5, max(1, multiprocessing.cpu_count() - 1))


@dataclass
class Sample:
    image: Path
    label: Path


class AsyncSaver:
    """Overlap disk writes with computation using a thread pool."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []

    def submit(self, name, img, mask, img_dir, lbl_dir, *, class_id, min_area, approx_eps):
        fut = self._pool.submit(
            _save_sample, name, img, mask, img_dir, lbl_dir,
            class_id=class_id, min_area=min_area, approx_eps=approx_eps,
        )
        self._futures.append(fut)
        return fut

    def collect_all(self) -> list[Sample]:
        results = []
        for f in self._futures:
            results.append(f.result())
        self._futures.clear()
        return results

    def shutdown(self):
        self._pool.shutdown(wait=True)


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
        # Keep full axis-aligned render; avoid tight-cropping that can distort geometry.
        self._plt.savefig(ram, format="png", dpi=100, pad_inches=0, transparent=False)
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
        # SlabBinaryImage returns RGB renders with dark slab geometry on light
        # background, so invert-threshold dark pixels as foreground.
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
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


def _parse_yolo_polygons_px(label_path: Path, w: int, h: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not label_path.exists():
        return polygons
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue
        pts: list[list[int]] = []
        for i in range(0, len(coords), 2):
            x = float(coords[i]) * w
            y = float(coords[i + 1]) * h
            xi = int(round(min(max(x, 0.0), float(w - 1))))
            yi = int(round(min(max(y, 0.0), float(h - 1))))
            pts.append([xi, yi])
        if len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.int32))
    return polygons


def _apply_label_style_to_image(
    image_path: Path,
    label_path: Path,
    *,
    edge_thickness: int,
    outer_white_px: int,
    bin_block_size: int,
    bin_c: int,
) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    h, w = img.shape
    polygons = _parse_yolo_polygons_px(label_path, w, h)
    if not polygons:
        return False

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    out = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        int(bin_block_size),
        int(bin_c),
    )

    label_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(label_mask, [poly], 255)

    # Keep background from binarized image, then force label interior white.
    out[label_mask > 0] = 255

    # Thin black label boundary (configurable thickness).
    for poly in polygons:
        cv2.polylines(out, [poly.reshape((-1, 1, 2))], isClosed=True, color=0, thickness=int(edge_thickness))

    # White ring outside labels.
    if outer_white_px > 0:
        k = int(outer_white_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(label_mask, kernel, iterations=1)
        outer_ring = cv2.subtract(dilated, label_mask)
        out[outer_ring > 0] = 255

    return bool(cv2.imwrite(str(image_path), out))


def _worker_generate_rects(worker_args: dict) -> list[dict]:
    """Worker process: loads its own DB, processes assigned floors, saves to disk."""
    fp_json = worker_args["fp_json"]
    floor_indices = worker_args["floor_indices"]
    rotations = worker_args["rotations"]
    image_size = worker_args["image_size"]
    crop_length = worker_args["crop_length"]
    img_dir = Path(worker_args["img_dir"])
    lbl_dir = Path(worker_args["lbl_dir"])
    max_rects = worker_args["max_rects"]
    min_area = worker_args["min_area"]
    approx_eps = worker_args["approx_eps"]
    rect_offset = worker_args["rect_offset"]
    worker_id = worker_args["worker_id"]

    db = DbLoader(fp_json)
    photo = RectFloorPhoto(image_size_px=image_size)
    photo.save = False
    rect_mask = RectBinaryImage(image_size_px=image_size).init()
    rect_mask.save = False

    all_floors = db.floors
    results: list[dict] = []
    rect_count = 0
    saver = AsyncSaver(max_workers=2)

    unique_rects = 0  # counts distinct rects (before rotation)
    for fi in floor_indices:
        if fi >= len(all_floors):
            continue
        floor = all_floors[fi]
        # Cap by unique rects, not total samples — each rect gets all rotations
        if max_rects > 0 and (rect_offset + unique_rects) >= max_rects:
            break
        for angle in rotations:
            floor.mutate(angle=angle, sx=1, sy=1)
            # Clear RectBinaryImage cache — it caches by floor.id only,
            # ignoring mutations. Without this, rotated rects get empty masks.
            rect_mask.close()
            rect_mask = RectBinaryImage(image_size_px=image_size).init()
            rect_mask.save = False
            for ri, rect in enumerate(floor.rect):
                if max_rects > 0 and (rect_offset + ri) >= max_rects:
                    break
                try:
                    _, img = photo.make_rect(rect, crop_length=crop_length)
                    _, mask = rect_mask.make_rect(rect, crop_length=crop_length)
                except Exception:
                    continue
                rect_count += 1
                idx = rect_offset + rect_count
                name = _sample_name("rect", floor.id, rect.id, angle, idx)
                saver.submit(
                    name, img, mask, img_dir, lbl_dir,
                    class_id=0, min_area=min_area, approx_eps=approx_eps,
                )
                if rect_count % 200 == 0:
                    print(f"  [worker {worker_id}] {rect_count} rects", flush=True)
        # Count unique rects for this floor (just once, not per rotation)
        unique_rects += len(floor.rect)

    samples = saver.collect_all()
    saver.shutdown()
    rect_mask.close()
    restore_plot_backend()

    print(f"  [worker {worker_id}] done — {rect_count} rects", flush=True)
    return [{"image": str(s.image), "label": str(s.label)} for s in samples]


def _postprocess_one(args_tuple: tuple) -> str:
    """Worker function for parallel post-processing."""
    img_p, lbl_p, edge_thickness, outer_white_px, bin_block_size, bin_c = args_tuple
    ip = Path(img_p)
    lp = Path(lbl_p)
    ok = _apply_label_style_to_image(
        ip, lp,
        edge_thickness=edge_thickness,
        outer_white_px=outer_white_px,
        bin_block_size=bin_block_size,
        bin_c=bin_c,
    )
    if ok:
        return "processed"
    elif not lp.exists() or not lp.read_text(encoding="utf-8").strip():
        return "skipped"
    return "failed"


def _postprocess_dataset_images(
    out_dir: Path,
    *,
    edge_thickness: int,
    outer_white_px: int,
    bin_block_size: int,
    bin_c: int,
) -> dict[str, int]:
    tasks = []
    for split in ("train", "val"):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        if not img_dir.exists():
            continue
        for image_path in img_dir.glob("*.png"):
            label_path = lbl_dir / f"{image_path.stem}.txt"
            tasks.append((
                str(image_path), str(label_path),
                edge_thickness, outer_white_px, bin_block_size, bin_c,
            ))

    if not tasks:
        return {"processed": 0, "skipped": 0, "failed": 0}

    print(f"Post-processing {len(tasks)} images with {NUM_POSTPROCESS_WORKERS} workers ...")
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=NUM_POSTPROCESS_WORKERS) as pool:
        for result in pool.map(_postprocess_one, tasks, chunksize=64):
            stats[result] += 1
    return stats


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
    ap.add_argument("--no-rects", action="store_true", help="Disable rect (wall) samples")
    ap.add_argument("--min-area", type=float, default=25.0, help="Min polygon area in px^2")
    ap.add_argument("--approx-eps", type=float, default=0.002, help="Polygon approx epsilon ratio")
    ap.add_argument("--style-edge-thickness", type=int, default=1, help="Label boundary thickness in px")
    ap.add_argument("--style-outer-white-px", type=int, default=3, help="White ring width outside labels")
    ap.add_argument("--style-bin-block-size", type=int, default=31, help="Adaptive threshold block size (odd)")
    ap.add_argument("--style-bin-c", type=int, default=5, help="Adaptive threshold C value")
    args = ap.parse_args()

    if not args.fp_json.exists():
        raise SystemExit(f"fp.json not found: {args.fp_json}")
    if args.style_edge_thickness < 1:
        raise SystemExit("--style-edge-thickness must be >= 1")
    if args.style_outer_white_px < 0:
        raise SystemExit("--style-outer-white-px must be >= 0")
    if args.style_bin_block_size < 3 or args.style_bin_block_size % 2 == 0:
        raise SystemExit("--style-bin-block-size must be odd and >= 3")

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

    include_rects = not args.no_rects
    samples: list[Sample] = []
    rect_count = 0

    if include_rects:
        # Count floors and pre-compute how many to process
        db = DbLoader(str(args.fp_json))
        num_floors = len(db.floors)

        # Determine floor cutoff if --max-rects is set
        if args.max_rects > 0:
            cumulative = 0
            cutoff = num_floors
            for i, floor in enumerate(db.floors):
                rects_in_floor = len(floor.rect) * len(rotations)
                cumulative += rects_in_floor
                if cumulative >= args.max_rects:
                    cutoff = i + 1
                    break
            floor_indices = list(range(cutoff))
        else:
            floor_indices = list(range(num_floors))

        del db  # free memory before spawning workers

        n_workers = min(NUM_RECT_WORKERS, len(floor_indices))
        # Round-robin partition for balanced load
        partitions: list[list[int]] = [[] for _ in range(n_workers)]
        for i, fi in enumerate(floor_indices):
            partitions[i % n_workers].append(fi)

        # Build worker args
        worker_args_list = []
        rect_offset = 0
        for wid, part in enumerate(partitions):
            worker_args_list.append({
                "fp_json": str(args.fp_json),
                "floor_indices": part,
                "rotations": rotations,
                "image_size": int(args.image_size),
                "crop_length": float(args.crop_length),
                "img_dir": str(img_train),
                "lbl_dir": str(lbl_train),
                "max_rects": int(args.max_rects),
                "min_area": args.min_area,
                "approx_eps": args.approx_eps,
                "rect_offset": rect_offset,
                "worker_id": wid,
            })

        print(f"Generating rects with {n_workers} workers across {len(floor_indices)} floors ...")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = [pool.submit(_worker_generate_rects, wa) for wa in worker_args_list]
            all_results: list[dict] = []
            for f in futures:
                all_results.extend(f.result())

        samples = [Sample(image=Path(r["image"]), label=Path(r["label"])) for r in all_results]
        rect_count = len(samples)
        print(f"Rect generation done: {rect_count} samples")

    # Split train/val
    rnd = random.Random(int(args.seed))
    rnd.shuffle(samples)
    n_train = int(len(samples) * float(args.train_split))
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    for s in val_samples:
        s.image.replace(img_val / s.image.name)
        s.label.replace(lbl_val / s.label.name)

    # Post-process train/val images using their labels:
    # - label interior white
    # - thin black boundary
    # - white outer ring outside labels
    style_stats = _postprocess_dataset_images(
        out_dir,
        edge_thickness=int(args.style_edge_thickness),
        outer_white_px=int(args.style_outer_white_px),
        bin_block_size=int(args.style_bin_block_size),
        bin_c=int(args.style_bin_c),
    )

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: wall\n",
        encoding="utf-8",
    )
    manifest = {
        "samples_total": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "rect_samples": rect_count,
        "rotations": rotations,
        "image_size": args.image_size,
        "crop_length": args.crop_length,
        "style_postprocess": {
            "edge_thickness": int(args.style_edge_thickness),
            "outer_white_px": int(args.style_outer_white_px),
            "bin_block_size": int(args.style_bin_block_size),
            "bin_c": int(args.style_bin_c),
            "processed": style_stats["processed"],
            "skipped": style_stats["skipped"],
            "failed": style_stats["failed"],
        },
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
    if img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(image_path), img)

    polygons = _mask_to_polygons(mask, min_area=min_area, approx_eps=approx_eps)
    h, w = img.shape[:2]
    _write_yolo_label(label_path, class_id, polygons, w, h)
    return Sample(image=image_path, label=label_path)


if __name__ == "__main__":
    main()
