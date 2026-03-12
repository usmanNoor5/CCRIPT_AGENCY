#!/usr/bin/env python3
"""
vector_extraction.py — Extract wall-like vector lines from CAD PDF pages.

Input:  PDF file (only pages with vector content are processed).
Output: Per-page PNGs — full page, drawing crop, vector overlay, optional tiles.

Usage:
    python3.12 vector_extraction.py "CHRISTENSEN MODEL.pdf"
    python3.12 vector_extraction.py "CHRISTENSEN MODEL.pdf" --pages 1,3,5
    python3.12 vector_extraction.py "CHRISTENSEN MODEL.pdf" --pages 1-4
    python3.12 vector_extraction.py "CHRISTENSEN MODEL.pdf" --tiles
"""

import argparse
import math
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

# ── Tuneable constants ──────────────────────────────────────────────
SCALE = 2                   # render resolution multiplier
MIN_LINE_WIDTH = 0.08       # pt – ignore hairlines thinner than this
MIN_LINE_LENGTH = 5.0       # pt – ignore tiny line segments
MAX_BRIGHT = 0.55           # stroke colour brightness ceiling (0=black, 1=white)
TILE_SIZE = 640             # px
TILE_OVERLAP = 128          # px

# Text keywords that indicate a right-panel / title-block region.
TITLE_BLOCK_KEYWORDS = [
    "sheet", "scale", "date", "drawn", "checked", "approved",
    "revision", "project", "architect", "engineer", "dwg",
    "no.", "copyright", "client",
]


# ── Helpers ─────────────────────────────────────────────────────────
def _brightness(color) -> float:
    """Return perceived brightness 0..1 of an RGB tuple/list (each 0..1)."""
    if color is None:
        return 1.0  # no stroke → treat as white (skip)
    if isinstance(color, (int, float)):
        return float(color)
    r, g, b = color[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _line_length(p1, p2) -> float:
    """Euclidean length between two fitz.Point objects, in PDF points."""
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def _is_dark(color, threshold=MAX_BRIGHT) -> bool:
    return _brightness(color) <= threshold


def parse_pages(spec: str, total: int) -> list[int]:
    """Parse '1,3,5' or '1-4' or '2-8' into 0-based page indices."""
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a) - 1, int(b) - 1
            indices.update(range(max(0, a), min(b + 1, total)))
        else:
            idx = int(part) - 1
            if 0 <= idx < total:
                indices.add(idx)
    return sorted(indices)


# ── Drawing-region crop ─────────────────────────────────────────────
def find_drawing_crop(page, page_rect, text_words) -> fitz.Rect:
    """
    Estimate the 'drawing only' region by excluding the title-block strip
    (usually the rightmost ~20 % or bottom ~15 % of the page where the
    legend / notes / title block lives).
    """
    pw, ph = page_rect.width, page_rect.height

    # Default: full page
    crop = fitz.Rect(page_rect)

    # Strategy: find words that belong to the title block, take the leftmost
    # x-coordinate of those words, and crop the drawing to everything left of it.
    title_xs = []
    title_ys = []
    for w in text_words:
        x0, y0, x1, y1, text, *_ = w
        if any(kw in text.lower() for kw in TITLE_BLOCK_KEYWORDS):
            title_xs.append(x0)
            title_ys.append(y0)

    if title_xs:
        # If most title-block words are in the right 30 %, crop there.
        right_words = [x for x in title_xs if x > pw * 0.70]
        bottom_words = [y for y in title_ys if y > ph * 0.80]

        if len(right_words) >= 3:
            crop.x1 = min(right_words) - 5
        if len(bottom_words) >= 3:
            crop.y1 = min(bottom_words) - 5

    # Clamp to page
    crop = crop & page_rect
    return crop


# ── Rotation helper ─────────────────────────────────────────────────
def _rotate_point(x, y, rotation, page_width, page_height):
    """Transform a point from unrotated PDF coords to rotated display coords."""
    if rotation == 90:
        return page_height - y, x
    elif rotation == 180:
        return page_width - x, page_height - y
    elif rotation == 270:
        return y, page_width - x
    return x, y  # 0°


# ── Extract filtered line segments ──────────────────────────────────
def extract_wall_lines(page, crop_rect=None):
    """
    Return list of ((x1,y1),(x2,y2)) tuples in rotated display coordinates.
    Only keeps line items that pass wall-like filters.
    """
    drawings = page.get_drawings()
    rotation = page.rotation
    # unrotated page dimensions
    if rotation in (90, 270):
        pw, ph = page.rect.height, page.rect.width
    else:
        pw, ph = page.rect.width, page.rect.height
    lines = []

    for d in drawings:
        stroke = d.get("color")        # stroke colour (RGB tuple, 0..1)
        width = d.get("width") or 0     # stroke width in pt

        # Skip hairlines
        if width < MIN_LINE_WIDTH:
            continue
        # Skip light / invisible strokes
        if not _is_dark(stroke):
            continue

        for item in d["items"]:
            if item[0] != "l":          # only line segments
                continue
            _, p1, p2 = item
            if _line_length(p1, p2) < MIN_LINE_LENGTH:
                continue

            # Transform to rotated coords
            rx1, ry1 = _rotate_point(p1.x, p1.y, rotation, pw, ph)
            rx2, ry2 = _rotate_point(p2.x, p2.y, rotation, pw, ph)

            # If crop supplied, keep only segments that touch it.
            if crop_rect is not None:
                seg_rect = fitz.Rect(
                    min(rx1, rx2), min(ry1, ry2),
                    max(rx1, rx2), max(ry1, ry2),
                )
                if not seg_rect.intersects(crop_rect):
                    continue

            lines.append(((rx1, ry1), (rx2, ry2)))

    return lines


# ── Visualise ───────────────────────────────────────────────────────
def render_page(page, scale=SCALE) -> Image.Image:
    """Render a PDF page to a PIL Image at given scale."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def draw_lines_on_image(img: Image.Image, lines, scale=SCALE,
                        color="lime", width=2, crop_rect=None) -> Image.Image:
    """Draw lines on a copy of img.  lines are in PDF-pt coords."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    ox = (crop_rect.x0 if crop_rect else 0) * scale
    oy = (crop_rect.y0 if crop_rect else 0) * scale
    for (x1, y1), (x2, y2) in lines:
        draw.line(
            (x1 * scale - ox, y1 * scale - oy,
             x2 * scale - ox, y2 * scale - oy),
            fill=color, width=width,
        )
    return out


def crop_image(img: Image.Image, crop_rect, page_rect, scale=SCALE) -> Image.Image:
    """Crop a rendered image to a sub-rect (both in PDF-pt coords)."""
    left = int(crop_rect.x0 * scale)
    top = int(crop_rect.y0 * scale)
    right = int(crop_rect.x1 * scale)
    bottom = int(crop_rect.y1 * scale)
    return img.crop((left, top, right, bottom))


# ── Tiling ──────────────────────────────────────────────────────────
def generate_tiles(img: Image.Image, tile_size=TILE_SIZE,
                   overlap=TILE_OVERLAP) -> list[tuple[Image.Image, int, int]]:
    """Yield (tile_image, row_y, col_x) for overlapping tiles."""
    w, h = img.size
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            box = img.crop((x, y, min(x + tile_size, w), min(y + tile_size, h)))
            # Pad to full tile size if needed
            if box.size[0] < tile_size or box.size[1] < tile_size:
                padded = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                padded.paste(box, (0, 0))
                box = padded
            tiles.append((box, y, x))
    return tiles


# ── Per-page processing ─────────────────────────────────────────────
def process_page(page, page_idx, out_dir: Path, do_tiles=False,
                 stem="page", scale=SCALE):
    """Process one PDF page: extract vectors, save outputs."""
    page_rect = page.rect
    text_words = page.get_text("words")

    # 1. Find drawing crop
    crop_rect = find_drawing_crop(page, page_rect, text_words)

    # 2. Extract wall-like lines inside crop
    lines = extract_wall_lines(page, crop_rect)
    print(f"  Page {page_idx + 1}: {len(lines)} wall-like lines "
          f"(crop {crop_rect.width:.0f}x{crop_rect.height:.0f} pt)")

    if len(lines) == 0:
        print(f"    >> No wall-like lines found, skipping outputs.")
        return

    # 3. Render full page
    full_img = render_page(page, scale=scale)
    tag = f"{stem}_p{page_idx + 1:02d}"

    # 4. Save full-page PNG
    full_path = out_dir / f"{tag}_full.png"
    full_img.save(str(full_path))
    print(f"    Saved: {full_path.name}")

    # 5. Save drawing crop
    crop_img = crop_image(full_img, crop_rect, page_rect, scale=scale)
    crop_path = out_dir / f"{tag}_drawing_crop.png"
    crop_img.save(str(crop_path))
    print(f"    Saved: {crop_path.name}")

    # 6. Save vector overlay on crop
    vec_img = draw_lines_on_image(crop_img, lines, scale=scale, crop_rect=crop_rect)
    vec_path = out_dir / f"{tag}_with_vectors.png"
    vec_img.save(str(vec_path))
    print(f"    Saved: {vec_path.name}")

    # 7. Optional tiles
    if do_tiles:
        tile_dir = out_dir / f"{tag}_tiles"
        tile_dir.mkdir(exist_ok=True)
        tiles = generate_tiles(crop_img)
        vec_tiles = generate_tiles(vec_img)
        for (tile, ty, tx), (vtile, _, _) in zip(tiles, vec_tiles):
            tname = f"{tag}_tile_r{ty:04d}_c{tx:04d}"
            tile.save(str(tile_dir / f"{tname}.png"))
            vtile.save(str(tile_dir / f"{tname}_with_vectors.png"))
        print(f"    Saved: {len(tiles)} tiles → {tile_dir.name}/")


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Extract wall-like vector lines from CAD PDF pages.")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--pages", default=None,
                        help="Page selection: '1,3,5' or '2-6' (1-based). "
                             "Default: all pages with vectors.")
    parser.add_argument("--tiles", action="store_true",
                        help="Also generate 640×640 tiles with overlaps")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <pdf_stem>_vectors/)")
    parser.add_argument("--scale", type=int, default=SCALE,
                        help=f"Render scale (default {SCALE})")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found.")
        sys.exit(1)

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    stem = pdf_path.stem

    # Decide which pages
    if args.pages:
        page_indices = parse_pages(args.pages, total)
    else:
        # Auto: only pages that have vector drawings
        page_indices = []
        for i in range(total):
            if len(doc[i].get_drawings()) > 0:
                page_indices.append(i)

    if not page_indices:
        print("No pages with vector content found.")
        sys.exit(0)

    print(f"PDF: {pdf_path.name} ({total} pages)")
    print(f"Processing pages: {[i + 1 for i in page_indices]}")

    out_dir = Path(args.out) if args.out else Path(f"{stem}_vectors")
    out_dir.mkdir(exist_ok=True, parents=True)

    render_scale = args.scale

    for idx in page_indices:
        process_page(doc[idx], idx, out_dir, do_tiles=args.tiles,
                     stem=stem, scale=render_scale)

    print(f"\nDone. Outputs in: {out_dir}/")


if __name__ == "__main__":
    main()
