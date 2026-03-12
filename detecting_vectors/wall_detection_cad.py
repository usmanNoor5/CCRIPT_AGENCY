#!/usr/bin/env python3
"""
wall_detection_cad.py — Detect walls from CAD PDF vector geometry.

Walls in CAD drawings are parallel pairs of lines separated by a
wall-thickness gap.  This script finds those pairs and outputs bounding
boxes around each wall segment.

Usage:
    python3.12 wall_detection_cad.py "CHRISTENSEN MODEL.pdf" --page 4
    python3.12 wall_detection_cad.py "CHRISTENSEN MODEL.pdf" --page 4 --min-length 20
"""

import argparse
import math
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw

# ── Tuneable constants ──────────────────────────────────────────────
SCALE = 3                       # render resolution for output PNG
MIN_STROKE_WIDTH = 0.12         # pt – skip very thin strokes
MIN_LENGTH = 15.0               # pt – minimum segment length for wall candidates
ANGLE_TOL = 8.0                 # degrees – tolerance from perfect H/V
WALL_THICKNESS_MIN = 2.0        # pt – minimum gap between parallel lines
WALL_THICKNESS_MAX = 25.0       # pt – maximum gap between parallel lines
OVERLAP_RATIO = 0.25            # minimum fractional overlap along the shared axis
MERGE_GAP = 5.0                 # pt – merge nearby wall boxes
MAX_BRIGHT = 0.35               # stroke brightness ceiling — only black/very dark lines


# ── Geometry helpers ────────────────────────────────────────────────
def _length(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def _angle_deg(x1, y1, x2, y2):
    """Angle in degrees w.r.t. positive x-axis (0..180)."""
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180


def _is_horizontal(angle):
    return angle < ANGLE_TOL or angle > (180 - ANGLE_TOL)


def _is_vertical(angle):
    return abs(angle - 90) < ANGLE_TOL


def _brightness(color) -> float:
    """Perceived brightness 0..1 of an RGB tuple (each 0..1)."""
    if color is None:
        return 1.0
    if isinstance(color, (int, float)):
        return float(color)
    r, g, b = color[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _overlap_1d(a_lo, a_hi, b_lo, b_hi):
    lo = max(a_lo, b_lo)
    hi = min(a_hi, b_hi)
    return max(0, hi - lo)


def _axis_overlap(seg_a, seg_b, orientation):
    """Fractional overlap along shared axis."""
    if orientation == "H":
        a_lo, a_hi = sorted([seg_a[0], seg_a[2]])
        b_lo, b_hi = sorted([seg_b[0], seg_b[2]])
    else:
        a_lo, a_hi = sorted([seg_a[1], seg_a[3]])
        b_lo, b_hi = sorted([seg_b[1], seg_b[3]])
    ov = _overlap_1d(a_lo, a_hi, b_lo, b_hi)
    min_len = min(a_hi - a_lo, b_hi - b_lo)
    if min_len < 1e-9:
        return 0.0
    return ov / min_len


def _wall_bbox(seg_a, seg_b, orientation):
    """
    Compute the bounding box of the wall region between two parallel segments.
    Returns (x0, y0, x1, y1) — top-left to bottom-right.
    """
    if orientation == "H":
        # Shared x-range (overlap region only)
        x_lo = max(min(seg_a[0], seg_a[2]), min(seg_b[0], seg_b[2]))
        x_hi = min(max(seg_a[0], seg_a[2]), max(seg_b[0], seg_b[2]))
        # Y spans from one line to the other
        all_y = [seg_a[1], seg_a[3], seg_b[1], seg_b[3]]
        y_lo = min(all_y)
        y_hi = max(all_y)
        return (x_lo, y_lo, x_hi, y_hi)
    else:
        y_lo = max(min(seg_a[1], seg_a[3]), min(seg_b[1], seg_b[3]))
        y_hi = min(max(seg_a[1], seg_a[3]), max(seg_b[1], seg_b[3]))
        all_x = [seg_a[0], seg_a[2], seg_b[0], seg_b[2]]
        x_lo = min(all_x)
        x_hi = max(all_x)
        return (x_lo, y_lo, x_hi, y_hi)


# ── Merge nearby wall boxes ────────────────────────────────────────
def _merge_wall_boxes(boxes, gap=MERGE_GAP):
    """Merge wall bounding boxes that are collinear and nearly touching."""
    if not boxes:
        return boxes

    h_boxes = []
    v_boxes = []
    for b in boxes:
        x0, y0, x1, y1 = b
        w, h = x1 - x0, y1 - y0
        if w > h:
            h_boxes.append(b)
        else:
            v_boxes.append(b)

    merged = []

    # Merge horizontal wall boxes (similar y-center, touching x-ranges)
    h_boxes.sort(key=lambda b: ((b[1] + b[3]) / 2, b[0]))
    used = [False] * len(h_boxes)
    for i in range(len(h_boxes)):
        if used[i]:
            continue
        x0, y0, x1, y1 = h_boxes[i]
        y_center = (y0 + y1) / 2
        thickness = y1 - y0
        for j in range(i + 1, len(h_boxes)):
            if used[j]:
                continue
            jx0, jy0, jx1, jy1 = h_boxes[j]
            jy_center = (jy0 + jy1) / 2
            # Must be same wall (similar y, similar thickness)
            if abs(jy_center - y_center) > gap + thickness / 2:
                continue
            if abs((jy1 - jy0) - thickness) > gap:
                continue
            # Must be touching or close in x
            if jx0 <= x1 + gap and jx1 >= x0 - gap:
                x0 = min(x0, jx0)
                x1 = max(x1, jx1)
                y0 = min(y0, jy0)
                y1 = max(y1, jy1)
                y_center = (y0 + y1) / 2
                thickness = y1 - y0
                used[j] = True
        merged.append((x0, y0, x1, y1))

    # Merge vertical wall boxes
    v_boxes.sort(key=lambda b: ((b[0] + b[2]) / 2, b[1]))
    used = [False] * len(v_boxes)
    for i in range(len(v_boxes)):
        if used[i]:
            continue
        x0, y0, x1, y1 = v_boxes[i]
        x_center = (x0 + x1) / 2
        thickness = x1 - x0
        for j in range(i + 1, len(v_boxes)):
            if used[j]:
                continue
            jx0, jy0, jx1, jy1 = v_boxes[j]
            jx_center = (jx0 + jx1) / 2
            if abs(jx_center - x_center) > gap + thickness / 2:
                continue
            if abs((jx1 - jx0) - thickness) > gap:
                continue
            if jy0 <= y1 + gap and jy1 >= y0 - gap:
                x0 = min(x0, jx0)
                x1 = max(x1, jx1)
                y0 = min(y0, jy0)
                y1 = max(y1, jy1)
                x_center = (x0 + x1) / 2
                thickness = x1 - x0
                used[j] = True
        merged.append((x0, y0, x1, y1))

    return merged


# ── Rotation + extraction ───────────────────────────────────────────
def _rotate_point(x, y, rotation, page_width, page_height):
    """Transform from unrotated PDF coords to rotated display coords."""
    if rotation == 90:
        return page_height - y, x
    elif rotation == 180:
        return page_width - x, page_height - y
    elif rotation == 270:
        return y, page_width - x
    return x, y


def extract_lines(page) -> list[tuple]:
    """Extract line segments with BLACK/dark stroke only.
    Returns (x1,y1,x2,y2) in rotated display coords."""
    drawings = page.get_drawings()
    rotation = page.rotation
    if rotation in (90, 270):
        pw, ph = page.rect.height, page.rect.width
    else:
        pw, ph = page.rect.width, page.rect.height
    segs = []
    for d in drawings:
        width = d.get("width") or 0
        if width < MIN_STROKE_WIDTH:
            continue
        # Only black / very dark strokes
        color = d.get("color")
        if _brightness(color) > MAX_BRIGHT:
            continue
        for item in d["items"]:
            if item[0] != "l":
                continue
            _, p1, p2 = item
            rx1, ry1 = _rotate_point(p1.x, p1.y, rotation, pw, ph)
            rx2, ry2 = _rotate_point(p2.x, p2.y, rotation, pw, ph)
            segs.append((rx1, ry1, rx2, ry2))
    return segs


# ── Main detection ──────────────────────────────────────────────────
def detect_walls(page) -> list[tuple]:
    """
    Detect walls as bounding boxes from parallel-pair geometry.
    Returns list of (x0, y0, x1, y1) wall bounding boxes in display coords.
    """
    raw_segs = extract_lines(page)
    print(f"  Raw dark line segments: {len(raw_segs)}")

    # Filter: long + orthogonal
    h_segs = []
    v_segs = []
    for seg in raw_segs:
        L = _length(*seg)
        if L < MIN_LENGTH:
            continue
        angle = _angle_deg(*seg)
        if _is_horizontal(angle):
            h_segs.append(seg)
        elif _is_vertical(angle):
            v_segs.append(seg)

    print(f"  After filter (len>={MIN_LENGTH}, ortho ±{ANGLE_TOL}°, dark only): "
          f"{len(h_segs)} H + {len(v_segs)} V = {len(h_segs) + len(v_segs)}")

    wall_boxes = []

    # Find parallel pairs — horizontal
    h_segs.sort(key=lambda s: (s[1] + s[3]) / 2)
    h_used = [False] * len(h_segs)
    for i in range(len(h_segs)):
        if h_used[i]:
            continue
        yi = (h_segs[i][1] + h_segs[i][3]) / 2
        for j in range(i + 1, len(h_segs)):
            if h_used[j]:
                continue
            yj = (h_segs[j][1] + h_segs[j][3]) / 2
            dist = abs(yj - yi)
            if dist > WALL_THICKNESS_MAX:
                break
            if dist < WALL_THICKNESS_MIN:
                continue
            ov = _axis_overlap(h_segs[i], h_segs[j], "H")
            if ov < OVERLAP_RATIO:
                continue
            bbox = _wall_bbox(h_segs[i], h_segs[j], "H")
            if _length(bbox[0], bbox[1], bbox[2], bbox[1]) >= MIN_LENGTH:
                wall_boxes.append(bbox)
                h_used[i] = True
                h_used[j] = True
                break

    # Find parallel pairs — vertical
    v_segs.sort(key=lambda s: (s[0] + s[2]) / 2)
    v_used = [False] * len(v_segs)
    for i in range(len(v_segs)):
        if v_used[i]:
            continue
        xi = (v_segs[i][0] + v_segs[i][2]) / 2
        for j in range(i + 1, len(v_segs)):
            if v_used[j]:
                continue
            xj = (v_segs[j][0] + v_segs[j][2]) / 2
            dist = abs(xj - xi)
            if dist > WALL_THICKNESS_MAX:
                break
            if dist < WALL_THICKNESS_MIN:
                continue
            ov = _axis_overlap(v_segs[i], v_segs[j], "V")
            if ov < OVERLAP_RATIO:
                continue
            bbox = _wall_bbox(v_segs[i], v_segs[j], "V")
            if _length(bbox[0], bbox[1], bbox[0], bbox[3]) >= MIN_LENGTH:
                wall_boxes.append(bbox)
                v_used[i] = True
                v_used[j] = True
                break

    print(f"  Parallel pairs found: {len(wall_boxes)}")

    # Merge nearby boxes
    merged = _merge_wall_boxes(wall_boxes)
    print(f"  After merge: {len(merged)} wall segments")

    return merged


# ── Visualisation ───────────────────────────────────────────────────
def render_walls(page, wall_boxes, scale=SCALE) -> Image.Image:
    """Render page with red boundary rectangles around each wall."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")

    for x0, y0, x1, y1 in wall_boxes:
        sx0, sy0 = x0 * scale, y0 * scale
        sx1, sy1 = x1 * scale, y1 * scale
        # Semi-transparent red fill + solid red border
        draw.rectangle([sx0, sy0, sx1, sy1], fill=(255, 0, 0, 60),
                       outline="red", width=2)

    return img


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect walls from CAD PDF vector geometry.")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--page", type=int, required=True,
                        help="Page number (1-based)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: <pdf_stem>_walls/)")
    parser.add_argument("--min-length", type=float, default=MIN_LENGTH,
                        help=f"Minimum segment length in pt (default {MIN_LENGTH})")
    parser.add_argument("--min-width", type=float, default=MIN_STROKE_WIDTH,
                        help=f"Minimum stroke width in pt (default {MIN_STROKE_WIDTH})")
    parser.add_argument("--thickness-min", type=float, default=WALL_THICKNESS_MIN,
                        help=f"Min wall thickness in pt (default {WALL_THICKNESS_MIN})")
    parser.add_argument("--thickness-max", type=float, default=WALL_THICKNESS_MAX,
                        help=f"Max wall thickness in pt (default {WALL_THICKNESS_MAX})")
    parser.add_argument("--scale", type=int, default=SCALE,
                        help=f"Render scale (default {SCALE})")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found.")
        sys.exit(1)

    doc = fitz.open(str(pdf_path))
    page_idx = args.page - 1
    if page_idx < 0 or page_idx >= len(doc):
        print(f"ERROR: Page {args.page} out of range (1-{len(doc)}).")
        sys.exit(1)

    # Apply overrides
    import wall_detection_cad as _self
    _self.MIN_LENGTH = args.min_length
    _self.MIN_STROKE_WIDTH = args.min_width
    _self.WALL_THICKNESS_MIN = args.thickness_min
    _self.WALL_THICKNESS_MAX = args.thickness_max
    _self.MAX_BRIGHT = 0.35
    render_scale = args.scale

    page = doc[page_idx]
    stem = pdf_path.stem

    print(f"PDF: {pdf_path.name}, Page {args.page}")
    print(f"Params: min_length={args.min_length}, min_width={args.min_width}, "
          f"thickness={args.thickness_min}-{args.thickness_max} pt, "
          f"dark_only (brightness<={MAX_BRIGHT})")

    # Detect walls
    wall_boxes = detect_walls(page)

    if not wall_boxes:
        print("\nNo walls detected. Try adjusting --min-length or --thickness-min/max.")
        sys.exit(0)

    # Compute total wall length
    total_length = 0
    for x0, y0, x1, y1 in wall_boxes:
        w, h = x1 - x0, y1 - y0
        total_length += max(w, h)  # wall length = longer dimension
    print(f"\nTotal wall length: {total_length:.1f} pt ({total_length / 72:.1f} inches)")
    print(f"Walls detected: {len(wall_boxes)}")

    # Save output
    out_dir = Path(args.out) if args.out else Path(f"{stem}_walls")
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save wall overlay image
    img = render_walls(page, wall_boxes, scale=render_scale)
    img_path = out_dir / f"{stem}_p{args.page:02d}_walls.png"
    img.save(str(img_path))
    print(f"Saved: {img_path}")

    # Print wall boxes
    print(f"\n--- Wall Bounding Boxes ({len(wall_boxes)}) ---")
    for i, (x0, y0, x1, y1) in enumerate(wall_boxes):
        w, h = x1 - x0, y1 - y0
        L = max(w, h)
        t = min(w, h)
        orient = "H" if w > h else "V"
        print(f"  {i + 1:3d}. [{orient}] ({x0:.1f},{y0:.1f})→({x1:.1f},{y1:.1f})  "
              f"len={L:.1f} thick={t:.1f} pt")


if __name__ == "__main__":
    main()
