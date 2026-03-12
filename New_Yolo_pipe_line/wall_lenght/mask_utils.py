from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 8-connectivity neighbors: (dy, dx, distance)
_NEIGHBORS_8 = [
    (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
    (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
]
_NEIGHBORS_8_NO_DIST = [(dy, dx) for dy, dx, _ in _NEIGHBORS_8]


def load_mask(mask_path: Path, dilate_pixels: int = 0) -> np.ndarray:
    """
    Load binary mask. Optionally dilate by dilate_pixels to recover under-segmented
    boundaries (model often misses 1–2 px at edges, which shortens the contour).
    """
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")
    out = (m > 0).astype(np.uint8) * 255
    if dilate_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_pixels + 1,) * 2)
        out = cv2.dilate(out, kernel)
    return out


def skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    img = binary_mask.copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _neighbor_count(s: set, y: int, x: int) -> int:
    return sum(1 for dy, dx in _NEIGHBORS_8_NO_DIST if (y + dy, x + dx) in s)


def prune_skeleton(skel: np.ndarray, max_iters: int = 50) -> np.ndarray:
    """
    Remove spur pixels (degree 1) iteratively so short branches and rungs
    between double edges are removed. Leaves the main centerline/loops.
    """
    out = (skel > 0).astype(np.uint8) * 255
    h, w = out.shape
    for _ in range(max_iters):
        ys, xs = np.where(out > 0)
        s = set(zip(ys.tolist(), xs.tolist()))
        to_remove = [
            (y, x) for y, x in s
            if _neighbor_count(s, y, x) == 1
        ]
        if not to_remove:
            break
        for y, x in to_remove:
            out[y, x] = 0
    return (out > 0).astype(np.uint8)


def _skeleton_component_length(pixels: list[tuple[int, int]]) -> float:
    """Length of one connected component using 8-connectivity (each edge once)."""
    s = set(pixels)
    length = 0.0
    for y, x in pixels:
        for dy, dx, dist in _NEIGHBORS_8:
            if (y + dy, x + dx) in s:
                length += dist
    return length / 2.0  # each edge counted twice


def _connected_components_8(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Label 8-connected components; return list of pixel lists per component."""
    labeled = np.zeros(skel.shape, dtype=np.int32)
    current = 0
    ys, xs = np.where(skel > 0)
    s = set(zip(ys.tolist(), xs.tolist()))
    components: list[list[tuple[int, int]]] = []

    def dfs(y: int, x: int, comp: list) -> None:
        if (y, x) not in s or labeled[y, x] != 0:
            return
        labeled[y, x] = 1
        comp.append((y, x))
        for dy, dx in _NEIGHBORS_8_NO_DIST:
            dfs(y + dy, x + dx, comp)

    for y, x in s:
        if labeled[y, x] == 0:
            current += 1
            comp: list[tuple[int, int]] = []
            dfs(y, x, comp)
            components.append(comp)
    return components


def skeleton_length_pixels(skel: np.ndarray) -> float:
    """
    Centerline length: sum each edge between skeleton pixels once using 8-connectivity.
    Horizontal/vertical = 1.0, diagonal = √2. Each edge counted once (halve the sum).
    """
    ys, xs = np.where(skel > 0)
    s = set(zip(ys.tolist(), xs.tolist()))
    length = 0.0
    for y, x in s:
        for dy, dx, dist in _NEIGHBORS_8:
            if (y + dy, x + dx) in s:
                length += dist
    return length / 2.0


def _centerline_length_one_contour(cnt: np.ndarray) -> float:
    """
    For one contour (blob), estimate centerline length.
    perimeter/2 overcounts for thick blobs (adds width). Use perimeter/2 - width_est.
    """
    p = cv2.arcLength(cnt, closed=True)
    a = cv2.contourArea(cnt)
    if p <= 0:
        return 0.0
    width_est = 2.0 * a / p  # rough width for a strip
    return max(0.0, p / 2.0 - width_est)


def wall_length_from_contours(mask: np.ndarray) -> tuple[float, str, float | None, float | None]:
    """
    Wall length from contour boundaries.
    - One band with hole: centerline = (outer + inner) / 2.
    - One contour (no hole): centerline = perimeter / 2.
    - Multiple contours (fragmented detection): centerline = sum over contours of
      (perimeter/2 - width_est) so we don't overcount thick pieces.
    Returns (length_pixels, method_used, outer_perim, inner_perim).
    """
    binary = (mask > 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    if hierarchy is None or len(contours) == 0:
        return 0.0, "contours_none", None, None

    h = hierarchy[0]
    outer_perims: list[float] = []
    inner_perims: list[float] = []
    outer_contours: list[np.ndarray] = []
    for i, cnt in enumerate(contours):
        p = cv2.arcLength(cnt, closed=True)
        if h[i][3] == -1:
            outer_perims.append(p)
            outer_contours.append(cnt)
        else:
            inner_perims.append(p)

    P_outer = sum(outer_perims)
    P_inner = sum(inner_perims)

    if P_outer > 0 and P_inner > 0:
        length = (P_outer + P_inner) / 2.0
        return length, "contours_centerline", P_outer, P_inner
    if P_outer > 0:
        if len(outer_contours) > 1:
            # Fragmented: multiple pieces. Use centerline = perim/2 - width per piece so we don't overcount.
            length = sum(_centerline_length_one_contour(c) for c in outer_contours)
            return length, "contours_fragmented_centerline", P_outer, None
        # Single contour (one connected band, no hole)
        return P_outer / 2.0, "contours_single_boundary_half", P_outer, None
    return 0.0, "contours_none", None, None


def validate_mask_quality(mask: np.ndarray, skel: np.ndarray) -> dict[str, Any]:
    """
    Validate mask quality to detect double edges, thick walls, or skeletonization issues.
    Returns dict with validation metrics and warnings.
    """
    mask_pixels = np.sum(mask > 0)
    skel_pixels = np.sum(skel > 0)
    thickness_estimate = mask_pixels / max(skel_pixels, 1)

    ys, xs = np.where(skel > 0)
    s = set(zip(ys.tolist(), xs.tolist()))
    branch_count = sum(1 for y, x in zip(ys, xs) if _neighbor_count(s, y, x) > 2)

    warnings = []
    if thickness_estimate > 3.0:
        warnings.append(f"Thick walls detected (avg thickness ~{thickness_estimate:.1f}px). Skeleton may not represent true centerline.")
    if branch_count > skel_pixels * 0.1:
        warnings.append(f"High branch point ratio ({branch_count}/{skel_pixels}={branch_count/skel_pixels:.1%}). Possible double edges or loops.")

    return {
        "mask_pixels": int(mask_pixels),
        "skeleton_pixels": int(skel_pixels),
        "thickness_estimate": float(thickness_estimate),
        "branch_points": branch_count,
        "warnings": warnings,
    }


def estimate_wall_length_pixels(
    mask_path: Path,
    return_validation: bool = False,
    dilate_pixels: int = 0,
) -> float | tuple[float, dict[str, Any]]:
    """
    Estimate wall length in pixels from mask = one centerline length (no double count).

    Primary: contour method. Wall mask is a band with outer and inner boundaries.
    Centerline length = (outer_perimeter + inner_perimeter) / 2. This is exact for
    a constant-thickness band and naturally excludes double-counting.

    Fallback: skeleton (prune spurs, then longest component) when contours don't
    give both outer and inner (e.g. no hole detected).

    Under-count can come from: (1) model under-segmentation at wall edges,
    (2) tile boundaries / stitching, (3) contour following the detected boundary
    (any missed edge shortens the contour). Use dilate_pixels=1 to recover ~1 px
    at boundaries (often recovers 1–3% length).
    """
    mask = load_mask(mask_path, dilate_pixels=dilate_pixels)
    length_contour, contour_method, p_outer, p_inner = wall_length_from_contours(mask)

    if contour_method in ("contours_centerline", "contours_single_boundary_half", "contours_fragmented_centerline") and length_contour > 0:
        length = length_contour
        length_used = contour_method
        validation_extra = {
            "contour_outer_perimeter_px": p_outer,
            "contour_inner_perimeter_px": p_inner,
        }
        if p_inner is None:
            validation_extra["contour_inner_perimeter_px"] = None
    else:
        # Fallback: skeleton + prune + longest component
        skel = skeletonize(mask)
        skel_pruned = prune_skeleton(skel)
        components = _connected_components_8(skel_pruned)
        if not components:
            length = 0.0
            length_used = "skeleton_none"
        elif len(components) == 1:
            length = _skeleton_component_length(components[0])
            length_used = "skeleton_single"
        else:
            lengths = [_skeleton_component_length(c) for c in components]
            length = max(lengths)
            length_used = "skeleton_longest_of_multiple"
        validation_extra = {
            "skeleton_after_prune_pixels": int(np.sum(skel_pruned > 0)),
            "components_after_prune": len(components),
        }

    if return_validation:
        skel_for_validation = skeletonize(mask)
        validation = validate_mask_quality(mask, skel_for_validation)
        validation["length_method"] = length_used
        validation["wall_length_pixels"] = float(length)
        validation["mask_dilate_pixels"] = dilate_pixels
        validation.update(validation_extra)
        return length, validation
    return length

