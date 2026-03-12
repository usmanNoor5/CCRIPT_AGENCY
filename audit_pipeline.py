#!/usr/bin/env python3
"""
Comprehensive audit of the data pipeline to find the source of NaN errors
in SAM3 training (ValueError: matrix contains invalid numeric entries).

Checks:
  1. YOLO label files for invalid values
  2. COCO annotations for invalid bboxes, areas, segmentations
  3. Cross-check images vs annotations
  4. YOLO-to-COCO conversion logic
  5. Specific edge cases (tiny bboxes, area mismatches)
  6. Simulate what SAM3 training actually sees (the full bbox transform chain)
"""

import json
import math
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
YOLO_LABELS_DIR = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_dataset/labels/train")
YOLO_IMAGES_DIR = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_dataset/images/train")
COCO_TRAIN_JSON = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_coco/train/_annotations.coco.json")
COCO_VALID_JSON = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_coco/valid/_annotations.coco.json")
COCO_TRAIN_DIR = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_coco/train")
COCO_VALID_DIR = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_coco/valid")

issues = []
warnings_list = []
stats = defaultdict(int)


def issue(msg):
    issues.append(msg)
    print(f"  [ISSUE] {msg}")


def warn(msg):
    warnings_list.append(msg)
    # Only print first 50 warnings to avoid spam
    if len(warnings_list) <= 50:
        print(f"  [WARN]  {msg}")
    elif len(warnings_list) == 51:
        print(f"  [WARN]  ... (suppressing further warnings, will report count)")


# =============================================================================
# 1. Audit YOLO label files
# =============================================================================
def audit_yolo_labels():
    print("\n" + "=" * 70)
    print("SECTION 1: YOLO Label File Audit")
    print("=" * 70)

    label_files = sorted(YOLO_LABELS_DIR.glob("*.txt"))
    print(f"Found {len(label_files)} label files")
    stats["yolo_label_files"] = len(label_files)

    total_annotations = 0
    bad_files = []

    for lf in label_files:
        try:
            text = lf.read_text()
        except Exception as e:
            issue(f"Cannot read {lf.name}: {e}")
            bad_files.append(lf.name)
            continue

        for line_num, line in enumerate(text.splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue

            total_annotations += 1

            # Check class ID
            try:
                class_id = int(parts[0])
            except ValueError:
                issue(f"{lf.name}:L{line_num}: non-integer class_id '{parts[0]}'")
                continue

            if class_id < 0:
                issue(f"{lf.name}:L{line_num}: negative class_id {class_id}")

            # Check coordinate count (must be even number of coords after class_id)
            coords_raw = parts[1:]
            if len(coords_raw) % 2 != 0:
                issue(f"{lf.name}:L{line_num}: odd number of coords ({len(coords_raw)})")
                continue

            if len(coords_raw) < 6:
                warn(f"{lf.name}:L{line_num}: fewer than 3 xy pairs ({len(coords_raw)//2} points)")
                stats["yolo_too_few_points"] += 1
                continue

            # Parse coordinates
            try:
                coords = [float(c) for c in coords_raw]
            except ValueError as e:
                issue(f"{lf.name}:L{line_num}: non-numeric coordinate: {e}")
                continue

            # Check for NaN / Inf
            for i, c in enumerate(coords):
                if math.isnan(c):
                    issue(f"{lf.name}:L{line_num}: NaN at coord index {i}")
                    stats["yolo_nan"] += 1
                if math.isinf(c):
                    issue(f"{lf.name}:L{line_num}: Inf at coord index {i}")
                    stats["yolo_inf"] += 1

            # Check range [0, 1]
            out_of_range = [(i, c) for i, c in enumerate(coords) if c < -0.001 or c > 1.001]
            if out_of_range:
                issue(f"{lf.name}:L{line_num}: coords out of [0,1]: {out_of_range[:5]}")
                stats["yolo_out_of_range"] += 1

            # Check for near-zero range (degenerate bbox)
            xs = coords[0::2]
            ys = coords[1::2]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)

            if x_range < 1e-6:
                warn(f"{lf.name}:L{line_num}: zero x-range (all x={xs[0]:.6f})")
                stats["yolo_zero_width"] += 1
            if y_range < 1e-6:
                warn(f"{lf.name}:L{line_num}: zero y-range (all y={ys[0]:.6f})")
                stats["yolo_zero_height"] += 1

            # Very large coordinate values (could indicate encoding issues)
            if any(abs(c) > 10 for c in coords):
                issue(f"{lf.name}:L{line_num}: very large coord value (max={max(abs(c) for c in coords):.2f})")
                stats["yolo_very_large"] += 1

    stats["yolo_total_annotations"] = total_annotations
    print(f"\nTotal YOLO annotations scanned: {total_annotations}")


# =============================================================================
# 2. Audit COCO JSON annotations
# =============================================================================
def audit_coco_json(json_path, img_dir, split_name):
    print(f"\n{'=' * 70}")
    print(f"SECTION 2: COCO JSON Audit ({split_name})")
    print(f"{'=' * 70}")

    if not json_path.exists():
        issue(f"COCO JSON not found: {json_path}")
        return

    with open(json_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Categories: {len(categories)}")
    stats[f"coco_{split_name}_images"] = len(images)
    stats[f"coco_{split_name}_annotations"] = len(annotations)

    # Build lookup
    img_by_id = {img["id"]: img for img in images}

    # Check categories
    cat_ids = set()
    for cat in categories:
        cat_ids.add(cat["id"])
        if "name" not in cat:
            issue(f"[{split_name}] Category missing 'name': {cat}")

    # --- Check annotations ---
    tiny_bboxes = []
    area_mismatches = []
    ann_image_ids = set()

    for ann in annotations:
        ann_id = ann.get("id", "?")
        img_id = ann.get("image_id")
        ann_image_ids.add(img_id)

        # Check image reference
        if img_id not in img_by_id:
            issue(f"[{split_name}] ann {ann_id}: references missing image_id {img_id}")
            continue

        img_info = img_by_id[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # --- Bbox checks ---
        bbox = ann.get("bbox")
        if bbox is None:
            issue(f"[{split_name}] ann {ann_id}: missing bbox")
            continue

        if len(bbox) != 4:
            issue(f"[{split_name}] ann {ann_id}: bbox has {len(bbox)} elements (expected 4)")
            continue

        x, y, w, h = bbox

        # NaN/Inf check
        for i, v in enumerate([x, y, w, h]):
            if math.isnan(v):
                issue(f"[{split_name}] ann {ann_id}: bbox has NaN at index {i}")
                stats[f"coco_{split_name}_bbox_nan"] += 1
            if math.isinf(v):
                issue(f"[{split_name}] ann {ann_id}: bbox has Inf at index {i}")
                stats[f"coco_{split_name}_bbox_inf"] += 1

        # Negative width/height
        if w <= 0:
            issue(f"[{split_name}] ann {ann_id}: bbox width <= 0 ({w})")
            stats[f"coco_{split_name}_neg_width"] += 1
        if h <= 0:
            issue(f"[{split_name}] ann {ann_id}: bbox height <= 0 ({h})")
            stats[f"coco_{split_name}_neg_height"] += 1

        # Negative x/y
        if x < 0:
            warn(f"[{split_name}] ann {ann_id}: bbox x < 0 ({x})")
            stats[f"coco_{split_name}_neg_x"] += 1
        if y < 0:
            warn(f"[{split_name}] ann {ann_id}: bbox y < 0 ({y})")
            stats[f"coco_{split_name}_neg_y"] += 1

        # Exceeds image dimensions
        if x + w > img_w + 1:
            warn(f"[{split_name}] ann {ann_id}: bbox right edge ({x+w:.1f}) > img width ({img_w})")
            stats[f"coco_{split_name}_exceeds_w"] += 1
        if y + h > img_h + 1:
            warn(f"[{split_name}] ann {ann_id}: bbox bottom edge ({y+h:.1f}) > img height ({img_h})")
            stats[f"coco_{split_name}_exceeds_h"] += 1

        # Tiny bboxes
        if w < 5 or h < 5:
            tiny_bboxes.append((ann_id, img_id, bbox, img_info["file_name"]))
            stats[f"coco_{split_name}_tiny_bbox"] += 1

        # --- Area checks ---
        area = ann.get("area", 0)
        if area <= 0:
            issue(f"[{split_name}] ann {ann_id}: area <= 0 ({area})")
            stats[f"coco_{split_name}_zero_area"] += 1

        if math.isnan(area) or math.isinf(area):
            issue(f"[{split_name}] ann {ann_id}: area is NaN/Inf ({area})")
            stats[f"coco_{split_name}_area_nan_inf"] += 1

        bbox_area = w * h
        if bbox_area > 0 and area > 0:
            ratio = area / bbox_area
            if ratio > 1.5 or ratio < 0.01:
                area_mismatches.append((ann_id, area, bbox_area, ratio))
                stats[f"coco_{split_name}_area_mismatch"] += 1

        # --- Category check ---
        cat_id = ann.get("category_id")
        if cat_id not in cat_ids:
            issue(f"[{split_name}] ann {ann_id}: unknown category_id {cat_id}")

        # --- Segmentation checks ---
        seg = ann.get("segmentation")
        if seg is not None and isinstance(seg, list):
            for poly_idx, poly in enumerate(seg):
                if len(poly) < 6:
                    issue(f"[{split_name}] ann {ann_id}, poly {poly_idx}: fewer than 3 points ({len(poly)//2} points)")
                    stats[f"coco_{split_name}_seg_too_few_points"] += 1

                for vi, v in enumerate(poly):
                    if math.isnan(v):
                        issue(f"[{split_name}] ann {ann_id}, poly {poly_idx}: NaN at index {vi}")
                        stats[f"coco_{split_name}_seg_nan"] += 1
                    if math.isinf(v):
                        issue(f"[{split_name}] ann {ann_id}, poly {poly_idx}: Inf at index {vi}")
                        stats[f"coco_{split_name}_seg_inf"] += 1

                # Check polygon coords vs image dimensions
                px = poly[0::2]
                py = poly[1::2]
                if any(v < -1 for v in px) or any(v < -1 for v in py):
                    warn(f"[{split_name}] ann {ann_id}, poly {poly_idx}: negative seg coords")
                    stats[f"coco_{split_name}_seg_negative"] += 1
                if any(v > img_w + 1 for v in px) or any(v > img_h + 1 for v in py):
                    warn(f"[{split_name}] ann {ann_id}, poly {poly_idx}: seg coords exceed image dims")
                    stats[f"coco_{split_name}_seg_exceeds"] += 1

                # Check if seg polygon is consistent with bbox
                seg_xmin, seg_xmax = min(px), max(px)
                seg_ymin, seg_ymax = min(py), max(py)
                seg_w = seg_xmax - seg_xmin
                seg_h = seg_ymax - seg_ymin

                # Large discrepancy between segmentation and bbox
                if w > 0 and h > 0:
                    if abs(seg_xmin - x) > 5 or abs(seg_ymin - y) > 5:
                        stats[f"coco_{split_name}_seg_bbox_offset"] += 1
                    if seg_w > 0 and seg_h > 0:
                        w_ratio = seg_w / w
                        h_ratio = seg_h / h
                        if w_ratio > 2.0 or w_ratio < 0.5 or h_ratio > 2.0 or h_ratio < 0.5:
                            stats[f"coco_{split_name}_seg_bbox_size_mismatch"] += 1

    # Check for images with no annotations
    imgs_with_anns = ann_image_ids
    imgs_without_anns = set(img_by_id.keys()) - imgs_with_anns
    if imgs_without_anns:
        print(f"  Images without annotations: {len(imgs_without_anns)}")
        stats[f"coco_{split_name}_imgs_no_anns"] = len(imgs_without_anns)

    # Report tiny bboxes
    if tiny_bboxes:
        print(f"\n  Tiny bboxes (w or h < 5px): {len(tiny_bboxes)}")
        for ann_id, img_id, bbox, fname in tiny_bboxes[:10]:
            print(f"    ann {ann_id} in {fname}: bbox={bbox}")

    # Report area mismatches
    if area_mismatches:
        print(f"\n  Area/bbox mismatches (ratio < 0.01 or > 1.5): {len(area_mismatches)}")
        for ann_id, area, bbox_area, ratio in area_mismatches[:10]:
            print(f"    ann {ann_id}: seg_area={area:.1f}, bbox_area={bbox_area:.1f}, ratio={ratio:.3f}")


# =============================================================================
# 3. Cross-check images vs annotations
# =============================================================================
def cross_check_images(json_path, img_dir, split_name):
    print(f"\n{'=' * 70}")
    print(f"SECTION 3: Image-Annotation Cross-Check ({split_name})")
    print(f"{'=' * 70}")

    if not json_path.exists():
        return

    with open(json_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    missing_files = []
    dimension_mismatches = []
    corrupted_images = []

    # Sample up to 100 images to verify
    sample_size = min(100, len(images))
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(images), size=sample_size, replace=False)

    for i in sample_indices:
        img_info = images[i]
        fname = img_info["file_name"]
        img_path = img_dir / fname

        if not img_path.exists():
            missing_files.append(fname)
            stats[f"crosscheck_{split_name}_missing"] += 1
            continue

        # Try to read the image
        img = cv2.imread(str(img_path))
        if img is None:
            corrupted_images.append(fname)
            stats[f"crosscheck_{split_name}_corrupted"] += 1
            continue

        actual_h, actual_w = img.shape[:2]
        expected_w = img_info["width"]
        expected_h = img_info["height"]

        if actual_w != expected_w or actual_h != expected_h:
            dimension_mismatches.append((fname, (actual_w, actual_h), (expected_w, expected_h)))
            stats[f"crosscheck_{split_name}_dim_mismatch"] += 1

    # Also check ALL image references exist (without loading)
    all_missing = []
    for img_info in images:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            all_missing.append(img_info["file_name"])

    if all_missing:
        issue(f"[{split_name}] {len(all_missing)} referenced images are MISSING")
        for f in all_missing[:5]:
            print(f"    Missing: {f}")

    if corrupted_images:
        issue(f"[{split_name}] {len(corrupted_images)} images are CORRUPTED (cv2 cannot read)")
        for f in corrupted_images[:5]:
            print(f"    Corrupted: {f}")

    if dimension_mismatches:
        issue(f"[{split_name}] {len(dimension_mismatches)} images have WRONG DIMENSIONS in JSON")
        for fname, actual, expected in dimension_mismatches[:5]:
            print(f"    {fname}: actual={actual}, json={expected}")
    else:
        print(f"  All {sample_size} sampled images have correct dimensions.")

    if not all_missing:
        print(f"  All {len(images)} referenced images exist on disk.")


# =============================================================================
# 4. Audit YOLO-to-COCO Converter Logic
# =============================================================================
def audit_converter():
    print(f"\n{'=' * 70}")
    print("SECTION 4: YOLO-to-COCO Converter Logic Audit")
    print("=" * 70)

    converter_issues = []

    # Potential issue 1: Odd number of coords handling
    print("\n  Checking coordinate parsing logic...")

    # The converter at line 39 does: for i in range(0, len(coords) - 1, 2)
    # If len(coords) is odd, it silently drops the last coord.
    # This is a potential source of misaligned polygons.
    # Example: if coords = [x1, y1, x2, y2, x3] -> parses x1,y1 x2,y2 (drops x3)
    # This means the label file has a corrupt line.

    print("  [CHECK] Odd-coord handling: converter drops trailing coord (safe but lossy)")

    # Potential issue 2: bbox_w < 1 or bbox_h < 1 filter
    print("  [CHECK] bbox min size filter: requires bbox_w >= 1px AND bbox_h >= 1px")
    print("          This is in PIXEL space, so very thin walls (<1px) are dropped (OK)")

    # Potential issue 3: area < 10 filter
    print("  [CHECK] area min filter: requires cv2.contourArea >= 10px^2 (OK)")

    # CRITICAL CHECK: category_id mapping
    # The COCO JSON uses category_id from the YOLO class_id (which is 0 for 'wall')
    # The categories list has {"id": 0, "name": "wall"}
    # The coco_json_loaders.py COCO_FROM_JSON iterates sorted cat_ids.
    # If category_id=0 is used, it works.
    print("  [CHECK] Category ID: YOLO class 0 -> COCO category_id 0 (consistent)")

    # CRITICAL CHECK: bbox format
    # yolo_to_coco.py produces: [x_min, y_min, bbox_w, bbox_h] in PIXEL coords (COCO xywh)
    # coco_json_loaders.py convert_boxlist_to_normalized_tensor normalizes:
    #   indices [0,2] / width  -> [x_min/W, bbox_w/W]
    #   indices [1,3] / height -> [y_min/H, bbox_h/H]
    # Result: [x_min/W, y_min/H, bbox_w/W, bbox_h/H] = normalized xywh
    # sam3_image_dataset.py then does: box_xywh_to_xyxy (converts to xyxy)
    # then multiplies by (w, h) to denormalize.
    # This is CORRECT: the flow is pixel_xywh -> norm_xywh -> norm_xyxy -> pixel_xyxy
    print("  [CHECK] Bbox format flow: pixel_xywh -> norm_xywh -> norm_xyxy -> pixel_xyxy (CORRECT)")

    # CRITICAL CHECK: What happens in the matcher?
    # The matcher in matcher.py expects boxes in cxcywh NORMALIZED format.
    # The NormalizeAPI transform converts from denormalized xyxy to normalized cxcywh.
    # If any bbox has zero width or height, cxcywh will have w=0 or h=0.
    # Then generalized_box_iou calls box_cxcywh_to_xyxy which gives degenerate boxes.
    # In generalized_box_iou: area = (rb - lt).clamp(min=0) ... then we divide by area.
    # If area = 0, we get division by zero -> NaN/Inf -> "matrix contains invalid numeric entries"
    print("\n  [CRITICAL] NaN root cause analysis:")
    print("    generalized_box_iou divides by enclosing area.")
    print("    If a target bbox has zero width OR height -> area=0 -> division by zero -> NaN")
    print("    This NaN propagates into the cost matrix -> linear_sum_assignment crashes")

    return converter_issues


# =============================================================================
# 5. Simulate SAM3 bbox processing to find problematic annotations
# =============================================================================
def simulate_sam3_processing(json_path, split_name):
    """Simulate exactly what SAM3 does with the annotations to find NaN sources."""
    print(f"\n{'=' * 70}")
    print(f"SECTION 5: SAM3 Processing Simulation ({split_name})")
    print(f"{'=' * 70}")

    if not json_path.exists():
        return

    with open(json_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    img_by_id = {img["id"]: img for img in images}

    nan_producers = []
    degenerate_boxes = []
    overflow_boxes = []

    for ann in annotations:
        ann_id = ann.get("id", "?")
        img_id = ann.get("image_id")
        if img_id not in img_by_id:
            continue

        img_info = img_by_id[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        bbox = ann["bbox"]  # [x, y, w, h] in pixel coords

        # Step 1: What coco_json_loaders.py does
        # convert_boxlist_to_normalized_tensor
        # Normalizes: [x/W, y/H, w/W, h/H]
        norm_bbox = [
            bbox[0] / img_w,  # x/W
            bbox[1] / img_h,  # y/H
            bbox[2] / img_w,  # w/W
            bbox[3] / img_h,  # h/H
        ]
        # Clamp to [0, 1]
        norm_bbox = [max(0, min(1, v)) for v in norm_bbox]

        # Step 2: What sam3_image_dataset.py does
        # box_xywh_to_xyxy: [x, y, w, h] -> [x, y, x+w, y+h]
        xyxy_norm = [
            norm_bbox[0],
            norm_bbox[1],
            norm_bbox[0] + norm_bbox[2],
            norm_bbox[1] + norm_bbox[3],
        ]
        # Denormalize: multiply by (w, h)
        xyxy_pixel = [
            xyxy_norm[0] * img_w,
            xyxy_norm[1] * img_h,
            xyxy_norm[2] * img_w,
            xyxy_norm[3] * img_h,
        ]
        # Clamp
        xyxy_pixel[0] = max(0, min(img_w, xyxy_pixel[0]))
        xyxy_pixel[1] = max(0, min(img_h, xyxy_pixel[1]))
        xyxy_pixel[2] = max(0, min(img_w, xyxy_pixel[2]))
        xyxy_pixel[3] = max(0, min(img_h, xyxy_pixel[3]))

        # Step 3: What the NormalizeAPI transform does
        # Converts xyxy to cxcywh normalized
        # cxcywh: [cx, cy, w, h]
        pw = xyxy_pixel[2] - xyxy_pixel[0]
        ph = xyxy_pixel[3] - xyxy_pixel[1]
        cx = (xyxy_pixel[0] + xyxy_pixel[2]) / 2
        cy = (xyxy_pixel[1] + xyxy_pixel[3]) / 2

        # After resize to 1008x1008 and normalization, the box is in [0,1] cxcywh
        # The important thing is whether w or h is zero/near-zero

        if pw < 0.5:
            degenerate_boxes.append({
                "ann_id": ann_id,
                "img_id": img_id,
                "file": img_info["file_name"],
                "bbox_xywh": bbox,
                "pixel_width": pw,
                "pixel_height": ph,
                "type": "near_zero_width"
            })

        if ph < 0.5:
            degenerate_boxes.append({
                "ann_id": ann_id,
                "img_id": img_id,
                "file": img_info["file_name"],
                "bbox_xywh": bbox,
                "pixel_width": pw,
                "pixel_height": ph,
                "type": "near_zero_height"
            })

        # Step 4: Simulate generalized_box_iou
        # box_cxcywh_to_xyxy: [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        # For the enclosing box, if both boxes are the same point:
        # area = 0 -> division by zero
        # Even if just the target has zero w or h, the area computation could produce NaN

        # Check if the normalized cxcywh would cause issues
        # After resize/normalization, approx:
        norm_w = pw / img_w  # This is the normalized width
        norm_h = ph / img_h

        if norm_w < 1e-6 or norm_h < 1e-6:
            nan_producers.append({
                "ann_id": ann_id,
                "img_id": img_id,
                "file": img_info["file_name"],
                "bbox": bbox,
                "norm_w": norm_w,
                "norm_h": norm_h,
                "will_produce_nan": True
            })

        # Check for very large bbox values that could overflow
        if any(abs(v) > 1e6 for v in bbox):
            overflow_boxes.append({
                "ann_id": ann_id,
                "bbox": bbox,
            })

    print(f"\n  Degenerate boxes (w or h < 0.5px after processing): {len(degenerate_boxes)}")
    for db in degenerate_boxes[:20]:
        print(f"    ann {db['ann_id']} ({db['type']}): bbox={db['bbox_xywh']}, "
              f"pixel_w={db['pixel_width']:.2f}, pixel_h={db['pixel_height']:.2f}, file={db['file']}")

    print(f"\n  Definite NaN producers (norm_w or norm_h < 1e-6): {len(nan_producers)}")
    for np_ in nan_producers[:20]:
        print(f"    ann {np_['ann_id']}: bbox={np_['bbox']}, "
              f"norm_w={np_['norm_w']:.8f}, norm_h={np_['norm_h']:.8f}")

    if overflow_boxes:
        print(f"\n  Overflow-risk boxes (values > 1e6): {len(overflow_boxes)}")

    stats[f"sim_{split_name}_degenerate"] = len(degenerate_boxes)
    stats[f"sim_{split_name}_nan_producers"] = len(nan_producers)

    return degenerate_boxes, nan_producers


# =============================================================================
# 6. Detailed statistics and histogram of bbox sizes
# =============================================================================
def bbox_statistics(json_path, split_name):
    print(f"\n{'=' * 70}")
    print(f"SECTION 6: BBox Size Distribution ({split_name})")
    print(f"{'=' * 70}")

    if not json_path.exists():
        return

    with open(json_path) as f:
        coco = json.load(f)

    annotations = coco.get("annotations", [])
    img_by_id = {img["id"]: img for img in coco.get("images", [])}

    widths = []
    heights = []
    areas = []
    aspect_ratios = []

    for ann in annotations:
        bbox = ann["bbox"]
        w, h = bbox[2], bbox[3]
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)
            areas.append(ann.get("area", 0))
            aspect_ratios.append(w / h)

    if not widths:
        print("  No valid annotations found.")
        return

    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)

    print(f"\n  BBox Width:  min={widths.min():.2f}, max={widths.max():.2f}, "
          f"mean={widths.mean():.2f}, median={np.median(widths):.2f}")
    print(f"  BBox Height: min={heights.min():.2f}, max={heights.max():.2f}, "
          f"mean={heights.mean():.2f}, median={np.median(heights):.2f}")
    print(f"  Area:        min={areas.min():.2f}, max={areas.max():.2f}, "
          f"mean={areas.mean():.2f}, median={np.median(areas):.2f}")
    print(f"  Aspect Ratio: min={aspect_ratios.min():.4f}, max={aspect_ratios.max():.4f}")

    # Distribution of tiny bboxes
    for threshold in [1, 2, 3, 5, 10]:
        n_w = np.sum(widths < threshold)
        n_h = np.sum(heights < threshold)
        print(f"  Bboxes with w < {threshold}px: {n_w}  |  h < {threshold}px: {n_h}")

    # Check for extreme aspect ratios (very thin walls)
    extreme_ar = np.sum((aspect_ratios > 50) | (aspect_ratios < 0.02))
    print(f"  Extreme aspect ratios (>50:1 or <1:50): {extreme_ar}")

    # Annotations with very small area
    small_area = np.sum(areas < 50)
    print(f"  Annotations with area < 50px^2: {small_area}")


# =============================================================================
# 7. Check if rotation augmentation breaks coordinates
# =============================================================================
def audit_rotation_labels():
    print(f"\n{'=' * 70}")
    print("SECTION 7: Rotation Augmentation Validation")
    print("=" * 70)

    # The augmentation code does:
    # rotate90: new_x = 1 - old_y, new_y = old_x
    # rotate180: new = 1 - old
    # rotate270: new_x = old_y, new_y = 1 - old_x
    #
    # After rotation, validate_polygon checks [0,1] range.
    # But if the original coord is exactly 0 or 1, the transformation should preserve [0,1].
    # Let's check for any rotated labels that have out-of-range coords.

    rotated_files = sorted(YOLO_LABELS_DIR.glob("*_r90.txt")) + \
                    sorted(YOLO_LABELS_DIR.glob("*_r180.txt")) + \
                    sorted(YOLO_LABELS_DIR.glob("*_r270.txt"))

    print(f"  Rotated label files: {len(rotated_files)}")

    out_of_range_count = 0
    for lf in rotated_files:
        text = lf.read_text()
        for line_num, line in enumerate(text.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = [float(c) for c in parts[1:]]
            for c in coords:
                if c < -0.001 or c > 1.001:
                    out_of_range_count += 1
                    if out_of_range_count <= 5:
                        issue(f"Rotated label {lf.name}:L{line_num}: coord out of range ({c:.6f})")
                    break

    if out_of_range_count == 0:
        print("  All rotated label coordinates are within [0, 1] range. (OK)")
    else:
        print(f"  Total rotated labels with out-of-range coords: {out_of_range_count}")


# =============================================================================
# 8. Check for duplicate annotations (same bbox on same image)
# =============================================================================
def check_duplicates(json_path, split_name):
    print(f"\n{'=' * 70}")
    print(f"SECTION 8: Duplicate Annotation Check ({split_name})")
    print(f"{'=' * 70}")

    if not json_path.exists():
        return

    with open(json_path) as f:
        coco = json.load(f)

    # Group by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        key = (ann["image_id"], tuple(ann["bbox"]))
        anns_by_image[key].append(ann["id"])

    dupes = {k: v for k, v in anns_by_image.items() if len(v) > 1}
    if dupes:
        print(f"  Found {len(dupes)} duplicate bbox entries:")
        for (img_id, bbox), ann_ids in list(dupes.items())[:5]:
            print(f"    image {img_id}, bbox={bbox}: {len(ann_ids)} duplicates")
        stats[f"coco_{split_name}_duplicates"] = len(dupes)
    else:
        print(f"  No duplicate annotations found. (OK)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("DATA PIPELINE AUDIT FOR SAM3 NaN ERRORS")
    print("=" * 70)
    print(f"Error: ValueError: matrix contains invalid numeric entries")
    print(f"Location: Hungarian matcher (linear_sum_assignment)")
    print(f"Root cause hypothesis: NaN/Inf in cost matrix from degenerate bboxes")

    # 1. YOLO labels
    audit_yolo_labels()

    # 2. COCO JSON
    audit_coco_json(COCO_TRAIN_JSON, COCO_TRAIN_DIR, "train")
    audit_coco_json(COCO_VALID_JSON, COCO_VALID_DIR, "valid")

    # 3. Cross-check
    cross_check_images(COCO_TRAIN_JSON, COCO_TRAIN_DIR, "train")
    cross_check_images(COCO_VALID_JSON, COCO_VALID_DIR, "valid")

    # 4. Converter logic
    audit_converter()

    # 5. SAM3 simulation
    train_degen, train_nan = simulate_sam3_processing(COCO_TRAIN_JSON, "train")
    valid_degen, valid_nan = simulate_sam3_processing(COCO_VALID_JSON, "valid")

    # 6. Statistics
    bbox_statistics(COCO_TRAIN_JSON, "train")
    bbox_statistics(COCO_VALID_JSON, "valid")

    # 7. Rotation check
    audit_rotation_labels()

    # 8. Duplicate check
    check_duplicates(COCO_TRAIN_JSON, "train")
    check_duplicates(COCO_VALID_JSON, "valid")

    # =============================================================================
    # FINAL REPORT
    # =============================================================================
    print("\n" + "=" * 70)
    print("FINAL AUDIT REPORT")
    print("=" * 70)

    print(f"\nCritical Issues: {len(issues)}")
    for i, iss in enumerate(issues):
        print(f"  {i+1}. {iss}")

    print(f"\nWarnings: {len(warnings_list)}")
    if len(warnings_list) > 20:
        print(f"  (showing first 20 of {len(warnings_list)})")
        for w in warnings_list[:20]:
            print(f"  - {w}")
    else:
        for w in warnings_list:
            print(f"  - {w}")

    print(f"\nStatistics:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Root cause summary
    print(f"\n{'=' * 70}")
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    total_nan_risk = stats.get("sim_train_nan_producers", 0) + stats.get("sim_valid_nan_producers", 0)
    total_degen = stats.get("sim_train_degenerate", 0) + stats.get("sim_valid_degenerate", 0)

    if total_nan_risk > 0:
        print(f"\n  CONFIRMED: {total_nan_risk} annotations will definitely produce NaN in the matcher.")
        print(f"  These have normalized width or height < 1e-6, causing division by zero")
        print(f"  in generalized_box_iou -> cost_giou -> cost matrix -> linear_sum_assignment crash.")
    elif total_degen > 0:
        print(f"\n  HIGH RISK: {total_degen} annotations have degenerate bboxes (w or h < 0.5px).")
        print(f"  After resize to 1008x1008 and normalization, these could become zero-area boxes.")
        print(f"  generalized_box_iou divides by enclosing area -> 0/0 -> NaN.")
    else:
        print(f"\n  No obvious NaN-producing annotations found in raw data.")
        print(f"  The NaN may come from the model predictions, not the ground truth.")
        print(f"  Check if the model outputs degenerate boxes during early training.")

    print(f"\n  ADDITIONAL RISK FACTORS:")
    tiny_train = stats.get("coco_train_tiny_bbox", 0)
    tiny_valid = stats.get("coco_valid_tiny_bbox", 0)
    if tiny_train + tiny_valid > 0:
        print(f"  - {tiny_train + tiny_valid} tiny bboxes (< 5px) that may become degenerate after resize")

    area_mm_train = stats.get("coco_train_area_mismatch", 0)
    area_mm_valid = stats.get("coco_valid_area_mismatch", 0)
    if area_mm_train + area_mm_valid > 0:
        print(f"  - {area_mm_train + area_mm_valid} area/bbox mismatches (inconsistent annotations)")

    seg_nan = stats.get("coco_train_seg_nan", 0) + stats.get("coco_valid_seg_nan", 0)
    if seg_nan > 0:
        print(f"  - {seg_nan} segmentation polygons with NaN coordinates")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
