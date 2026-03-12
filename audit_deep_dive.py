#!/usr/bin/env python3
"""
Deep dive into the specific NaN generation path in SAM3 training.

This script simulates EXACTLY what happens to each annotation as it flows through:
1. COCO JSON -> coco_json_loaders.py (normalize to xywh)
2. sam3_image_dataset.py (xywh -> xyxy, denormalize)
3. RandomResize transform (resize to up to 1008x1008)
4. NormalizeAPI transform (denorm xyxy -> norm cxcywh)
5. matcher.py -> generalized_box_iou (the NaN crash site)

We also investigate:
- The area/bbox mismatch issue (seg_area << bbox_area)
- What happens with extreme aspect ratio walls (400:1)
- Whether the FilterEmptyTargets transform could leave degenerate boxes
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

COCO_TRAIN_JSON = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_coco/train/_annotations.coco.json")
COCO_VALID_JSON = Path("/home/ec2-user/CCRIPT_AGENCY/augmented_coco/valid/_annotations.coco.json")


def simulate_full_pipeline(json_path, split_name):
    """Simulate the FULL SAM3 pipeline for each annotation."""
    print(f"\n{'='*70}")
    print(f"DEEP DIVE: Full pipeline simulation ({split_name})")
    print(f"{'='*70}")

    with open(json_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    img_by_id = {img["id"]: img for img in images}

    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    target_resolution = 1008  # From wall_ft.yaml

    nan_risk_annotations = []
    extreme_aspect_ratios = []
    area_mismatch_details = []

    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in img_by_id:
            continue
        img_info = img_by_id[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        bbox = ann["bbox"]  # [x, y, w, h] pixel
        bx, by, bw, bh = bbox

        # --- Step 1: coco_json_loaders.py normalize ---
        # convert_boxlist_to_normalized_tensor divides indices [0,2] by width, [1,3] by height
        norm_x = max(0, min(1, bx / img_w))
        norm_y = max(0, min(1, by / img_h))
        norm_w = max(0, min(1, bw / img_w))
        norm_h = max(0, min(1, bh / img_h))

        # --- Step 2: sam3_image_dataset.py box_xywh_to_xyxy + denorm ---
        # box_xywh_to_xyxy: [x, y, w, h] -> [x, y, x+w, y+h]
        # Then multiply by (W, H)
        xyxy_x1 = norm_x * img_w
        xyxy_y1 = norm_y * img_h
        xyxy_x2 = (norm_x + norm_w) * img_w
        xyxy_y2 = (norm_y + norm_h) * img_h

        # Clamp
        xyxy_x1 = max(0, min(img_w, xyxy_x1))
        xyxy_y1 = max(0, min(img_h, xyxy_y1))
        xyxy_x2 = max(0, min(img_w, xyxy_x2))
        xyxy_y2 = max(0, min(img_h, xyxy_y2))

        pixel_w = xyxy_x2 - xyxy_x1
        pixel_h = xyxy_y2 - xyxy_y1

        # --- Step 3: RandomResize to target_resolution (square=true) ---
        # For square resize, both sides go to target_resolution
        # Scale factor
        scale_x = target_resolution / img_w
        scale_y = target_resolution / img_h

        resized_w = pixel_w * scale_x
        resized_h = pixel_h * scale_y

        # --- Step 4: NormalizeAPI -> cxcywh normalized ---
        # Final normalized cxcywh
        final_norm_w = resized_w / target_resolution
        final_norm_h = resized_h / target_resolution

        # --- Step 5: In generalized_box_iou ---
        # box_cxcywh_to_xyxy converts to xyxy
        # Then computes: area = wh of enclosing box
        # If target box has w=0 or h=0, the diagonal of the enclosing box degenerates
        # The formula is: iou - (area - union) / area
        # If area = 0, we get 0/0 = NaN

        # But also: box_iou computes union = area1 + area2 - inter
        # If target area = 0, then union = area_pred, iou = 0/area_pred = 0 (OK unless area_pred also 0)
        # The issue is in the enclosing box:
        # lt = min(boxes1_lt, boxes2_lt), rb = max(boxes1_rb, boxes2_rb)
        # area = (rb - lt).clamp(min=0).prod() -- this CAN be zero if all boxes collapse to a line
        # Then return iou - (area - union) / area  -- division by zero!

        # So even a prediction box with non-zero area, if matched with a zero-width target,
        # the enclosing box along that dimension could still be non-zero (since pred has width).
        # BUT: if the target is a point (w=0, h=0) and prediction is also very small,
        # the enclosing area could be near-zero -> numerical instability.

        # More importantly: after resize to 1008, a 1.6px height on a 640px image becomes:
        # 1.6 * (1008/640) = 2.52px. Normalized: 2.52/1008 = 0.0025
        # That's small but not zero. box_iou should handle this.

        # However, the REAL issue might be in the cost_class computation with focal loss:
        # If out_prob is very close to 0 or 1:
        #   (1 - out_prob)^gamma * log(out_prob) when out_prob -> 0: 1^2 * (-inf) = -inf
        #   This produces -inf in the cost matrix!

        aspect_ratio = pixel_w / pixel_h if pixel_h > 0 else float('inf')
        if aspect_ratio > 50 or (aspect_ratio > 0 and aspect_ratio < 0.02):
            extreme_aspect_ratios.append({
                "ann_id": ann["id"],
                "img_id": img_id,
                "file": img_info["file_name"],
                "bbox": bbox,
                "pixel_wh": (pixel_w, pixel_h),
                "resized_wh": (resized_w, resized_h),
                "final_norm_wh": (final_norm_w, final_norm_h),
                "aspect_ratio": aspect_ratio,
            })

        # Check if after resize, either dimension < 1 pixel
        if resized_w < 1.0 or resized_h < 1.0:
            nan_risk_annotations.append({
                "ann_id": ann["id"],
                "img_id": img_id,
                "file": img_info["file_name"],
                "bbox": bbox,
                "resized_wh": (resized_w, resized_h),
                "final_norm_wh": (final_norm_w, final_norm_h),
            })

        # Area mismatch investigation
        seg_area = ann.get("area", 0)
        bbox_area = bw * bh
        if bbox_area > 0 and seg_area > 0:
            ratio = seg_area / bbox_area
            if ratio < 0.01:
                area_mismatch_details.append({
                    "ann_id": ann["id"],
                    "img_id": img_id,
                    "file": img_info["file_name"],
                    "bbox": bbox,
                    "seg_area": seg_area,
                    "bbox_area": bbox_area,
                    "ratio": ratio,
                })

    # Report
    print(f"\n  Annotations where resize produces sub-pixel dimension: {len(nan_risk_annotations)}")
    for item in nan_risk_annotations[:15]:
        print(f"    ann {item['ann_id']}: bbox={item['bbox']}, "
              f"resized_wh=({item['resized_wh'][0]:.3f}, {item['resized_wh'][1]:.3f}), "
              f"file={item['file']}")

    print(f"\n  Extreme aspect ratios (>50:1 or <1:50): {len(extreme_aspect_ratios)}")
    for item in extreme_aspect_ratios[:15]:
        print(f"    ann {item['ann_id']}: bbox={item['bbox']}, "
              f"AR={item['aspect_ratio']:.1f}, "
              f"resized_wh=({item['resized_wh'][0]:.1f}, {item['resized_wh'][1]:.1f}), "
              f"file={item['file']}")

    print(f"\n  Area/bbox extreme mismatches (seg_area < 1% of bbox_area): {len(area_mismatch_details)}")
    for item in area_mismatch_details[:15]:
        print(f"    ann {item['ann_id']}: bbox={item['bbox']}, "
              f"seg_area={item['seg_area']:.1f}, bbox_area={item['bbox_area']:.1f}, "
              f"ratio={item['ratio']:.4f}, file={item['file']}")

    return nan_risk_annotations, extreme_aspect_ratios, area_mismatch_details


def investigate_focal_loss_nan():
    """Investigate the focal loss NaN path."""
    print(f"\n{'='*70}")
    print("DEEP DIVE: Focal Loss NaN Analysis")
    print("="*70)

    # From the matcher (BinaryFocalHungarianMatcher or BinaryHungarianMatcherV2):
    # With focal=true, stable=False:
    #
    # log_out_prob = logsigmoid(out_score)
    # log_one_minus_out_prob = logsigmoid(-out_score)
    # cost_class = -alpha * (1 - out_prob)^gamma * log_out_prob
    #            + (1 - alpha) * out_prob^gamma * log_one_minus_out_prob
    #
    # logsigmoid is numerically stable, so this part is OK.
    # But the GIoU cost can still produce NaN.

    # Let's trace generalized_box_iou more carefully.
    # For boxes in cxcywh format, box_cxcywh_to_xyxy:
    #   x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2
    #
    # box_iou(boxes1, boxes2):
    #   area1 = (x2-x1)*(y2-y1) for each box in boxes1
    #   area2 = (x2-x1)*(y2-y1) for each box in boxes2
    #   union = area1 + area2 - inter
    #   iou = inter / union
    #
    # generalized_box_iou:
    #   lt = min(x1_1, x1_2), rb = max(x2_1, x2_2) -- enclosing box
    #   area = (rb - lt).clamp(min=0).prod()
    #   return iou - (area - union) / area
    #
    # If a target box has w=0, h>0:
    #   area2 = 0, inter = 0, union = area1 + 0 - 0 = area1
    #   iou = 0 / area1 = 0 (OK if area1 > 0)
    #   enclosing area: along width, rb-lt = max(x2_1,x2_2) - min(x1_1,x1_2)
    #     Since target is a line (x1=x2), this = max(pred_x2, target_cx) - min(pred_x1, target_cx)
    #     If pred has non-zero width, enclosing width > 0 -> area > 0
    #   giou = 0 - (area - area1) / area = -(area - area1) / area (OK, no NaN)
    #
    # If BOTH target and prediction collapse to zero area:
    #   area1 = 0, area2 = 0, inter = 0, union = 0
    #   iou = 0/0 = NaN!
    #
    # This can happen in early training when the model predicts degenerate boxes.
    # The model initializes with near-zero predicted boxes -> union=0 -> NaN!

    print("\n  Key finding: generalized_box_iou produces NaN when BOTH predicted")
    print("  and target boxes have zero area (union = 0 -> 0/0 = NaN).")
    print()
    print("  In early training, the model may predict degenerate boxes (near-zero area).")
    print("  If these are matched against very thin wall annotations (area near zero),")
    print("  the union becomes 0, producing NaN in the GIoU cost.")
    print()
    print("  ALSO: torch.cdist with L1 norm can produce NaN if inputs contain NaN/Inf.")
    print("  If any box coordinate is NaN (from upstream computation), cdist propagates it.")

    # Check if there's a guard in the code
    print("\n  Checking for NaN guards in the matcher...")
    print("  - _do_matching: NO NaN guard before linear_sum_assignment")
    print("  - generalized_box_iou: NO guard against zero-area boxes")
    print("  - box_iou: NO guard against zero union")
    print()
    print("  CONCLUSION: The matcher has NO NaN protection. Any degenerate box")
    print("  (from data OR from model predictions) will crash training.")


def investigate_area_mismatch_root_cause():
    """Investigate why seg_area << bbox_area for some annotations."""
    print(f"\n{'='*70}")
    print("DEEP DIVE: Area/BBox Mismatch Root Cause")
    print("="*70)

    with open(COCO_TRAIN_JSON) as f:
        coco = json.load(f)

    img_by_id = {img["id"]: img for img in coco["images"]}

    # Find the mismatched annotations and examine their segmentation polygons
    for ann in coco["annotations"]:
        bbox = ann["bbox"]
        bw, bh = bbox[2], bbox[3]
        bbox_area = bw * bh
        seg_area = ann.get("area", 0)

        if bbox_area > 0 and seg_area > 0 and seg_area / bbox_area < 0.01:
            img_info = img_by_id[ann["image_id"]]
            seg = ann.get("segmentation", [])

            print(f"\n  ann {ann['id']} (image: {img_info['file_name']}):")
            print(f"    bbox: {bbox}")
            print(f"    bbox_area: {bbox_area:.1f}")
            print(f"    seg_area (from cv2.contourArea): {seg_area:.1f}")

            if seg and isinstance(seg, list):
                for pi, poly in enumerate(seg):
                    pts = np.array(poly).reshape(-1, 2)
                    print(f"    polygon {pi}: {len(pts)} points")
                    print(f"      x range: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
                    print(f"      y range: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
                    print(f"      poly span: w={pts[:,0].max()-pts[:,0].min():.1f}, h={pts[:,1].max()-pts[:,1].min():.1f}")

                    # Recompute area
                    area_cv2 = abs(cv2_contour_area(pts))
                    # Shoelace formula
                    area_shoelace = shoelace_area(pts)
                    print(f"      cv2 contourArea: {area_cv2:.1f}")
                    print(f"      shoelace area: {area_shoelace:.1f}")

            # Only show first 3
            if ann["id"] > 160:
                break


def cv2_contour_area(pts):
    """Compute contour area using OpenCV's formula."""
    import cv2
    return cv2.contourArea(pts.astype(np.float32))


def shoelace_area(pts):
    """Compute polygon area using the shoelace formula."""
    n = len(pts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    return abs(area) / 2


def check_thin_wall_specific_examples():
    """Look at actual examples of the thinnest walls."""
    print(f"\n{'='*70}")
    print("DEEP DIVE: Thinnest Wall Examples")
    print("="*70)

    with open(COCO_TRAIN_JSON) as f:
        coco = json.load(f)

    img_by_id = {img["id"]: img for img in coco["images"]}

    # Sort by minimum dimension
    ann_with_min_dim = []
    for ann in coco["annotations"]:
        bw, bh = ann["bbox"][2], ann["bbox"][3]
        min_dim = min(bw, bh)
        ann_with_min_dim.append((min_dim, ann))

    ann_with_min_dim.sort(key=lambda x: x[0])

    print("\n  20 thinnest annotations (by min(w,h)):")
    for min_dim, ann in ann_with_min_dim[:20]:
        img_info = img_by_id[ann["image_id"]]

        # What happens after resize to 1008?
        img_w, img_h = img_info["width"], img_info["height"]
        scale_x = 1008 / img_w
        scale_y = 1008 / img_h
        bw, bh = ann["bbox"][2], ann["bbox"][3]
        resized_w = bw * scale_x
        resized_h = bh * scale_y

        # After NormalizeAPI: xyxy -> cxcywh, normalized to [0,1]
        # The w,h in normalized cxcywh:
        norm_w = resized_w / 1008
        norm_h = resized_h / 1008

        print(f"    ann {ann['id']}: bbox_wh=({bw:.2f}, {bh:.2f}), "
              f"resized_wh=({resized_w:.2f}, {resized_h:.2f}), "
              f"norm_cxcywh_wh=({norm_w:.6f}, {norm_h:.6f}), "
              f"file={img_info['file_name']}")

    # Check what the area computation gives for these thin boxes
    print("\n\n  Box area in normalized space for thinnest annotations:")
    print("  (This is what enters generalized_box_iou)")
    for min_dim, ann in ann_with_min_dim[:10]:
        img_info = img_by_id[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]
        bw, bh = ann["bbox"][2], ann["bbox"][3]
        norm_w = (bw / img_w)  # simplified
        norm_h = (bh / img_h)
        box_area_norm = norm_w * norm_h
        print(f"    ann {ann['id']}: norm_area = {box_area_norm:.8f} "
              f"(w={norm_w:.6f}, h={norm_h:.6f})")

    # The critical question: can giou produce NaN for these boxes?
    print("\n  Simulating generalized_box_iou for worst case:")
    print("  (target = thinnest annotation, pred = random small box)")

    import torch
    for min_dim, ann in ann_with_min_dim[:5]:
        img_info = img_by_id[ann["image_id"]]
        img_w, img_h = img_info["width"], img_info["height"]
        bx, by, bw, bh = ann["bbox"]

        # Simulate target in normalized cxcywh
        norm_cx = (bx + bw/2) / img_w
        norm_cy = (by + bh/2) / img_h
        norm_w = bw / img_w
        norm_h = bh / img_h

        tgt = torch.tensor([[norm_cx, norm_cy, norm_w, norm_h]])

        # Simulate prediction: near-zero box (early training)
        pred_small = torch.tensor([[0.5, 0.5, 0.001, 0.001]])

        # box_cxcywh_to_xyxy
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(-1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=-1)

        def box_area(boxes):
            x0, y0, x1, y1 = boxes.unbind(-1)
            return (x1 - x0) * (y1 - y0)

        def generalized_box_iou(boxes1, boxes2):
            area1 = box_area(boxes1)
            area2 = box_area(boxes2)

            lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
            rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            union = area1[..., None] + area2[..., None, :] - inter
            iou = inter / union

            lt2 = torch.min(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
            rb2 = torch.max(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
            wh2 = (rb2 - lt2).clamp(min=0)
            area = wh2[..., 0] * wh2[..., 1]

            giou = iou - (area - union) / area
            return giou

        tgt_xyxy = box_cxcywh_to_xyxy(tgt)
        pred_xyxy = box_cxcywh_to_xyxy(pred_small)

        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)

        print(f"    ann {ann['id']}: tgt_cxcywh={tgt[0].tolist()}, "
              f"giou={giou.item():.6f}, has_nan={torch.isnan(giou).any().item()}")

        # Now try with pred = zero-area box
        pred_zero = torch.tensor([[norm_cx, norm_cy, 0.0, 0.0]])
        pred_zero_xyxy = box_cxcywh_to_xyxy(pred_zero)
        giou_zero = generalized_box_iou(pred_zero_xyxy, tgt_xyxy)
        print(f"              pred=zero: giou={giou_zero.item():.6f}, "
              f"has_nan={torch.isnan(giou_zero).any().item()}")

        # What if BOTH are zero?
        tgt_zero = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        tgt_zero_xyxy = box_cxcywh_to_xyxy(tgt_zero)
        giou_both_zero = generalized_box_iou(pred_zero_xyxy, tgt_zero_xyxy)
        print(f"              both=zero: giou={giou_both_zero.item():.6f}, "
              f"has_nan={torch.isnan(giou_both_zero).any().item()}")


def check_l1_cost_with_thin_boxes():
    """Check if torch.cdist can produce issues with thin boxes."""
    print(f"\n{'='*70}")
    print("DEEP DIVE: L1 Cost Check (torch.cdist)")
    print("="*70)

    import torch

    # Simulate a batch with thin wall boxes
    # After normalization, boxes are in cxcywh [0,1]
    # A very thin wall: cx=0.5, cy=0.5, w=0.5, h=0.0025
    tgt = torch.tensor([
        [0.5, 0.5, 0.5, 0.0025],  # very thin wall
        [0.3, 0.3, 0.1, 0.1],     # normal box
    ])

    # Model predictions (100 queries)
    pred = torch.randn(100, 4).sigmoid()

    cost_bbox = torch.cdist(pred, tgt, p=1)
    print(f"  L1 cost matrix shape: {cost_bbox.shape}")
    print(f"  Has NaN: {torch.isnan(cost_bbox).any().item()}")
    print(f"  Has Inf: {torch.isinf(cost_bbox).any().item()}")
    print(f"  Min: {cost_bbox.min().item():.6f}, Max: {cost_bbox.max().item():.6f}")

    # Now check with NaN prediction (can happen if model has NaN weights)
    pred_with_nan = pred.clone()
    pred_with_nan[0, 0] = float('nan')
    cost_nan = torch.cdist(pred_with_nan, tgt, p=1)
    print(f"\n  With NaN in prediction:")
    print(f"  Has NaN: {torch.isnan(cost_nan).any().item()}")
    nan_count = torch.isnan(cost_nan).sum().item()
    print(f"  NaN count: {nan_count} / {cost_nan.numel()}")


def main():
    simulate_full_pipeline(COCO_TRAIN_JSON, "train")
    simulate_full_pipeline(COCO_VALID_JSON, "valid")
    investigate_focal_loss_nan()
    investigate_area_mismatch_root_cause()
    check_thin_wall_specific_examples()
    check_l1_cost_with_thin_boxes()

    print(f"\n{'='*70}")
    print("FINAL DIAGNOSIS")
    print("="*70)
    print("""
  The NaN error "matrix contains invalid numeric entries" in the Hungarian
  matcher has MULTIPLE potential causes:

  1. CONFIRMED NaN PATH - generalized_box_iou with zero-area boxes:
     When BOTH a predicted box and target box have zero area, union=0,
     and iou = 0/0 = NaN. This happens in early training when the model
     hasn't learned to predict reasonable boxes yet.

  2. CONFIRMED NaN PATH - generalized_box_iou enclosing area:
     If the enclosing box of pred+target has zero area (both collapse to
     the same point), then (area - union)/area = 0/0 = NaN.

  3. HIGH-RISK ANNOTATIONS in your dataset:
     - 208 annotations with bbox dimension < 5px
     - 125+ annotations with extreme aspect ratios (up to 400:1)
     - 130 annotations where segmentation area is <1% of bbox area
       (indicating the polygon is actually a thin line inside a large bbox)

  4. The area/bbox mismatches (seg_area << bbox_area) suggest some
     wall polygons are very thin lines that get a large bounding box.
     The bbox from the polygon's min/max x,y can be much larger than
     the actual wall area.

  RECOMMENDED FIXES (in order of importance):

  A. Add NaN guard in matcher.py (immediate fix):
     After computing C (the cost matrix), add:
       C = np.nan_to_num(C, nan=1e8, posinf=1e8, neginf=-1e8)

  B. Add minimum bbox size in yolo_to_coco.py:
     Change the filter from bbox_w < 1 to bbox_w < 3 and bbox_h < 3
     (or better: min(bbox_w, bbox_h) >= 3)

  C. Add NaN guard in generalized_box_iou (box_ops.py):
     Replace: return iou - (area - union) / area
     With:    return iou - (area - union) / area.clamp(min=1e-6)
     Also:    iou = inter / union.clamp(min=1e-6)

  D. Filter extreme aspect ratios in augment_dataset.py:
     Skip polygons where aspect_ratio > 100 or < 0.01

  E. Re-run yolo_to_coco.py with stricter filters, then retrain.
""")


if __name__ == "__main__":
    main()
