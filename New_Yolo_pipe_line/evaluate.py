#!/usr/bin/env python3
"""
evaluate.py — Evaluate YOLO-seg model at multiple confidence thresholds.

Computes per-threshold: IoU, Dice/F1, Precision, Recall, F1 (detection).
Runs on the test split tiles (already 640x640 with ground truth labels).

Usage:
    python3.12 evaluate.py
    python3.12 evaluate.py --dataset yolo_raw_dataset --model runs/segment/wall_seg_raw_v1/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_gt_mask(label_path: str, img_size: int = 640) -> np.ndarray:
    """Load YOLO-seg label file and render as binary mask."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    label_file = Path(label_path)
    if not label_file.exists():
        return mask
    text = label_file.read_text().strip()
    if not text:
        return mask

    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 7:  # class + at least 3 points
            continue
        coords = list(map(float, parts[1:]))
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_size)
            y = int(coords[i + 1] * img_size)
            points.append([x, y])
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def pred_mask_from_results(results, img_size: int = 640) -> np.ndarray:
    """Extract combined binary mask from YOLO prediction results."""
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for result in results:
        if result.masks is not None:
            for mask_data in result.masks.data:
                m = mask_data.cpu().numpy()
                m_resized = cv2.resize(m, (img_size, img_size))
                mask = np.maximum(mask, (m_resized > 0.5).astype(np.uint8) * 255)
    return mask


def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    """Compute IoU, Dice, Precision, Recall from binary masks."""
    gt_bool = gt > 0
    pred_bool = pred > 0

    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    gt_sum = gt_bool.sum()
    pred_sum = pred_bool.sum()

    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1.0
    precision = intersection / pred_sum if pred_sum > 0 else 1.0
    recall = intersection / gt_sum if gt_sum > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_pixels": int(gt_sum),
        "pred_pixels": int(pred_sum),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yolo_bin_dataset")
    parser.add_argument("--model", default="runs/segment/wall_seg_v1/weights/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    img_dir = dataset / "images" / args.split
    lbl_dir = dataset / "labels" / args.split

    images = sorted(img_dir.glob("*.png"))
    # Evaluate on all samples
    all_images = []
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        has_label = lbl_path.exists() and lbl_path.read_text().strip() != ""
        all_images.append((img_path, lbl_path, has_label))

    total = len(all_images)
    pos = sum(1 for _, _, h in all_images if h)
    neg = total - pos
    print(f"Dataset: {dataset} / {args.split}")
    print(f"Tiles: {total} ({pos} positive, {neg} negative)")
    print(f"Model: {args.model}")
    print()

    model = YOLO(args.model)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    print(f"{'Conf':>6} | {'IoU':>7} | {'Dice':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'FP tiles':>8} | {'FN tiles':>8}")
    print("-" * 80)

    best_f1 = 0
    best_thresh = 0

    for conf in thresholds:
        metrics_list = []
        false_pos_tiles = 0  # negative tiles with predictions
        false_neg_tiles = 0  # positive tiles with no predictions

        for img_path, lbl_path, has_label in all_images:
            img = cv2.imread(str(img_path))
            results = model.predict(img, conf=conf, verbose=False, imgsz=640)
            pred = pred_mask_from_results(results)
            gt = load_gt_mask(str(lbl_path))

            m = compute_metrics(gt, pred)
            metrics_list.append(m)

            pred_has = pred.sum() > 0
            if not has_label and pred_has:
                false_pos_tiles += 1
            if has_label and not pred_has:
                false_neg_tiles += 1

        # Average metrics (only over tiles that have GT or predictions)
        relevant = [m for m in metrics_list if m["gt_pixels"] > 0 or m["pred_pixels"] > 0]
        if relevant:
            avg_iou = np.mean([m["iou"] for m in relevant])
            avg_dice = np.mean([m["dice"] for m in relevant])
            avg_prec = np.mean([m["precision"] for m in relevant])
            avg_rec = np.mean([m["recall"] for m in relevant])
            avg_f1 = np.mean([m["f1"] for m in relevant])
        else:
            avg_iou = avg_dice = avg_prec = avg_rec = avg_f1 = 0.0

        print(f"{conf:>6.2f} | {avg_iou:>7.4f} | {avg_dice:>7.4f} | {avg_prec:>7.4f} | {avg_rec:>7.4f} | {avg_f1:>7.4f} | {false_pos_tiles:>8} | {false_neg_tiles:>8}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_thresh = conf

    print("-" * 80)
    print(f"\nBest F1: {best_f1:.4f} at conf={best_thresh:.2f}")
    print(f"\nRecommended: --conf {best_thresh}")


if __name__ == "__main__":
    main()
