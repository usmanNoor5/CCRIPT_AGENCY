#!/usr/bin/env python3
"""
train.py — Launch YOLO26x-seg wall segmentation training.

Usage:
    screen -S yolo_train
    python3.12 train.py
    # Ctrl+A, D to detach
"""

from pathlib import Path
from ultralytics import YOLO


def main():
    base = Path(__file__).parent
    data_yaml = base / "yolo_raw_dataset" / "data.yaml"
    model_weights = base / "yolo26x-seg.pt"

    if not data_yaml.exists():
        raise SystemExit(f"Dataset not found: {data_yaml}")
    if not model_weights.exists():
        raise SystemExit(f"Model weights not found: {model_weights}")

    print("=== YOLO26x-seg Wall Segmentation Training ===")
    print(f"Dataset: {data_yaml}")
    print(f"Model:   {model_weights}")
    print()

    model = YOLO(str(model_weights))

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        patience=20,
        workers=8,
        device=0,
        project=str(base / "runs" / "segment"),
        name="wall_seg_raw_v1",
    )

    print()
    print("=== Training complete ===")
    print(f"Results: {base / 'runs' / 'segment' / 'wall_seg_v1'}")


if __name__ == "__main__":
    main()
