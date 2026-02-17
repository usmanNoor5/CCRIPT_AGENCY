from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLO-seg on wall segmentation dataset.")
    ap.add_argument("--data", type=Path, default=Path("Angle_project/full_dataset/data.yaml"), help="Path to data.yaml")
    ap.add_argument("--model", type=str, default="yolo26s-seg.pt", help="Base YOLO-seg model")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4, help="Batch size (4 for 8GB Orin)")
    ap.add_argument("--device", type=str, default="0", help="CUDA: 0 | CPU: cpu | Apple: mps")
    ap.add_argument("--optimizer", type=str, default="AdamW")
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--close-mosaic", type=int, default=5)
    ap.add_argument("--mosaic", type=float, default=0.5)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (0.1 = 10%)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--project", type=str, default="runs_api")
    ap.add_argument("--name", type=str, default="yoloseg_wall")
    ap.add_argument("--export-engine", action="store_true", help="Export TensorRT engine after training")
    args = ap.parse_args()

    if not args.data.exists():
        raise SystemExit(f"data.yaml not found: {args.data}")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        raise SystemExit("Missing ultralytics. Install with `pip install ultralytics`.") from e

    model = YOLO(args.model, task="segment")
    model.train(
        data=str(args.data),
        epochs=int(args.epochs),
        patience=int(args.patience),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        optimizer=str(args.optimizer),
        lr0=float(args.lr0),
        lrf=float(args.lrf),
        cos_lr=bool(args.cos_lr),
        warmup_epochs=float(args.warmup_epochs),
        weight_decay=float(args.weight_decay),
        close_mosaic=int(args.close_mosaic),
        mosaic=float(args.mosaic),
        mixup=float(args.mixup),
        fraction=float(args.fraction),
        seed=int(args.seed),
        project=str(args.project),
        name=str(args.name),
    )

    # Load best weights and run validation
    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.exists():
        print(f"\nBest model saved at: {best_path}")
        best_model = YOLO(str(best_path), task="segment")
        print("Running validation on best model...")
        best_model.val(data=str(args.data), imgsz=int(args.imgsz), device=str(args.device))

        if args.export_engine:
            print("Exporting TensorRT engine...")
            best_model.export(format="engine", imgsz=int(args.imgsz), device=str(args.device))
            print(f"Engine saved at: {best_path.with_suffix('.engine')}")
    else:
        print(f"\nWARNING: best.pt not found at {best_path}")


if __name__ == "__main__":
    main()
