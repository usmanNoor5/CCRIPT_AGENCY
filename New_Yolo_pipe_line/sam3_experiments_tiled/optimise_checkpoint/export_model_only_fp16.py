#!/usr/bin/env python3
"""
Export a training checkpoint to model-only FP16.

Input checkpoint usually has keys like:
['model', 'optimizer', 'epoch', ...]

Output checkpoint contains exactly one top-level key:
{'model': <state_dict>}

All floating tensors in state_dict are converted to float16 and kept as float16
(no cast back to float32).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def dtype_histogram(state_dict: dict[str, torch.Tensor]) -> Counter:
    hist: Counter = Counter()
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            hist[str(tensor.dtype)] += 1
    return hist


def convert_state_dict_to_fp16(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            converted[key] = value.to(dtype=torch.float16)
        else:
            converted[key] = value
    return converted


def main() -> None:
    parser = argparse.ArgumentParser("Export model-only FP16 checkpoint")
    parser.add_argument(
        "--input-checkpoint",
        default="/home/ec2-user/adb_/New_Yolo_pipe_lineexperiments/footing_cement_finetune/checkpoints/checkpoint.pt",
        help="Path to full training checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ec2-user/adb_/New_Yolo_pipe_line/sam3_experiments_tiled/optimise_checkpoint/fp16_model_only",
        help="Directory to save exported checkpoint",
    )
    parser.add_argument(
        "--output-name",
        default="checkpoint_model_only_fp16.pt",
        help="Output checkpoint filename",
    )
    args = parser.parse_args()

    input_path = Path(args.input_checkpoint)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    if not input_path.exists():
        raise SystemExit(f"Input checkpoint not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(str(input_path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise SystemExit("Input checkpoint is not a dict.")
    if "model" not in ckpt or not isinstance(ckpt["model"], dict):
        raise SystemExit("Input checkpoint does not contain a valid 'model' state_dict.")

    original_state = ckpt["model"]
    print(f"Original model tensor count: {len(original_state)}")
    print(f"Original dtypes: {dict(dtype_histogram(original_state))}")

    fp16_state = convert_state_dict_to_fp16(original_state)
    model_only_fp16 = {"model": fp16_state}

    torch.save(model_only_fp16, str(output_path))

    print(f"Saved model-only FP16 checkpoint: {output_path}")
    print(f"Top-level keys: {list(model_only_fp16.keys())}")
    print(f"Converted dtypes: {dict(dtype_histogram(fp16_state))}")
    print("FP16 tensors use 2 bytes per element (float32 uses 4 bytes).")
    print("No cast-back to float32 performed.")


if __name__ == "__main__":
    main()
