#!/usr/bin/env python3
"""
Convert a SAM3 model-only checkpoint to FP16.

Input checkpoint format expected:
{
  "model": { ... state_dict tensors ... }
}
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser("Convert model-only checkpoint to fp16")
    parser.add_argument(
        "--input-checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/optimise_checkpoint/checkpoint_model_only.pt",
        help="Path to source model-only checkpoint",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoint_model_only_fp16.pt",
        help="Path to save fp16 checkpoint",
    )
    args = parser.parse_args()

    input_path = Path(args.input_checkpoint)
    output_path = Path(args.output_checkpoint)

    if not input_path.exists():
        raise SystemExit(f"Input checkpoint not found: {input_path}")

    print(f"Loading checkpoint: {input_path}")
    ckpt = torch.load(str(input_path), map_location="cpu")

    if not isinstance(ckpt, dict) or "model" not in ckpt or not isinstance(ckpt["model"], dict):
        raise SystemExit("Input checkpoint must be a dict with key 'model' containing state_dict.")

    model_state = ckpt["model"]
    fp16_state = {}
    converted = 0
    kept = 0

    for key, value in model_state.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            fp16_state[key] = value.half()
            converted += 1
        else:
            fp16_state[key] = value
            kept += 1

    out_ckpt = {"model": fp16_state}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(output_path))

    print(f"Saved FP16 checkpoint: {output_path}")
    print(f"Converted tensors to fp16: {converted}")
    print(f"Kept non-floating tensors: {kept}")


if __name__ == "__main__":
    main()
