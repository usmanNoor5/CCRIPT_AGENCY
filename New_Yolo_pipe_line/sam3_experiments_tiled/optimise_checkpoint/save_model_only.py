#!/usr/bin/env python3
"""
Save a model-only checkpoint from a full training checkpoint.

Input checkpoint can contain keys like:
['model', 'optimizer', 'epoch', 'loss', ...]

This script writes a new .pt that only contains:
{'model': <state_dict>}
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser("Extract model-only checkpoint")
    parser.add_argument(
        "--input-checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoints/checkpoint.pt",
        help="Path to full checkpoint",
    )
    parser.add_argument(
        "--output-checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/optimise_checkpoint/checkpoint_model_only.pt",
        help="Path to write new model-only checkpoint",
    )
    args = parser.parse_args()

    input_path = Path(args.input_checkpoint)
    output_path = Path(args.output_checkpoint)

    if not input_path.exists():
        raise SystemExit(f"Input checkpoint not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    ckpt = torch.load(str(input_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise SystemExit("Input checkpoint is not a dict.")
    if "model" not in ckpt:
        raise SystemExit("Input checkpoint does not contain 'model' key.")
    if not isinstance(ckpt["model"], dict):
        raise SystemExit("'model' key exists but is not a state_dict dict.")

    model_only = {"model": ckpt["model"]}
    torch.save(model_only, str(output_path))

    print(f"Saved model-only checkpoint: {output_path}")
    print(f"Top-level keys in new file: {list(model_only.keys())}")
    print(f"Number of model tensors: {len(model_only['model'])}")


if __name__ == "__main__":
    main()
