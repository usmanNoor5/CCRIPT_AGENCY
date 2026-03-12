#!/usr/bin/env python3
import argparse
import torch


def main():
    parser = argparse.ArgumentParser("Inspect checkpoint keys")
    parser.add_argument(
        "--checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoints/checkpoint.pt",
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict):
        print("Top-level keys:")
        print(list(ckpt.keys()))
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            print("\n' model ' key exists and contains state_dict.")
            print(f"Number of model weights: {len(ckpt['model'])}")
            sample_keys = list(ckpt["model"].keys())[:20]
            print("\nFirst 20 model keys:")
            for k in sample_keys:
                print(k)
    else:
        print("Checkpoint is not a dict. Type:", type(ckpt))


if __name__ == "__main__":
    main()
