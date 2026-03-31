#!/usr/bin/env python3
"""
Export SAM3 checkpoint to model-only FP16 and compare with original via cosine similarity.

Steps:
1) Load original checkpoint and extract only {'model': state_dict}
2) Convert floating tensors in model state_dict to FP16 (no cast-back to FP32)
3) Save the FP16 model-only checkpoint in a target folder
4) Compute cosine similarity between original and FP16 model tensors
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch


def dtype_histogram(state_dict: dict[str, torch.Tensor]) -> Counter:
    hist: Counter = Counter()
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            hist[str(tensor.dtype)] += 1
    return hist


def flatten_for_cosine(tensor: torch.Tensor) -> torch.Tensor:
    # Convert to float representation for cosine computation only.
    if tensor.is_complex():
        return torch.view_as_real(tensor).reshape(-1).to(torch.float32)
    return tensor.reshape(-1).to(torch.float32)


def model_cosine_similarity(
    model_a: dict[str, torch.Tensor], model_b: dict[str, torch.Tensor]
) -> float:
    keys_a = set(model_a.keys())
    keys_b = set(model_b.keys())
    common_keys = sorted(keys_a & keys_b)
    if not common_keys:
        raise RuntimeError("No common model keys found for cosine similarity.")

    dot = torch.tensor(0.0, dtype=torch.float64)
    norm_a_sq = torch.tensor(0.0, dtype=torch.float64)
    norm_b_sq = torch.tensor(0.0, dtype=torch.float64)
    finite_count = 0

    for key in common_keys:
        ta = model_a[key]
        tb = model_b[key]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            continue

        va = flatten_for_cosine(ta)
        vb = flatten_for_cosine(tb)
        if va.numel() != vb.numel():
            raise RuntimeError(f"Shape mismatch for key '{key}': {ta.shape} vs {tb.shape}")

        finite_mask = torch.isfinite(va) & torch.isfinite(vb)
        if finite_mask.sum().item() == 0:
            continue

        va_f = va[finite_mask].to(torch.float64)
        vb_f = vb[finite_mask].to(torch.float64)
        finite_count += int(va_f.numel())

        dot += torch.dot(va_f, vb_f)
        norm_a_sq += torch.dot(va_f, va_f)
        norm_b_sq += torch.dot(vb_f, vb_f)

    if finite_count == 0:
        raise RuntimeError("Cannot compute cosine similarity: no finite values found.")
    if norm_a_sq.item() == 0.0 or norm_b_sq.item() == 0.0:
        raise RuntimeError("Cannot compute cosine similarity: zero norm encountered.")

    cosine = dot / (torch.sqrt(norm_a_sq) * torch.sqrt(norm_b_sq))
    return float(cosine.item())


def convert_model_to_fp16(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            converted[key] = value.to(torch.float16)
        else:
            converted[key] = value
    return converted


def main() -> None:
    parser = argparse.ArgumentParser("Export model-only FP16 and compute cosine similarity")
    parser.add_argument(
        "--input-checkpoint",
        default="/home/ec2-user/adb_/New_Yolo_pipe_lineexperiments/concrete_finetune_v1/checkpoints/checkpoint.pt",
        help="Path to original full checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/ec2-user/adb_/New_Yolo_pipe_lineexperiments/testing ones/testing_conc/fp16_model_only_concrete_v1",
        help="Directory to save output FP16 checkpoint",
    )
    parser.add_argument(
        "--output-name",
        default="checkpoint_model_only_fp16.pt",
        help="Output filename",
    )
    args = parser.parse_args()

    input_path = Path(args.input_checkpoint)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name

    if not input_path.exists():
        raise SystemExit(f"Input checkpoint not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading original checkpoint: {input_path}")
    ckpt = torch.load(str(input_path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise SystemExit("Original checkpoint is not a dict.")
    if "model" not in ckpt or not isinstance(ckpt["model"], dict):
        raise SystemExit("Original checkpoint does not contain a valid 'model' key.")

    original_model = ckpt["model"]
    print(f"Original top-level keys: {list(ckpt.keys())}")
    print(f"Original model tensor count: {len(original_model)}")
    print(f"Original dtypes: {dict(dtype_histogram(original_model))}")

    fp16_model = convert_model_to_fp16(original_model)
    model_only_fp16 = {"model": fp16_model}
    torch.save(model_only_fp16, str(output_path))

    print(f"Saved FP16 model-only checkpoint: {output_path}")
    print(f"Saved top-level keys: {list(model_only_fp16.keys())}")
    print(f"Saved dtypes: {dict(dtype_histogram(fp16_model))}")

    fp32_count = sum(
        1
        for v in fp16_model.values()
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32
    )
    print(f"float32 tensor count in saved model: {fp32_count}")

    # Cosine similarity between original model and converted FP16 model.
    cosine = model_cosine_similarity(original_model, fp16_model)
    print(f"cosine_similarity(original_vs_fp16): {cosine:.12f}")
    print("Done.")


if __name__ == "__main__":
    main()
