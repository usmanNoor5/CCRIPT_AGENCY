#!/usr/bin/env python3
"""
A/B test SAM3 fp32 vs fp16 checkpoints on the same test set.

Metrics:
  - Pixel agreement
  - IoU
  - Dice
  - Cosine similarity of flattened output mask vectors
  - Per-image inference time (fp32/fp16)
"""

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate


def binarize_image(img: np.ndarray, block_size: int = 51, c_val: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_val,
    )
    inverted = cv2.bitwise_not(binary)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def gather_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


class Sam3Predictor:
    def __init__(self, model, resolution=1008, device="cuda", conf=0.3, use_amp: bool = False):
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = model
        self.device = device
        self.conf = conf
        self.use_amp = use_amp and device.startswith("cuda")
        self._processor = Sam3Processor(
            model,
            resolution=resolution,
            device=device,
            confidence_threshold=conf,
        )

    @torch.inference_mode()
    def predict_tile(self, pil_image, text_prompt="walls"):
        from sam3.model import box_ops

        autocast_enabled = self.use_amp
        # Use torch.amp.autocast for forward; when enabled, this runs compute in fp16 on CUDA.
        from torch.amp import autocast
        with autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            state = self._processor.set_image(pil_image)
            text_outputs = self.model.backbone.forward_text([text_prompt], device=self.device)
            state["backbone_out"].update(text_outputs)
            state["geometric_prompt"] = self.model._get_dummy_prompt()

            outputs = self.model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=self._processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.conf
        out_bbox = out_bbox[keep]
        out_masks = out_masks[keep]

        img_h = state["original_height"]
        img_w = state["original_width"]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        _ = boxes * scale_fct[None, :]

        if len(out_masks) > 0:
            masks = interpolate(
                out_masks.unsqueeze(1),
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            masks = (masks > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
        else:
            masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

        return masks


def load_predictor(checkpoint_path: Path, bpe_path: Path, conf: float, device: str, precision: str):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        device="cpu",
        eval_mode=False,
        load_from_HF=False,
        checkpoint_path=None,
        enable_segmentation=True,
    )

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    # Keep weights in fp32 for numerical stability; use AMP for fp16 branch.
    model = model.float().to(device).eval()
    use_amp = precision == "fp16" and device.startswith("cuda")
    return Sam3Predictor(model, resolution=1008, device=device, conf=conf, use_amp=use_amp)


def infer_mask(predictor, image_bgr: np.ndarray, tile_size: int, overlap: int, prompt: str):
    h, w = image_bgr.shape[:2]
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile-size")

    wall_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = image_bgr[y:y2, x:x2]

            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            masks = predictor.predict_tile(Image.fromarray(tile_rgb), text_prompt=prompt)

            tile_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            for i in range(len(masks)):
                tile_mask = np.maximum(tile_mask, masks[i] * 255)

            wall_mask[y:y2, x:x2] = np.maximum(wall_mask[y:y2, x:x2], tile_mask[:th, :tw])

    return wall_mask


def cosine_similarity_binary(mask_a: np.ndarray, mask_b: np.ndarray):
    va = (mask_a.reshape(-1) > 0).astype(np.float32)
    vb = (mask_b.reshape(-1) > 0).astype(np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def main():
    parser = argparse.ArgumentParser("A/B test fp32 vs fp16 SAM3")
    parser.add_argument(
        "--fp32-checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/optimise_checkpoint/checkpoint_model_only.pt",
    )
    parser.add_argument(
        "--fp16-checkpoint",
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoint_model_only_fp16.pt",
    )
    parser.add_argument(
        "--input-folder",
        default="/home/ec2-user/CCRIPT_AGENCY/npw_data/single_pages",
    )
    parser.add_argument(
        "--bpe-path",
        default="/home/ec2-user/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument("--output-csv", default=None, help="Path to write per-image CSV report")
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--prompt", default="walls")
    parser.add_argument("--no-bin", action="store_true")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    images = gather_images(input_folder)
    if not images:
        raise SystemExit(f"No images found in: {input_folder}")

    fp32_ckpt = Path(args.fp32_checkpoint)
    fp16_ckpt = Path(args.fp16_checkpoint)
    bpe_path = Path(args.bpe_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Images: {len(images)}")
    print(f"fp32: {fp32_ckpt}")
    print(f"fp16: {fp16_ckpt}")

    print("Loading fp32 model...")
    pred32 = load_predictor(fp32_ckpt, bpe_path, conf=args.conf, device=device, precision="fp32")
    print("Loading fp16 model...")
    pred16 = load_predictor(fp16_ckpt, bpe_path, conf=args.conf, device=device, precision="fp16")

    rows = []
    for i, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[{i}/{len(images)}] SKIP unreadable: {img_path.name}")
            continue

        if args.no_bin:
            model_input = img
        else:
            model_input = binarize_image(img)

        t0 = time.time()
        m32 = infer_mask(pred32, model_input, args.tile_size, args.overlap, args.prompt)
        t32 = time.time() - t0

        t0 = time.time()
        m16 = infer_mask(pred16, model_input, args.tile_size, args.overlap, args.prompt)
        t16 = time.time() - t0

        a = m32 > 0
        b = m16 > 0
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        iou = float(inter / union) if union > 0 else 1.0
        dice = float((2.0 * inter) / (a.sum() + b.sum())) if (a.sum() + b.sum()) > 0 else 1.0
        pix_acc = float((a == b).mean())
        cos = cosine_similarity_binary(m32, m16)

        rows.append(
            {
                "image": img_path.name,
                "iou": iou,
                "dice": dice,
                "pixel_agreement": pix_acc,
                "cosine_similarity": cos,
                "fp32_seconds": t32,
                "fp16_seconds": t16,
                "speedup_x": (t32 / t16) if t16 > 0 else math.inf,
            }
        )
        print(
            f"[{i}/{len(images)}] {img_path.name} | "
            f"IoU={iou:.6f} Dice={dice:.6f} Cos={cos:.6f} "
            f"| t32={t32:.2f}s t16={t16:.2f}s"
        )

    if not rows:
        raise SystemExit("No images were processed.")

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print("\n=== Aggregate A/B Summary ===")
    print(f"Mean IoU            : {avg('iou'):.6f}")
    print(f"Mean Dice           : {avg('dice'):.6f}")
    print(f"Mean Pixel Agreement: {avg('pixel_agreement'):.6f}")
    print(f"Mean Cosine Sim     : {avg('cosine_similarity'):.6f}")
    print(f"Mean fp32 time (s)  : {avg('fp32_seconds'):.3f}")
    print(f"Mean fp16 time (s)  : {avg('fp16_seconds'):.3f}")
    print(f"Mean speedup (x)    : {avg('speedup_x'):.3f}")

    csv_path = (
        Path(args.output_csv)
        if args.output_csv
        else Path("/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/ab_test_fp32_vs_fp16.csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-image CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
