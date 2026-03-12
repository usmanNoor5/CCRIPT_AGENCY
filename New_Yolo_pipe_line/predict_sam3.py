#!/usr/bin/env python3
"""
predict_sam3.py — Run fine-tuned SAM3 model on a floor plan image.

Steps:
  1. Load image, optionally binarize
  2. Tile into 1008x1008 patches
  3. Run SAM3 inference on each tile with text prompt "walls"
  4. Stitch segmentation masks back onto full image
  5. Save result with wall overlay

Usage:
    python3.12 predict_sam3.py <image_path>
    python3.12 predict_sam3.py <image_path> --no-bin
    python3.12 predict_sam3.py <image_path> --conf 0.3
    python3.12 predict_sam3.py <image_path> --save-tiles
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate
from PIL import Image


def binarize_image(img: np.ndarray, block_size: int = 51, C: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C,
    )
    inverted = cv2.bitwise_not(binary)
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


class Sam3Predictor:
    """Predictor for SAM3 with segmentation head (pixel masks)."""

    def __init__(self, model, resolution=1008, device="cuda", conf=0.5):
        from sam3.model.sam3_image_processor import Sam3Processor
        self.model = model
        self.device = device
        self.conf = conf
        self._processor = Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=conf)

    @torch.inference_mode()
    def predict_tile(self, pil_image, text_prompt="walls"):
        from sam3.model import box_ops

        # Set image
        state = self._processor.set_image(pil_image)

        # Set text prompt
        text_outputs = self.model.backbone.forward_text([text_prompt], device=self.device)
        state["backbone_out"].update(text_outputs)
        state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Run grounding
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
        out_probs = out_probs[keep]
        out_bbox = out_bbox[keep]
        out_masks = out_masks[keep]

        # Convert boxes cxcywh -> xyxy and scale to image size
        img_h = state["original_height"]
        img_w = state["original_width"]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        # Interpolate masks to original image size and threshold
        if len(out_masks) > 0:
            masks = interpolate(
                out_masks.unsqueeze(1),
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            masks = (masks > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)  # (N, H, W)
        else:
            masks = np.zeros((0, img_h, img_w), dtype=np.uint8)

        return boxes.cpu().numpy(), out_probs.cpu().numpy(), masks


def load_sam3_model(checkpoint_path, device="cuda", conf=0.5):
    from sam3.model_builder import build_sam3_image_model

    bpe_path = "/home/ec2-user/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    print("Building SAM3 model (with segmentation)...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cpu",
        eval_mode=False,
        load_from_HF=False,
        checkpoint_path=None,
        enable_segmentation=True,
    )

    print(f"Loading fine-tuned weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"  Loaded: {len(ckpt) - len(missing)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    predictor = Sam3Predictor(model, resolution=1008, device=device, conf=conf)
    print("SAM3 model loaded.")
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Predict walls using fine-tuned SAM3")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoints/checkpoint.pt",
                        help="Path to SAM3 checkpoint")
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--no-bin", action="store_true", help="Skip binarization")
    parser.add_argument("--prompt", type=str, default="walls", help="Text prompt for detection")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--save-tiles", action="store_true", help="Save individual tiles")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    orig = cv2.imread(str(img_path))
    if orig is None:
        raise SystemExit(f"Cannot read image: {img_path}")

    h, w = orig.shape[:2]
    print(f"Image: {img_path.name} ({w}x{h})")

    if args.no_bin:
        model_input = orig.copy()
        print("Binarize: No (raw image)")
    else:
        model_input = binarize_image(orig)
        print("Binarize: Yes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_sam3_model(args.checkpoint, device=device, conf=args.conf)

    # Tile and predict
    tile_sz = args.tile_size
    stride = tile_sz - args.overlap
    wall_mask = np.zeros((h, w), dtype=np.uint8)
    total_detections = 0
    tile_count = 0

    if args.save_tiles:
        tiles_dir = Path(f"{img_path.stem}_sam3_tiles")
        tiles_dir.mkdir(exist_ok=True)
        print(f"Saving tiles to: {tiles_dir}/")

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_sz, h)
            x2 = min(x + tile_sz, w)
            tile = model_input[y:y2, x:x2]

            th, tw = tile.shape[:2]
            if th < tile_sz or tw < tile_sz:
                padded = np.zeros((tile_sz, tile_sz, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_pil = Image.fromarray(tile_rgb)

            boxes, scores, masks = predictor.predict_tile(tile_pil, text_prompt=args.prompt)
            tile_count += 1

            n_dets = len(boxes)
            total_detections += n_dets

            # Combine all instance masks into single tile mask
            tile_mask = np.zeros((tile_sz, tile_sz), dtype=np.uint8)
            for i in range(n_dets):
                tile_mask = np.maximum(tile_mask, masks[i] * 255)

            # Stitch into full mask (crop to actual tile size)
            mask_crop = tile_mask[:th, :tw]
            wall_mask[y:y2, x:x2] = np.maximum(wall_mask[y:y2, x:x2], mask_crop)

            # Save individual tiles
            if args.save_tiles:
                tile_name = f"tile_r{y:05d}_c{x:05d}"
                cv2.imwrite(str(tiles_dir / f"{tile_name}_input.png"), tile)
                cv2.imwrite(str(tiles_dir / f"{tile_name}_mask.png"), tile_mask)
                tile_overlay = tile.copy()
                tile_colored = np.zeros_like(tile)
                tile_colored[:, :, 2] = 255
                tm_bool = tile_mask > 0
                if tm_bool.any():
                    tile_overlay[tm_bool] = cv2.addWeighted(tile, 0.5, tile_colored, 0.5, 0)[tm_bool]
                for i in range(n_dets):
                    box = boxes[i].astype(int)
                    cv2.rectangle(tile_overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    if i < len(scores):
                        cv2.putText(tile_overlay, f"{scores[i]:.2f}", (box[0], box[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(str(tiles_dir / f"{tile_name}_overlay.png"), tile_overlay)

            if tile_count % 10 == 0:
                print(f"  Processed {tile_count} tiles, {total_detections} detections so far...")

    print(f"Tiles processed: {tile_count}")
    print(f"Total wall detections: {total_detections}")

    # Create overlay on original image
    overlay = orig.copy()
    wall_colored = np.zeros_like(orig)
    wall_colored[:, :, 2] = 255  # Red channel
    mask_bool = wall_mask > 0
    overlay[mask_bool] = cv2.addWeighted(orig, 0.5, wall_colored, 0.5, 0)[mask_bool]

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    out_path = args.output or str(img_path.stem) + "_sam3_predicted.png"
    cv2.imwrite(out_path, overlay)
    print(f"Output saved: {out_path}")

    mask_path = str(Path(out_path).stem) + "_mask.png"
    cv2.imwrite(mask_path, wall_mask)
    print(f"Mask saved: {mask_path}")


if __name__ == "__main__":
    main()
