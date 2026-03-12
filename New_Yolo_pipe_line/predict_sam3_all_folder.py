#!/usr/bin/env python3
"""
predict_sam3_all_folder.py — Run fine-tuned SAM3 model on all images in a folder.

This script mirrors predict_sam3.py behavior, but processes a whole directory.

Steps per image:
  1. Load image, optionally binarize
  2. Tile into 1008x1008 patches
  3. Run SAM3 inference on each tile with text prompt "walls"
  4. Stitch segmentation masks back onto full image
  5. Save result with wall overlay + mask

Usage:
  python3.12 predict_sam3_all_folder.py <input_folder>
  python3.12 predict_sam3_all_folder.py <input_folder> --no-bin
  python3.12 predict_sam3_all_folder.py <input_folder> --conf 0.3
  python3.12 predict_sam3_all_folder.py <input_folder> --output-dir sam3_preds
"""

import argparse
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


class Sam3Predictor:
    """Predictor for SAM3 with segmentation head (pixel masks)."""

    def __init__(self, model, resolution=1008, device="cuda", conf=0.5):
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = model
        self.device = device
        self.conf = conf
        self._processor = Sam3Processor(
            model,
            resolution=resolution,
            device=device,
            confidence_threshold=conf,
        )

    @torch.inference_mode()
    def predict_tile(self, pil_image, text_prompt="walls"):
        from sam3.model import box_ops

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
        out_probs = out_probs[keep]
        out_bbox = out_bbox[keep]
        out_masks = out_masks[keep]

        img_h = state["original_height"]
        img_w = state["original_width"]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

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

        return boxes.cpu().numpy(), out_probs.cpu().numpy(), masks


def load_sam3_model(checkpoint_path, bpe_path, device="cuda", conf=0.5):
    from sam3.model_builder import build_sam3_image_model

    print("Building SAM3 model (with segmentation)...")
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
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
    print(
        f"  Loaded: {len(ckpt) - len(missing)} keys, "
        f"missing: {len(missing)}, unexpected: {len(unexpected)}"
    )

    model = model.to(device)
    model.eval()
    predictor = Sam3Predictor(model, resolution=1008, device=device, conf=conf)
    print("SAM3 model loaded.")
    return predictor


def run_single_image(
    predictor,
    img_path: Path,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    prompt: str,
    no_bin: bool,
    save_tiles: bool,
):
    orig = cv2.imread(str(img_path))
    if orig is None:
        print(f"[SKIP] Cannot read image: {img_path}")
        return False

    h, w = orig.shape[:2]
    print(f"\nImage: {img_path.name} ({w}x{h})")

    if no_bin:
        model_input = orig.copy()
        print("Binarize: No (raw image)")
    else:
        model_input = binarize_image(orig)
        print("Binarize: Yes")

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile-size")

    wall_mask = np.zeros((h, w), dtype=np.uint8)
    total_detections = 0
    tile_count = 0

    tiles_dir = None
    if save_tiles:
        tiles_dir = output_dir / f"{img_path.stem}_sam3_tiles"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving tiles to: {tiles_dir}/")

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            tile = model_input[y:y2, x:x2]

            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:th, :tw] = tile
                tile = padded

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_pil = Image.fromarray(tile_rgb)
            boxes, scores, masks = predictor.predict_tile(tile_pil, text_prompt=prompt)
            tile_count += 1

            n_dets = len(boxes)
            total_detections += n_dets

            tile_mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            for i in range(n_dets):
                tile_mask = np.maximum(tile_mask, masks[i] * 255)

            mask_crop = tile_mask[:th, :tw]
            wall_mask[y:y2, x:x2] = np.maximum(wall_mask[y:y2, x:x2], mask_crop)

            if save_tiles and tiles_dir is not None:
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
                        cv2.putText(
                            tile_overlay,
                            f"{scores[i]:.2f}",
                            (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                cv2.imwrite(str(tiles_dir / f"{tile_name}_overlay.png"), tile_overlay)

            if tile_count % 10 == 0:
                print(f"  Processed {tile_count} tiles, {total_detections} detections so far...")

    print(f"Tiles processed: {tile_count}")
    print(f"Total wall detections: {total_detections}")

    overlay = orig.copy()
    wall_colored = np.zeros_like(orig)
    wall_colored[:, :, 2] = 255
    mask_bool = wall_mask > 0
    overlay[mask_bool] = cv2.addWeighted(orig, 0.5, wall_colored, 0.5, 0)[mask_bool]

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    out_path = output_dir / f"{img_path.stem}_sam3_predicted.png"
    mask_path = output_dir / f"{img_path.stem}_sam3_predicted_mask.png"
    cv2.imwrite(str(out_path), overlay)
    cv2.imwrite(str(mask_path), wall_mask)

    print(f"Output saved: {out_path}")
    print(f"Mask saved:   {mask_path}")
    return True


def gather_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="Predict walls using fine-tuned SAM3 on a folder")
    parser.add_argument("input_folder", type=str, help="Path to input folder containing images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoints/checkpoint.pt",
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument(
        "--bpe-path",
        type=str,
        default="/home/ec2-user/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        help="Path to SAM3 BPE vocab file",
    )
    parser.add_argument("--tile-size", type=int, default=1008)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--no-bin", action="store_true", help="Skip binarization")
    parser.add_argument("--prompt", type=str, default="walls", help="Text prompt for detection")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs. Default: <input_folder>/sam3_outputs",
    )
    parser.add_argument("--save-tiles", action="store_true", help="Save individual tiles")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder not found or not a directory: {input_folder}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    bpe_path = Path(args.bpe_path)
    if not bpe_path.exists():
        raise SystemExit(f"BPE file not found: {bpe_path}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_folder / "sam3_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = gather_images(input_folder)
    if not images:
        raise SystemExit(f"No images found in folder: {input_folder}")

    print(f"Input folder : {input_folder}")
    print(f"Output folder: {output_dir}")
    print(f"Images found : {len(images)}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"BPE path     : {bpe_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    predictor = load_sam3_model(str(checkpoint_path), bpe_path, device=device, conf=args.conf)

    ok_count = 0
    for idx, img_path in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] Processing {img_path.name}")
        try:
            ok = run_single_image(
                predictor=predictor,
                img_path=img_path,
                output_dir=output_dir,
                tile_size=args.tile_size,
                overlap=args.overlap,
                prompt=args.prompt,
                no_bin=args.no_bin,
                save_tiles=args.save_tiles,
            )
            if ok:
                ok_count += 1
        except Exception as exc:
            print(f"[ERROR] {img_path.name}: {exc}")

    print("\n=== Done ===")
    print(f"Processed successfully: {ok_count}/{len(images)}")
    print(f"Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
