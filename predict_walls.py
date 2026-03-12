"""
predict_walls.py
----------------
Run wall detection on tiles using fine-tuned SAM3 checkpoint,
then restitch predictions back onto the full original image.

Usage:
    # Stitch mode (tile directory from prepare_for_inference.py):
    python3.12 predict_walls.py --tile-dir testing/tiles/

    # Single image mode:
    python3.12 predict_walls.py --image path/to/image.jpg

    # Custom checkpoint / confidence:
    python3.12 predict_walls.py --tile-dir testing/tiles/ --checkpoint checkpoint_10.pt --confidence 0.5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent / "sam3"))

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops


DEFAULT_CKPT = "sam3_train/logs/checkpoints/checkpoint_20.pt"
CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned SAM3 model from training checkpoint."""
    print(f"Building SAM3 model...")
    model = build_sam3_image_model(
        device="cpu",
        eval_mode=False,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=False,
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded (epoch {ckpt.get('epoch', '?')}) — "
          f"missing: {len(missing)}, unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Single-tile inference
# ---------------------------------------------------------------------------

def predict_tile(model, processor, pil_img, confidence: float, device: str):
    """Run wall detection on a single PIL image, return boxes and scores."""
    try:
        state = processor.set_image(pil_img)
        state = processor.set_text_prompt(prompt="wall", state=state)
        boxes = state["boxes"].cpu().float().numpy()
        scores = state["scores"].cpu().float().numpy()
        return boxes, scores
    except KeyError:
        # pred_masks missing when enable_segmentation=False — run manually
        pass

    img_w, img_h = pil_img.size
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(pil_img)
        text_outputs = model.backbone.forward_text(["wall"], device=device)
        state["backbone_out"].update(text_outputs)
        state["geometric_prompt"] = model._get_dummy_prompt()

        outputs = model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=processor.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > confidence
        out_probs = out_probs[keep]
        out_bbox = out_bbox[keep]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h],
                                 device=out_bbox.device)
        boxes = boxes * scale_fct[None, :]

    return boxes.cpu().float().numpy(), out_probs.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Stitch tiles back into full image
# ---------------------------------------------------------------------------

def stitch_predictions(tile_dir: Path, model, processor, confidence: float,
                       device: str, out_dir: Path, min_area: int = 200,
                       epsilon_factor: float = 0.01):
    """
    Run SAM3 on each tile, paint boxes onto a full-size canvas,
    then produce a stitched result.
    """
    # Find the cropped base image
    cropped_files = sorted(tile_dir.glob("*_cropped.png"))
    if not cropped_files:
        print("No *_cropped.png found — cannot stitch. Run prepare_for_inference.py first.")
        return None

    base_img = cv2.imread(str(cropped_files[0]))
    if base_img is None:
        return None

    stem = cropped_files[0].stem
    if stem.endswith("_cropped"):
        stem = stem[:-8]
    h_full, w_full = base_img.shape[:2]

    # Find tiles matching the pattern
    tile_pattern = re.compile(rf"{re.escape(stem)}_tile_r(\d+)_c(\d+)\.(png|jpg)")
    tiles = []
    for f in sorted(tile_dir.iterdir()):
        m = tile_pattern.match(f.name)
        if m:
            tiles.append((f, int(m.group(1)), int(m.group(2))))

    if not tiles:
        print("No tiles found matching pattern.")
        return None

    print(f"Stitching {len(tiles)} tiles onto {w_full}x{h_full} canvas ({stem})")

    # Canvas for box-based mask (paint detected wall boxes)
    mask_full = np.zeros((h_full, w_full), dtype=np.float32)
    count_full = np.zeros((h_full, w_full), dtype=np.float32)
    all_boxes_global = []
    all_scores_global = []
    total_walls = 0

    for tile_path, ty, tx in tiles:
        tile_img_cv = cv2.imread(str(tile_path))
        if tile_img_cv is None:
            continue
        tile_h, tile_w = tile_img_cv.shape[:2]

        # Run SAM3 on tile
        pil_tile = Image.open(tile_path).convert("RGB")
        boxes, scores = predict_tile(model, processor, pil_tile, confidence, device)
        n_walls = len(boxes)
        total_walls += n_walls

        # Paint boxes onto mask canvas (offset to global coords)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to tile bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(tile_w, x2), min(tile_h, y2)
            # Offset to global position
            gx1, gy1 = tx + x1, ty + y1
            gx2, gy2 = tx + x2, ty + y2
            # Clamp to full image
            gx1, gy1 = max(0, gx1), max(0, gy1)
            gx2, gy2 = min(w_full, gx2), min(h_full, gy2)

            mask_full[gy1:gy2, gx1:gx2] += score
            count_full[gy1:gy2, gx1:gx2] += 1.0
            all_boxes_global.append([gx1, gy1, gx2, gy2])
            all_scores_global.append(score)

        status = f"{n_walls} walls" if n_walls else "---"
        print(f"  {tile_path.name} -> {status}")

    # Average overlapping regions
    count_full[count_full == 0] = 1
    mask_binary = (mask_full / count_full > 0.3).astype(np.uint8)

    # Trace clean wall outlines
    print("Tracing wall outlines ...")
    line_canvas = trace_walls(mask_binary, min_area=min_area,
                              epsilon_factor=epsilon_factor)

    # Create side-by-side output
    overlay = base_img.copy()
    overlay[line_canvas == 255] = (0, 0, 255)  # red outlines

    # Also draw bounding boxes on a separate overlay
    box_overlay = base_img.copy()
    for box, score in zip(all_boxes_global, all_scores_global):
        x1, y1, x2, y2 = [int(v) for v in box]
        thickness = max(2, int(score * 5))
        cv2.rectangle(box_overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    combined = np.hstack([base_img, overlay, box_overlay])
    fs = max(0.8, min(h_full, w_full) / 2000)
    th = max(2, int(fs * 2.5))
    cv2.putText(combined, "Original", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th)
    cv2.putText(combined, f"Wall Outlines ({total_walls})", (w_full + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th)
    cv2.putText(combined, "Bounding Boxes", (w_full * 2 + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), th)

    # Save outputs
    combined_path = out_dir / f"{stem}_sam3_combined.png"
    cv2.imwrite(str(combined_path), combined)
    print(f"\nSaved: {combined_path}")

    mask_path = out_dir / f"{stem}_sam3_mask.png"
    cv2.imwrite(str(mask_path), mask_binary * 255)
    print(f"Saved: {mask_path}")

    box_path = out_dir / f"{stem}_sam3_boxes.png"
    cv2.imwrite(str(box_path), box_overlay)
    print(f"Saved: {box_path}")

    print(f"\nTotal walls detected: {total_walls}")
    return combined


# ---------------------------------------------------------------------------
# Wall tracing (from infer_walls.py)
# ---------------------------------------------------------------------------

def trace_walls(mask_binary: np.ndarray, min_area: int = 200,
                epsilon_factor: float = 0.01) -> np.ndarray:
    """Retrace wall blobs as clean outlines using Douglas-Peucker simplification."""
    h, w = mask_binary.shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        epsilon = epsilon_factor * cv2.arcLength(cnt, closed=True)
        simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)
        cv2.drawContours(canvas, [simplified], 0, 255, 2)

    return canvas


# ---------------------------------------------------------------------------
# Single image mode (no tiling)
# ---------------------------------------------------------------------------

def predict_single(model, processor, image_path: Path, confidence: float,
                   device: str, out_dir: Path):
    """Run prediction on a single image and save result."""
    pil_img = Image.open(image_path).convert("RGB")
    boxes, scores = predict_tile(model, processor, pil_img, confidence, device)

    img = cv2.imread(str(image_path))
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        thickness = max(2, int(score * 5))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = f"wall {score:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = out_dir / f"pred_{image_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path} ({len(boxes)} walls detected)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Wall detection with fine-tuned SAM3")
    ap.add_argument("--image", type=str, help="Single image path")
    ap.add_argument("--tile-dir", type=str,
                    help="Directory of tiles from prepare_for_inference.py (will stitch)")
    ap.add_argument("--image-dir", type=str, help="Directory of images (no stitching)")
    ap.add_argument("--out", type=str, default="sam3_predictions",
                    help="Output directory (default: sam3_predictions)")
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                    help=f"Checkpoint path (default: {DEFAULT_CKPT})")
    ap.add_argument("--confidence", type=float, default=CONFIDENCE,
                    help=f"Confidence threshold (default: {CONFIDENCE})")
    ap.add_argument("--min-area", type=int, default=200,
                    help="Min blob area for wall tracing (default: 200)")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="Douglas-Peucker simplification (default: 0.01)")
    args = ap.parse_args()

    if not args.image and not args.tile_dir and not args.image_dir:
        ap.error("Provide --image, --tile-dir, or --image-dir")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.checkpoint, device="cuda")
    processor = Sam3Processor(model, confidence_threshold=args.confidence)

    # Stitch mode: tile directory
    if args.tile_dir:
        tile_dir = Path(args.tile_dir)
        stitch_predictions(tile_dir, model, processor, args.confidence,
                           "cuda", out_dir, args.min_area, args.epsilon)
        return

    # Collect individual images
    images = []
    if args.image:
        images.append(Path(args.image))
    if args.image_dir:
        img_dir = Path(args.image_dir)
        images.extend(sorted(img_dir.glob("*.jpg")))
        images.extend(sorted(img_dir.glob("*.png")))

    print(f"Found {len(images)} image(s)")
    for img_path in images:
        print(f"Processing: {img_path.name}")
        try:
            predict_single(model, processor, img_path, args.confidence,
                           "cuda", out_dir)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone — results saved to {out_dir}/")


if __name__ == "__main__":
    main()
