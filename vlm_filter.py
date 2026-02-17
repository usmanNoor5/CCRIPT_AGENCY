
"""
Use Qwen2-VL to filter floor plan pages and crop to the drawing area
before handing off to the YOLO-seg wall detection pipeline.

Pipeline:
    python3.12 vlm_filter.py --input-dir /path/to/pages/
    python3.12 testing/prepare_for_inference.py --input testing/vlm_filtered/<image>.png
    python3.12 infer_walls.py --source testing/tiles/

Usage examples:
    # Filter + crop a folder of floor plan pages
    python3.12 vlm_filter.py --input-dir /path/to/pages/

    # Just filter (no crop), save results to custom output dir
    python3.12 vlm_filter.py --input-dir /path/to/pages/ --no-crop --out-dir my_filtered/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    print(f"Loading {model_id} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model ready.\n")
    return model, processor


# ---------------------------------------------------------------------------
# VLM inference helpers
# ---------------------------------------------------------------------------

def _run_inference(model, processor, image_path: Path, question: str,
                   max_new_tokens: int, skip_special: bool) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=skip_special,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def ask_vlm(model, processor, image_path: Path, question: str, max_new_tokens: int = 256) -> str:
    return _run_inference(model, processor, image_path, question, max_new_tokens, skip_special=True)


def ask_vlm_raw(model, processor, image_path: Path, question: str, max_new_tokens: int = 128) -> str:
    """Return raw output including Qwen2.5-VL grounding special tokens."""
    return _run_inference(model, processor, image_path, question, max_new_tokens, skip_special=False)


# ---------------------------------------------------------------------------
# Step 1 – Filter: does this page contain a floor plan with walls?
# ---------------------------------------------------------------------------

FILTER_QUESTION = (
    "Does this page contain an architectural floor plan drawing that shows "
    "walls or room layouts? Answer with only 'yes' or 'no'."
)


def is_floor_plan(model, processor, image_path: Path) -> bool:
    answer = ask_vlm(model, processor, image_path, FILTER_QUESTION, max_new_tokens=8)
    return answer.lower().startswith("y")


# ---------------------------------------------------------------------------
# Step 2 – Crop: locate the floor plan area and crop to it
# ---------------------------------------------------------------------------

# The image is downscaled to this before querying the VLM so that coords
# the model reports are in a known pixel space that we can scale back from.
VLM_QUERY_MAX_PX = 1024


def resize_for_vlm(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Resize image so its longest side is VLM_QUERY_MAX_PX. Returns (resized, sx, sy)."""
    h, w = img.shape[:2]
    scale = min(VLM_QUERY_MAX_PX / w, VLM_QUERY_MAX_PX / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return resized, w / nw, h / nh   # scale factors back to original
    return img, 1.0, 1.0


def build_crop_question(qw: int, qh: int) -> str:
    return (
        f"This architectural drawing image is {qw}x{qh} pixels. "
        "Find ONLY the floor plan drawing area — the part showing room outlines and walls. "
        "Exclude: title block, project text, north arrow, scale bar, revision table, "
        "door/window schedules, room finish schedules, material legends, "
        "any grid of boxes on the right or bottom, and all blank margins. "
        "The floor plan is usually the large open drawing, often on the left or centre. "
        f"Output ONLY a JSON array of 4 integers [x1, y1, x2, y2] in pixel coords "
        f"within this {qw}x{qh} image. "
        "Prefer cutting the right boundary tighter to exclude schedule boxes. "
        "No explanation — just the array."
    )


def parse_bbox_pixels(response: str, qw: int, qh: int) -> tuple[int, int, int, int] | None:
    """Parse [x1,y1,x2,y2] pixel coords from VLM response, validated against query image size."""
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", response)
    if not m:
        return None
    x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
    if x1 >= x2 or y1 >= y2:
        return None
    # Clamp to query image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(qw, x2), min(qh, y2)
    # Reject if the box covers almost the entire image (model gave up)
    if (x2 - x1) > qw * 0.95 and (y2 - y1) > qh * 0.95:
        return None
    return x1, y1, x2, y2


def vlm_crop(model, processor, image_path: Path, img: np.ndarray) -> np.ndarray:
    """Resize image to query resolution, ask VLM for bbox, scale result to original."""
    orig_h, orig_w = img.shape[:2]

    # Downscale for VLM query
    q_img, sx, sy = resize_for_vlm(img)
    qh, qw = q_img.shape[:2]

    # Save resized image to temp file for VLM
    tmp_path = image_path.parent / f"_vlm_tmp_{image_path.name}"
    cv2.imwrite(str(tmp_path), q_img)

    try:
        question = build_crop_question(qw, qh)
        response = ask_vlm(model, processor, tmp_path, question, max_new_tokens=32)
        print(f"    VLM bbox response: {response!r}")
        bbox_q = parse_bbox_pixels(response, qw, qh)
    finally:
        tmp_path.unlink(missing_ok=True)

    if bbox_q is None:
        print("    Could not parse bbox — using full image.")
        return img

    # Scale back to original image coords
    x1 = max(0,      int(bbox_q[0] * sx) - 20)
    y1 = max(0,      int(bbox_q[1] * sy) - 20)
    x2 = min(orig_w, int(bbox_q[2] * sx) + 20)
    y2 = min(orig_h, int(bbox_q[3] * sy) + 20)
    print(f"    Query bbox: {bbox_q}  →  Original crop: [{x1},{y1},{x2},{y2}] (img {orig_w}x{orig_h})")
    return img[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="VLM-based floor plan filter and crop.")
    ap.add_argument("--input-dir",  type=Path, required=True,
                    help="Directory containing floor plan page images (png/jpg)")
    ap.add_argument("--out-dir",    type=Path, default=Path("testing/vlm_filtered"),
                    help="Output directory for filtered (and optionally cropped) images")
    ap.add_argument("--model",      type=str,  default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HuggingFace model ID")
    ap.add_argument("--no-crop",    action="store_true",
                    help="Skip VLM crop step — just filter and copy full images")
    ap.add_argument("--exts",       type=str,  default="png,jpg,jpeg",
                    help="Comma-separated image extensions to process")
    args = ap.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    exts = [f".{e.strip().lstrip('.')}" for e in args.exts.split(",")]
    images = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in exts
    )
    if not images:
        raise SystemExit(f"No images found in {args.input_dir} (extensions: {exts})")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model)

    kept, skipped = [], []
    log = []

    for img_path in images:
        print(f"[{img_path.name}]")

        # Step 1 — Filter
        has_walls = is_floor_plan(model, processor, img_path)
        if not has_walls:
            print("  -> SKIPPED (no floor plan / walls detected)\n")
            skipped.append(img_path.name)
            log.append({"file": img_path.name, "kept": False, "reason": "no floor plan"})
            continue

        print("  -> KEPT (floor plan detected)")

        img = cv2.imread(str(img_path))
        if img is None:
            print("  -> Could not read image, skipping.\n")
            skipped.append(img_path.name)
            continue

        # Step 2 — Crop
        if args.no_crop:
            cropped = img
            print("  -> Crop skipped (--no-crop)")
        else:
            cropped = vlm_crop(model, processor, img_path, img)

        out_path = args.out_dir / img_path.name
        cv2.imwrite(str(out_path), cropped)
        print(f"  -> Saved: {out_path}\n")

        kept.append(img_path.name)
        log.append({"file": img_path.name, "kept": True,
                    "cropped": not args.no_crop,
                    "out": str(out_path)})

    # Save run log
    log_path = args.out_dir / "vlm_filter_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    # Summary
    print("=" * 60)
    print(f"Done.  Kept: {len(kept)}  |  Skipped: {len(skipped)}")
    print(f"Output:  {args.out_dir}/")
    print(f"Log:     {log_path}")
    if kept:
        print("\nNext step — tile each kept image:")
        for name in kept:
            print(f"  python3.12 testing/prepare_for_inference.py --input {args.out_dir / name}")
        print("\nThen run wall detection:")
        print("  python3.12 infer_walls.py --source testing/tiles/")


if __name__ == "__main__":
    main()
