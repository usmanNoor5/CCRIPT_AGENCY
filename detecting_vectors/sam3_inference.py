#!/usr/bin/env python3
"""
SAM3 Inference Script — uses the local Roboflow inference server (Docker).

Usage:
    python3.12 sam3_inference.py /path/to/image.jpg --prompt "Walls"
    python3.12 sam3_inference.py /path/to/image.png --prompt "Walls" "Slab"
    python3.12 sam3_inference.py /path/to/image.jpg --prompt "Walls" --threshold 0.3
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import requests
from PIL import Image, ImageDraw
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
API_KEY = "rQeVr4DrCdtRsfk96koy"
MODEL_ID = "my-first-project-2-06mc5/4"
INFERENCE_URL = "http://localhost:9001"

# ── Helpers ─────────────────────────────────────────────────────────
def encode_image(image_path: str) -> str:
    """Read image file and return base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_sam3_inference(image_path: str, prompts: list[str], threshold: float = 0.5,
                       model_id: str = MODEL_ID, server_url: str = INFERENCE_URL):
    """
    Send image to local SAM3 inference server and return results.
    """
    print(f"Image: {image_path}")
    print(f"Prompts: {prompts}")
    print(f"Threshold: {threshold}")
    print(f"Model: {model_id}")
    print()

    # Build prompt list
    prompt_list = [{"type": "text", "text": p} for p in prompts]

    # Encode image as base64
    img_b64 = encode_image(image_path)

    # Send request
    url = f"{server_url}/sam3/concept_segment?api_key={API_KEY}"
    payload = {
        "image": {
            "type": "base64",
            "value": img_b64,
        },
        "model_id": model_id,
        "prompts": prompt_list,
    }

    print("Sending request to inference server...")
    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code != 200:
        print(f"ERROR: Server returned {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()
    return result


def draw_masks_on_image(image_path: str, result: dict, output_path: str):
    """Draw detected masks/predictions on the image."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = [
        (255, 0, 0, 80),    # red
        (0, 255, 0, 80),    # green
        (0, 0, 255, 80),    # blue
        (255, 255, 0, 80),  # yellow
        (255, 0, 255, 80),  # magenta
    ]

    total_predictions = 0

    for pr in result.get("prompt_results", []):
        prompt_text = pr.get("echo", {}).get("text", "unknown")
        predictions = pr.get("predictions", [])
        print(f"Prompt '{prompt_text}': {len(predictions)} predictions")

        color = colors[pr.get("prompt_index", 0) % len(colors)]

        for pred in predictions:
            total_predictions += 1
            confidence = pred.get("confidence", 0)
            print(f"  - Confidence: {confidence:.3f}")

            # Draw polygon points if available
            points = pred.get("points", [])
            if points:
                poly = [(p["x"], p["y"]) for p in points]
                draw.polygon(poly, fill=color, outline=(color[0], color[1], color[2], 255))

            # Draw bounding box if available
            x = pred.get("x")
            y = pred.get("y")
            w = pred.get("width")
            h = pred.get("height")
            if all(v is not None for v in [x, y, w, h]):
                x0 = x - w / 2
                y0 = y - h / 2
                x1 = x + w / 2
                y1 = y + h / 2
                draw.rectangle([x0, y0, x1, y1],
                               fill=color,
                               outline=(color[0], color[1], color[2], 255),
                               width=2)

    # Composite overlay onto original
    result_img = Image.alpha_composite(img, overlay).convert("RGB")
    result_img.save(output_path)
    print(f"\nTotal predictions: {total_predictions}")
    print(f"Saved: {output_path}")
    return total_predictions


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SAM3 inference via local Roboflow server")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--prompt", nargs="+", default=["Walls"],
                        help="Text prompts (default: Walls)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--model", default=None,
                        help=f"Model ID (default: {MODEL_ID})")
    parser.add_argument("--server", default=None,
                        help=f"Inference server URL (default: {INFERENCE_URL})")
    parser.add_argument("--out", default=None,
                        help="Output image path (default: <input>_sam3.png)")
    args = parser.parse_args()

    model_id = args.model or MODEL_ID
    server_url = args.server or INFERENCE_URL

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: {image_path} not found")
        sys.exit(1)

    # Run inference
    result = run_sam3_inference(str(image_path), args.prompt, args.threshold,
                                model_id=model_id, server_url=server_url)

    # Print raw JSON
    print("\n--- Raw Response ---")
    print(json.dumps(result, indent=2)[:2000])

    # Draw results on image
    out_path = args.out or f"{image_path.stem}_sam3.png"
    draw_masks_on_image(str(image_path), result, out_path)


if __name__ == "__main__":
    main()
