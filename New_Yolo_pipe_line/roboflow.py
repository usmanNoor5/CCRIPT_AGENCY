import gradio as gr
import requests
import base64
import io
import cv2
import numpy as np
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
ROBOFLOW_API_KEY = "rQeVr4DrCdtRsfk96koy"
MODEL_ID = "reannotate-ahe6g/3"
INFERENCE_SERVER_URL = f"http://localhost:9001/sam3/concept_segment?api_key={ROBOFLOW_API_KEY}"


def draw_masks(image_pil, api_response):
    """Parses the SAM 3 JSON response and draws translucent masks on the image."""
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()

    predictions = []
    if "prompt_results" in api_response:
        for pr in api_response["prompt_results"]:
            predictions.extend(pr.get("predictions", []))
    elif "predictions" in api_response:
        predictions = api_response["predictions"]

    for pred in predictions:
        if "points" in pred:
            pts = np.array([[p["x"], p["y"]] for p in pred["points"]], np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.polylines(img_cv, [pts], True, (0, 200, 0), 2)

        if "masks" in pred:
            for mask in pred["masks"]:
                if isinstance(mask, list) and len(mask) > 0:
                    if isinstance(mask[0], dict):
                        pts = np.array([[p["x"], p["y"]] for p in mask], np.int32)
                    else:
                        pts = np.array(mask, np.int32)

                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.polylines(img_cv, [pts], True, (0, 200, 0), 2)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def process_image_and_segment(image_path, class_name):
    """Send a PNG/JPG image to the Roboflow inference server for SAM3 segmentation."""
    try:
        # Load image
        img_pil = Image.open(image_path).convert("RGB")

        # Convert to base64
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Request inference
        payload = {
            "image": {
                "type": "base64",
                "value": img_base64
            },
            "model_id": MODEL_ID,
            "prompts": [
                {
                    "type": "text",
                    "text": class_name
                }
            ]
        }

        response = requests.post(INFERENCE_SERVER_URL, json=payload)
        response.raise_for_status()
        raw_results = response.json()

        # Draw the masks
        annotated_img = draw_masks(img_pil, raw_results)

        return annotated_img, raw_results

    except requests.exceptions.HTTPError as e:
        return None, {"error": f"HTTP Error: {str(e)}", "server_response": response.text}
    except Exception as e:
        return None, {"error": str(e)}


# ==========================================
# GRADIO UI DEFINITION
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SAM 3 Wall Segmenter")
    gr.Markdown("Upload a PNG/JPG floor plan image and provide a class name to segment.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="filepath")
            class_input = gr.Textbox(label="Class Name", value="walls",
                                     placeholder="Enter the class name...")
            submit_btn = gr.Button("Analyze Image", variant="primary")

        with gr.Column(scale=2):
            image_output = gr.Image(label="Annotated Result", type="pil")
            json_output = gr.JSON(label="Raw SAM 3 JSON Output")

    submit_btn.click(
        fn=process_image_and_segment,
        inputs=[image_input, class_input],
        outputs=[image_output, json_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, root_path="/gui")
