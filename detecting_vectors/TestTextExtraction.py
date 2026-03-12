# import fitz  # PyMuPDF
# from PIL import Image, ImageDraw

# def test_text_extraction(pdf_path, page_index):
#     # 1. Open Document
#     try:
#         doc = fitz.open(pdf_path)
#         page = doc[page_index]  # Load specific page
#     except Exception as e:
#         print(f"Error loading PDF: {e}")
#         return\
    
    

#     # 2. Setup Image for Visualization
#     # Render page to high-res image (300 DPI equivalent)
#     pix = page.get_pixmap(matrix=fitz.Matrix(1, 1)) 
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     draw = ImageDraw.Draw(img)

#     print(f"--- EXTRACTED TEXT FROM PAGE {page_index + 1} ---")

#     # 3. Extract "Words" (Contains coordinates: x0, y0, x1, y1, "text", ...)
#     words = page.get_text("words")

#     if not words:
#         print("!! NO NATIVE TEXT FOUND !! This page might be an image scan.")
#     else:
#         # Loop through found words
#         for w in words:
#             x0, y0, x1, y1, text, block_no, line_no, word_no = w
            
#             # Print to Terminal
#             # We add a pipe '|' to see if there is whitespace or strange formatting
#             print(f"[{x0:.1f}, {y0:.1f}] : {text}")

#             # 4. Draw Box on Image
#             # Note: We must scale coordinates by 2 because we used Matrix(2, 2) for the image
#             draw.rectangle(
#                 [x0 * 2, y0 * 2, x1 * 2, y1 * 2],
#                 outline="red",
#                 width=2
#             )

#     # 5. Save output
#     output_filename = f"text_extraction_test_page_{page_index + 1}.png"
#     img.save(output_filename)
#     print(f"\n--- VISUALIZATION SAVED: {output_filename} ---")
#     print("Check this image. If you see red boxes around the text, extraction is working perfectly.")

# # --- RUN CONFIGURATION ---
# # Page 22 (Index 21) is Sheet A-501 (Elevations)
# pdf_file = "Arch Hampton_Surfside_032825.pdf"
# target_page_index = 0

# if __name__ == "__main__":
#     test_text_extraction(pdf_file, target_page_index)

import gradio as gr
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

def process_pdf_page(pdf_file, page_number):
    """
    Wraps the original test_text_extraction logic for Gradio.
    """
    if pdf_file is None:
        return None, "Please upload a PDF file."
    
    # Gradio passes the file path via pdf_file.name
    pdf_path = pdf_file.name
    
    # Convert user input (1-based) to Python index (0-based)
    try:
        page_index = int(page_number) - 1
    except ValueError:
        return None, "Page number must be an integer."

    log_output = []
    readable_text = ""  # Variable to build the "Paragraph" view

    # 1. Open Document
    try:
        doc = fitz.open(pdf_path)
        
        if page_index < 0 or page_index >= len(doc):
            return None, f"Error: Page {page_number} does not exist. PDF has {len(doc)} pages."
            
        page = doc[page_index]
    except Exception as e:
        return None, f"Error loading PDF: {e}"

    original_or = page.rotation
    page.set_rotation(0)
    
    # 2. Setup Image for Visualization
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
 # Ensure no rotation for accurate coordinate mapping

    print(f"Original Page Rotation: {original_or} degrees. Temporarily set to 0 for processing.")
    log_output.append(f"--- EXTRACTED DATA FROM PAGE {page_number} ---")

    # 3. Extract "Words" with SORTING ENABLED
    # 'sort=True' forces the extractor to read Top-Left -> Bottom-Right
    # instead of the order the text was originally typed (Creation Order).
    words = page.get_text("words", sort=True)

    if not words:
        log_output.append("!! NO NATIVE TEXT FOUND !! This page might be an image scan.")
        final_log = "\n".join(log_output)
    else:
        # Variables to help reconstruct the sentence structure
        previous_y = words[0][1]
        
        # Loop through found words
        for w in words:
            x0, y0, x1, y1, text, block_no, line_no, word_no = w
            
            # --- Paragraph Reconstruction Logic ---
            # If the new word starts on a new line (y position changes by > 5 pixels), add a newline
            if abs(y0 - previous_y) > 5:
                readable_text += "\n"
            else:
                readable_text += " " # Just add a space between words on the same line
            
            readable_text += text
            previous_y = y0
            # --------------------------------------

            # Add technical data to log
            log_output.append(f"[{x0:.1f}, {y0:.1f}] : {text}")

            # 4. Draw Box on Image
            draw.rectangle(
                [x0 * 2, y0 * 2, x1 * 2, y1 * 2],
                outline="red",
                width=2
            )

        # Combine the Readable Text and the Technical Logs
        final_log = (
            "--- RECONSTRUCTED TEXT (READING ORDER) ---\n"
            f"{readable_text}\n\n"
            "--- DETAILED COORDINATE DATA ---\n" + 
            "\n".join(log_output)
        )

    return img, final_log

# --- GRADIO INTERFACE CONFIGURATION ---

with gr.Blocks() as demo:
    gr.Markdown("## PDF Text Extraction Visualizer")
    gr.Markdown("Upload a PDF to see text order and bounding boxes.")

    with gr.Row():
        with gr.Column():
            input_pdf = gr.File(label="Upload PDF", file_count="single", type="filepath")
            input_page = gr.Number(label="Page Number to Analyze", value=1, precision=0)
            btn_process = gr.Button("Process Page", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Visualized Output", type="pil")
            # Increased lines to 20 so you can see the paragraph clearly
            output_log = gr.Textbox(label="Extraction Logs", lines=20)

    btn_process.click(
        fn=process_pdf_page,
        inputs=[input_pdf, input_page],
        outputs=[output_image, output_log]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# import fitz  # PyMuPDF
# from PIL import Image, ImageDraw

# def search_and_highlight_text(pdf_path, page_index):
#     # 1. User Input
#     search_term = input("Enter the word or phrase you want to find: ").strip()
#     if not search_term:
#         print("No search term entered. Exiting.")
#         return

#     try:
#         doc = fitz.open(pdf_path)
#         page = doc[page_index]
#     except Exception as e:
#         print(f"Error loading PDF: {e}")
#         return

#     # 2. Setup Image for Visualization
#     pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     draw = ImageDraw.Draw(img, "RGBA")

#     # 3. Search for the specific text
#     # search_for returns a list of Rect objects where the text was found
#     text_instances = page.search_for(search_term)

#     if not text_instances:
#         print(f"'{search_term}' was NOT found on page {page_index + 1}.")
#     else:
#         print(f"Found {len(text_instances)} instances of '{search_term}'.")
        
#         for rect in text_instances:
#             # Scale coordinates for our 2x resolution image
#             x0, y0, x1, y1 = rect.x0 * 2, rect.y0 * 2, rect.x1 * 2, rect.y1 * 2
            
#             # Draw a yellow highlight (semi-transparent)
#             # (255, 255, 0, 100) is Yellow with ~40% opacity
#             draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 0, 100))
            
#             # Optional: Add a red border so it really pops
#             draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

#     # 4. Save output
#     output_filename = f"highlight_test_page_{page_index + 1}.png"
#     img.save(output_filename)
#     print(f"\n--- VISUALIZATION SAVED: {output_filename} ---")

# # --- RUN CONFIGURATION ---
# pdf_file = "Arch Hampton_Surfside_032825.pdf"
# target_page_index = 0

# if __name__ == "__main__":
#     search_and_highlight_text(pdf_file, target_page_index)