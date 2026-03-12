# import fitz
# from PIL import Image, ImageDraw

# def get_line_category(color_tuple):
#     # (Same color classification logic as before)
#     if color_tuple is None: return None 
#     if isinstance(color_tuple, (float, int)): r=g=b=color_tuple
#     elif len(color_tuple) == 3: r,g,b = color_tuple
#     else: return "OTHER"
    
#     brightness = (r+g+b)/3
#     saturation = max(r,g,b) - min(r,g,b)
    
#     if brightness < 0.35: return "BLACK"
#     if 0.35 <= brightness < 0.95 and saturation < 0.1: return "GRAY"
#     return "OTHER"

# # --- MAIN ---

# pdf_path = "/Users/sharjeelbokhari/Documents/CCRIPT/Projects/gray-mechanical/Plumbing Combined - Health & Wellness.pdf"
# doc = fitz.open(pdf_path)
# page = doc[2] 
# ZOOM = 2.0  # High res for accuracy

# # 1. Handle Rotation & Setup
# original_rotation = page.rotation
# page.set_rotation(0)

# # 2. Create Two Images:
# #    A. The Visual Image (What you see)
# pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
# visual_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
# visual_draw = ImageDraw.Draw(visual_img)

# #    B. The "Hot Zone" Mask (Hidden lookup map)
# #       Mode '1' = 1-bit pixels (Black/White), much faster
# mask_img = Image.new("1", (pix.width, pix.height), 0) # 0 = Black background
# mask_draw = ImageDraw.Draw(mask_img)

# # 3. Process Vectors
# paths = page.get_drawings()
# print(f"Processing {len(paths)} vectors...")

# for p in paths:
#     if p["type"] == "f" or p.get("opacity", 1) == 0: continue
    
#     category = get_line_category(p.get("color"))
    
#     if category == "BLACK":
#         # Draw on Visual Image (Green, Thin)
#         # Draw on MASK Image (White, THICK) -> This creates the "detection zone"
        
#         for item in p["items"]:
#             if item[0] == "l":
#                 p1, p2 = item[1], item[2]
#                 coords = (p1.x*ZOOM, p1.y*ZOOM, p2.x*ZOOM, p2.y*ZOOM)
                
#                 # Visual: Nice thin green line
#                 visual_draw.line(coords, fill="green", width=2)
                
#                 # Mask: Thick white "search area" (e.g. 60px = 30px radius)
#                 mask_draw.line(coords, fill=1, width=60) 

# # 4. Filter Text using the Mask
# text_blocks = page.get_text("dict")["blocks"]
# found_count = 0

# for block in text_blocks:
#     if block["type"] == 0: # Text
#         for line in block["lines"]:
#             for span in line["spans"]:
#                 bbox = span["bbox"]
#                 text = span["text"]
                
#                 # Calculate center point of the text label
#                 cx = (bbox[0] + bbox[2]) / 2 * ZOOM
#                 cy = (bbox[1] + bbox[3]) / 2 * ZOOM
                
#                 # Bounds check (to avoid crash at edges)
#                 if 0 <= cx < mask_img.width and 0 <= cy < mask_img.height:
                    
#                     # THE MAGIC TRICK:
#                     # Check the pixel value at the text's center on the mask.
#                     # If it is 1 (White), it's inside a "Fat Pipe".
#                     if mask_img.getpixel((cx, cy)) > 0:
                        
#                         # Draw Blue Box on the VISUAL image
#                         visual_draw.rectangle(
#                             (bbox[0]*ZOOM, bbox[1]*ZOOM, bbox[2]*ZOOM, bbox[3]*ZOOM),
#                             outline="blue",
#                             width=2
#                         )
#                         found_count += 1

# # 5. Restore Rotation & Save
# if original_rotation != 0:
#     visual_img = visual_img.rotate(-original_rotation, expand=True)
#     # (We don't need to rotate the mask, we are done with it)

# visual_img.save("page_smart_highlight.png")
# print(f"Done. Found {found_count} labels near pipes using Hot Zone masking.")

import fitz
from PIL import Image, ImageDraw

def get_line_category(color_tuple):
    """Classifies VECTOR lines (0.0-1.0 floats)"""
    if color_tuple is None: return None 
    if isinstance(color_tuple, (float, int)): r=g=b=color_tuple
    elif len(color_tuple) == 3: r,g,b = color_tuple
    else: return "OTHER"
    
    brightness = (r+g+b)/3
    saturation = max(r,g,b) - min(r,g,b)
    
    if brightness < 0.35: return "BLACK"
    if 0.35 <= brightness < 0.95 and saturation < 0.1: return "GRAY"
    return "OTHER"

def is_text_black(srgb_int):
    """Classifies TEXT integers (0-255 sRGB)"""
    r = (srgb_int >> 16) & 0xFF
    g = (srgb_int >> 8) & 0xFF
    b = srgb_int & 0xFF
    
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    brightness = (r_norm + g_norm + b_norm) / 3.0
    
    return brightness < 0.35

# --- MAIN ---

pdf_path = "/home/ec2-user/CCRIPT_AGENCY/detecting_vectors/CHRISTENSEN MODEL.pdf"
doc = fitz.open(pdf_path)
page = doc[3] 
ZOOM = 2.0

original_rotation = page.rotation
page.set_rotation(0)

pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
visual_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
visual_draw = ImageDraw.Draw(visual_img)

mask_img = Image.new("1", (pix.width, pix.height), 0)
mask_draw = ImageDraw.Draw(mask_img)

paths = page.get_drawings()
print(f"Processing {len(paths)} vectors...")

for p in paths:
    if p["type"] == "f" or p.get("opacity", 1) == 0: continue
    
    category = get_line_category(p.get("color"))
    
    if category == "BLACK":
        for item in p["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                coords = (p1.x*ZOOM, p1.y*ZOOM, p2.x*ZOOM, p2.y*ZOOM)
                
                visual_draw.line(coords, fill="green", width=2)
                
                mask_draw.line(coords, fill=1, width=60) 

text_blocks = page.get_text("dict")["blocks"]
found_count = 0

for block in text_blocks:
    if block["type"] == 0: 
        for line in block["lines"]:
            for span in line["spans"]:
                
                
                if not is_text_black(span["color"]):
                    continue

                bbox = span["bbox"]
                
                cx = (bbox[0] + bbox[2]) / 2 * ZOOM
                cy = (bbox[1] + bbox[3]) / 2 * ZOOM
                
                if 0 <= cx < mask_img.width and 0 <= cy < mask_img.height:
                    
                    if mask_img.getpixel((cx, cy)) > 0:                        
                        visual_draw.rectangle(
                            (bbox[0]*ZOOM, bbox[1]*ZOOM, bbox[2]*ZOOM, bbox[3]*ZOOM),
                            outline="blue",
                            width=2
                        )
                        found_count += 1

if original_rotation != 0:
    visual_img = visual_img.rotate(-original_rotation, expand=True)

visual_img.save("highlighted_image.png")
print(f"Done. Highlighted {found_count} BLACK text labels near pipes.")