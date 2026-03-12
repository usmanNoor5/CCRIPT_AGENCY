import fitz
from PIL import Image, ImageDraw

# doc = fitz.open("Candlewood Suites Drawings Combined.pdf")
doc = fitz.open("CHRISTENSEN MODEL.pdf")
page = doc[4]  # zero-based

pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # high-res
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img.save("page.png")

drawings = page.get_drawings()
draw = ImageDraw.Draw(img)

for d in drawings:
    for item in d["items"]:
        if item[0] == "l":  # line
            _, p1, p2 = item
            draw.line(
                (p1.x * 2, p1.y * 2, p2.x * 2, p2.y * 2),
                fill="green",
                width=2
            )

img.save("page_with_vectors.png")

for d in drawings:
    rect = d["rect"]
    draw.rectangle(
        [rect.x0*2, rect.y0*2, rect.x1*2, rect.y1*2],
        outline="blue",
        width=2
    )
img.save("page_with_bboxes.png")



# Drawing boxing with merging nearby boxes (to get whole diagrams instead of parts)
# import fitz
# from PIL import Image, ImageDraw

# SCALE = 2
# MERGE_DISTANCE =100  # px (tune this)

# doc = fitz.open("Candlewood Suites Drawings Combined.pdf")
# page = doc[10]

# pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
# img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
# draw = ImageDraw.Draw(img)

# drawings = page.get_drawings()

# # ---- STEP 1: collect rects (scaled) ----
# rects = []
# for d in drawings:
#     r = d["rect"]
#     rects.append([
#         r.x0 * SCALE,
#         r.y0 * SCALE,
#         r.x1 * SCALE,
#         r.y1 * SCALE
#     ])

# # ---- STEP 2: rectangle utilities ----
# def expand(rect, d):
#     return [
#         rect[0] - d,
#         rect[1] - d,
#         rect[2] + d,
#         rect[3] + d
#     ]

# def overlaps(a, b):
#     return not (
#         a[2] < b[0] or
#         a[0] > b[2] or
#         a[3] < b[1] or
#         a[1] > b[3]
#     )

# def merge(a, b):
#     return [
#         min(a[0], b[0]),
#         min(a[1], b[1]),
#         max(a[2], b[2]),
#         max(a[3], b[3])
#     ]

# # ---- STEP 3: cluster rects ----
# clusters = []

# for rect in rects:
#     expanded = expand(rect, MERGE_DISTANCE)
#     merged = False

#     for i, c in enumerate(clusters):
#         if overlaps(expanded, c):
#             clusters[i] = merge(c, rect)
#             merged = True
#             break

#     if not merged:
#         clusters.append(rect)

# # ---- STEP 4: draw final drawing-level boxes ----
# for c in clusters:
#     draw.rectangle(c, outline="red", width=4)

# img.save("page_with_drawing_boxes.png")


# import fitz
# from PIL import Image, ImageDraw

# # ---------------- CONFIG ----------------
# PDF_PATH = "Candlewood Suites Drawings Combined.pdf"
# PAGE_NUM = 29
# TARGET_WORD = "Canopy"
# FUZZ_FACTOR = 15      # Keeps the drawings grouped correctly
# SCALE = 3             # High resolution for the zoom
# PADDING = 50          # Extra breathing room around the edges (pixels)
# # ---------------------------------------

# def get_smart_viewport(pdf_path, page_num, word, fuzz=15):
#     doc = fitz.open(pdf_path)
#     page = doc[page_num]
#     page_rect = page.rect

#     # 1. Find the Anchor Word
#     found = page.search_for(word)
#     if not found:
#         print(f"ERROR: Word '{word}' not found.")
#         return
#     anchor = found[0]

#     # 2. Get Vectors (with Frame Breaker logic)
#     raw_drawings = page.get_drawings()
#     inflated_rects = []
    
#     for d in raw_drawings:
#         r = d["rect"]
#         # Ignore page borders/frames
#         if r.width > (page_rect.width * 0.5) or r.height > (page_rect.height * 0.5):
#             continue
#         # Inflate
#         inflated = fitz.Rect(r.x0 - fuzz, r.y0 - fuzz, r.x1 + fuzz, r.y1 + fuzz)
#         inflated_rects.append(inflated)

#     # 3. Cluster/Merge
#     changed = True
#     while changed:
#         changed = False
#         new_pool = []
#         while inflated_rects:
#             current = inflated_rects.pop(0)
#             merged = False
#             for i, other in enumerate(new_pool):
#                 if current.intersects(other):
#                     new_pool[i] = current | other
#                     merged = True
#                     changed = True
#                     break
#             if not merged:
#                 new_pool.append(current)
#         inflated_rects = new_pool

#     # 4. Find the 2 closest drawing groups
#     candidates = []
#     for r in inflated_rects:
#         dist = (r.x0 - anchor.x0)**2 + (r.y0 - anchor.y0)**2
#         # Deflate to get the real bounds
#         real_crop = fitz.Rect(r.x0 + fuzz, r.y0 + fuzz, r.x1 - fuzz, r.y1 - fuzz)
#         candidates.append((dist, real_crop))

#     candidates.sort(key=lambda x: x[0])
    
#     # Select the top 2 (or 1 if only 1 exists)
#     targets = [candidates[0][1]]
#     if len(candidates) > 1:
#         targets.append(candidates[1][1])

#     # 5. Calculate the "Super-Bounding Box" (The Viewport)
#     # Start with the anchor word itself
#     viewport = fitz.Rect(anchor)
    
#     # Expand the viewport to include the identified drawings
#     for t in targets:
#         viewport = viewport | t

#     # Add Padding
#     viewport.x0 -= PADDING
#     viewport.y0 -= PADDING
#     viewport.x1 += PADDING
#     viewport.y1 += PADDING
    
#     # Intersect with page bounds to avoid errors (don't go off the page)
#     viewport = viewport & page_rect

#     print(f"Calculated viewport area: {viewport}")

#     # 6. Render ONLY that Viewport
#     # We use the 'clip' parameter to render only the part we need
#     # This is much faster than rendering the whole page and cropping later
#     mat = fitz.Matrix(SCALE, SCALE)
#     pix = page.get_pixmap(matrix=mat, clip=viewport)
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#     draw = ImageDraw.Draw(img)

#     # 7. Draw the Highlight
#     # We need to map the word's coordinates relative to our new viewport
#     # Formula: (Word_Coord - Viewport_Origin) * Scale
#     wx0 = (anchor.x0 - viewport.x0) * SCALE
#     wy0 = (anchor.y0 - viewport.y0) * SCALE
#     wx1 = (anchor.x1 - viewport.x0) * SCALE
#     wy1 = (anchor.y1 - viewport.y0) * SCALE
    
#     draw.rectangle([wx0-5, wy0-5, wx1+5, wy1+5], outline="red", width=5)

#     img.save("smart_viewport_zoom.jpg")
#     print("Success! Saved 'smart_viewport_zoom.jpg'")

# get_smart_viewport(PDF_PATH, PAGE_NUM, TARGET_WORD, fuzz=FUZZ_FACTOR)