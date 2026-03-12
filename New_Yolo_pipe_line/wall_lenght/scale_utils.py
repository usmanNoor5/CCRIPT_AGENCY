from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

_VLM_CACHE: Dict[str, Dict[str, Any]] = {}


_SCALE_PATTERNS = [
    # 1/4" = 1'-0"
    re.compile(
        r"(?P<num>\d+)\s*/\s*(?P<den>\d+)\s*[\"']?\s*=\s*(?P<ft>\d+)\s*['’]\s*[- ]?\s*(?P<in>\d*)\s*[\"”]?",
        re.I,
    ),
    # 1" = 10'
    re.compile(
        r"(?P<num>\d+(?:\.\d+)?)\s*[\"”]?\s*=\s*(?P<ft>\d+)\s*['’]",
        re.I,
    ),
]


def extract_page_text(pdf_path: Path, page_index: int) -> str:
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                txt = pdf.pages[page_index].extract_text() or ""
                if txt.strip():
                    return txt
        except Exception:
            pass
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            page = doc.load_page(page_index)
            txt = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE) or ""
            doc.close()
            return txt
        except Exception:
            pass
    return ""


def parse_scale_from_text(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (real_inches_per_paper_inch, raw_scale_string).
    """
    t = text.replace("”", '"').replace("’", "'")
    for pat in _SCALE_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        gd = m.groupdict()
        num = float(gd.get("num", 1) or 1)
        den = float(gd.get("den", 1) or 1)
        ft = float(gd.get("ft", 0) or 0)
        inch = float(gd.get("in", 0) or 0)
        paper_inch = num / den if "den" in gd else num
        real_inches = ft * 12.0 + inch
        if paper_inch > 0 and real_inches > 0:
            return (real_inches / paper_inch), m.group(0)
    return None, None


def feet_per_pixel(real_inches_per_paper_inch: float, dpi: float) -> float:
    return real_inches_per_paper_inch / (dpi * 12.0)


def _load_vlm(model_id: str) -> Dict[str, Any]:
    if model_id in _VLM_CACHE:
        return _VLM_CACHE[model_id]
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    payload = {"model": model, "processor": processor}
    _VLM_CACHE[model_id] = payload
    return payload


def parse_scale_from_image_vlm(
    image_path: Path,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
) -> Tuple[Optional[float], Optional[str]]:
    """
    VLM fallback to read scale text directly from drawing image.
    """
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        return None, None

    vlm = _load_vlm(model_id)
    proc = vlm["processor"]
    model = vlm["model"]
    prompt = (
        "Read the drawing scale text from this plan sheet. "
        "Return only one scale expression like: 1/4\" = 1'-0\" or 1\" = 10'. "
        "If no scale is visible, return NONE."
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    try:
        # copy image to temp because some runtimes prefer local temp path handling
        tmp.write_bytes(Path(image_path).read_bytes())
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(tmp)},
                {"type": "text", "text": prompt},
            ],
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output = model.generate(**inputs, max_new_tokens=64)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
        decoded = proc.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        if decoded.upper().startswith("NONE"):
            return None, None
        return parse_scale_from_text(decoded)
    except Exception:
        return None, None
    finally:
        tmp.unlink(missing_ok=True)

