from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from wall_length_pipeline.mask_utils import estimate_wall_length_pixels
from wall_length_pipeline.report_utils import write_excel, write_pdf_report
from wall_length_pipeline.scale_utils import (
    extract_page_text,
    feet_per_pixel,
    parse_scale_from_image_vlm,
    parse_scale_from_text,
)

# PDF points per inch (standard)
PDF_POINTS_PER_INCH = 72.0


def page_index_from_stem(stem: str) -> Optional[int]:
    m = re.search(r"_p(\d{4})$", stem)
    if not m:
        return None
    return int(m.group(1)) - 1


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_page_to_pdf(summary: Dict[str, Any]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in summary.get("pdfs", []):
        pdf_path = Path(p["pdf"])
        for img in p.get("selected_pages", []):
            out[Path(img).stem] = pdf_path
    return out


def find_mask(results_dir: Path, page_stem: str) -> Optional[Path]:
    candidate = results_dir / f"{page_stem}_walls_mask.png"
    if candidate.exists():
        return candidate
    masks = sorted(results_dir.glob("*_walls_mask.png"))
    return masks[0] if masks else None


def infer_dpi_from_rendered_page(
    image_path: Path,
    pdf_path: Path,
    page_index: int,
) -> Optional[float]:
    """
    Infer DPI from a rendered page image and the PDF page dimensions.
    Formula: DPI = (image pixels) * 72 / (page size in points).
    Returns None if inference fails (missing file, no PyMuPDF, etc.).
    """
    if not image_path.exists():
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h_px, w_px = img.shape[:2]

    if fitz is None:
        return None
    try:
        doc = fitz.open(str(pdf_path))
        try:
            page = doc.load_page(page_index)
            rect = page.rect
            w_pt = rect.width
            h_pt = rect.height
        finally:
            doc.close()
    except Exception:
        return None

    if w_pt <= 0 or h_pt <= 0:
        return None
    # DPI = pixels per inch; 1 inch = 72 points, so pixels = (points/72) * dpi => dpi = pixels * 72 / points
    dpi_w = w_px * PDF_POINTS_PER_INCH / w_pt
    dpi_h = h_px * PDF_POINTS_PER_INCH / h_pt
    # Use average in case of small rounding differences
    return (dpi_w + dpi_h) / 2.0


def infer_dpi_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    """
    When pipeline summary has no 'dpi_used', infer DPI from the first run's
    rendered page image and PDF page dimensions.
    """
    page_to_pdf = build_page_to_pdf(summary)
    for run in summary.get("runs", []):
        image_path = Path(run["image"])
        page_stem = image_path.stem
        pdf_path = page_to_pdf.get(page_stem)
        if pdf_path is None:
            continue
        pidx = page_index_from_stem(page_stem)
        if pidx is None:
            continue
        dpi = infer_dpi_from_rendered_page(image_path, pdf_path, pidx)
        if dpi is not None:
            return dpi
    return None


def quantify(
    summary_path: Path,
    out_dir: Path,
    *,
    use_vlm_scale_fallback: bool = True,
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    mask_dilate: int = 0,
) -> List[Dict[str, Any]]:
    summary = load_summary(summary_path)
    # DPI: use pipeline summary if present, else infer from rendered page image + PDF page dimensions
    dpi = summary.get("dpi_used")
    if dpi is not None:
        dpi = float(dpi)
        print(f"Using DPI from pipeline summary: {dpi}")
    else:
        dpi = infer_dpi_from_summary(summary)
        if dpi is not None:
            print(f"Using DPI inferred from rendered page + PDF: {dpi:.1f}")
        else:
            dpi = 200.0
            print("Warning: Could not get DPI from summary or from PDF/page image. Assuming 200 DPI.")

    page_to_pdf = build_page_to_pdf(summary)
    pdf_scale_cache: Dict[str, tuple[Optional[float], Optional[str]]] = {}
    rows: List[Dict[str, Any]] = []

    for run in summary.get("runs", []):
        if run.get("skipped"):
            continue
        if "results_dir" not in run:
            continue
        image_path = Path(run["image"])
        page_stem = image_path.stem
        results_dir = Path(run["results_dir"])
        mask_path = find_mask(results_dir, page_stem)
        if mask_path is None or not mask_path.exists():
            continue

        pdf_path = page_to_pdf.get(page_stem)
        if pdf_path is None:
            continue
        pidx = page_index_from_stem(page_stem)
        if pidx is None:
            continue

        text = extract_page_text(pdf_path, pidx)
        inches_per_paper_inch, scale_text = parse_scale_from_text(text)
        if inches_per_paper_inch is None and use_vlm_scale_fallback:
            inches_per_paper_inch, scale_text = parse_scale_from_image_vlm(
                image_path,
                model_id=vlm_model,
            )
        if inches_per_paper_inch is None:
            cached = pdf_scale_cache.get(str(pdf_path))
            if cached is not None:
                inches_per_paper_inch, scale_text = cached
        else:
            pdf_scale_cache[str(pdf_path)] = (inches_per_paper_inch, scale_text)

        # Get length with validation (optional dilation recovers under-segmented edges)
        length_px, validation = estimate_wall_length_pixels(
            mask_path, return_validation=True, dilate_pixels=mask_dilate
        )
        
        ft_per_px = feet_per_pixel(inches_per_paper_inch, dpi) if inches_per_paper_inch else None
        length_ft = (length_px * ft_per_px) if ft_per_px else None
        length_m = (length_ft * 0.3048) if length_ft is not None else None

        row = {
            "pdf": str(pdf_path),
            "page_stem": page_stem,
            "scale_text": scale_text or "NOT_FOUND",
            "feet_per_pixel": ft_per_px,
            "wall_length_pixels": float(length_px),
            "total_wall_length_ft": float(length_ft) if length_ft is not None else None,
            "total_wall_length_m": float(length_m) if length_m is not None else None,
            "mask_path": str(mask_path),
            "validation": validation,
        }
        
        # Print warnings if validation issues detected
        if validation["warnings"]:
            print(f"\nWarning for {page_stem}:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_excel(rows, out_dir / "wall_lengths.xlsx")
    write_pdf_report(rows, out_dir / "wall_lengths_report.pdf")
    (out_dir / "wall_lengths.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Estimate total wall lengths from stitched mask outputs."
    )
    ap.add_argument(
        "--pipeline-summary",
        type=Path,
        required=True,
        help="Path to pdf_input_pipeline summary json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("testing/pdf_pipeline_runs/quantification"),
        help="Output directory for Excel/PDF reports",
    )
    ap.add_argument(
        "--mask-dilate",
        type=int,
        default=0,
        metavar="N",
        help="Dilate mask by N pixels before length (e.g. 1 recovers under-segmented edges; can add ~1-3%% length). Default 0.",
    )
    ap.add_argument(
        "--no-vlm-scale-fallback",
        action="store_true",
        help="Disable VLM fallback for scale extraction from page image",
    )
    ap.add_argument(
        "--vlm-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="VLM model for image-based scale extraction fallback",
    )
    args = ap.parse_args()

    rows = quantify(
        args.pipeline_summary,
        args.out_dir,
        use_vlm_scale_fallback=not args.no_vlm_scale_fallback,
        vlm_model=args.vlm_model,
        mask_dilate=args.mask_dilate,
    )
    print(f"Pages quantified: {len(rows)}")
    print(f"Excel: {args.out_dir / 'wall_lengths.xlsx'}")
    print(f"PDF:   {args.out_dir / 'wall_lengths_report.pdf'}")
    print(f"JSON:  {args.out_dir / 'wall_lengths.json'}")


if __name__ == "__main__":
    main()

