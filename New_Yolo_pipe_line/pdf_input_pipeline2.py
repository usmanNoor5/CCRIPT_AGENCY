"""
Full PDF pipeline for CCRIPT_AGENCY.

Combines all steps into one script:
  1) PDF -> architectural page extraction
  2) VLM filter + crop (Bedrock Claude)
  3) SAM3 wall segmentation
  4) Wall length estimation (scale + quantification)

Usage:
  python3.12 pdf_input_pipeline2.py --pdf-dir . --use-sam3
  python3.12 pdf_input_pipeline2.py --pdf myfile.pdf --use-sam3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

ARCH_DESIGN_TERMS = (
    "floor plan",
    "ground plan",
    "floor plan 1st floor",
    "floor plan 1st",
    "1st floor",
    "2nd floor",
    "ground floor",
    "groundfloor",
    "basement",
    "architecture plan",
    "architectural plan",
    "site plan",
)

FLOOR_HINT_TERMS = (
    "floor plan",
    "ground floor",
    "groundfloor",
    "ground plan",
    "first floor",
    "second floor",
    "third floor",
    "1st floor",
    "2nd floor",
    "3rd floor",
    "storey plan",
    "story plan",
    "foundation plan",
)

DISCIPLINE_TERMS = (
    "electrical",
    "mechanical",
    "plumbing",
    "hvac",
)

EXCLUDE_TERMS = (
    "elevation",
    "elevation drawing",
    "section drawing",
)

PDF_POINTS_PER_INCH = 72.0


@dataclass
class PageCandidate:
    page_index: int
    text: str
    storey_key: str | None
    discipline: str | None


def _is_arch_page(text: str) -> bool:
    t = text.lower()
    if any(ex in t for ex in EXCLUDE_TERMS):
        return False
    if any(term in t for term in ARCH_DESIGN_TERMS):
        return True

    has_floor_hint = any(term in t for term in FLOOR_HINT_TERMS) or bool(
        re.search(
            r"\b(1st|2nd|3rd|first|second|third)\s+(floor|storey|story)\b",
            t,
        )
    )
    has_plan_context = any(
        k in t for k in ("plan", "layout", "architectural", "mechanical", "electrical")
    )
    return has_floor_hint and has_plan_context


def _extract_storey_key(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\bground\s+floor\b|\bgroundfloor\b|\bground\s+plan\b", t):
        return "ground"
    if re.search(r"\bbasement\b", t):
        return "basement"

    m = re.search(r"\b(\d+)(st|nd|rd|th)?\s*(floor|storey|story)\b", t)
    if m:
        return f"floor_{int(m.group(1))}"

    words = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    }
    for w, n in words.items():
        if re.search(rf"\b{w}\s+(floor|storey|story)\b", t):
            return f"floor_{n}"
    return None


def _extract_discipline(text: str) -> str | None:
    t = text.lower()
    for term in DISCIPLINE_TERMS:
        if term in t:
            return term
    return None


def _select_single_plan_per_storey(candidates: List[PageCandidate]) -> List[PageCandidate]:
    by_storey: dict[str, List[PageCandidate]] = {}
    selected_misc: List[PageCandidate] = []
    for c in candidates:
        if c.storey_key is None:
            selected_misc.append(c)
            continue
        by_storey.setdefault(c.storey_key, []).append(c)

    selected: List[PageCandidate] = []
    for storey in sorted(by_storey.keys()):
        selected.append(sorted(by_storey[storey], key=lambda x: x.page_index)[0])

    selected.extend(selected_misc)
    return sorted(selected, key=lambda x: x.page_index)


def _get_page_text(pdf_path: Path, page_index: int, page_obj: "fitz.Page") -> str:
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page = pdf.pages[page_index]
                txt = (page.extract_text() or "").strip()
                if txt:
                    return txt
        except Exception:
            pass

    try:
        txt = page_obj.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        return (txt or "").strip()
    except Exception:
        return ""


def extract_arch_pages(pdf_path: Path, pages_dir: Path, dpi: float) -> List[Path]:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is required for PDF pipeline. Install with: pip install pymupdf"
        )

    pages_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[PageCandidate] = []
    selected: List[Path] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = _get_page_text(pdf_path, page_index, page)
            if not _is_arch_page(text):
                continue
            candidates.append(
                PageCandidate(
                    page_index=page_index,
                    text=text,
                    storey_key=_extract_storey_key(text),
                    discipline=_extract_discipline(text),
                )
            )

        chosen = _select_single_plan_per_storey(candidates)
        scale = dpi / PDF_POINTS_PER_INCH
        matrix = fitz.Matrix(scale, scale)
        for c in chosen:
            page = doc.load_page(c.page_index)
            sheet_id = f"{pdf_path.stem}_p{c.page_index + 1:04d}"
            out_path = pages_dir / f"{sheet_id}.png"
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))
            txt_path = out_path.with_suffix(".txt")
            txt_path.write_text(c.text, encoding="utf-8")
            selected.append(out_path)
    finally:
        doc.close()
    return selected


def vlm_select_pages_from_pdf(
    repo_dir: Path,
    pdf_path: Path,
    work_dir: Path,
    dpi: float,
    vlm_model: str,
) -> List[int]:
    if fitz is None:
        return []

    all_pages_dir = work_dir / "all_pages"
    kept_dir = work_dir / "vlm_kept"
    if all_pages_dir.exists():
        shutil.rmtree(all_pages_dir)
    if kept_dir.exists():
        shutil.rmtree(kept_dir)
    all_pages_dir.mkdir(parents=True, exist_ok=True)
    kept_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        scale = dpi / PDF_POINTS_PER_INCH
        matrix = fitz.Matrix(scale, scale)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            sheet_id = f"{pdf_path.stem}_p{page_index + 1:04d}"
            out_path = all_pages_dir / f"{sheet_id}.png"
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))
    finally:
        doc.close()

    cmd = [
        sys.executable,
        str(repo_dir / "vlm_filter_bedrock.py"),
        "--input-dir",
        str(all_pages_dir),
        "--out-dir",
        str(kept_dir),
        "--model",
        vlm_model,
        "--no-crop",
    ]
    run_cmd(cmd, cwd=repo_dir)

    selected_indices: List[int] = []
    for p in sorted(kept_dir.glob("*.png")):
        m = re.search(r"_p(\d{4})$", p.stem)
        if not m:
            continue
        selected_indices.append(int(m.group(1)) - 1)
    return sorted(set(selected_indices))


def run_cmd(cmd: List[str], cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    print(">", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def preprocess_image_for_inference(
    img_path: Path,
    out_path: Path,
    *,
    mode: str,
    blur: int,
    block_size: int,
    c: int,
) -> Path:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    work = norm
    if blur > 0:
        work = cv2.GaussianBlur(norm, (blur, blur), 0)

    if mode == "gray":
        out = gray
    elif mode == "norm":
        out = norm
    elif mode == "bin":
        out = cv2.adaptiveThreshold(
            work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c,
        )
    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_bgr)
    return out_path


def run_vlm_crop(
    repo_dir: Path,
    image_path: Path,
    run_dir: Path,
    vlm_model: str,
) -> Optional[Path]:
    vlm_input_dir = run_dir / "vlm_input"
    vlm_out_dir = run_dir / "vlm_filtered"
    vlm_input_dir.mkdir(parents=True, exist_ok=True)
    vlm_out_dir.mkdir(parents=True, exist_ok=True)

    local_input = vlm_input_dir / image_path.name
    shutil.copy2(image_path, local_input)
    txt_src = image_path.with_suffix(".txt")
    if txt_src.exists():
        shutil.copy2(txt_src, vlm_input_dir / txt_src.name)

    cmd = [
        sys.executable,
        str(repo_dir / "vlm_filter_bedrock.py"),
        "--input-dir",
        str(vlm_input_dir),
        "--out-dir",
        str(vlm_out_dir),
        "--model",
        vlm_model,
    ]
    try:
        run_cmd(cmd, cwd=repo_dir)
    except subprocess.CalledProcessError as e:
        print(f"Warning: VLM crop failed; falling back to original. Reason: {e}")
        return image_path

    candidate = vlm_out_dir / image_path.name
    if candidate.exists():
        return candidate
    print(f"  -> VLM says not a floor plan — skipping {image_path.name}")
    return None


def run_image_pipeline(
    repo_dir: Path,
    image_path: Path,
    run_dir: Path,
    *,
    tile_size: int,
    overlap: int,
    no_crop: bool,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    min_area: int,
    epsilon: float,
    use_vlm_floor_crop: bool,
    vlm_model: str,
    preprocess_mode: str,
    preprocess_blur: int,
    preprocess_block_size: int,
    preprocess_c: int,
    multi_offset_tiling: bool,
    max_white_ratio: float,
    no_skip_white: bool = True,
    use_sam3: bool = False,
    sam3_checkpoint: Optional[Path] = None,
) -> dict:
    tiles_dir = run_dir / "tiles"
    infer_dir = run_dir / "inference_results"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    if infer_dir.exists():
        shutil.rmtree(infer_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)

    image_for_tiling = image_path
    page_type = "floor_plan"  # default
    if use_vlm_floor_crop:
        vlm_result = run_vlm_crop(
            repo_dir=repo_dir,
            image_path=image_path,
            run_dir=run_dir,
            vlm_model=vlm_model,
        )
        if vlm_result is None:
            return {"image": str(image_path), "skipped": True, "reason": "no label"}
        image_for_tiling = vlm_result

        # Check VLM filter log for page_type (floor_plan vs foundation)
        vlm_log_path = run_dir / "vlm_filtered" / "vlm_filter_log.json"
        if vlm_log_path.exists():
            try:
                vlm_log = json.loads(vlm_log_path.read_text(encoding="utf-8"))
                for entry in vlm_log:
                    if entry.get("file") == image_path.name and entry.get("kept"):
                        page_type = entry.get("page_type", "floor_plan")
                        break
            except Exception:
                pass

        if page_type == "foundation":
            print(f"  -> Foundation plan — VLM cropped, skipping SAM3: {image_path.name}")
            return {
                "image": str(image_path),
                "skipped": True,
                "reason": "foundation",
                "vlm_cropped": str(vlm_result),
            }

    # Optional preprocessing before tiling.
    if preprocess_mode != "none":
        preprocessed_dir = run_dir / "preprocessed"
        preprocessed_path = preprocessed_dir / image_for_tiling.name
        image_for_tiling = preprocess_image_for_inference(
            image_for_tiling,
            preprocessed_path,
            mode=preprocess_mode,
            blur=preprocess_blur,
            block_size=preprocess_block_size,
            c=preprocess_c,
        )

    if use_sam3:
        page_stem = image_path.stem
        overlay_path = infer_dir / f"{page_stem}_overlay.png"
        sam3_cmd = [
            sys.executable,
            str(repo_dir / "predict_sam3.py"),
            str(image_for_tiling),
            "--checkpoint", str(sam3_checkpoint),
            "--conf", str(conf),
            "--output", str(overlay_path),
        ]
        run_cmd(sam3_cmd, cwd=infer_dir)

        raw_mask = infer_dir / f"{page_stem}_overlay_mask.png"
        walls_mask = infer_dir / f"{page_stem}_walls_mask.png"
        if raw_mask.exists():
            raw_mask.rename(walls_mask)

        return {
            "image": str(image_path),
            "image_for_tiling": str(image_for_tiling),
            "preprocess_mode": preprocess_mode,
            "tiles_dir": str(tiles_dir),
            "results_dir": str(infer_dir),
        }

    prep_cmd = [
        sys.executable,
        str(repo_dir / "testing" / "prepare_for_inference.py"),
        "--input", str(image_for_tiling),
        "--tile-size", str(tile_size),
        "--overlap", str(overlap),
        "--out", str(tiles_dir),
    ]
    if no_skip_white:
        prep_cmd.append("--no-skip-white")
    else:
        prep_cmd.extend(["--max-white-ratio", str(max_white_ratio)])
    if multi_offset_tiling:
        prep_cmd.append("--multi-offset")
    if no_crop:
        prep_cmd.append("--no-crop")
    run_cmd(prep_cmd, cwd=repo_dir)

    infer_cmd = [
        sys.executable,
        str(repo_dir / "infer_walls.py"),
        "--source", str(tiles_dir),
        "--model", str(model_path),
        "--conf", str(conf),
        "--imgsz", str(imgsz),
        "--device", str(device),
        "--out", str(infer_dir),
        "--min-area", str(min_area),
        "--epsilon", str(epsilon),
    ]
    run_cmd(
        infer_cmd,
        cwd=repo_dir,
        extra_env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    )

    return {
        "image": str(image_path),
        "image_for_tiling": str(image_for_tiling),
        "preprocess_mode": preprocess_mode,
        "tiles_dir": str(tiles_dir),
        "results_dir": str(infer_dir),
    }


def collect_pdfs(pdf: Path | None, pdf_dir: Path | None) -> List[Path]:
    if pdf is not None:
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf}")
        return [pdf]
    if pdf_dir is None:
        raise ValueError("Provide either --pdf or --pdf-dir")
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")
    return pdfs


def run_wall_length_estimation(summary_path: Path, out_dir: Path, vlm_model: str) -> None:
    """Run wall length estimation on all completed SAM3 runs."""
    quant_dir = out_dir / "quantification"
    print(f"\n{'=' * 60}")
    print("Step 4: Wall length estimation")
    print("=" * 60)
    try:
        from wall_length_pipeline.estimate_wall_lengths import quantify
        rows = quantify(
            summary_path,
            quant_dir,
            use_vlm_scale_fallback=True,
            vlm_model=vlm_model,
        )
        print(f"\nPages quantified: {len(rows)}")
        print("-" * 60)
        for row in rows:
            length_ft = row.get("total_wall_length_ft")
            length_m = row.get("total_wall_length_m")
            scale = row.get("scale_text", "N/A")
            stem = row.get("page_stem", "?")
            if length_ft is not None:
                print(f"  {stem}: {length_ft:.1f} ft ({length_m:.1f} m) [scale: {scale}]")
            else:
                print(f"  {stem}: scale not found — length in pixels only")
        print("-" * 60)
        print(f"Excel:  {quant_dir / 'wall_lengths.xlsx'}")
        print(f"PDF:    {quant_dir / 'wall_lengths_report.pdf'}")
        print(f"JSON:   {quant_dir / 'wall_lengths.json'}")
    except Exception as e:
        print(f"Warning: Wall length estimation failed: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full CCRIPT pipeline: PDF -> page selection -> VLM crop -> SAM3 -> wall lengths"
    )
    parser.add_argument("--pdf", type=Path, default=None, help="Single input PDF")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="Directory of input PDFs")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("testing/pdf_pipeline_runs"),
        help="Output root directory",
    )
    parser.add_argument("--dpi", type=float, default=200.0, help="Render DPI for selected pages")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for prepare step")
    parser.add_argument("--overlap", type=int, default=64, help="Tile overlap for prepare step")
    parser.add_argument("--no-crop", action="store_true", help="Pass --no-crop to prepare step")
    parser.add_argument(
        "--model", type=Path,
        default=Path("runs/segment/runs_api/yoloseg_wall5/weights/best.pt"),
        help="YOLO segmentation model path",
    )
    parser.add_argument("--conf", type=float, default=0.2, help="Inference confidence")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Inference device")
    parser.add_argument("--min-area", type=int, default=200, help="infer_walls --min-area")
    parser.add_argument("--epsilon", type=float, default=0.0001, help="infer_walls --epsilon")
    parser.add_argument(
        "--preprocess-mode", type=str, choices=["none", "gray", "norm", "bin"],
        default="none", help="Preprocess after VLM crop, before tiling",
    )
    parser.add_argument("--bin", action="store_true", help="Shortcut for --preprocess-mode bin")
    parser.add_argument("--preprocess-blur", type=int, default=3)
    parser.add_argument("--preprocess-block-size", type=int, default=31)
    parser.add_argument("--preprocess-c", type=int, default=5)
    parser.add_argument("--multi-offset-tiling", action="store_true")
    parser.add_argument("--max-white-ratio", type=float, default=0.995)
    parser.add_argument("--skip-white", action="store_true")
    parser.add_argument("--no-skip-white", action="store_true")
    parser.add_argument("--no-vlm-floor-crop", action="store_true",
                        help="Disable VLM crop before tiling")
    parser.add_argument(
        "--vlm-model", type=str, default="eu.anthropic.claude-opus-4-6-v1",
        help="VLM model for floor-region crop and scale extraction",
    )
    parser.add_argument("--no-vlm-page-select-fallback", action="store_true")
    parser.add_argument("--use-sam3", action="store_true",
                        help="Use SAM3 instead of YOLO for wall segmentation")
    parser.add_argument(
        "--sam3-checkpoint", type=Path,
        default=Path("/home/ec2-user/CCRIPT_AGENCY/New_Yolo_pipe_line/sam3_experiments_tiled/checkpoints/checkpoint.pt"),
        help="Path to SAM3 checkpoint",
    )
    parser.add_argument("--no-wall-lengths", action="store_true",
                        help="Skip wall length estimation at the end")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    if args.bin:
        args.preprocess_mode = "bin"
    if args.preprocess_block_size % 2 == 0 or args.preprocess_block_size < 3:
        raise SystemExit("--preprocess-block-size must be odd and >= 3")
    if args.preprocess_blur < 0 or (args.preprocess_blur % 2 == 0 and args.preprocess_blur != 0):
        raise SystemExit("--preprocess-blur must be 0 or an odd integer")

    repo_dir = Path(__file__).resolve().parent
    out_dir = args.out_dir if args.out_dir.is_absolute() else (repo_dir / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.pdf, args.pdf_dir)
    model_path = args.model if args.model.is_absolute() else (repo_dir / args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # ---- Step 1: PDF page extraction ----
    print("=" * 60)
    print("Step 1: Extracting architectural pages from PDFs")
    print("=" * 60)

    summary: dict = {
        "pdfs": [],
        "total_selected_pages": 0,
        "runs": [],
        "dpi_used": args.dpi,
    }
    for pdf_path in pdfs:
        pages_dir = out_dir / "selected_pages" / pdf_path.stem
        selected_pages = extract_arch_pages(pdf_path, pages_dir, dpi=args.dpi)
        if (
            len(selected_pages) == 0
            and not args.no_vlm_page_select_fallback
        ):
            fallback_work_dir = out_dir / "vlm_page_select" / pdf_path.stem
            pidxs = vlm_select_pages_from_pdf(
                repo_dir=repo_dir,
                pdf_path=pdf_path,
                work_dir=fallback_work_dir,
                dpi=args.dpi,
                vlm_model=args.vlm_model,
            )
            if pidxs:
                if fitz is None:
                    raise RuntimeError("PyMuPDF required for rendering selected pages")
                if pages_dir.exists():
                    shutil.rmtree(pages_dir)
                pages_dir.mkdir(parents=True, exist_ok=True)
                doc = fitz.open(str(pdf_path))
                try:
                    scale = args.dpi / PDF_POINTS_PER_INCH
                    matrix = fitz.Matrix(scale, scale)
                    for pidx in pidxs:
                        page = doc.load_page(pidx)
                        sheet_id = f"{pdf_path.stem}_p{pidx + 1:04d}"
                        out_path = pages_dir / f"{sheet_id}.png"
                        pix = page.get_pixmap(matrix=matrix, alpha=False)
                        pix.save(str(out_path))
                        selected_pages.append(out_path)
                finally:
                    doc.close()

        pdf_entry = {
            "pdf": str(pdf_path),
            "selected_pages": [str(p) for p in selected_pages],
            "selected_count": len(selected_pages),
        }
        summary["pdfs"].append(pdf_entry)
        summary["total_selected_pages"] += len(selected_pages)

        # ---- Steps 2 & 3: VLM filter/crop + SAM3 wall segmentation ----
        print(f"\n{'=' * 60}")
        print(f"Steps 2-3: VLM filter + wall segmentation for {pdf_path.name}")
        print("=" * 60)

        for page_img in selected_pages:
            run_dir = out_dir / "runs" / page_img.stem
            run_info = run_image_pipeline(
                repo_dir=repo_dir,
                image_path=page_img,
                run_dir=run_dir,
                tile_size=args.tile_size,
                overlap=args.overlap,
                no_crop=args.no_crop,
                model_path=model_path,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                min_area=args.min_area,
                epsilon=args.epsilon,
                use_vlm_floor_crop=not args.no_vlm_floor_crop,
                vlm_model=args.vlm_model,
                preprocess_mode=args.preprocess_mode,
                preprocess_blur=args.preprocess_blur,
                preprocess_block_size=args.preprocess_block_size,
                preprocess_c=args.preprocess_c,
                multi_offset_tiling=args.multi_offset_tiling,
                max_white_ratio=args.max_white_ratio,
                no_skip_white=args.no_skip_white or not args.skip_white,
                use_sam3=args.use_sam3,
                sam3_checkpoint=args.sam3_checkpoint,
            )
            if run_info.get("skipped"):
                if run_info.get("reason") == "foundation":
                    vlm_cropped = run_info.get("vlm_cropped")
                    if vlm_cropped:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        infer_dir = run_dir / "inference_results"
                        infer_dir.mkdir(parents=True, exist_ok=True)
                        dest = infer_dir / page_img.name
                        shutil.copy2(vlm_cropped, dest)
                        print(f"  -> Saved VLM-cropped foundation plan: {dest}")
                        summary["runs"].append({
                            "image": str(page_img),
                            "page_type": "foundation",
                            "results_dir": str(infer_dir),
                        })
                    else:
                        print(f"  -> Kept image (foundation plan): {page_img.name}")
                else:
                    if run_dir.exists():
                        shutil.rmtree(run_dir)
                    if page_img.exists():
                        page_img.unlink()
                    txt_file = page_img.with_suffix(".txt")
                    if txt_file.exists():
                        txt_file.unlink()
            else:
                summary["runs"].append(run_info)

    summary_path = out_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Pipeline summary:")
    print(f"  Processed PDFs: {len(summary['pdfs'])}")
    print(f"  Selected pages: {summary['total_selected_pages']}")
    print(f"  Inference runs: {len(summary['runs'])}")
    print(f"  Summary file:   {summary_path}")

    if args.json:
        print(json.dumps(summary, indent=2))

    # ---- Step 4: Wall length estimation ----
    if not args.no_wall_lengths:
        run_wall_length_estimation(summary_path, out_dir, args.vlm_model)
    else:
        print("\nWall length estimation skipped (--no-wall-lengths)")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
