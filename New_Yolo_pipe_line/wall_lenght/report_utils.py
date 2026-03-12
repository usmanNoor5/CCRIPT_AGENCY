from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def write_excel(rows: List[Dict[str, Any]], out_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Wall Lengths"
    headers = [
        "pdf",
        "page_stem",
        "scale_text",
        "feet_per_pixel",
        "wall_length_pixels",
        "total_wall_length_ft",
        "total_wall_length_m",
        "mask_path",
    ]
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h) for h in headers])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


def write_pdf_report(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Wall Length Estimation Report")
    y -= 24
    c.setFont("Helvetica", 9)
    for r in rows:
        ft = r.get("total_wall_length_ft")
        m = r.get("total_wall_length_m")
        ft_txt = f"{ft:.2f} ft" if isinstance(ft, (int, float)) else "N/A"
        m_txt = f"{m:.2f} m" if isinstance(m, (int, float)) else "N/A"
        line = (
            f"{r['page_stem']} | scale={r.get('scale_text','N/A')} | "
            f"total={ft_txt} ({m_txt})"
        )
        c.drawString(40, y, line)
        y -= 14
        if y < 60:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = height - 40
    c.save()

