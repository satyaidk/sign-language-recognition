"""
Build training/docs/TRAINING_DOCUMENTATION.pdf from the markdown files in
training/docs/.

A small, purpose-built Markdown -> reportlab renderer covering exactly the
subset used in these docs: headings, paragraphs, bullet/numbered lists, fenced
code blocks, pipe tables, images, blockquotes, horizontal rules, inline `code`
and **bold**.  The training diagrams and the metrics plots are embedded.

    python training/docs/build_pdf.py
"""
import re
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

DOCS = Path(__file__).resolve().parent
TRAIN_DIR = DOCS.parent
OUT = DOCS / "TRAINING_DOCUMENTATION.pdf"
ORDER = ["01_overview.md", "02_architecture.md", "03_pipeline.md",
         "04_file_reference.md", "05_data_formats.md", "06_developer_guide.md",
         "07_results_and_future.md"]

INK = colors.HexColor("#1B2631")
ACCENT = colors.HexColor("#2E86C1")
CODE_BG = colors.HexColor("#F4F6F7")
HEAD_BG = colors.HexColor("#EAF2F8")
GREY = colors.HexColor("#5D6D7E")

# Glyphs the standard PDF fonts can't show -> ASCII equivalents.
TEXT_MAP = {0x2265: ">=", 0x2264: "<=", 0x00D7: "x", 0x2192: "->", 0x2190: "<-",
            0x2208: " in ", 0x2026: "...", 0x2022: "-", 0x00B1: "+/-"}
BOX = {0x2502: "|", 0x2500: "-", 0x250C: "+", 0x2510: "+", 0x2514: "+",
       0x2518: "+", 0x251C: "+", 0x2524: "+", 0x252C: "+", 0x2534: "+",
       0x253C: "+", 0x25BA: ">", 0x25B6: ">", 0x25BC: "v", 0x25B2: "^"}
CODE_MAP = {**TEXT_MAP, **BOX}


def inline(t):
    t = t.translate(TEXT_MAP)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" color="#A0228C">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    return t


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("H1x", parent=s["Heading1"], fontSize=19, spaceBefore=6,
                         spaceAfter=10, textColor=INK))
    s.add(ParagraphStyle("H2x", parent=s["Heading2"], fontSize=14, spaceBefore=12,
                         spaceAfter=6, textColor=ACCENT))
    s.add(ParagraphStyle("H3x", parent=s["Heading3"], fontSize=11.5, spaceBefore=9,
                         spaceAfter=4, textColor=INK))
    s.add(ParagraphStyle("Body", parent=s["BodyText"], fontSize=9.5, leading=14,
                         spaceAfter=6, textColor=INK))
    s.add(ParagraphStyle("Bull", parent=s["BodyText"], fontSize=9.5, leading=13,
                         leftIndent=16, bulletIndent=4, spaceAfter=2, textColor=INK))
    s.add(ParagraphStyle("Quote", parent=s["BodyText"], fontSize=9, leading=13,
                         leftIndent=14, textColor=GREY, spaceAfter=6,
                         borderColor=ACCENT, borderWidth=0, leftPadding=8))
    s.add(ParagraphStyle("Codeblk", parent=s["Code"], fontSize=8, leading=10,
                         textColor=INK, backColor=CODE_BG, borderPadding=6,
                         spaceAfter=8, spaceBefore=2))
    s.add(ParagraphStyle("Cell", parent=s["BodyText"], fontSize=8.3, leading=11,
                         textColor=INK))
    s.add(ParagraphStyle("CellH", parent=s["BodyText"], fontSize=8.5, leading=11,
                         textColor=INK, fontName="Helvetica-Bold"))
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], fontSize=26, textColor=INK,
                         alignment=TA_CENTER, spaceAfter=4))
    s.add(ParagraphStyle("Sub", parent=s["BodyText"], fontSize=12, textColor=GREY,
                         alignment=TA_CENTER))
    return s


def image_flowable(path, max_w):
    if not path.exists():
        return None
    iw, ih = ImageReader(str(path)).getSize()
    w = min(max_w, iw)
    return Image(str(path), width=w, height=w * ih / iw)


def table_flowable(rows, st, max_w):
    header, *body = rows
    ncol = len(header)
    data = [[Paragraph(inline(c), st["CellH"]) for c in header]]
    data += [[Paragraph(inline(c), st["Cell"]) for c in r] for r in body]
    t = Table(data, colWidths=[max_w / ncol] * ncol, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D5DBDB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def parse_md(path, st, max_w):
    lines = path.read_text(encoding="utf-8").split("\n")
    flow, i, n = [], 0, len(lines)
    para = []

    def flush():
        if para:
            flow.append(Paragraph(inline(" ".join(para)), st["Body"]))
            para.clear()

    while i < n:
        ln = lines[i]
        st_ln = ln.strip()

        if st_ln.startswith("```"):                       # code block
            flush(); i += 1; buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i].translate(CODE_MAP)); i += 1
            i += 1
            flow.append(Preformatted("\n".join(buf), st["Codeblk"]))
            continue

        if not st_ln:                                     # blank
            flush(); i += 1; continue

        if st_ln.startswith("!["):                        # image
            flush()
            m = re.search(r"\(([^)]+)\)", st_ln)
            if m:
                img = image_flowable((path.parent / m.group(1)).resolve(), max_w)
                if img:
                    flow += [Spacer(1, 4), img, Spacer(1, 4)]
            i += 1; continue

        if st_ln.startswith("|"):                         # table
            flush(); rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", lines[i].strip()):   # skip separator
                    rows.append(cells)
                i += 1
            if rows:
                flow += [table_flowable(rows, st, max_w), Spacer(1, 6)]
            continue

        if re.match(r"^#{1,6}\s", st_ln):                 # heading
            flush()
            lvl = len(st_ln) - len(st_ln.lstrip("#"))
            txt = st_ln[lvl:].strip()
            flow.append(Paragraph(inline(txt), st[f"H{min(lvl,3)}x"]))
            i += 1; continue

        if st_ln in ("---", "***", "___"):                # hr
            flush()
            flow.append(Spacer(1, 6)); i += 1; continue

        if st_ln.startswith(">"):                         # blockquote
            flush()
            flow.append(Paragraph(inline(st_ln.lstrip(">").strip()), st["Quote"]))
            i += 1; continue

        m = re.match(r"^(\d+)\.\s+(.*)", st_ln)           # numbered list
        if m:
            flush()
            flow.append(Paragraph(inline(m.group(2)), st["Bull"],
                                  bulletText=f"{m.group(1)}."))
            i += 1; continue

        if re.match(r"^[-*]\s+", st_ln):                  # bullet list
            flush()
            flow.append(Paragraph(inline(st_ln[2:]), st["Bull"], bulletText="-"))
            i += 1; continue

        para.append(st_ln); i += 1                        # paragraph text

    flush()
    return flow


def title_page(st, max_w):
    flow = [Spacer(1, 5 * cm),
            Paragraph("Sign-Language Recognition", st["TitleBig"]),
            Paragraph("Training &amp; Inference — Technical Documentation", st["Sub"]),
            Spacer(1, 0.6 * cm),
            Paragraph(f"Generated {date.today().isoformat()}", st["Sub"]),
            Spacer(1, 1.4 * cm)]
    contents = [["#", "Section"]]
    for f in ORDER:
        title = f[3:-3].replace("_", " ").title()
        contents.append([f[:2], title])
    contents.append(["A", "Appendix — Training Metrics"])
    flow.append(table_flowable(contents, st, max_w * 0.7))
    flow.append(PageBreak())
    return flow


def appendix(st, max_w):
    flow = [Paragraph("Appendix — Training Metrics", st["H1x"]),
            Paragraph("Auto-generated charts from <font face='Courier'>train.py</font> "
                      "(cross-validation) under <font face='Courier'>training/artifacts/metrics/</font>.",
                      st["Body"])]
    reports = sorted((TRAIN_DIR / "artifacts" / "metrics").glob("*.png"))
    if not reports:
        flow.append(Paragraph("No metrics plots found — run "
                              "<font face='Courier'>python training/train.py</font> first.", st["Body"]))
    for p in reports:
        flow.append(Paragraph(p.stem.replace("_", " ").title(), st["H3x"]))
        img = image_flowable(p, max_w)
        if img:
            flow += [img, Spacer(1, 10)]
    return flow


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(1.8 * cm, 1.1 * cm,
                      "Sign-Language Recognition — Training & Inference Documentation")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, str(doc.page))
    canvas.restoreState()


def main():
    st = styles()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, leftMargin=1.8 * cm,
                            rightMargin=1.8 * cm, title="Sign-Language Recognition — Training")
    max_w = doc.width
    flow = title_page(st, max_w)
    for f in ORDER:
        flow += parse_md(DOCS / f, st, max_w)
        flow.append(PageBreak())
    flow += appendix(st, max_w)
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(f"[OK] PDF -> {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
