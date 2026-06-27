"""
Build training/docs/LIVE_SEGMENTATION_FIX.pdf from LIVE_FIX.md.

Reuses the markdown -> reportlab renderer from build_pdf.py (same look as the
main training documentation) and regenerates the state-machine diagram first.

    python training/docs/build_live_fix_pdf.py
"""
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

import make_live_diagram
from build_pdf import GREY, parse_md, styles

DOCS = Path(__file__).resolve().parent
OUT = DOCS / "LIVE_SEGMENTATION_FIX.pdf"


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(1.8 * cm, 1.1 * cm,
                      "Sign-Language Recognition — Live Translation Fix")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, str(doc.page))
    canvas.restoreState()


def main():
    make_live_diagram.main()                       # refresh the diagram first

    st = styles()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, leftMargin=1.8 * cm,
                            rightMargin=1.8 * cm,
                            title="Sign-Language Recognition — Live Translation Fix")
    flow = [Spacer(1, 3 * cm),
            Paragraph("Live Translation Fix", st["TitleBig"]),
            Paragraph("Motion-Gated Signs, Margin Gating &amp; De-duplication", st["Sub"]),
            Spacer(1, 0.5 * cm),
            Paragraph(f"Generated {date.today().isoformat()}", st["Sub"]),
            Spacer(1, 1.0 * cm)]
    flow += parse_md(DOCS / "LIVE_FIX.md", st, doc.width)
    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[OK] PDF -> {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
