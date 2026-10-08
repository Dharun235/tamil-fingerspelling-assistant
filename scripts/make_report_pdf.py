#!/usr/bin/env python3
"""Render the technical report as a formatted Unicode PDF."""

from io import BytesIO
from pathlib import Path
import html
import re
import sys

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
PAGE_WIDTH, PAGE_HEIGHT = A4
TEXT_WIDTH = PAGE_WIDTH - 34 * mm
TAMIL_FONT_PATH = "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc"


def inline_markdown(value: str) -> str:
    value = html.escape(value, quote=False)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#135f88">\1</link>', value)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", value)
    return value


def tamil_label(label: str):
    """Return a small raster flowable because ReportLab cannot embed Apple's TTC/CFF font."""
    font = ImageFont.truetype(TAMIL_FONT_PATH, 12, index=0)
    bbox = font.getbbox(label)
    width = max(24, bbox[2] - bbox[0] + 8)
    height = max(18, bbox[3] - bbox[1] + 5)
    bitmap = PILImage.new("RGBA", (width, height), (255, 255, 255, 0))
    ImageDraw.Draw(bitmap).text((4, 1), label, font=font, fill=(20, 30, 40, 255))
    stream = BytesIO()
    bitmap.save(stream, format="PNG")
    stream.seek(0)
    flowable = Image(stream, width=min(width, 42), height=min(height, 18))
    return flowable


def image_flowable(path: Path, max_width=TEXT_WIDTH, max_height=85 * mm):
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(str(path), width=width * scale, height=height * scale)


def table_flowable(rows, mapping=False):
    converted = []
    for row_index, row in enumerate(rows):
        cells = []
        for col_index, value in enumerate(row):
            if mapping and row_index > 0 and col_index == 1:
                cells.append(tamil_label(value))
            else:
                cells.append(Paragraph(inline_markdown(value), STYLES["table"] if row_index else STYLES["table_header"]))
        converted.append(cells)
    table = Table(converted, repeatRows=1, hAlign="LEFT", colWidths=None)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123b5d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#b8c7cf")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fa")]),
    ]))
    return table


def parse_table(lines, start, mapping=False):
    rows = []
    index = start
    while index < len(lines) and lines[index].startswith("|"):
        raw = lines[index]
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        if not all(set(cell) <= set("-:") for cell in cells):
            rows.append(cells)
        index += 1
    return table_flowable(rows, mapping=mapping), index


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, textColor=colors.HexColor("#123b5d"), spaceAfter=12),
        "h2": ParagraphStyle("ReportH2", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor("#123b5d"), spaceBefore=13, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("ReportH3", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=colors.HexColor("#245c77"), spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("ReportBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, spaceAfter=6, alignment=TA_LEFT),
        "bullet": ParagraphStyle("ReportBullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, leftIndent=13, firstLineIndent=-8, spaceAfter=3),
        "note": ParagraphStyle("ReportNote", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=colors.HexColor("#405563"), leftIndent=10, borderPadding=5, borderColor=colors.HexColor("#b8c7cf"), borderWidth=0.5, spaceAfter=7),
        "code": ParagraphStyle("ReportCode", parent=base["Code"], fontName="Courier", fontSize=7.5, leading=10, backColor=colors.HexColor("#f1f4f6"), borderPadding=6, spaceBefore=3, spaceAfter=7),
        "caption": ParagraphStyle("ReportCaption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.HexColor("#405563"), alignment=TA_CENTER, spaceAfter=9),
        "table": ParagraphStyle("TableBody", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9, spaceAfter=0),
        "table_header": ParagraphStyle("TableHeader", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9, textColor=colors.white, spaceAfter=0),
    }


STYLES = styles()


def report_story(report_path: Path):
    lines = report_path.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    in_code = False
    code = []
    while index < len(lines):
        raw = lines[index]
        if raw.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code), STYLES["code"]))
                code = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code.append(raw)
            index += 1
            continue
        if raw.strip() == "<!-- FULL_MAPPING_TABLE -->":
            story.append(PageBreak())
            story.append(Paragraph("Appendix - complete TLFS23 class mapping", STYLES["h2"]))
            mapping_path = report_path.parent / "TLFS23_CLASS_MAPPING.md"
            mapping_lines = mapping_path.read_text(encoding="utf-8").splitlines()
            table_start = next(i for i, value in enumerate(mapping_lines) if value.startswith("| Class ID"))
            table, _ = parse_table(mapping_lines, table_start, mapping=True)
            story.append(table)
            story.append(Spacer(1, 5))
            story.append(Paragraph("Source: TLFS23 on Mendeley Data, CC BY 4.0.", STYLES["note"]))
            index += 1
            continue
        if not raw.strip():
            story.append(Spacer(1, 4))
            index += 1
            continue
        if raw.startswith("# "):
            story.append(Paragraph(inline_markdown(raw[2:]), STYLES["title"]))
        elif raw.startswith("## "):
            story.append(Paragraph(inline_markdown(raw[3:]), STYLES["h2"]))
        elif raw.startswith("### "):
            story.append(Paragraph(inline_markdown(raw[4:]), STYLES["h3"]))
        elif raw.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", raw)
            if match:
                figure_path = report_path.parent / match.group(2)
                if figure_path.exists():
                    story.append(Spacer(1, 4))
                    story.append(image_flowable(figure_path))
                    story.append(Paragraph(match.group(1), STYLES["caption"]))
        elif raw.startswith("*") and raw.endswith("*"):
            story.append(Paragraph(inline_markdown(raw), STYLES["caption"]))
        elif raw.startswith("- "):
            story.append(Paragraph("• " + inline_markdown(raw[2:]), STYLES["bullet"]))
        elif re.match(r"^\d+\. ", raw):
            story.append(Paragraph(inline_markdown(raw), STYLES["bullet"]))
        elif raw.startswith("> "):
            story.append(Paragraph(inline_markdown(raw[2:]), STYLES["note"]))
        elif raw.startswith("|"):
            table, index = parse_table(lines, index)
            story.append(table)
            story.append(Spacer(1, 8))
            continue
        elif raw.startswith("<"):
            # HTML comments and bare angle-bracket links are not useful in the PDF body.
            if raw.startswith("<http"):
                story.append(Paragraph(inline_markdown(raw[1:-1]), STYLES["body"]))
        else:
            story.append(Paragraph(inline_markdown(raw), STYLES["body"]))
        index += 1
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e0e5"))
    canvas.line(17 * mm, 13 * mm, PAGE_WIDTH - 17 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#64747d"))
    canvas.drawString(17 * mm, 8 * mm, "Real-Time Tamil Fingerspelling Assistant")
    canvas.drawRightString(PAGE_WIDTH - 17 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm, topMargin=16 * mm, bottomMargin=18 * mm, title="Real-Time Tamil Fingerspelling Assistant - Technical Report")
    document.build(report_story(source), onFirstPage=footer, onLaterPages=footer)
