#!/usr/bin/env python3
"""Create polished architecture diagrams for the report."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def f(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def centered(draw, box, text, font, fill):
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=5, align="center")
    x = left + (right - left - (bbox[2] - bbox[0])) / 2
    y = top + (bottom - top - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=5, align="center")


def box(draw, xy, title, subtitle, fill, outline="#b9c8d2"):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    centered(draw, (x1 + 12, y1 + 12, x2 - 12, y1 + 58), title, f(25, True), "#123b5d")
    centered(draw, (x1 + 12, y1 + 62, x2 - 12, y2 - 12), subtitle, f(18), "#405563")


def arrow(draw, start, end, color="#2b78a0", width=6):
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - 18 * direction, y2 - 11), (x2 - 18 * direction, y2 + 11)]
    else:
        direction = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 11, y2 - 18 * direction), (x2 + 11, y2 - 18 * direction)]
    draw.polygon(points, fill=color)


def pill(draw, xy, text, fill="#e8f3f8", outline="#7aa9bd"):
    draw.rounded_rectangle(xy, radius=15, fill=fill, outline=outline, width=2)
    centered(draw, xy, text, f(17, True), "#245c77")


def recognition_flow():
    image = Image.new("RGB", (1800, 940), "#f7fafb")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "Recognition and interaction flow", font=f(38, True), fill="#123b5d")
    draw.text((72, 92), "One webcam frame becomes a Tamil character through inspectable stages", font=f(22), fill="#405563")

    boxes = [
        (70, 190, 340, 370, "Browser", "webcam JPEG\nuser feedback", "#e8f3f8"),
        (405, 190, 675, 370, "FastAPI", "WebSocket\nper-user state", "#eaf4ee"),
        (740, 190, 1010, 370, "OpenCV 5", "decode +\n640x480 normalize", "#fff3df"),
        (1075, 190, 1345, 370, "Hand pose", "RTMPose/RTMDet\nup to 2 hands", "#f2eafa"),
        (1410, 190, 1730, 370, "Geometry", "finger states\nT I M R P", "#fcebea"),
    ]
    for x1, y1, x2, y2, title, subtitle, fill in boxes:
        box(draw, (x1, y1, x2, y2), title, subtitle, fill)
    for first, second in zip(boxes, boxes[1:]):
        arrow(draw, (first[2] + 12, 280), (second[0] - 12, 280))

    box(draw, (360, 540, 710, 745), "Class mapping", "left/right bit vectors\nclass ID -> Tamil label", "#eaf4ee")
    box(draw, (805, 540, 1155, 745), "Stability", "350 ms hold\nduplicate suppression", "#fff3df")
    box(draw, (1250, 540, 1600, 745), "Tamil UI", "preview + transcript\nreverse reference sign", "#e8f3f8")
    arrow(draw, (1570, 370), (535, 540), color="#6c8c9b")
    arrow(draw, (710, 642), (805, 642))
    arrow(draw, (1155, 642), (1250, 642))
    pill(draw, (78, 815, 500, 870), "TLFS23 labels + reference images")
    arrow(draw, (500, 842), (535, 745), color="#6c8c9b", width=4)
    pill(draw, (1250, 815, 1690, 870), "User can reframe before commit", fill="#fff3df", outline="#d7ab62")
    arrow(draw, (1420, 745), (1470, 815), color="#d09a3a", width=4)
    image.save(OUT / "recognition-flow.png", optimize=True)


def aws_flow():
    image = Image.new("RGB", (1800, 760), "#f7fafb")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "AWS deployment flow", font=f(38, True), fill="#123b5d")
    draw.text((72, 92), "Reproducible container path from source code to public HTTPS demo", font=f(22), fill="#405563")

    boxes = [
        (80, 220, 390, 440, "Source", "Dockerfile\npinned dependencies", "#e8f3f8"),
        (500, 220, 810, 440, "Amazon ECR", "private image\n`tamil-fingerspelling:v5`", "#fff3df"),
        (920, 220, 1230, 440, "ECS Express Mode", "Fargate x86-64\nStockholm eu-north-1", "#eaf4ee"),
        (1340, 220, 1720, 440, "Public endpoint", "HTTPS ingress\nALB-managed URL", "#f2eafa"),
    ]
    for x1, y1, x2, y2, title, subtitle, fill in boxes:
        box(draw, (x1, y1, x2, y2), title, subtitle, fill)
    for first, second in zip(boxes, boxes[1:]):
        arrow(draw, (first[2] + 12, 330), (second[0] - 12, 330))

    box(draw, (560, 555, 920, 690), "CloudWatch", "container logs +\noperational visibility", "#fcebea")
    box(draw, (1120, 555, 1480, 690), "Browser user", "camera stays client-side\nframes processed in memory", "#e8f3f8")
    arrow(draw, (1075, 440), (750, 555), color="#b45f5f", width=4)
    arrow(draw, (1530, 440), (1300, 555), color="#6c8c9b", width=4)
    pill(draw, (80, 555, 420, 610), "aws_start.sh / aws_stop.sh", fill="#fff3df", outline="#d7ab62")
    arrow(draw, (250, 440), (250, 555), color="#d09a3a", width=4)
    image.save(OUT / "aws-deployment-flow.png", optimize=True)


if __name__ == "__main__":
    recognition_flow()
    aws_flow()
