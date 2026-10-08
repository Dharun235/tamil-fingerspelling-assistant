#!/usr/bin/env python3
"""Create a compact TLFS23 class-to-sign mapping figure."""

from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/TLFS23 - Tamil Language Finger Spelling Image Dataset"
REFS = DATA / "Refrence Image"
OUT = ROOT / "docs/assets/tlfs23-sign-mapping.png"


def labels():
    result = {}
    for line in (DATA / "ReadMe.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(\d+)\s+([^\s(]+)", line)
        if match:
            result[int(match.group(1))] = match.group(2)
    return result


def font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Tamil MN.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def find_image(class_id):
    prefixes = (f"{class_id}_", f"{class_id}-")
    return next((path for path in REFS.iterdir() if path.name.startswith(prefixes)), None)


def main():
    chosen = [1, 2, 14, 15, 32, 33, 34, 104, 128, 200]
    names = labels()
    width, cell_w, cell_h = 1500, 290, 285
    image = Image.new("RGB", (width, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(34)
    label_font = font(24)
    small_font = font(19)
    draw.text((45, 25), "TLFS23 class IDs -> Tamil labels -> reference signs", fill="#123b5d", font=title_font)
    draw.text((45, 73), "Examples from the 247-character mapping; bit order: T I M R P (1 = finger open)", fill="#405563", font=small_font)

    for index, class_id in enumerate(chosen):
        row, col = divmod(index, 5)
        x, y = 35 + col * cell_w, 115 + row * cell_h
        draw.rounded_rectangle((x, y, x + cell_w - 18, y + cell_h - 18), radius=12, outline="#c8d6de", width=2, fill="#f7fafb")
        ref = find_image(class_id)
        if ref:
            with Image.open(ref) as source:
                source = source.convert("RGB")
                source.thumbnail((235, 165))
                px = x + ((cell_w - 18) - source.width) // 2
                image.paste(source, (px, y + 12))
        label = names.get(class_id, "?")
        draw.text((x + 14, y + 183), f"Class {class_id}: {label}", fill="#123b5d", font=label_font)
        draw.text((x + 14, y + 218), "TLFS23 reference image", fill="#405563", font=small_font)

    image.save(OUT, optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
