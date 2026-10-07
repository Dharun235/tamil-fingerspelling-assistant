"""Tamil class labels and macOS font discovery for realtime display."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = ROOT / "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/ReadMe.txt"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc"),
    Path("/System/Library/Fonts/Supplemental/Tamil MN.ttc"),
)


def load_labels(path: Path = DEFAULT_README) -> dict[str, str]:
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(\d+)\s+([^\s(]+)", line)
        if match:
            labels[match.group(1)] = match.group(2)
    return labels


def tamil_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(
        "Tamil font not found. Install Tamil Sangam MN or pass a Tamil font path."
    )
