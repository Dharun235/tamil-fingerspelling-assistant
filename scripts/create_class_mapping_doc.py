#!/usr/bin/env python3
"""Write the complete TLFS23 class-to-finger-state mapping table."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.tamil_labels import load_labels  # noqa: E402
from scripts.create_finger_mapping import FINGERS, mapping  # noqa: E402


def normalized(value: str) -> str:
    return ", ".join(value.split(",")) if value else "-"


def bits(value: str) -> str:
    active = {part.strip() for part in value.split(",") if part.strip()}
    return "".join("1" if finger in active else "0" for finger in FINGERS) if active else "-"


def main():
    destination = ROOT / "docs/TLFS23_CLASS_MAPPING.md"
    labels = load_labels()
    expected = mapping()
    lines = [
        "# TLFS23 complete class mapping",
        "",
        "Finger order: `T I M R P`; `1` means finger open and `0` means finger closed.",
        "",
        "This table records the deterministic mapping used by the runtime. Class IDs 1-31 are one-hand patterns. IDs 32-247 combine one left-hand pattern with one right-hand pattern.",
        "",
        "| Class ID | Tamil label | Left fingers | Left bits | Right fingers | Right bits |",
        "|---:|---|---|---|---|---|",
    ]
    for class_id in range(1, 248):
        left, right = expected[class_id]
        lines.append(f"| {class_id} | {labels.get(str(class_id), '?')} | {normalized(left)} | `{bits(left)}` | {normalized(right)} | `{bits(right)}` |")
    lines.extend([
        "",
        "| Background | no Tamil label | - | - | - | - |",
        "",
        "Source: [TLFS23 on Mendeley Data](https://data.mendeley.com/datasets/39kzs5pxmk/2), CC BY 4.0."
    ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
