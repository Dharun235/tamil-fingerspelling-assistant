"""Create TLFS23 finger ON/OFF mapping and compare detected reference states."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FINGERS = ("T", "I", "M", "R", "P")
RIGHT = {
    1: "I", 2: "T,I", 3: "I,M", 4: "T,I,M", 5: "I,M,R",
    6: "T,I,M,R", 7: "I,M,R,P", 8: "T,I,M,R,P", 9: "T",
    10: "P", 11: "T,P", 12: "I,P", 13: "M,R,P",
}
LEFT = {
    14: "T", 15: "P", 16: "P,R,M", 17: "P,R", 18: "T,I",
    19: "I,M,R", 20: "T,I,M,R", 21: "I", 22: "I,P",
    23: "T,I,M,P", 24: "I,M,R,P", 25: "T,M", 26: "T,P",
    27: "T,M,R,P", 28: "T,M,R", 29: "T,I,P", 30: "T,I,M", 31: "I,M",
}


def clean(value: str) -> str:
    return ",".join(f for f in FINGERS if f in {x.strip() for x in value.split(",") if x.strip()})


def bits(value: str) -> str:
    active = {x.strip() for x in value.split(",") if x.strip()}
    return "".join("1" if finger in active else "0" for finger in FINGERS)


def mapping() -> dict[int, tuple[str, str]]:
    result = {label: ("", value) for label, value in RIGHT.items()}
    result.update({label: (value, "") for label, value in LEFT.items()})
    label = 32
    # 18 left patterns × 12 right patterns = 216 two-hand classes.
    # Right pattern 13 is a right-only class and is not used in combinations.
    for left_label in range(14, 32):
        for right_label in range(1, 13):
            result[label] = (LEFT[left_label], RIGHT[right_label])
            label += 1
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-csv", type=Path, default=Path("output/reference_finger_analysis/finger_scores.csv"))
    parser.add_argument("--mapping-csv", type=Path, default=Path("output/finger_state_mapping.csv"))
    parser.add_argument("--check-csv", type=Path, default=Path("output/reference_mapping_check.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = mapping()
    args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "left_on", "left_bits_T I M R P", "right_on", "right_bits_T I M R P"])
        for label in range(1, 248):
            left, right = expected[label]
            writer.writerow([label, clean(left), bits(left), clean(right), bits(right)])

    if not args.states_csv.exists():
        print(f"mapping saved: {args.mapping_csv}")
        return

    with args.states_csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    args.check_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class", "image_path", "expected_left", "observed_left", "expected_right", "observed_right", "status"]
    counts: dict[str, int] = {}
    with args.check_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            label = int(row["label"])
            expected_left, expected_right = ({x.strip() for x in expected[label][0].split(",") if x.strip()}, {x.strip() for x in expected[label][1].split(",") if x.strip()})
            observed: dict[str, set[str]] = {}
            available: dict[str, bool] = {}
            missing: set[str] = set()
            for side in ("left", "right"):
                observed[side] = set()
                present = False
                for finger in FINGERS:
                    raw = row[f"{side}_{finger}_score"]
                    try:
                        present |= math.isfinite(float(raw))
                    except ValueError:
                        pass
                    if row[f"{side}_{finger}_state"] == "HIGH":
                        observed[side].add(finger)
                if not present:
                    missing.add(side)
                available[side] = present
            status = "PASS"
            expected_sides = {side for side, value in (("left", expected_left), ("right", expected_right)) if value}
            if missing & expected_sides:
                status = "MISSING_HAND"
            elif any(available[side] for side in ("left", "right") if side not in expected_sides):
                status = "EXTRA_HAND"
            elif observed["left"] != expected_left or observed["right"] != expected_right:
                status = "STATE_MISMATCH"
            counts[status] = counts.get(status, 0) + 1
            writer.writerow({
                "class": label,
                "image_path": row["image_path"],
                "expected_left": clean(",".join(expected_left)),
                "observed_left": clean(",".join(observed["left"])),
                "expected_right": clean(",".join(expected_right)),
                "observed_right": clean(",".join(observed["right"])),
                "status": status,
            })
    print(f"mapping saved: {args.mapping_csv}")
    print(f"check saved: {args.check_csv}")
    print("summary:", counts)


if __name__ == "__main__":
    main()
