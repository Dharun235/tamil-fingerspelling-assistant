"""Extract fixed-size hand-pose features from class-organized images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import HandLandmark, PalmDetection  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LANDMARK_FEATURES = 21 * 3
MAX_HANDS = 2


def feature_names() -> list[str]:
    names = ["image_path", "label", "detected_hands"]
    for side in ("left", "right"):
        names.extend([f"{side}_score", f"{side}_handedness"])
        names.extend(f"{side}_{axis}{i}" for i in range(21) for axis in "xyz")
    return names


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Make landmarks translation- and scale-independent using the wrist."""
    points = landmarks.astype(np.float32).copy()
    points -= points[0]
    scale = float(np.linalg.norm(points[:, :2], axis=1).max())
    if scale > 1e-6:
        points /= scale
    return points.reshape(-1)


def make_row(image_path: Path, label: str, detector: PalmDetection, landmarker: HandLandmark, args: argparse.Namespace) -> list[object] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"skip unreadable: {image_path}", file=sys.stderr)
        return None

    palms = detector.detect(image, threshold=args.det_threshold)
    landmarks, _world, scores, handedness = landmarker.predict(image, palms)
    keep = scores >= args.threshold
    landmarks, scores, handedness = landmarks[keep], scores[keep], handedness[keep]

    # Keep one deterministic slot per side. If handedness duplicates, use x-position.
    hands: dict[str, tuple[np.ndarray, float, float]] = {}
    order = np.argsort(landmarks[:, 0, 0]) if len(landmarks) else []
    for index in order:
        side = "left" if handedness[index] > 0.5 else "right"
        candidate = (landmarks[index], float(scores[index]), float(handedness[index]))
        if side not in hands:
            hands[side] = candidate

    row: list[object] = [str(image_path.relative_to(args.dataset_root)), label, int(len(landmarks))]
    for side in ("left", "right"):
        if side in hands:
            points, score, hand_prob = hands[side]
            row.extend([score, hand_prob, *normalize_landmarks(points)])
        else:
            row.extend([0.0, 0.0, *([0.0] * LANDMARK_FEATURES)])
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Folder containing one subfolder per class")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data" / "landmarks.csv")
    parser.add_argument("--palm-model", type=Path, default=ROOT / "weights" / "palm_detection_full_Nx3x192x192.onnx")
    parser.add_argument("--hand-model", type=Path, default=ROOT / "weights" / "hand_landmark_full_Nx3x224x224.onnx")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images (useful for a speed test)")
    parser.add_argument("--resume", action="store_true", help="Append to existing CSV and skip image paths already present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    providers = ["CPUExecutionProvider"]
    detector = PalmDetection(str(args.palm_model), providers=providers)
    landmarker = HandLandmark(str(args.hand_model), providers=providers)
    images = sorted(path for path in args.dataset_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    existing: set[str] = set()
    if args.resume and args.output_csv.exists():
        with args.output_csv.open(newline="") as file:
            existing = {row[0] for row in csv.reader(file) if row and row[0] != "image_path"}
        images = [path for path in images if str(path.relative_to(args.dataset_root)) not in existing]
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No images found under {args.dataset_root}")

    written = 0
    mode = "a" if args.resume and args.output_csv.exists() else "w"
    with args.output_csv.open(mode, newline="") as file:
        writer = csv.writer(file)
        if mode == "w":
            writer.writerow(feature_names())
        for number, image_path in enumerate(images, start=1):
            relative = image_path.relative_to(args.dataset_root)
            label = relative.parts[0]
            row = make_row(image_path, label, detector, landmarker, args)
            if row is not None:
                writer.writerow(row)
                written += 1
                file.flush()
            if number % 100 == 0 or number == len(images):
                print(f"processed {number}/{len(images)} images; saved {written} rows")


if __name__ == "__main__":
    main()
