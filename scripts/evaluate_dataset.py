"""Evaluate hand detection, finger states, Tamil mapping, and latency.

The TLFS23 folder number supplies the expected hand-state combination. This is
therefore a proxy evaluation: it measures whether the model/rules obey the
dataset's class mapping, not manually annotated landmark correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.finger_rules import FINGERS, classify  # noqa: E402
from models.geometry_features import coordinate_features  # noqa: E402
from models.rtmpose_utils import suppress_duplicate_hands  # noqa: E402
from scripts.create_finger_mapping import mapping  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_default = ROOT / "data/TLFS23 - Tamil Language Finger Spelling Image Dataset"
    parser.add_argument("--dataset-root", type=Path, default=dataset_default)
    parser.add_argument("--limit-per-class", type=int, default=20,
                        help="Images per class; 0 evaluates every image.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/dataset_eval_20")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--det-model", type=Path)
    parser.add_argument("--pose-model", type=Path)
    return parser.parse_args()


def expected_states(label: int) -> dict[str, set[str]]:
    left, right = mapping()[label]
    return {
        "left": {finger for finger in left.split(",") if finger},
        "right": {finger for finger in right.split(",") if finger},
    }


def image_rows(dataset_root: Path, limit: int):
    folders = sorted(
        (path for path in (dataset_root / "Dataset Folders").iterdir() if path.is_dir()),
        key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name),
    )
    for folder in folders:
        if not folder.name.isdigit() or not 1 <= int(folder.name) <= 247:
            continue
        images = sorted(path for path in folder.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        selected = images if limit == 0 else images[:limit]
        for image in selected:
            yield int(folder.name), image


def hand_side(hand: np.ndarray) -> str:
    first = hand[9, :2] - hand[0, :2]
    second = hand[17, :2] - hand[0, :2]
    cross = float(first[0] * second[1] - first[1] * second[0])
    return "right" if cross >= 0 else "left"


def infer(detector, frame: np.ndarray, thresholds: dict, score_threshold: float):
    keypoints, scores = detector(frame)
    detections = [
        (hand.astype(np.float32), float(np.mean(score)))
        for hand, score in zip(keypoints, scores, strict=True)
        if float(np.mean(score)) >= score_threshold
    ]
    detections = suppress_duplicate_hands(detections, 0.30)
    chosen: dict[str, tuple[np.ndarray, float]] = {}
    for hand, score in detections:
        side = hand_side(hand)
        if side not in chosen or score > chosen[side][1]:
            chosen[side] = hand, score

    observed = {side: set() for side in SIDES}
    scores_by_side = {side: None for side in SIDES}
    gestures = {}
    for side, (hand, score) in chosen.items():
        labels, gesture = classify(
            coordinate_features(hand),
            hand,
            side,
            thresholds,
        )
        observed[side] = {finger for finger, value in labels.items() if value}
        scores_by_side[side] = score
        gestures[side] = gesture
    return observed, scores_by_side, gestures, len(chosen)


def predicted_label(observed: dict[str, set[str]]) -> int | None:
    for label, (left, right) in mapping().items():
        if observed["left"] == {x for x in left.split(",") if x} \
                and observed["right"] == {x for x in right.split(",") if x}:
            return label
    return None


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, fraction * 100))


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    thresholds_path = ROOT / "data/coordinate_thresholds.json"
    thresholds = json.loads(thresholds_path.read_text())
    rows = list(image_rows(dataset_root, args.limit_per_class))
    if not rows:
        raise FileNotFoundError(f"No class images found under {dataset_root}")

    from rtmlib import Hand

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector_kwargs = {"backend": "onnxruntime", "device": args.device}
    if args.det_model:
        detector_kwargs["det"] = str(args.det_model)
    if args.pose_model:
        detector_kwargs["pose"] = str(args.pose_model)
    with redirect_stdout(StringIO()):
        detector = Hand(**detector_kwargs)

    prediction_fields = [
        "class_label", "image_path", "detected_hands", "expected_hands",
        "hand_set_correct", "predicted_class", "class_correct", "latency_ms",
    ]
    for side in SIDES:
        prediction_fields.extend([f"{side}_score", f"{side}_expected", f"{side}_observed"])
        prediction_fields.extend(f"{side}_{finger}_correct" for finger in FINGERS)

    predictions = []
    latencies = []
    class_counts = Counter()
    hand_set_correct = 0
    hand_expected_count = Counter()
    hand_detected_count = Counter()
    finger_counts = {finger: Counter() for finger in FINGERS}
    confusion = Counter()

    for index, (label, image_path) in enumerate(rows, start=1):
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        start = time.perf_counter()
        observed, scores, gestures, detected_count = infer(
            detector, frame, thresholds, args.score_threshold)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        expected = expected_states(label)
        expected_sides = {side for side in SIDES if expected[side]}
        observed_sides = {side for side in SIDES if scores[side] is not None}
        sides_correct = expected_sides == observed_sides
        hand_set_correct += int(sides_correct)
        predicted = predicted_label(observed)
        class_correct = predicted == label
        confusion[(label, predicted if predicted is not None else 0)] += 1
        class_counts[label] += 1

        row = {
            "class_label": label,
            "image_path": str(image_path.relative_to(ROOT)),
            "detected_hands": detected_count,
            "expected_hands": len(expected_sides),
            "hand_set_correct": int(sides_correct),
            "predicted_class": predicted or "UNKNOWN",
            "class_correct": int(class_correct),
            "latency_ms": round(latency_ms, 3),
        }
        for side in SIDES:
            expected_text = "".join(finger for finger in FINGERS if finger in expected[side])
            observed_text = "".join(finger for finger in FINGERS if finger in observed[side])
            row[f"{side}_score"] = "" if scores[side] is None else round(scores[side], 4)
            row[f"{side}_expected"] = expected_text
            row[f"{side}_observed"] = observed_text
            if expected[side]:
                hand_expected_count[side] += 1
            if scores[side] is not None:
                hand_detected_count[side] += 1
            for finger in FINGERS:
                if scores[side] is None:
                    row[f"{side}_{finger}_correct"] = ""
                    continue
                actual = finger in expected[side]
                predicted_state = finger in observed[side]
                row[f"{side}_{finger}_correct"] = int(actual == predicted_state)
                finger_counts[finger]["tp"] += int(actual and predicted_state)
                finger_counts[finger]["fp"] += int(not actual and predicted_state)
                finger_counts[finger]["fn"] += int(actual and not predicted_state)
                finger_counts[finger]["tn"] += int(not actual and not predicted_state)
        predictions.append(row)
        if index % 100 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}")

    with (args.output_dir / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(predictions)

    labels = sorted(class_counts)
    total = len(predictions)
    exact_class_accuracy = sum(row["class_correct"] for row in predictions) / max(total, 1)
    per_class = []
    for label in labels:
        support = class_counts[label]
        correct = sum(row["class_correct"] for row in predictions if row["class_label"] == label)
        per_class.append({"class_label": label, "support": support,
                          "accuracy": correct / max(support, 1)})
    with (args.output_dir / "per_class.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class_label", "support", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class)

    class_metrics = []
    for label in labels:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in labels if other != label)
        fn = sum(confusion[(label, other)] for other in labels + [0] if other != label)
        class_metrics.append(binary_metrics(tp, fp, fn, 0))
    macro = {name: statistics.mean(metric[name] for metric in class_metrics)
             for name in ("precision", "recall", "f1")}

    summary = {
        "dataset_root": str(dataset_root),
        "sample_limit_per_class": args.limit_per_class,
        "images_evaluated": total,
        "classes_evaluated": len(labels),
        "score_threshold": args.score_threshold,
        "ground_truth_note": "Folder labels define expected hand sides/finger states; this is a proxy evaluation, not manual landmark ground truth.",
        "hand_stage": {
            "exact_hand_set_accuracy": hand_set_correct / max(total, 1),
            "expected_hand_images": dict(hand_expected_count),
            "detected_hand_images": dict(hand_detected_count),
        },
        "finger_stage": {
            finger: binary_metrics(**counts) for finger, counts in finger_counts.items()
        },
        "class_stage": {
            "exact_accuracy": exact_class_accuracy,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
            "unknown_predictions": sum(confusion[(label, 0)] for label in labels),
        },
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "images_per_second_mean": 1000 / max(statistics.mean(latencies), 1e-9),
        },
        "outputs": ["predictions.csv", "per_class.csv", "summary.json"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
