"""Save misclassified held-out images with hand landmarks and labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hand_pose_onnx import annotate, process  # noqa: E402
from models import HandLandmark, PalmDetection  # noqa: E402
from realtime_predict import make_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "output" / "mlp.joblib")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "splits" / "test.npz")
    parser.add_argument("--test-paths", type=Path, default=ROOT / "data" / "splits" / "test_paths.npy")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "TLFS23 - Tamil Language Finger Spelling Image Dataset" / "Dataset Folders")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "cases")
    parser.add_argument("--limit", type=int, default=100, help="Maximum wrong examples")
    parser.add_argument("--correct-limit", type=int, default=100, help="Maximum correct examples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = np.load(args.test, allow_pickle=False)
    paths = np.load(args.test_paths, allow_pickle=False)
    model = joblib.load(args.model)
    predictions = model.predict(split["x"])
    probabilities = model.predict_proba(split["x"])
    wrong = np.flatnonzero(predictions != split["y"])
    wrong = wrong[np.argsort(-probabilities[wrong].max(axis=1))][: args.limit]
    correct = np.flatnonzero(predictions == split["y"])
    correct = correct[: args.correct_limit]

    # Prefer a correctly classified test image as the visual example for each class.
    example_paths: dict[str, str] = {}
    for index in np.flatnonzero(predictions == split["y"]):
        example_paths.setdefault(str(split["y"][index]), str(paths[index]))
    for index in range(len(paths)):
        example_paths.setdefault(str(split["y"][index]), str(paths[index]))

    detector = PalmDetection(str(ROOT / "weights" / "palm_detection_full_Nx3x192x192.onnx"), providers=["CPUExecutionProvider"])
    landmarker = HandLandmark(str(ROOT / "weights" / "hand_landmark_full_Nx3x224x224.onnx"), providers=["CPUExecutionProvider"])
    options = argparse.Namespace(det_threshold=0.5, threshold=0.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in np.concatenate((correct, wrong)):
        image_path = args.dataset_root / str(paths[index])
        image = cv2.imread(str(image_path))
        original = image.copy()
        landmarks, scores, handedness = process(image, detector, landmarker, options)
        for points, score, hand_prob in zip(landmarks, scores, handedness, strict=True):
            annotate(image, points, float(score), float(hand_prob))
        confidence = float(probabilities[index].max())
        actual, predicted = str(split["y"][index]), str(predictions[index])
        text = f"actual {actual} | predicted {predicted} | {confidence:.1%}"
        cv2.rectangle(image, (8, 8), (630, 55), (0, 0, 0), -1)
        cv2.putText(image, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        example_path = args.dataset_root / example_paths.get(predicted, str(paths[index]))
        example = cv2.imread(str(example_path))
        example_landmarks, example_scores, example_handedness = process(example, detector, landmarker, options)
        for points, score, hand_prob in zip(example_landmarks, example_scores, example_handedness, strict=True):
            annotate(example, points, float(score), float(hand_prob))
        example_text = f"predicted class {predicted} example"
        cv2.rectangle(example, (8, 8), (440, 55), (0, 0, 0), -1)
        cv2.putText(example, example_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        is_correct = actual == predicted
        group = "correct" if is_correct else "wrong"
        if is_correct:
            folder = args.output_dir / group / f"class_{actual}"
        else:
            folder = args.output_dir / group / f"actual_{actual}_pred_{predicted}"
        folder.mkdir(parents=True, exist_ok=True)
        case_dir = folder / f"{len(rows):04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        original_output = case_dir / "original.jpg"
        prediction_output = case_dir / "prediction.jpg"
        example_output = case_dir / "predicted_class_example.jpg"
        comparison_output = case_dir / "comparison.jpg"
        cv2.imwrite(str(original_output), original)
        cv2.imwrite(str(prediction_output), image)
        cv2.imwrite(str(example_output), example)
        original_labeled = original.copy()
        original_text = f"actual class {actual}"
        cv2.rectangle(original_labeled, (8, 8), (330, 55), (0, 0, 0), -1)
        cv2.putText(original_labeled, original_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        if original_labeled.shape[0] != example.shape[0]:
            example = cv2.resize(example, (example.shape[1], original_labeled.shape[0]))
        comparison = np.concatenate((original_labeled, example), axis=1)
        cv2.imwrite(str(comparison_output), comparison)
        rows.append((group, actual, predicted, confidence, str(paths[index]), str(original_output), str(prediction_output), str(example_output), str(comparison_output)))
    with (args.output_dir / "index.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "actual", "predicted", "confidence", "image_path", "original", "prediction", "predicted_class_example", "comparison"])
        writer.writerows(rows)
    print(f"wrong={len(wrong)} correct={len(correct)} output={args.output_dir}")


if __name__ == "__main__":
    main()
