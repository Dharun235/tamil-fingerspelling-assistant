"""Save misclassified held-out images with hand landmarks and labels."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "failures")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = np.load(args.test, allow_pickle=False)
    paths = np.load(args.test_paths, allow_pickle=False)
    model = joblib.load(args.model)
    predictions = model.predict(split["x"])
    probabilities = model.predict_proba(split["x"])
    failed = np.flatnonzero(predictions != split["y"])
    failed = failed[np.argsort(-probabilities[failed].max(axis=1))][: args.limit]

    detector = PalmDetection(str(ROOT / "weights" / "palm_detection_full_Nx3x192x192.onnx"), providers=["CPUExecutionProvider"])
    landmarker = HandLandmark(str(ROOT / "weights" / "hand_landmark_full_Nx3x224x224.onnx"), providers=["CPUExecutionProvider"])
    options = argparse.Namespace(det_threshold=0.5, threshold=0.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in failed:
        image_path = args.dataset_root / str(paths[index])
        image = cv2.imread(str(image_path))
        landmarks, scores, handedness = process(image, detector, landmarker, options)
        for points, score, hand_prob in zip(landmarks, scores, handedness, strict=True):
            annotate(image, points, float(score), float(hand_prob))
        confidence = float(probabilities[index].max())
        actual, predicted = str(split["y"][index]), str(predictions[index])
        text = f"actual {actual} | predicted {predicted} | {confidence:.1%}"
        cv2.rectangle(image, (8, 8), (630, 55), (0, 0, 0), -1)
        cv2.putText(image, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        output = args.output_dir / f"{len(rows):04d}_actual-{actual}_pred-{predicted}.jpg"
        cv2.imwrite(str(output), image)
        rows.append((actual, predicted, confidence, str(paths[index]), str(output)))
    np.savetxt(args.output_dir / "index.csv", np.asarray(rows, dtype=str), fmt="%s", delimiter=",", header="actual,predicted,confidence,image_path,output", comments="")
    print(f"failures={len(failed)} saved={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
