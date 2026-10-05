"""Display live webcam hand-pose classification."""

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

from hand_pose_onnx import process, annotate  # noqa: E402
from models import HandLandmark, PalmDetection  # noqa: E402


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = landmarks.astype(np.float32).copy()
    points -= points[0]
    scale = float(np.linalg.norm(points[:, :2], axis=1).max())
    if scale > 1e-6:
        points /= scale
    return points.reshape(-1)


def make_features(landmarks: np.ndarray, scores: np.ndarray, handedness: np.ndarray) -> np.ndarray:
    hands: dict[str, tuple[np.ndarray, float, float]] = {}
    order = np.argsort(landmarks[:, 0, 0]) if len(landmarks) else []
    for index in order:
        side = "left" if handedness[index] > 0.5 else "right"
        if side not in hands:
            hands[side] = (landmarks[index], float(scores[index]), float(handedness[index]))

    values: list[float] = [float(len(landmarks))]
    for side in ("left", "right"):
        if side in hands:
            points, score, hand_prob = hands[side]
            values.extend([score, hand_prob, *normalize_landmarks(points)])
        else:
            values.extend([0.0, 0.0, *([0.0] * (21 * 3))])
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "output" / "mlp.joblib")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--palm-model", type=Path, default=ROOT / "weights" / "palm_detection_full_Nx3x192x192.onnx")
    parser.add_argument("--hand-model", type=Path, default=ROOT / "weights" / "hand_landmark_full_Nx3x224x224.onnx")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    detector = PalmDetection(str(args.palm_model), providers=["CPUExecutionProvider"])
    landmarker = HandLandmark(str(args.hand_model), providers=["CPUExecutionProvider"])
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    print("Press q or ESC to quit")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        options = argparse.Namespace(det_threshold=args.det_threshold, threshold=args.threshold)
        landmarks, scores, handedness = process(frame, detector, landmarker, options)
        if len(landmarks):
            for points, score, hand_prob in zip(landmarks, scores, handedness, strict=True):
                annotate(frame, points, float(score), float(hand_prob))
            features = make_features(landmarks, scores, handedness)
            probabilities = model.predict_proba(features)[0]
            best = int(np.argmax(probabilities))
            label = model.classes_[best]
            confidence = float(probabilities[best])
            text = f"Class {label}  {confidence:.1%}"
        else:
            text = "No hand"
        cv2.rectangle(frame, (10, 10), (390, 58), (0, 0, 0), -1)
        cv2.putText(frame, text, (20, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Tamil Fingerspelling Assistant", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
