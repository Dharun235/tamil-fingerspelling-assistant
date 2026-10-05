# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import argparse
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import HAND_CONNECTIONS, HandLandmark, PalmDetection

BONE_COLOR, POINT_COLOR, TEXT_COLOR = (110, 218, 130), (0, 209, 255), (255, 255, 255)  # BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MediaPipe Hands ONNX inference - palm detection + 21 hand landmarks for images or a webcam',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'sources',
        type=str,
        nargs='+',
        help='Image paths and/or webcam indices (e.g. 0 opens the default camera)',
    )
    parser.add_argument(
        '--palm-model',
        type=str,
        default='weights/palm_detection_full_Nx3x192x192.onnx',
        help='Path to the palm detection ONNX model',
    )
    parser.add_argument(
        '--hand-model',
        type=str,
        default='weights/hand_landmark_full_Nx3x224x224.onnx',
        help='Path to the hand landmark ONNX model',
    )
    parser.add_argument(
        '--det-threshold',
        type=float,
        default=0.5,
        help='Palm detection confidence threshold',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Hand-presence probability below which a hand is discarded',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='If set, save annotated images to this directory',
    )
    return parser.parse_args()


def annotate(image: np.ndarray, landmarks: np.ndarray, score: float, handedness: float) -> None:
    """Draw one hand's skeleton, joints, and handedness label onto `image`."""
    points = np.rint(landmarks[:, :2]).astype(np.int32)
    hand_size = max(points[:, 1].max() - points[:, 1].min(), points[:, 0].max() - points[:, 0].min())
    thickness = max(1, round(hand_size / 150))
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(image, points[start_idx], points[end_idx], BONE_COLOR, thickness, cv2.LINE_AA)
    for x, y in points:
        cv2.circle(image, (x, y), thickness + 1, POINT_COLOR, -1, cv2.LINE_AA)

    label = f'{"Left" if handedness > 0.5 else "Right"} {score:.2f}'
    anchor = (int(points[:, 0].min()), max(20, int(points[:, 1].min()) - 8))
    cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)


def process(image: np.ndarray, detector: PalmDetection, landmarker: HandLandmark, args: argparse.Namespace):
    palms = detector.detect(image, threshold=args.det_threshold)
    landmarks, _world, scores, handedness = landmarker.predict(image, palms)
    kept = scores >= args.threshold
    return landmarks[kept], scores[kept], handedness[kept]


def run_image(image_path: str, detector: PalmDetection, landmarker: HandLandmark, args: argparse.Namespace) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f'{image_path}: READ_ERR')
        return

    landmarks, scores, handedness = process(image, detector, landmarker, args)
    if len(landmarks) == 0:
        print(f'{image_path}: NO_HAND')
        return

    for idx, (hand_landmarks, score, handed) in enumerate(zip(landmarks, scores, handedness, strict=True), start=1):
        print(f'{image_path} [hand {idx}]: score={score:.4f}, handedness={handed:.4f}')
        if args.save_dir:
            annotate(image, hand_landmarks, score, handed)

    if args.save_dir:
        output_path = os.path.join(args.save_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, image)
        print(f'Saved {output_path}')


def run_webcam(camera_index: int, detector: PalmDetection, landmarker: HandLandmark, args: argparse.Namespace) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f'webcam {camera_index}: OPEN_ERR')
        return

    window = f'Hand Landmark - webcam {camera_index}'
    controls = 'q/ESC to quit' + (', s to save a snapshot' if args.save_dir else '')
    print(f'webcam {camera_index}: {controls}')

    while True:
        ok, frame = capture.read()
        if not ok:
            print(f'webcam {camera_index}: READ_ERR')
            break

        landmarks, scores, handedness = process(frame, detector, landmarker, args)
        for hand_landmarks, score, handed in zip(landmarks, scores, handedness, strict=True):
            annotate(frame, hand_landmarks, score, handed)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord('s') and args.save_dir:
            output_path = os.path.join(args.save_dir, f'webcam{camera_index}_{time.strftime("%Y%m%d-%H%M%S")}.jpg')
            cv2.imwrite(output_path, frame)
            print(f'Saved {output_path}')

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    # CPU keeps behavior consistent across macOS, Linux, and headless runs.
    providers = ['CPUExecutionProvider']
    detector = PalmDetection(args.palm_model, providers=providers)
    landmarker = HandLandmark(args.hand_model, providers=providers)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for source in args.sources:
        if source.isdigit():
            run_webcam(int(source), detector, landmarker, args)
        else:
            run_image(source, detector, landmarker, args)


if __name__ == '__main__':
    main()
