"""Frame-to-Tamil realtime pipeline, independent of desktop/web UI."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Hand

ROOT = Path(__file__).resolve().parents[1]
from models.finger_rules import classify  # noqa: E402
from models.geometry_features import coordinate_features  # noqa: E402
from models.rtmpose_utils import suppress_duplicate_hands  # noqa: E402
from models.tamil_labels import load_labels  # noqa: E402
from scripts.create_finger_mapping import FINGERS, mapping  # noqa: E402

EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))


def hand_side(hand: np.ndarray) -> str:
    first = hand[9, :2] - hand[0, :2]
    second = hand[17, :2] - hand[0, :2]
    cross = float(first[0] * second[1] - first[1] * second[0])
    return "right" if cross >= 0 else "left"


def tamil_class(states: dict[str, set[str]]) -> str | None:
    for label, (left, right) in mapping().items():
        expected = {
            "left": set(left.split(",")) - {""},
            "right": set(right.split(",")) - {""},
        }
        if states == expected:
            return str(label)
    return None


class RealtimePipeline:
    """Own model and per-user transcript/timer state."""

    def __init__(self, thresholds_path: Path | str | None = None,
                 labels_path: Path | str | None = None,
                 det_model: Path | str | None = None,
                 pose_model: Path | str | None = None,
                 device: str = "cpu", commit_ms: int = 350,
                 release_ms: int = 300):
        thresholds_path = Path(thresholds_path or ROOT / "data/coordinate_thresholds.json")
        labels_path = Path(labels_path or ROOT / "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/ReadMe.txt")
        self.thresholds = json.loads(thresholds_path.read_text())
        self.tamil_labels = load_labels(labels_path)
        self.commit_ms = commit_ms
        self.release_ms = release_ms
        self.detector = Hand(det=str(det_model) if det_model else None,
                             pose=str(pose_model) if pose_model else None,
                             backend="onnxruntime", device=device)
        self.transcript: list[str] = []
        self.candidate: str | None = None
        self.candidate_since = time.monotonic()
        self.release_since: float | None = None
        self.last_committed: str | None = None
        self.locked = False

    def _commit(self, token: str) -> None:
        self.transcript.append(" " if token == "SPACE" else self.tamil_labels.get(token, token))
        self.last_committed = token
        self.locked = True

    def command(self, name: str) -> None:
        if name in {"backspace", "undo"} and self.transcript:
            self.transcript.pop()
        elif name == "clear":
            self.transcript.clear()
        elif name == "space":
            self.transcript.append(" ")

    def process(self, frame: np.ndarray) -> dict:
        keypoints, scores = self.detector(frame)
        detections = [
            (hand.astype(np.float32), float(np.mean(score)))
            for hand, score in zip(keypoints, scores, strict=True)
            if float(np.mean(score)) >= .30
        ]
        detections = suppress_duplicate_hands(detections, .30)
        chosen: dict[str, tuple[np.ndarray, float]] = {}
        for hand, score in detections:
            side = hand_side(hand)
            if side not in chosen or score > chosen[side][1]:
                chosen[side] = (hand, score)

        states = {"left": set(), "right": set()}
        hands = []
        fist_detected = False
        height, width = frame.shape[:2]
        for side, (hand, score) in chosen.items():
            features = coordinate_features(hand)
            labels, gesture = classify(features, hand, side, self.thresholds)
            states[side] = {finger for finger, value in labels.items() if value}
            fist_detected = fist_detected or gesture == "fist_like"
            hands.append({
                "side": side,
                "score": round(score, 4),
                "gesture": gesture,
                "states": labels,
                "landmarks": [[round(float(point[0] / width), 6), round(float(point[1] / height), 6)] for point in hand],
            })

        now = time.monotonic()
        token = "SPACE" if fist_detected else (tamil_class(states) if chosen else None)
        if token != self.candidate:
            self.candidate = token
            self.candidate_since = now
            if token is not None and token != self.last_committed:
                self.locked = False
                self.release_since = None

        if token is None:
            if self.release_since is None:
                self.release_since = now
            elif self.locked and (now - self.release_since) * 1000 >= self.release_ms:
                self.locked = False
                self.last_committed = None
        else:
            self.release_since = None
            stable_ms = (now - self.candidate_since) * 1000
            if not self.locked and stable_ms >= self.commit_ms:
                self._commit(token)

        stable_ms = (now - self.candidate_since) * 1000 if token else 0.0
        preview = "SPACE" if token == "SPACE" else (self.tamil_labels.get(token, token) if token else None)
        status = "LOCKED" if self.locked else ("REGISTERING" if token else "UNLOCKED")
        return {
            "hands": hands,
            "preview": preview,
            "message": "".join(self.transcript),
            "status": status,
            "progress": round(min(stable_ms / self.commit_ms, 1.0) if token and not self.locked else 0.0, 3),
            "stable_ms": round(stable_ms),
            "commit_ms": self.commit_ms,
            "edges": EDGES,
        }
