"""Realtime RTMPose hand landmarks and deterministic finger ON/OFF display."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rtmlib import Hand

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.finger_rules import FINGERS, classify  # noqa: E402
from models.rtmpose_utils import suppress_duplicate_hands  # noqa: E402
from models.tamil_labels import load_labels, tamil_font  # noqa: E402
from scripts.create_finger_mapping import mapping  # noqa: E402
from models.geometry_features import coordinate_features  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
COLORS = {"T": (0, 165, 255), "I": (255, 100, 0), "M": (0, 220, 0),
          "R": (255, 0, 180), "P": (180, 0, 255)}
COMMIT_MS = 350
RELEASE_MS = 300


def side(hand: np.ndarray) -> str:
    first = hand[9, :2] - hand[0, :2]
    second = hand[17, :2] - hand[0, :2]
    # Explicit scalar keeps working with NumPy 2.x, where np.cross rejects 2D vectors.
    cross = float(first[0] * second[1] - first[1] * second[0])
    return "right" if cross >= 0 else "left"


def tamil_class(states: dict[str, set[str]]) -> str | None:
    """Return mapped class token, or None while pose is incomplete/unknown."""
    for label, (left, right) in mapping().items():
        expected = {
            "left": set(left.split(",")) - {""},
            "right": set(right.split(",")) - {""},
        }
        if states == expected:
            return str(label)
    return None


def wrap_unicode(draw, text: str, font, max_width: int) -> list[str]:
    lines, current = [], ""
    for char in text:
        trial = current + char
        if current and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current or not lines:
        lines.append(current)
    return lines


def draw_transcript(frame, transcript: list[str], candidate: str | None,
                    stable_ms: float, locked: bool, line_width: int,
                    commit_ms: int):
    """Draw large Tamil message, preview, progress, and controls."""
    height, width = frame.shape[:2]
    panel_top = max(0, height - 245)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, .86, frame, .14, 0, frame)
    preview = candidate or "—"
    progress = min(stable_ms / commit_ms, 1.0) if candidate and not locked else 0.0
    status = "LOCKED" if locked else ("REGISTERING" if candidate else "UNLOCKED")
    status_color = (0, 180, 0) if locked else ((0, 220, 255) if candidate else (180, 180, 180))
    cv2.putText(frame, status, (width - 170, panel_top + 28),
                cv2.FONT_HERSHEY_SIMPLEX, .55, status_color, 2)
    cv2.rectangle(frame, (16, panel_top + 48), (316, panel_top + 59), (70, 70, 70), -1)
    cv2.rectangle(frame, (16, panel_top + 48), (16 + int(300 * progress), panel_top + 59), (0, 220, 255), -1)
    if candidate and not locked:
        cv2.putText(frame, f"{int(stable_ms):03d}/{commit_ms} ms",
                    (326, panel_top + 59), cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1)
    cv2.putText(frame, "Space: space   B/U: undo   C: clear   Q: quit",
                (16, height - 9), cv2.FONT_HERSHEY_SIMPLEX, .48, (190, 190, 190), 1)

    # Pillow renders Tamil shaping/Unicode; cv2.putText does not.
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    message_font = tamil_font(43)
    preview_font = tamil_font(30)
    small_font = tamil_font(22)
    preview_text = f"Preview: {preview}"
    draw.text((16, panel_top + 3), preview_text, font=preview_font, fill=(0, 220, 255))
    draw.text(((width - (draw.textbbox((0, 0), "Message", font=small_font)[2])) // 2,
               panel_top + 3), "Message", font=small_font, fill=(180, 180, 180))
    message = "".join(transcript) if transcript else "(empty)"
    lines = wrap_unicode(draw, message, message_font, min(width - 32, line_width * 43))
    visible = max(1, (height - panel_top - 70) // 50)
    for row, line in enumerate(lines[-visible:]):
        box = draw.textbbox((0, 0), line, font=message_font)
        x = max(16, (width - (box[2] - box[0])) // 2)
        draw.text((x, panel_top + 70 + row * 50), line,
                  font=message_font, fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_hand(frame, hand: np.ndarray, hand_side: str, score: float,
              labels: dict[str, int], gesture: str):
    points = hand[:, :2].astype(int)
    for a, b in EDGES:
        cv2.line(frame, tuple(points[a]), tuple(points[b]), (190, 190, 190), 2)
    chains = {"T": (1, 2, 3, 4), "I": (5, 6, 7, 8), "M": (9, 10, 11, 12),
              "R": (13, 14, 15, 16), "P": (17, 18, 19, 20)}
    # Translucent state mask directly over finger pose: green ON, red OFF.
    mask = frame.copy()
    for finger, chain in chains.items():
        color = (55, 205, 75) if labels[finger] else (55, 55, 220)
        for a, b in zip(chain, chain[1:]):
            cv2.line(mask, tuple(points[a]), tuple(points[b]), color, 18, cv2.LINE_AA)
    cv2.addWeighted(mask, .42, frame, .58, 0, frame)
    for point in points:
        cv2.circle(frame, tuple(point), 4, (240, 240, 240), -1)
    x = max(10, int(points[:, 0].min()))
    y = max(28, int(points[:, 1].min()) - 10)
    cv2.putText(frame, f"{hand_side}  pose:{score:.2f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    cv2.putText(frame, gesture, (x, y + 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (220, 220, 220), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "data/coordinate_thresholds.json")
    parser.add_argument("--det-model", type=Path, default=None)
    parser.add_argument("--pose-model", type=Path, default=None)
    parser.add_argument("--commit-ms", type=int, default=COMMIT_MS)
    parser.add_argument("--release-ms", type=int, default=RELEASE_MS)
    parser.add_argument("--line-width", type=int, default=28)
    parser.add_argument("--labels", type=Path, default=ROOT / "data/TLFS23 - Tamil Language Finger Spelling Image Dataset/ReadMe.txt")
    args = parser.parse_args()

    thresholds = json.loads(args.thresholds.read_text())
    tamil_labels = load_labels(args.labels)
    detector = Hand(det=str(args.det_model) if args.det_model else None,
                    pose=str(args.pose_model) if args.pose_model else None,
                    backend="onnxruntime", device=args.device)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    print("Realtime running. Hold sign to commit; release hand before next sign.")

    transcript: list[str] = []
    candidate = None
    candidate_since = time.monotonic()
    release_since = None
    locked = False
    last_committed_token = None

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        keypoints, scores = detector(frame)
        detections = [(hand.astype(np.float32), float(np.mean(score)))
                      for hand, score in zip(keypoints, scores, strict=True)
                      if float(np.mean(score)) >= .30]
        detections = suppress_duplicate_hands(detections, .30)
        chosen = {}
        for hand, score in detections:
            hand_side = side(hand)
            if hand_side not in chosen or score > chosen[hand_side][1]:
                chosen[hand_side] = (hand, score)

        states = {"left": set(), "right": set()}
        fist_detected = False
        for hand_side, (hand, score) in chosen.items():
            features = coordinate_features(hand)
            labels, gesture = classify(features, hand, hand_side, thresholds)
            draw_hand(frame, hand, hand_side, score, labels, gesture)

            states[hand_side] = {finger for finger, value in labels.items() if value}
            fist_detected = fist_detected or gesture == "fist_like"

        now = time.monotonic()
        # Fist is reserved as explicit SPACE control gesture.
        token = "SPACE" if fist_detected else (tamil_class(states) if chosen else None)
        if token != candidate:
            candidate = token
            candidate_since = now
            # Different sign starts a fresh registration immediately.
            if token is not None and token != last_committed_token:
                locked = False
                release_since = None

        if token is None:
            if release_since is None:
                release_since = now
            elif locked and (now - release_since) * 1000 >= args.release_ms:
                locked = False
                last_committed_token = None
        else:
            release_since = None
            stable_ms = (now - candidate_since) * 1000
            if not locked and stable_ms >= args.commit_ms:
                transcript.append(" " if token == "SPACE" else tamil_labels.get(token, token))
                locked = True
                last_committed_token = token

        stable_ms = (now - candidate_since) * 1000 if token else 0.0
        display_candidate = "SPACE" if token == "SPACE" else (tamil_labels.get(token, token) if token else None)
        draw_transcript(frame, transcript, display_candidate, stable_ms, locked,
                        args.line_width, args.commit_ms)

        cv2.putText(frame, f"hands: {len(chosen)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        legend_x = max(12, frame.shape[1] - 270)
        cv2.putText(frame, "FINGER STATE", (legend_x, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.rectangle(frame, (legend_x, 36), (legend_x + 18, 54), (45, 190, 70), -1)
        cv2.putText(frame, "ON / open", (legend_x + 25, 51), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.rectangle(frame, (legend_x + 105, 36), (legend_x + 123, 54), (55, 55, 210), -1)
        cv2.putText(frame, "OFF / closed", (legend_x + 130, 51), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.imshow("Tamil fingerspelling - finger states", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == 32:  # keyboard space: temporary explicit space action
            transcript.append(" ")
        elif key in (ord("b"), 8):
            if transcript:
                transcript.pop()
        elif key == ord("u"):
            if transcript:
                transcript.pop()
        elif key == ord("c"):
            transcript.clear()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
