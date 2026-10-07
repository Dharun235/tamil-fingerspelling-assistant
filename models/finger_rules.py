"""Shared deterministic finger ON/OFF rules."""

from __future__ import annotations

import numpy as np

FINGERS = ("T", "I", "M", "R", "P")
NON_THUMB = ("I", "M", "R", "P")
PALM_NEAR_THRESHOLD = 0.70
DISTAL_COMPACT_THRESHOLD = 0.65
THUMB_CONTACT_THRESHOLD = 0.30
FIST_REACH_THRESHOLD = 0.25
FIST_COMPACT_THRESHOLD = 0.55


def classify(features, hand: np.ndarray, hand_side: str, thresholds: dict[str, float]):
    """Classify five fingers from normalized geometric features."""
    labels = {
        finger: int(features[finger]["shape"] >= thresholds[f"{hand_side}_{finger}_shape"]
                    and ((features[finger]["reach"] >= 0 and features[finger]["separation"] >= 0.5)
                         if finger == "T" else features[finger]["reach"] >= 0))
        for finger in FINGERS
    }
    fist = [f for f in NON_THUMB if features[f]["reach"] < FIST_REACH_THRESHOLD
            and features[f]["compact"] < FIST_COMPACT_THRESHOLD]
    gesture = "fist_like" if len(fist) >= 3 else "normal"
    if gesture == "fist_like":
        for finger in NON_THUMB:
            labels[finger] = 0
    scale = max(float(np.linalg.norm(hand[0, :2] - hand[9, :2])), 1e-6)
    for finger, tip in {"I": 8, "M": 12, "R": 16, "P": 20}.items():
        if np.linalg.norm(hand[4, :2] - hand[tip, :2]) / scale < THUMB_CONTACT_THRESHOLD:
            labels[finger] = 0
        if (features[finger]["tip_palm"] < PALM_NEAR_THRESHOLD
                and features[finger]["compact"] < DISTAL_COMPACT_THRESHOLD):
            labels[finger] = 0
    return labels, gesture
