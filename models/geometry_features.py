"""Normalized geometric features used by deterministic finger rules."""

from __future__ import annotations

import numpy as np

CHAINS = {"T": [1, 2, 3, 4], "I": [5, 6, 7, 8], "M": [9, 10, 11, 12],
          "R": [13, 14, 15, 16], "P": [17, 18, 19, 20]}


def angle(points, a, b, c):
    first = points[a] - points[b]
    second = points[c] - points[b]
    cosine = np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def radial_reach(hand, chain):
    wrist = hand[0, :2]
    scale = max(float(np.linalg.norm(hand[0, :2] - hand[9, :2])), 1e-6)
    return float((np.linalg.norm(hand[chain[-1], :2] - wrist)
                  - np.linalg.norm(hand[chain[0], :2] - wrist)) / scale)


def distal_compactness(hand, chain):
    points = hand[list(chain), :2]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float((lengths[1] + lengths[2]) / max(2.0 * lengths[0], 1e-6))


def coordinate_features(hand):
    """Return normalized shape, reach, contact, and palm-distance features."""
    xy = hand[:, :2] - hand[0:1, :2]
    palm = hand[[0, 5, 9, 13, 17], :2].mean(axis=0)
    palm_scale = max(float(np.linalg.norm(hand[0, :2] - hand[9, :2])), 1e-6)
    result = {}
    for finger, chain in CHAINS.items():
        if finger == "T":
            palm_distance = float(np.linalg.norm(hand[4, :2] - palm) / palm_scale)
            bend = min(angle(xy, 1, 2, 3), angle(xy, 2, 3, 4)) / 180.0
            shape = palm_distance * bend
        else:
            points = hand[chain, :2]
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            straightness = float(np.linalg.norm(points[-1] - points[0]) / max(lengths.sum(), 1e-6))
            bend = min(angle(points, 0, 1, 2), angle(points, 1, 2, 3)) / 180.0
            shape = straightness * bend
            palm_distance = float(np.linalg.norm(hand[chain[-1], :2] - palm) / palm_scale)
        separation = (float(min(np.linalg.norm(hand[4, :2] - hand[i, :2])
                                for i in (8, 12, 16, 20)) / palm_scale)
                      if finger == "T" else 0.0)
        tip_palm = float(np.linalg.norm(hand[chain[-1], :2] - palm) / palm_scale)
        result[finger] = {
            "shape": shape,
            "distance": palm_distance,
            "reach": radial_reach(hand, chain),
            "compact": distal_compactness(hand, chain),
            "separation": separation,
            "tip_palm": tip_palm,
        }
    return result
