"""Small RTMPose post-processing helpers."""

from __future__ import annotations

import numpy as np


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    def box(points: np.ndarray) -> np.ndarray:
        xy = points[:, :2]
        return np.r_[xy.min(axis=0), xy.max(axis=0)]

    a, b = box(first), box(second)
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda value: max(0.0, value[2] - value[0]) * max(0.0, value[3] - value[1])
    return float(intersection / max(area(a) + area(b) - intersection, 1e-9))


def suppress_duplicate_hands(detections: list[tuple[np.ndarray, float]], iou_threshold: float = 0.3) -> list[tuple[np.ndarray, float]]:
    """Keep the highest-confidence detection when two hand boxes overlap heavily."""
    kept: list[tuple[np.ndarray, float]] = []
    for detection in sorted(detections, key=lambda item: item[1], reverse=True):
        if all(bbox_iou(detection[0], previous[0]) < iou_threshold for previous in kept):
            kept.append(detection)
    return kept
