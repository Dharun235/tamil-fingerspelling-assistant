# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

"""MediaPipe Hands ONNX Runtime inference (palm detection + 21-point hand landmarks).

Both models are the pure networks exported from this repo's PyTorch modules
(models/model.py, see onnx_export.py); all post-processing is done here in Python.
Two-stage pipeline, mirroring MediaPipe Hands:

1. `PalmDetection` (`palm_detection_full_Nx3x192x192.onnx`) — a BlazePalm
   detector returning raw SSD outputs (regressors + classificators); anchor
   decode and MediaPipe's WEIGHTED NMS are done here. Input: 192x192 RGB in
   [0, 1] (letterboxed). Output rows: `(score, cx, cy, w, wrist_x, wrist_y,
   middle_mcp_x, middle_mcp_y)` in image pixels.
2. `HandLandmark` (`hand_landmark_full_Nx3x224x224.onnx`) — takes a rotated
   square crop per palm (box scaled 2.6x, center shifted half a box toward the
   fingers, rotated so wrist -> middle-finger MCP points up: MediaPipe's
   `palm_detection_detection_to_roi` rule). Outputs 21 screen landmarks in
   crop pixels, a presence probability, a handedness probability, and 21
   world landmarks in meters — probabilities already sigmoided.

Landmarks are mapped back to full-image float pixels through the inverse crop
affine.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

NUM_LANDMARKS = 21
PALM_INPUT_SIZE = 192
HAND_INPUT_SIZE = 224

# MediaPipe palm_detection_detection_to_roi: scale 2.6, shift half a box toward
# the fingers, rotate wrist -> middle-finger MCP to point up.
ROI_SCALE = 2.6
ROI_SHIFT = 0.5

# The 21-point hand skeleton (MediaPipe HAND_CONNECTIONS).
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm edge
]


def _generate_palm_anchors() -> np.ndarray:
    """SSD anchor centers for palm_detection_full at 192x192.

    MediaPipe's SsdAnchorsCalculator config: strides [8, 16, 16, 16], aspect
    ratio 1.0 plus one interpolated scale per layer, fixed_anchor_size — so
    only the (x, y) centers matter. Consecutive layers with equal stride share
    one feature map, stacking their anchors per cell: 24x24x2 + 12x12x6 = 2016.

    Returns:
        Array of shape (2016, 2) — normalized anchor centers.
    """
    anchors = []
    strides = [8, 16, 16, 16]
    idx = 0
    while idx < len(strides):
        last = idx
        while last < len(strides) and strides[last] == strides[idx]:
            last += 1
        repeats = 2 * (last - idx)
        cells = PALM_INPUT_SIZE // strides[idx]
        for y in range(cells):
            for x in range(cells):
                anchors.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * repeats)
        idx = last
    return np.array(anchors)


def _weighted_nms(detections: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """MediaPipe's WEIGHTED non-max suppression.

    Instead of discarding overlapping candidates (hard NMS), all candidates
    overlapping the current top detection are averaged, weighted by score —
    box and keypoints alike. This blending is what the original pipeline does;
    hard NMS produces slightly different palm boxes/keypoints and therefore a
    slightly different landmark ROI.

    Args:
        detections: Rows of (score, cx, cy, w, h, kp0x, kp0y, kp2x, kp2y),
            normalized coordinates, sorted or unsorted.

    Returns:
        Blended detections, same row layout, highest score first.
    """
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1, ty1, tx2, ty2 = top[1] - top[3] / 2, top[2] - top[4] / 2, top[1] + top[3] / 2, top[2] + top[4] / 2
        inter = np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1)) * np.maximum(
            0, np.minimum(y2, ty2) - np.maximum(y1, ty1)
        )
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / np.maximum(union, 1e-9)

        overlapping = remaining[iou > iou_threshold]
        weights = overlapping[:, :1]
        blended = top.copy()
        blended[1:] = (overlapping[:, 1:] * weights).sum(axis=0) / weights.sum()
        output.append(blended)
        remaining = remaining[iou <= iou_threshold]
    return np.array(output)


def warp_roi(image: np.ndarray, roi: tuple[float, float, float, float], out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a rotated square ROI directly at the model input size.

    Single bilinear resampling straight to `out_size` — mirroring MediaPipe's
    `ImageToTensorCalculator`, which never materializes an intermediate crop.
    (A warp-then-resize implementation both interpolates twice and quantizes
    the crop side to whole pixels; measurably worse.) Out-of-image regions are
    zero-padded.

    Args:
        image: Full BGR image, shape (H, W, 3).
        roi: The (center_x, center_y, side, angle_degrees) ROI to cut.
        out_size: Model input side length (the output is out_size x out_size).

    Returns:
        A tuple of the (out_size, out_size, 3) crop and the inverse 2x3 affine
        mapping crop pixel coordinates back to full-image coordinates.
    """
    cx, cy, side, angle = roi
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, out_size / side)
    matrix[0, 2] += out_size / 2.0 - cx
    matrix[1, 2] += out_size / 2.0 - cy

    crop = cv2.warpAffine(image, matrix, (out_size, out_size))
    return crop, cv2.invertAffineTransform(matrix)


def palm_roi(palm: np.ndarray) -> tuple[float, float, float, float]:
    """Hand ROI for the landmark model from one palm row, per MediaPipe's rule.

    Args:
        palm: One detection row from `PalmDetection.detect` — (score, cx, cy,
            width, wrist_x, wrist_y, mcp_x, mcp_y) in image pixels.

    Returns:
        A square rotated ROI as (center_x, center_y, side, angle_degrees): the
        palm box scaled 2.6x, center shifted half a box toward the fingers,
        rotated so wrist -> middle-finger MCP points up.
    """
    _, cx, cy, width, wrist_x, wrist_y, mcp_x, mcp_y = (float(v) for v in palm[:8])
    direction = np.array([mcp_x - wrist_x, mcp_y - wrist_y])
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 0 else np.array([0.0, -1.0])
    center_x, center_y = cx + ROI_SHIFT * width * unit[0], cy + ROI_SHIFT * width * unit[1]
    angle = float(np.degrees(np.arctan2(unit[1], unit[0]))) + 90.0
    return center_x, center_y, ROI_SCALE * width, angle


class PalmDetection:
    """BlazePalm palm detector: raw SSD outputs decoded with MediaPipe's weighted NMS here.

    Source model: exported from this repo's PyTorch `PalmDetectionNet`
    (models/model.py), an architecture recovered from MediaPipe's palm_detection_full
    (https://github.com/google-ai-edge/mediapipe).
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Load the ONNX model.

        Args:
            model_path: Path to the ONNX file.
            providers: ONNX Runtime execution providers. Defaults to
                auto-detected (CUDA > CoreML > CPU).
        """
        if providers is None:
            available = ort.get_available_providers()
            preferred = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in preferred if p in available]

        options = ort.SessionOptions()
        options.log_severity_level = 3  # silence CoreML partition warnings
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.anchors = _generate_palm_anchors()

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop into a CHW float32 slice in [0, 1], RGB order."""
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def _decode_raw(self, regressors: np.ndarray, scores_raw: np.ndarray, threshold: float) -> np.ndarray:
        """MediaPipe's TensorsToDetections decode + WEIGHTED NMS.

        Returns rows of (score, cx, cy, w, h, kp0x, kp0y, kp2x, kp2y),
        normalized to the letterboxed frame.
        """
        scores = 1.0 / (1.0 + np.exp(-scores_raw.ravel().astype(np.float64)))
        keep = scores >= threshold
        if not keep.any():
            return np.empty((0, 9))

        reg, anchor = regressors[keep].astype(np.float64), self.anchors[keep]
        rows = np.empty((keep.sum(), 9))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = reg[:, 0:2] / PALM_INPUT_SIZE + anchor  # cx, cy
        rows[:, 3:5] = reg[:, 2:4] / PALM_INPUT_SIZE  # w, h
        rows[:, 5:7] = reg[:, 4:6] / PALM_INPUT_SIZE + anchor  # wrist center
        rows[:, 7:9] = reg[:, 8:10] / PALM_INPUT_SIZE + anchor  # middle-finger MCP (keypoint 2)
        return _weighted_nms(rows)

    def detect(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Detect palms in a full image.

        The image is letterboxed to 192x192 to preserve aspect ratio (the
        detector's square-box geometry breaks under non-uniform stretch).

        Args:
            image: Full BGR image, shape (H, W, 3).
            threshold: Minimum detection confidence.

        Returns:
            Array of shape (N, 8), float32 — one row per palm as (score, cx,
            cy, width, wrist_x, wrist_y, mcp_x, mcp_y) in full-image pixels;
            `width` is the palm box side (the detector emits square boxes) and
            wrist -> MCP is the hand's "up" direction.
        """
        # Fractional letterbox via a single affine, like MediaPipe's — integer
        # pad offsets would shift the whole frame by up to half a source pixel.
        h, w = image.shape[:2]
        scale = PALM_INPUT_SIZE / max(h, w)
        pad_x, pad_y = (PALM_INPUT_SIZE - w * scale) / 2.0, (PALM_INPUT_SIZE - h * scale) / 2.0
        matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]])
        canvas = cv2.warpAffine(image, matrix, (PALM_INPUT_SIZE, PALM_INPUT_SIZE))

        blob = self._preprocess(canvas)[np.newaxis]
        regressors, classificators = self.session.run(None, {self.input_name: blob})
        rows = self._decode_raw(regressors[0], classificators[0], threshold)

        if len(rows) == 0:
            return np.empty((0, 8), dtype=np.float32)

        def unletterbox(xy_normalized: np.ndarray) -> np.ndarray:
            return (xy_normalized * PALM_INPUT_SIZE - (pad_x, pad_y)) / scale

        palms = np.empty((len(rows), 8))
        palms[:, 0] = rows[:, 0]
        palms[:, 1:3] = unletterbox(rows[:, 1:3])
        palms[:, 3] = rows[:, 3:5].max(axis=1) * PALM_INPUT_SIZE / scale
        palms[:, 4:6] = unletterbox(rows[:, 5:7])
        palms[:, 6:8] = unletterbox(rows[:, 7:9])
        return palms.astype(np.float32)


class HandLandmark:
    """MediaPipe hand landmark model (21 3D keypoints + handedness).

    All hands of an image are batched into a single session run — the graph has
    a dynamic batch dimension.

    Source model: exported from this repo's PyTorch `HandLandmarkNet`
    (models/model.py), an architecture recovered from MediaPipe's hand_landmark_full
    (https://github.com/google-ai-edge/mediapipe).
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Load the ONNX model.

        Args:
            model_path: Path to the ONNX file.
            providers: ONNX Runtime execution providers. Defaults to
                auto-detected (CUDA > CoreML > CPU).
        """
        if providers is None:
            available = ort.get_available_providers()
            preferred = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in preferred if p in available]

        options = ort.SessionOptions()
        options.log_severity_level = 3  # silence CoreML partition warnings
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop into a CHW float32 slice in [0, 1], RGB order."""
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))

    def predict(self, image: np.ndarray, palms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict 21 landmarks for every detected palm of a full image.

        Args:
            image: Full BGR image, shape (H, W, 3).
            palms: Palm detections from `PalmDetection.detect`, shape (N, 8).

        Returns:
            A tuple of four arrays, float32: landmarks (N, 21, 3) — `x`/`y` in
            full-image pixels, `z` relative depth on the same scale; world
            landmarks (N, 21, 3) in meters, origin at the hand center; presence
            scores (N,); and handedness (N,) — the probability MediaPipe would
            label the hand "Left" (both probabilities already sigmoided
            in-graph).
        """
        if len(palms) == 0:
            empty = np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)
            return empty, empty.copy(), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        blobs, inverses, z_scales = [], [], []
        for palm in palms:
            roi = palm_roi(palm)
            crop, inverse = warp_roi(image, roi, HAND_INPUT_SIZE)
            blobs.append(self._preprocess(crop))
            inverses.append(inverse)
            z_scales.append(roi[2] / HAND_INPUT_SIZE)

        screen, presence, handedness, world = self.session.run(None, {self.input_name: np.stack(blobs)})

        # x/y: 224-crop pixels -> image pixels via the inverse warp affine;
        # z: scaled by the same crop-to-image factor.
        landmarks = screen.reshape(-1, NUM_LANDMARKS, 3).astype(np.float64)
        for idx, inverse in enumerate(inverses):
            landmarks[idx, :, :2] = landmarks[idx, :, :2] @ inverse[:, :2].T + inverse[:, 2]
            landmarks[idx, :, 2] *= z_scales[idx]
        return (
            landmarks.astype(np.float32),
            world.reshape(-1, NUM_LANDMARKS, 3).astype(np.float32),
            presence.ravel().astype(np.float32),
            handedness.ravel().astype(np.float32),
        )
