# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ['HandLandmarkNet', 'PalmDetectionNet']

# BatchNorm does not appear here because the released weights have it folded
# into the convolutions — verified: Google's own .tflite files contain no
# normalization ops (TFLite always folds BN on export), which is why the convs
# carry bias. The folding cannot be inverted, so the original BN statistics are
# unrecoverable. Structures were recovered from this repo's verified ONNX
# graphs, and the shipped weights were mapped from them and checked for parity
# against ONNX Runtime.


class PalmBlock(nn.Module):
    """Residual depthwise-separable block of BlazePalm: DW 5x5 -> PW 1x1, skip, PReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels),
            # pw-linear
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(x))


class PalmDown(nn.Module):
    """Stride-2 transition of BlazePalm: DW 5x5 s2 -> PW 1x1, MaxPool skip, PReLU.

    The depthwise uses TF-style 'same' padding; the skip is zero-padded along
    channels when the block widens.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            # dw
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=2, groups=in_channels),
            # pw-linear
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.extra_channels = out_channels - in_channels
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.pool(x)
        if self.extra_channels > 0:
            skip = F.pad(skip, (0, 0, 0, 0, 0, self.extra_channels))
        return self.act(self.conv(x) + skip)


class PalmDetectionNet(nn.Module):
    """BlazePalm full: SSD palm detector on a 192x192 letterboxed image.

    Input is (N, 3, 192, 192) RGB in [0, 1]. Returns raw box/keypoint
    regressors (N, 2016, 18) and score logits (N, 2016, 1) over two anchor
    grids — 24x24 with 2 anchors per cell first, then 12x12 with 6 — matching
    the anchor order of `models/onnx_model.py`, where decode and weighted NMS
    live. The backbone runs to 6x6 and an FPN decoder (bilinear x2 -> PW ->
    PReLU -> add skip -> 2 blocks) restores the 12x12 and 24x24 maps.

    Example:
        >>> model = PalmDetectionNet()
        >>> model.load_state_dict(torch.load('weights/palm_detection_full.pt'))
        >>> regressors, logits = model(x)

    Reference:
        https://arxiv.org/abs/2006.10214
        https://github.com/google-ai-edge/mediapipe
    """

    def __init__(self) -> None:
        super().__init__()

        feature_setting = [
            # c, n — channels and number of residual blocks per stage
            [32, 4],
            [64, 4],
            [128, 4],
            [256, 4],
            [256, 4],
        ]

        stages: list[nn.Module] = []
        in_channels = 32
        for stage, (channels, num_blocks) in enumerate(feature_setting):
            layers: list[nn.Module] = []
            if stage == 0:
                layers.extend(
                    [
                        nn.ZeroPad2d((1, 2, 1, 2)),
                        nn.Conv2d(3, 32, kernel_size=5, stride=2),
                        nn.PReLU(32),
                    ]
                )
            else:
                layers.append(PalmDown(in_channels, channels))
            layers.extend(PalmBlock(channels) for _ in range(num_blocks))
            stages.append(nn.Sequential(*layers))
            in_channels = channels
        # features_24 / features_12 / features_6 are the FPN skip levels.
        self.features_24 = nn.Sequential(*stages[:3])
        self.features_12 = stages[3]
        self.features_6 = stages[4]

        self.up_12 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.PReLU(256))
        self.fpn_12 = nn.Sequential(PalmBlock(256), PalmBlock(256))
        self.cls_12 = nn.Conv2d(256, 6 * 1, kernel_size=1)
        self.reg_12 = nn.Conv2d(256, 6 * 18, kernel_size=1)
        self.up_24 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.PReLU(128))
        self.fpn_24 = nn.Sequential(PalmBlock(128), PalmBlock(128))
        self.cls_24 = nn.Conv2d(128, 2 * 1, kernel_size=1)
        self.reg_24 = nn.Conv2d(128, 2 * 18, kernel_size=1)

    @staticmethod
    def _flatten(x: Tensor, num_coords: int) -> Tensor:
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, num_coords)

    @staticmethod
    def _upsample(x: Tensor) -> Tensor:
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x24 = self.features_24(x)
        x12 = self.features_12(x24)
        x6 = self.features_6(x12)

        p12 = self.fpn_12(self.up_12(self._upsample(x6)) + x12)
        p24 = self.fpn_24(self.up_24(self._upsample(p12)) + x24)

        regressors = torch.cat([self._flatten(self.reg_24(p24), 18), self._flatten(self.reg_12(p12), 18)], dim=1)
        logits = torch.cat([self._flatten(self.cls_24(p24), 1), self._flatten(self.cls_12(p12), 1)], dim=1)
        return regressors, logits


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual with ReLU6 and linear projection.

    Stride-2 depthwise convolutions use TF-style 'same' padding (asymmetric),
    matching the original TFLite graph.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')

        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.extend([nn.Conv2d(in_channels, hidden_dim, kernel_size=1), nn.ReLU6(inplace=True)])
        if stride == 2:
            layers.append(nn.ZeroPad2d((0, 1, 0, 1) if kernel_size == 3 else (1, 2, 1, 2)))
        layers.extend(
            [
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2 if stride == 1 else 0,
                    groups=hidden_dim,
                ),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class HandLandmarkNet(nn.Module):
    """MediaPipe hand landmark model: 21 3D keypoints + handedness from 224x224.

    Input is (N, 3, 224, 224) RGB in [0, 1]. Returns screen landmarks (N, 63)
    in 224-crop pixels, presence and handedness probabilities (N, 1) — sigmoid
    applied, as in the original graph — and world landmarks (N, 63) in meters.

    Example:
        >>> model = HandLandmarkNet()
        >>> model.load_state_dict(torch.load('weights/hand_landmark_full.pt'))
        >>> screen, presence, handedness, world = model(x)

    Reference:
        https://arxiv.org/abs/2006.10214
        https://github.com/google-ai-edge/mediapipe
    """

    def __init__(self) -> None:
        super().__init__()

        inverted_residual_setting = [
            # t, c, n, s, k
            [1, 16, 1, 1, 3],
            [4, 24, 1, 2, 3],
            [6, 24, 1, 1, 3],
            [6, 40, 1, 2, 5],
            [6, 40, 1, 1, 5],
            [6, 80, 1, 2, 3],
            [6, 80, 2, 1, 3],
            [6, 112, 1, 1, 5],
            [6, 112, 2, 1, 5],
            [6, 192, 1, 2, 5],
            [6, 192, 3, 1, 5],
        ]

        features: list[nn.Module] = [
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(3, 24, kernel_size=3, stride=2),
            nn.ReLU6(inplace=True),
        ]
        in_channels = 24
        for t, c, n, s, k in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, k, stride, expand_ratio=t))
                in_channels = c
        # building last several layers
        features.extend(
            [
                nn.Conv2d(192, 1152, kernel_size=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(1152, 1152, kernel_size=3, padding=1, groups=1152),
                nn.ReLU6(inplace=True),
            ]
        )
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.screen_head = nn.Linear(1152, 63)
        self.presence_head = nn.Linear(1152, 1)
        self.handedness_head = nn.Linear(1152, 1)
        self.world_head = nn.Linear(1152, 63)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        screen = self.screen_head(x)
        presence = torch.sigmoid(self.presence_head(x))
        handedness = torch.sigmoid(self.handedness_head(x))
        world = self.world_head(x)
        return screen, presence, handedness, world
