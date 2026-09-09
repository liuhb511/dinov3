"""Lightweight high-resolution local CNN branch."""

import torch.nn as nn


def _gn(channels: int) -> nn.GroupNorm:
    # Stable for small batches. Prefer 8/4/2 groups when divisible.
    for groups in (8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            _gn(out_ch),
            nn.GELU(),
        )


class DepthwiseSeparableDown(nn.Module):
    """Depthwise 3x3 stride-2 downsample + pointwise channel projection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = ConvGNAct(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=2,
            groups=in_ch,
        )
        self.pw = ConvGNAct(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class LightweightLocalEncoderMD(nn.Module):
    """
    For a 1024x1024 input:
        stem   : 16 x 512 x 512  (intermediate only, not fused)
        cnn_f4 : 32 x 256 x 256  (true 1/4 spatial feature)
        cnn_f8 : 64 x 128 x 128  (true 1/8 spatial feature)
    """

    def __init__(
        self,
        stem_channels=16,
        f4_channels=32,
        f8_channels=64,
    ):
        super().__init__()

        self.stem = ConvGNAct(
            3,
            stem_channels,
            kernel_size=3,
            stride=2,
        )
        self.stage4 = DepthwiseSeparableDown(
            stem_channels,
            f4_channels,
        )
        self.stage8 = DepthwiseSeparableDown(
            f4_channels,
            f8_channels,
        )

    def forward(self, x):
        x2 = self.stem(x)
        cnn_f4 = self.stage4(x2)
        cnn_f8 = self.stage8(cnn_f4)
        return {
            "cnn_f4": cnn_f4,
            "cnn_f8": cnn_f8,
        }
