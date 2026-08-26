"""
inclusion_v2/models/shared_decoder.py

共享 4 级 Residual Decoder（512 → 256 → 128 → 64 → 32）。
输入为 LightFusion 输出（patch 分辨率），输出为 2^4 倍分辨率特征图。
"""
import torch.nn as nn

from .residual_block import UpResidualBlock


class SharedDecoder(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 32):
        super().__init__()
        self.dec1 = UpResidualBlock(in_channels, 256)
        self.dec2 = UpResidualBlock(256, 128)
        self.dec3 = UpResidualBlock(128, 64)
        self.dec4 = UpResidualBlock(64, out_channels)

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x
