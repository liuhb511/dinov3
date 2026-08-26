"""
inclusion_v2/models/fusion.py

轻量三层融合（替代旧 CBAM + 三层可学习权重融合）：
    1×1 投影到统一通道 → 逐元素求和 → 1×1/3×3 融合
不引入 attention，推理开销极低。
"""
import torch.nn as nn


class LightFusion(nn.Module):
    def __init__(self, feat_dim: int = 768, proj_dim: int = 512):
        super().__init__()
        self.proj1 = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.GroupNorm(8, proj_dim),
            nn.GELU(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.GroupNorm(8, proj_dim),
            nn.GELU(),
        )
        self.proj3 = nn.Sequential(
            nn.Conv2d(feat_dim, proj_dim, 1, bias=False),
            nn.GroupNorm(8, proj_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, proj_dim),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, proj_dim),
            nn.GELU(),
        )

    def forward(self, f1, f2, f3):
        x = self.proj1(f1) + self.proj2(f2) + self.proj3(f3)
        return self.fuse(x)
