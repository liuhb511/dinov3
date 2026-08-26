"""
inclusion_v2/models/heads.py

Gate / Strip / Point / Boundary 四个轻量 head（全 1x1 卷积，开销极低）。
共享 decoder 特征后分叉，实现"一次 backbone、一次 decoder、多 head"。

类别数从 data.label_mapping 推导，保证"单一事实来源"：
修改类别体系只需改 label_mapping.py。
"""
import torch.nn as nn

from ..data.label_mapping import (
    NUM_GATE_CLASSES,
    NUM_STRIP_CLASSES,
    NUM_POINT_CLASSES,
)


class _Conv1x1Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        return self.head(x)


class GateHead(_Conv1x1Head):
    """像素级三分组：bg / strip / point。"""

    def __init__(self, in_channels):
        super().__init__(in_channels, num_classes=NUM_GATE_CLASSES)


class StripHead(_Conv1x1Head):
    """条状专家：bg + A/B/C/TINB-C/TIND/HH/XW/XQL。"""

    def __init__(self, in_channels):
        super().__init__(in_channels, num_classes=NUM_STRIP_CLASSES)


class PointHead(_Conv1x1Head):
    """点状专家：bg + D/HC/SZ。"""

    def __init__(self, in_channels):
        super().__init__(in_channels, num_classes=NUM_POINT_CLASSES)


class BoundaryHead(nn.Module):
    """轻量边界辅助监督头（训练辅助，推理可用可不用）。"""

    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, in_channels // 4),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, 1),
        )

    def forward(self, x):
        return self.head(x)
