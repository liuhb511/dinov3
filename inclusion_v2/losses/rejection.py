"""
inclusion_v2/losses/rejection.py

单向 rejection penalty（贴合业务错误方向，不做类别加权）：
- StripRejectionLoss: 只在 GT=HH 区域额外压制 P(A)+P(C)（划痕→A/C 是明显单向错误）
- PointRejectionLoss: 只在 GT∈{HC,SZ} 区域额外压制 P(D)（灰尘/水渍→D 单向错误）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StripRejectionLoss(nn.Module):
    """GT=HH(7) 区域，压制 Strip Head 中 P(A)+P(C)。A->head1, C->head3。"""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, strip_logits, unified_mask):
        """
        strip_logits: [B, 9, H, W]
        unified_mask: [B, H, W]（12 类编码，HH=7）
        """
        region = (unified_mask == 7)
        if not region.any():
            return strip_logits.sum() * 0.0
        probs = F.softmax(strip_logits, dim=1)
        pa = probs[:, 1]   # A
        pc = probs[:, 3]   # C
        return (pa + pc)[region].mean()


class PointRejectionLoss(nn.Module):
    """GT∈{HC(10), SZ(11)} 区域，压制 Point Head 中 P(D)。D->head1。"""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, point_logits, unified_mask):
        region = (unified_mask == 10) | (unified_mask == 11)
        if not region.any():
            return point_logits.sum() * 0.0
        probs = F.softmax(point_logits, dim=1)
        pd = probs[:, 1]  # D
        return pd[region].mean()
