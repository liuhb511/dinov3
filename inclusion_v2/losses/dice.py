"""
inclusion_v2/losses/dice.py

Soft Dice Loss，支持 ignore_index：
- ignore 区域在 one-hot 中置零，不参与分子分母（概率与 GT 均被 mask）。
- 无 GT 像素的类别（one-hot 求和为 0）不参与平均。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, smooth: float = 1.0,
                 include_background: bool = False):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        B, C, H, W = logits.shape
        target = target.long()
        valid = target != self.ignore_index

        tgt = target.clone()
        tgt[~valid] = 0

        probs = F.softmax(logits, dim=1)
        # ignore 区域不参与
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        onehot = F.one_hot(tgt, num_classes=C).permute(0, 3, 1, 2).float()
        onehot = onehot * valid_f

        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        denominator = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # 无 GT 的类别不参与平均
        has_gt = onehot.sum(dims) > 0
        if not self.include_background:
            dice = dice[1:]
            has_gt = has_gt[1:]

        if has_gt.any():
            return 1.0 - dice[has_gt].mean()
        return dice.sum() * 0.0
