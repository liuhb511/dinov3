"""
inclusion_v2/losses/boundary.py

Boundary Head 辅助损失：用 Laplacian 从 unified mask（12 类）提取边缘，
与 Boundary Head 输出做 BCEWithLogitsLoss（低权重训练辅助，缓解面积偏大）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]]],
            dtype=torch.float32,
        )
        self.register_buffer("laplacian", kernel.unsqueeze(0))

    def forward(self, boundary_logits, unified_mask):
        """
        boundary_logits: [B, 1, H, W]
        unified_mask:    [B, H, W] 或 [B, 1, H, W]
        内部强制 fp32（对 AMP 鲁棒）。
        """
        if unified_mask.dim() == 3:
            unified_mask = unified_mask.unsqueeze(1)
        gt = unified_mask.to(dtype=torch.float32)
        kern = self.laplacian.to(device=gt.device, dtype=torch.float32)

        gt_edge = F.conv2d(gt, kern, padding=1).abs()
        gt_edge = (gt_edge > 0).to(dtype=torch.float32)

        if boundary_logits.shape[2:] != gt_edge.shape[2:]:
            gt_edge = F.interpolate(
                gt_edge, size=boundary_logits.shape[2:], mode="nearest",
            )

        return F.binary_cross_entropy_with_logits(
            boundary_logits.to(dtype=torch.float32), gt_edge,
        )
