import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_target(target):
    """
    将 target 统一转换为 [B, H, W]。
    """

    if target.dim() == 4:
        if target.shape[1] != 1:
            raise ValueError(
                f"Target 通道数必须为1，实际 shape={target.shape}"
            )
        target = target.squeeze(1)

    if target.dim() != 3:
        raise ValueError(
            f"Target 应为 [B,H,W] 或 [B,1,H,W]，实际 shape={target.shape}"
        )

    return target.long()


class DiceLoss(nn.Module):
    """
    多类别 Soft Dice Loss。
    """

    def __init__(self, smooth=1.0, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W] 或 [B, 1, H, W]
        """

        target = normalize_target(target)
        num_classes = logits.shape[1]

        target_min = int(target.min().item())
        target_max = int(target.max().item())

        if target_min < 0 or target_max >= num_classes:
            unique_values = torch.unique(target).detach().cpu().tolist()

            raise ValueError(
                f"GT 类别索引越界："
                f"min={target_min}，"
                f"max={target_max}，"
                f"unique={unique_values}，"
                f"num_classes={num_classes}"
            )

        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(
                logits,
                size=target.shape[1:],
                mode="bilinear",
                align_corners=False,
            )

        probabilities = F.softmax(logits, dim=1)

        target_onehot = F.one_hot(
            target,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).to(dtype=probabilities.dtype)

        dims = (0, 2, 3)

        intersection = torch.sum(
            probabilities * target_onehot,
            dim=dims,
        )

        denominator = (
            torch.sum(probabilities, dim=dims)
            + torch.sum(target_onehot, dim=dims)
        )

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )

        if not self.include_background:
            dice = dice[1:]

        return 1.0 - dice.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Head 辅助损失。

    用 Laplacian 从 GT mask 提取边缘 →
    与 Boundary Head 输出做 BCEWithLogitsLoss。
    """

    def __init__(self):
        super().__init__()

        kernel = torch.tensor(
            [[[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]]],
            dtype=torch.float32
        )
        self.register_buffer("laplacian", kernel.unsqueeze(0))

    def forward(self, boundary_logits, target):
        """
        boundary_logits: [B, 1, H, W] — Boundary Head 输出
        target:          [B, H, W] 或 [B, 1, H, W] — GT mask
        """
        target = normalize_target(target)
        gt = target.unsqueeze(1).to(dtype=torch.float32)

        # 用 Laplacian 提取 GT 边缘
        gt_edge = F.conv2d(
            gt,
            self.laplacian.to(device=gt.device),
            padding=1,
        ).abs()
        gt_edge = (gt_edge > 0).to(dtype=torch.float32)

        # 对齐尺寸
        if boundary_logits.shape[2:] != gt_edge.shape[2:]:
            gt_edge = F.interpolate(
                gt_edge,
                size=boundary_logits.shape[2:],
                mode="nearest",
            )

        return F.binary_cross_entropy_with_logits(
            boundary_logits.to(dtype=torch.float32),
            gt_edge,
        )


class TotalLoss(nn.Module):
    """
    总损失：

    TotalLoss =
        ce_weight × CE
        + dice_weight × Dice
        + boundary_weight × BoundaryLoss（boundary_weight > 0 时启用）
    """

    def __init__(self, cfg):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = DiceLoss(
            smooth=1.0,
            include_background=False,
        )

        self.boundary = BoundaryLoss()

        self.ce_weight = getattr(
            cfg,
            "ce_weight",
            getattr(cfg, "bce_weight", 0.5),
        )

        self.dice_weight = cfg.dice_weight
        self.boundary_weight = cfg.boundary_weight

    def forward(self, seg, boundary, target):
        """
        seg:      [B, num_classes, H, W]
        boundary: [B, 1, H, W] — Boundary Head 输出
        target:   [B, H, W] 或 [B, 1, H, W]
        """

        target = normalize_target(target)

        ce_loss = self.ce(seg, target)
        dice_loss = self.dice(seg, target)

        total_loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )

        if self.boundary_weight > 0 and boundary is not None:
            boundary_loss = self.boundary(boundary, target)
            total_loss = total_loss + self.boundary_weight * boundary_loss

        return total_loss
