import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_target(target):
    """
    将 target 统一转换为 [B,H,W]。
    """

    if target.dim() == 4:
        if target.shape[1] != 1:
            raise ValueError(
                f"Target 通道数必须为 1，实际 shape={target.shape}"
            )
        target = target.squeeze(1)

    if target.dim() != 3:
        raise ValueError(
            f"Target 应为 [B,H,W] 或 [B,1,H,W]，"
            f"实际 shape={target.shape}"
        )

    return target.long()


class DiceLoss(nn.Module):
    """
    多类别 Soft Dice Loss。
    """

    def __init__(
        self,
        smooth=1.0,
        include_background=False,
    ):
        super().__init__()

        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, target):
        """
        logits: [B,C,H,W]
        target: [B,H,W] 或 [B,1,H,W]
        """

        target = normalize_target(target)

        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(
                logits,
                size=target.shape[1:],
                mode="bilinear",
                align_corners=False,
            )

        num_classes = logits.shape[1]

        target_min = int(target.min().item())
        target_max = int(target.max().item())

        if target_min < 0 or target_max >= num_classes:
            unique_values = (
                torch.unique(target)
                .detach()
                .cpu()
                .tolist()
            )

            raise ValueError(
                f"GT 类别索引越界："
                f"min={target_min}，"
                f"max={target_max}，"
                f"unique={unique_values}，"
                f"num_classes={num_classes}"
            )

        probabilities = F.softmax(logits, dim=1)

        target_onehot = F.one_hot(
            target,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2)

        target_onehot = target_onehot.to(
            dtype=probabilities.dtype
        )

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
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        if not self.include_background:
            if num_classes <= 1:
                raise ValueError(
                    "include_background=False 时，"
                    "num_classes 必须大于 1"
                )

            dice = dice[1:]

        return 1.0 - dice.mean()


class MissingLabelRobustCE(nn.Module):
    """
    针对“前景漏标为背景”的鲁棒多类别 CE。

    规则：
    1. 已标前景像素：权重 1.0
    2. 普通背景像素：权重 background_weight
    3. GT 是背景，但 Teacher 高置信预测为前景：
       权重 conflict_weight

    注意：
    疑似漏标像素只降权，不修改原标签。
    """

    def __init__(
        self,
        background_index=0,
        background_weight=0.7,
        conflict_weight=0.15,
        confidence_threshold=0.95,
    ):
        super().__init__()

        if not 0.0 <= conflict_weight <= background_weight:
            raise ValueError(
                "应满足："
                "0 <= conflict_weight <= background_weight"
            )

        if not 0.0 <= background_weight <= 1.0:
            raise ValueError(
                "background_weight 必须位于 [0,1]"
            )

        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError(
                "confidence_threshold 必须位于 (0,1)"
            )

        self.background_index = background_index
        self.background_weight = background_weight
        self.conflict_weight = conflict_weight
        self.confidence_threshold = confidence_threshold

    def forward(
        self,
        student_logits,
        teacher_logits,
        target,
        use_teacher,
    ):
        target = normalize_target(target)

        if student_logits.shape[2:] != target.shape[1:]:
            student_logits = F.interpolate(
                student_logits,
                size=target.shape[1:],
                mode="bilinear",
                align_corners=False,
            )

        pixel_ce = F.cross_entropy(
            student_logits,
            target,
            reduction="none",
        )

        background_mask = (
            target == self.background_index
        )

        foreground_mask = ~background_mask

        # 默认所有像素权重为 1
        pixel_weight = torch.ones_like(pixel_ce)

        # 普通背景像素降低权重
        pixel_weight[background_mask] = (
            self.background_weight
        )

        conflict_mask = torch.zeros_like(
            background_mask,
            dtype=torch.bool,
        )

        if use_teacher:
            if teacher_logits is None:
                raise ValueError(
                    "启用 Teacher 漏标检测时，"
                    "teacher_logits 不能为空"
                )

            if (
                teacher_logits.shape[2:]
                != target.shape[1:]
            ):
                teacher_logits = F.interpolate(
                    teacher_logits,
                    size=target.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )

            with torch.no_grad():
                teacher_probability = F.softmax(
                    teacher_logits.float(),
                    dim=1,
                )

                teacher_confidence, teacher_class = (
                    teacher_probability.max(dim=1)
                )

                # GT 是背景，但 Teacher 高置信认为是前景
                conflict_mask = (
                    background_mask
                    & (
                        teacher_class
                        != self.background_index
                    )
                    & (
                        teacher_confidence
                        >= self.confidence_threshold
                    )
                )

            # 疑似漏标区域进一步降低背景监督
            pixel_weight[conflict_mask] = (
                self.conflict_weight
            )

        weighted_ce = (
            pixel_ce * pixel_weight
        ).sum() / pixel_weight.sum().clamp_min(1.0)

        total_pixels = target.numel()

        stats = {
            "foreground_ratio": (
                foreground_mask.sum().float()
                / total_pixels
            ).detach(),

            "background_ratio": (
                background_mask.sum().float()
                / total_pixels
            ).detach(),

            "conflict_ratio": (
                conflict_mask.sum().float()
                / total_pixels
            ).detach(),
        }

        return weighted_ce, stats


class TotalLoss(nn.Module):
    """
    总损失：

    total =
        ce_weight * robust_ce
        + dice_weight * foreground_dice

    boundary 参数保留，但不参与损失。
    """

    def __init__(self, cfg):
        super().__init__()

        self.ema_warmup_epochs = getattr(
            cfg,
            "ema_warmup_epochs",
            8,
        )

        self.robust_ce = MissingLabelRobustCE(
            background_index=getattr(
                cfg,
                "background_index",
                0,
            ),
            background_weight=getattr(
                cfg,
                "background_weight",
                0.7,
            ),
            conflict_weight=getattr(
                cfg,
                "conflict_weight",
                0.15,
            ),
            confidence_threshold=getattr(
                cfg,
                "confidence_threshold",
                0.95,
            ),
        )

        self.dice = DiceLoss(
            smooth=1.0,
            include_background=False,
        )

        self.ce_weight = getattr(
            cfg,
            "ce_weight",
            0.7,
        )

        self.dice_weight = getattr(
            cfg,
            "dice_weight",
            0.3,
        )

    def forward(
        self,
        student_seg,
        teacher_seg,
        boundary,
        target,
        epoch,
        force_disable_teacher=False,
    ):
        """
        student_seg:
            Student logits [B,C,H,W]

        teacher_seg:
            Teacher logits [B,C,H,W]，可以为 None

        boundary:
            保留兼容参数，当前不参与损失

        target:
            [B,H,W] 或 [B,1,H,W]

        epoch:
            当前 epoch，从 0 开始

        force_disable_teacher:
            验证时可强制禁用 Teacher 冲突筛选
        """

        del boundary

        target = normalize_target(target)

        if student_seg.shape[2:] != target.shape[1:]:
            student_seg = F.interpolate(
                student_seg,
                size=target.shape[1:],
                mode="bilinear",
                align_corners=False,
            )

        use_teacher = (
            epoch >= self.ema_warmup_epochs
            and not force_disable_teacher
            and teacher_seg is not None
        )

        ce_loss, stats = self.robust_ce(
            student_logits=student_seg,
            teacher_logits=teacher_seg,
            target=target,
            use_teacher=use_teacher,
        )

        dice_loss = self.dice(
            student_seg,
            target,
        )

        total_loss = (
            self.ce_weight * ce_loss
            + self.dice_weight * dice_loss
        )

        stats.update({
            "ce_loss": ce_loss.detach(),
            "dice_loss": dice_loss.detach(),
            "total_loss": total_loss.detach(),
            "teacher_enabled": use_teacher,
        })

        return total_loss, stats