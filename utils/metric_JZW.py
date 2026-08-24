"""
utils/metric.py

多类别语义分割评估指标
- 全验证集混淆矩阵
- 每类别 IoU、Dice
- mIoU、mDice
"""

import torch


def normalize_mask(mask):
    """
    将 mask 统一转换为 [B, H, W]。
    """

    if mask.dim() == 4:
        if mask.shape[1] != 1:
            raise ValueError(
                f"Mask 通道数必须为1，实际 shape={mask.shape}"
            )
        mask = mask.squeeze(1)

    if mask.dim() != 3:
        raise ValueError(
            f"Mask 应为 [B,H,W] 或 [B,1,H,W]，实际 shape={mask.shape}"
        )

    return mask.long()


@torch.no_grad()
def update_confusion_matrix(confusion_matrix, pred, target, num_classes, ignore_index=None):
    """
    在 GPU 上累计多类别混淆矩阵。

    confusion_matrix:
        [C, C]
        行表示真实类别，列表示预测类别。

    pred:
        [B,H,W] 或 [B,1,H,W]

    target:
        [B,H,W] 或 [B,1,H,W]
    """

    pred = normalize_mask(pred).reshape(-1)
    target = normalize_mask(target).reshape(-1)

    valid = (
        (target >= 0)
        & (target < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )

    if ignore_index is not None:
        valid &= target != ignore_index

    target = target[valid]
    pred = pred[valid]

    encoded = target * num_classes + pred

    batch_matrix = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(
        num_classes,
        num_classes,
    )

    confusion_matrix += batch_matrix

    return confusion_matrix


@torch.no_grad()
def calculate_metrics(confusion_matrix, include_background=False, eps=1e-6):
    """
    根据整个验证集的混淆矩阵计算 mIoU 和 mDice。

    include_background=False：
        宏平均时不统计背景类别0。

    Returns:
        mean_iou
        mean_dice
        class_iou
        class_dice
    """

    matrix = confusion_matrix.float()

    true_positive = torch.diag(matrix)

    target_pixels = matrix.sum(dim=1)
    prediction_pixels = matrix.sum(dim=0)

    union = (
        target_pixels
        + prediction_pixels
        - true_positive
    )

    dice_denominator = (
        target_pixels
        + prediction_pixels
    )

    class_iou = (
        true_positive + eps
    ) / (
        union + eps
    )

    class_dice = (
        2.0 * true_positive + eps
    ) / (
        dice_denominator + eps
    )

    valid_iou_classes = union > 0
    valid_dice_classes = dice_denominator > 0

    if not include_background:
        valid_iou_classes[0] = False
        valid_dice_classes[0] = False

    if valid_iou_classes.any():
        mean_iou = class_iou[
            valid_iou_classes
        ].mean()
    else:
        mean_iou = matrix.new_tensor(0.0)

    if valid_dice_classes.any():
        mean_dice = class_dice[
            valid_dice_classes
        ].mean()
    else:
        mean_dice = matrix.new_tensor(0.0)

    return (
        mean_iou.item(),
        mean_dice.item(),
        class_iou.cpu().tolist(),
        class_dice.cpu().tolist(),
    )