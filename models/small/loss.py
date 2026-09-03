"""Loss for A-Small-v1: baseline Dice + size-aware CE + tiny-target focal auxiliary loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.loss_JZW import BoundaryLoss, DiceLoss, normalize_target


def weighted_cross_entropy(logits, target, pixel_weight=None):
    target = normalize_target(target)
    ce = F.cross_entropy(logits, target, reduction="none")
    if pixel_weight is None:
        return ce.mean()
    if pixel_weight.dim() == 4:
        pixel_weight = pixel_weight.squeeze(1)
    pixel_weight = pixel_weight.to(device=ce.device, dtype=ce.dtype)
    return (ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)


def sigmoid_focal_loss(logits, targets, weights=None, alpha=0.75, gamma=2.0):
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - pt).pow(gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * focal * bce

    if weights is None:
        return loss.mean()
    weights = weights.to(device=logits.device, dtype=logits.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


class SmallTargetLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dice = DiceLoss(smooth=1.0, include_background=False)
        self.boundary = BoundaryLoss()

        self.ce_weight = float(getattr(cfg, "ce_weight", 0.7))
        self.dice_weight = float(getattr(cfg, "dice_weight", 0.3))
        self.boundary_weight = float(getattr(cfg, "boundary_weight", 0.0))

        self.aux_d_weight = float(getattr(cfg, "aux_d_loss_weight", 0.30))
        self.aux_tind_weight = float(getattr(cfg, "aux_tind_loss_weight", 0.15))
        self.focal_alpha = float(getattr(cfg, "aux_focal_alpha", 0.75))
        self.focal_gamma = float(getattr(cfg, "aux_focal_gamma", 2.0))

    def main_loss(self, seg, boundary, target, pixel_weight=None):
        target = normalize_target(target)
        ce_loss = weighted_cross_entropy(seg, target, pixel_weight)
        dice_loss = self.dice(seg, target)
        total = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        boundary_loss = seg.new_tensor(0.0)
        if self.boundary_weight > 0 and boundary is not None:
            boundary_loss = self.boundary(boundary, target)
            total = total + self.boundary_weight * boundary_loss

        return total, {
            "ce": ce_loss.detach(),
            "dice": dice_loss.detach(),
            "boundary": boundary_loss.detach(),
        }

    def forward(
        self,
        seg,
        boundary,
        aux_logits,
        target,
        pixel_weight=None,
        aux_target=None,
        aux_weight=None,
    ):
        main, parts = self.main_loss(seg, boundary, target, pixel_weight)

        aux_d = seg.new_tensor(0.0)
        aux_tind = seg.new_tensor(0.0)
        total = main

        if aux_logits is not None and aux_target is not None:
            if aux_logits.shape[1] != 2:
                raise ValueError(f"small aux head 应输出2通道，实际 {tuple(aux_logits.shape)}")

            if aux_target.shape[-2:] != aux_logits.shape[-2:]:
                aux_target = F.interpolate(aux_target.float(), size=aux_logits.shape[-2:], mode="nearest")
                if aux_weight is not None:
                    aux_weight = F.interpolate(aux_weight.float(), size=aux_logits.shape[-2:], mode="nearest")

            d_weight = aux_weight[:, 0:1] if aux_weight is not None else None
            tind_weight = aux_weight[:, 1:2] if aux_weight is not None else None

            aux_d = sigmoid_focal_loss(
                aux_logits[:, 0:1], aux_target[:, 0:1], d_weight,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )
            aux_tind = sigmoid_focal_loss(
                aux_logits[:, 1:2], aux_target[:, 1:2], tind_weight,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )
            total = total + self.aux_d_weight * aux_d + self.aux_tind_weight * aux_tind

        parts.update({
            "main": main.detach(),
            "aux_d": aux_d.detach(),
            "aux_tind": aux_tind.detach(),
            "total": total.detach(),
        })
        return total, parts
