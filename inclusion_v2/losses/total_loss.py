"""
inclusion_v2/losses/total_loss.py

MVP-1 总损失：

    L_total =
        λ_gate  * L_gate
      + λ_strip * (0.6 * OHEM-CE(strip) + 0.4 * Dice(strip) + λ_HH * Rejection(HH→A/C))
      + λ_point * (0.6 * OHEM-CE(point) + 0.4 * Dice(point) + λ_D  * Rejection(HC/SZ→D))
      + λ_bnd   * L_boundary

各 head 使用自己的 target 与 ignore mask（见 data/label_mapping.py）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ohem import OhemCrossEntropy
from .dice import SoftDiceLoss
from .rejection import StripRejectionLoss, PointRejectionLoss
from .boundary import BoundaryLoss
from ..data.label_mapping import IGNORE_INDEX


class InclusionTotalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gate_weight = getattr(cfg, "gate_weight", 1.0)
        self.strip_weight = getattr(cfg, "strip_weight", 1.0)
        self.point_weight = getattr(cfg, "point_weight", 1.0)
        self.boundary_weight = getattr(cfg, "boundary_weight", 0.1)

        self.strip_ce_weight = getattr(cfg, "strip_ce_weight", 0.6)
        self.strip_dice_weight = getattr(cfg, "strip_dice_weight", 0.4)
        self.strip_rejection_weight = getattr(cfg, "strip_rejection_weight", 0.1)

        self.point_ce_weight = getattr(cfg, "point_ce_weight", 0.6)
        self.point_dice_weight = getattr(cfg, "point_dice_weight", 0.4)
        self.point_rejection_weight = getattr(cfg, "point_rejection_weight", 0.1)

        use_focal = getattr(cfg, "use_focal", False)

        if use_focal:
            from .ohem import FocalLoss
            self.strip_ce = FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)
            self.point_ce = FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)
        else:
            self.strip_ce = OhemCrossEntropy(
                ignore_index=IGNORE_INDEX,
                min_kept=getattr(cfg, "ohem_min_kept", 100000),
            )
            self.point_ce = OhemCrossEntropy(
                ignore_index=IGNORE_INDEX,
                min_kept=getattr(cfg, "ohem_min_kept", 100000),
            )

        self.strip_dice = SoftDiceLoss(ignore_index=IGNORE_INDEX, include_background=False)
        self.point_dice = SoftDiceLoss(ignore_index=IGNORE_INDEX, include_background=False)

        self.gate_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.strip_rejection = StripRejectionLoss()
        self.point_rejection = PointRejectionLoss()
        self.boundary = BoundaryLoss()

    def forward(self, outputs, targets):
        """
        outputs: dict(gate=[B,3,H,W], strip=[B,9,H,W], point=[B,4,H,W], boundary=[B,1,H,W])
        targets: dict(gate, strip, point, mask=[B,H,W])
        Returns:
            total_loss: Tensor
            loss_dict:  dict(可打印的各分量)
        """
        gate = outputs["gate"]
        strip = outputs["strip"]
        point = outputs["point"]
        boundary = outputs.get("boundary")

        H, W = targets["mask"].shape[-2:]
        # AMP/autocast 下统一转 fp32 计算损失：
        # - 避免 OHEM/CE 大和溢出、Dice/BCE 精度问题
        # - 转回 fp32 是微分的，不影响梯度
        gate = F.interpolate(gate.float(), size=(H, W), mode="bilinear", align_corners=False)
        strip = F.interpolate(strip.float(), size=(H, W), mode="bilinear", align_corners=False)
        point = F.interpolate(point.float(), size=(H, W), mode="bilinear", align_corners=False)
        if boundary is not None:
            boundary = F.interpolate(boundary.float(), size=(H, W), mode="bilinear", align_corners=False)

        # ---------- gate ----------
        l_gate = self.gate_weight * self.gate_loss(gate, targets["gate"])

        # ---------- strip ----------
        l_strip_ce = self.strip_ce(strip, targets["strip"])
        l_strip_dice = self.strip_dice(strip, targets["strip"])
        l_strip_rej = self.strip_rejection_weight * self.strip_rejection(strip, targets["mask"])
        l_strip = self.strip_weight * (
            self.strip_ce_weight * l_strip_ce
            + self.strip_dice_weight * l_strip_dice
            + l_strip_rej
        )

        # ---------- point ----------
        l_point_ce = self.point_ce(point, targets["point"])
        l_point_dice = self.point_dice(point, targets["point"])
        l_point_rej = self.point_rejection_weight * self.point_rejection(point, targets["mask"])
        l_point = self.point_weight * (
            self.point_ce_weight * l_point_ce
            + self.point_dice_weight * l_point_dice
            + l_point_rej
        )

        # ---------- boundary ----------
        l_boundary = torch.zeros_like(l_gate)
        if boundary is not None and self.boundary_weight > 0:
            l_boundary = self.boundary_weight * self.boundary(boundary, targets["mask"])

        total = l_gate + l_strip + l_point + l_boundary

        loss_dict = {
            "gate": l_gate.detach(),
            "strip_ce": l_strip_ce.detach(),
            "strip_dice": l_strip_dice.detach(),
            "strip_rej": l_strip_rej.detach(),
            "point_ce": l_point_ce.detach(),
            "point_dice": l_point_dice.detach(),
            "point_rej": l_point_rej.detach(),
            "boundary": l_boundary.detach(),
        }
        return total, loss_dict
