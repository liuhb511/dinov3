import torch
import torch.nn as nn
import torch.nn.functional as F

# Existing repository model is reused but NOT modified.
from models.dinov3_segmentation import DINOv3Seg

class LabRedBranch(nn.Module):
    """
    Lightweight color branch. Input is a single LAB a* channel.
    It predicts a RED evidence logit map.
    """
    def __init__(self, channels=32):
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),

            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),

            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.GELU(),

            nn.Conv2d(c * 2, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.GELU(),

            nn.Conv2d(c, 1, 1),
        )

    def forward(self, a):
        y = self.net(a)
        return F.interpolate(
            y, size=a.shape[-2:], mode="bilinear", align_corners=False
        )

class HMSLabRedSeg(nn.Module):
    """
    RGB path remains exactly the repository's DINOv3Seg.
    LAB-a* only provides an additive residual to the RED class logit.

    This keeps the existing inclusion-oriented models/ directory untouched.
    """
    def __init__(
        self,
        cfg,
        red_class=4,
        lab_channels=32,
        fusion_init=0.5,
    ):
        super().__init__()
        self.base = DINOv3Seg(cfg)
        self.red_class = int(red_class)
        self.lab_branch = LabRedBranch(lab_channels)

        # Positive learnable fusion scale. softplus keeps it >= 0.
        init = torch.tensor(float(fusion_init)).clamp_min(1e-4)
        self._fusion_raw = nn.Parameter(torch.log(torch.expm1(init)))

    @property
    def fusion_scale(self):
        return F.softplus(self._fusion_raw)

    @property
    def encoder(self):
        # compatibility with two-stage freeze/unfreeze code
        return self.base.encoder

    @property
    def decoder(self):
        return self.base.decoder

    def forward(self, rgb, lab_a):
        seg, boundary = self.base(rgb)
        red_residual = self.lab_branch(lab_a)

        # Avoid unsafe in-place modification of autograd views.
        residual = torch.zeros_like(seg)
        residual[:, self.red_class:self.red_class+1] = (
            self.fusion_scale * red_residual
        )
        seg = seg + residual
        return seg, boundary, red_residual
