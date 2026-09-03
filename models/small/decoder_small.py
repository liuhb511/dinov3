"""DecoderV3-compatible decoder with a training-only small-target auxiliary head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder_v3 import DecoderV3


class SmallTargetAuxHead(nn.Module):
    """Two binary channels: [tiny-D, small-TIND].

    The head is intentionally tiny and is skipped completely when
    ``return_aux=False``. Therefore deployment/inference can use exactly the
    original segmentation path.
    """

    def __init__(self, in_channels: int = 32, hidden_channels: int = 32, prior_prob: float = 0.01):
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_channels, 2, kernel_size=1)

        # Sparse-target prior: avoid starting with 50% positive probability.
        prior_prob = min(max(float(prior_prob), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.out.bias, math.log(prior_prob / (1.0 - prior_prob)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.block(x))


class DecoderV3Small(DecoderV3):
    """Same main segmentation path as DecoderV3 + training-only auxiliary head."""

    def __init__(self, num_classes: int, feat_dim: int = 768, aux_hidden: int = 32):
        super().__init__(num_classes=num_classes, feat_dim=feat_dim)
        self.small_aux_head = SmallTargetAuxHead(32, aux_hidden)

    def forward(self, feats, output_size=None, return_aux: bool = True):
        f4 = feats["f4"]
        f8 = feats["f8"]
        f16 = feats["f16"]

        x = self.fusion(f4, f8, f16)
        x = self.proj(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        seg = self.seg_head(x)
        boundary = self.boundary_head(x)
        aux = self.small_aux_head(x) if return_aux else None

        if output_size is not None:
            seg = F.interpolate(seg, size=output_size, mode="bilinear", align_corners=False)
            boundary = F.interpolate(boundary, size=output_size, mode="bilinear", align_corners=False)
            if aux is not None:
                aux = F.interpolate(aux, size=output_size, mode="bilinear", align_corners=False)

        return seg, boundary, aux
