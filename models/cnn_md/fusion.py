"""Gated residual fusion between decoder semantic features and local CNN features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class GatedResidualFusion(nn.Module):
    """
    Decoder + CNN local feature fusion:

        concat(decoder, local)
            -> 1x1 projection
            -> depthwise 3x3 refinement
            -> local_update

        output = decoder + alpha * local_update

    Alpha starts at 0.1 so the new CNN branch enters conservatively.
    """

    def __init__(self, decoder_ch: int, local_ch: int, init_alpha: float = 0.1):
        super().__init__()

        self.project = nn.Sequential(
            nn.Conv2d(
                decoder_ch + local_ch, decoder_ch,
                kernel_size=1, bias=False
            ),
            _gn(decoder_ch),
            nn.GELU(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                decoder_ch, decoder_ch,
                kernel_size=3, padding=1,
                groups=decoder_ch, bias=False
            ),
            _gn(decoder_ch),
            nn.GELU(),
        )

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, decoder_feat, local_feat):
        if local_feat.shape[-2:] != decoder_feat.shape[-2:]:
            local_feat = F.interpolate(
                local_feat,
                size=decoder_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        local_update = torch.cat([decoder_feat, local_feat], dim=1)
        local_update = self.project(local_update)
        local_update = self.refine(local_update)

        return decoder_feat + self.alpha * local_update
