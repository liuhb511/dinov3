"""A-Small-v1 model.

Main inference path remains DINOv3 + DecoderV3 segmentation head.
The small-target auxiliary head is used only when explicitly requested during
training.
"""

from __future__ import annotations

import torch.nn as nn

from .encoder import DINOv3EncoderSmall
from .decoder_small import DecoderV3Small


class DINOv3SegSmall(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = DINOv3EncoderSmall(
            cfg.backbone_name,
            trainable=not cfg.freeze_backbone,
        )
        self.decoder = DecoderV3Small(
            num_classes=cfg.num_classes,
            feat_dim=getattr(cfg, "feat_dim", 768),
            aux_hidden=getattr(cfg, "small_aux_hidden", 32),
        )

    def forward(self, x, return_aux=None):
        # Default behavior: train -> auxiliary on; eval/inference -> auxiliary off.
        if return_aux is None:
            return_aux = self.training

        feats = self.encoder(x)
        seg, boundary, aux = self.decoder(
            feats,
            output_size=x.shape[2:],
            return_aux=bool(return_aux),
        )

        if return_aux:
            return seg, boundary, aux
        return seg, boundary
