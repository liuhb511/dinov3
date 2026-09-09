"""A-CNN-MD-v1: Simple-7 Correct + L4/L8/L12 + local CNN F4/F8."""

import torch.nn as nn

from .encoder import DINOv3EncoderMD
from .local_encoder import LightweightLocalEncoderMD
from .decoder import DecoderV3CNNMD


class DINOv3SegCNNMD(nn.Module):
    """Unified 7-class direct segmentation model."""

    def __init__(self, cfg):
        super().__init__()

        feature_layers = getattr(cfg, "feature_layers", (4, 8, 12))
        local_stem_channels = int(
            getattr(cfg, "local_stem_channels", 16)
        )
        local_f4_channels = int(
            getattr(cfg, "local_f4_channels", 32)
        )
        local_f8_channels = int(
            getattr(cfg, "local_f8_channels", 64)
        )
        fusion_init_alpha = float(
            getattr(cfg, "fusion_init_alpha", 0.1)
        )

        self.encoder = DINOv3EncoderMD(
            cfg.backbone_name,
            trainable=not cfg.freeze_backbone,
            feature_layers=feature_layers,
        )

        self.local_encoder = LightweightLocalEncoderMD(
            stem_channels=local_stem_channels,
            f4_channels=local_f4_channels,
            f8_channels=local_f8_channels,
        )

        self.decoder = DecoderV3CNNMD(
            num_classes=cfg.num_classes,
            feat_dim=768,
            local_f4_channels=local_f4_channels,
            local_f8_channels=local_f8_channels,
            fusion_init_alpha=fusion_init_alpha,
        )

    def forward(self, x):
        dino_feats = self.encoder(x)
        local_feats = self.local_encoder(x)
        return self.decoder(
            dino_feats,
            local_feats,
            output_size=x.shape[2:],
        )
