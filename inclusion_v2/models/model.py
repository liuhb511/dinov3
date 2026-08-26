"""
inclusion_v2/models/model.py

InclusionDualExpertNet —— 共享 DINOv3 + 共享 Decoder + 双专家 Head。

forward 返回 dict：
    gate:     [B, 3, H, W]
    strip:    [B, 9, H, W]
    point:    [B, 4, H, W]
    boundary: [B, 1, H, W]
"""
import torch.nn as nn

from .encoder import DINOv3Encoder
from .fusion import LightFusion
from .shared_decoder import SharedDecoder
from .heads import GateHead, StripHead, PointHead, BoundaryHead


class InclusionDualExpertNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = DINOv3Encoder(
            cfg.backbone_name,
            trainable=not cfg.freeze_backbone,
            layers=getattr(cfg, "encoder_layers", (4, 8, 12)),
        )
        feat_dim = getattr(cfg, "feat_dim", 768)
        fusion_dim = getattr(cfg, "fusion_dim", 512)
        decoder_dim = getattr(cfg, "decoder_dim", 32)

        self.fusion = LightFusion(feat_dim=feat_dim, proj_dim=fusion_dim)
        self.decoder = SharedDecoder(in_channels=fusion_dim, out_channels=decoder_dim)

        self.gate_head = GateHead(decoder_dim)
        self.strip_head = StripHead(decoder_dim)
        self.point_head = PointHead(decoder_dim)
        self.boundary_head = BoundaryHead(decoder_dim)

    def forward(self, x):
        feats = self.encoder(x)
        fused = self.fusion(feats["f1"], feats["f2"], feats["f3"])
        dec = self.decoder(fused)

        return {
            "gate": self.gate_head(dec),
            "strip": self.strip_head(dec),
            "point": self.point_head(dec),
            "boundary": self.boundary_head(dec),
        }
