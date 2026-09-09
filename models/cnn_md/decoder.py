"""DecoderV3 + CNN F8/F4 gated residual fusion."""

import torch.nn.functional as F

from models.decoder_v3 import DecoderV3
from .fusion import GatedResidualFusion


class DecoderV3CNNMD(DecoderV3):
    """
    Keeps the original DecoderV3 main path:
        AttentionFusion -> proj -> dec1 -> dec2 -> dec3 -> dec4 -> heads

    Only inserts:
        CNN F8 after dec1 (true 1/8)
        CNN F4 after dec2 (true 1/4)
    """

    def __init__(self, num_classes, feat_dim=768, local_f4_channels=32,
                 local_f8_channels=64, fusion_init_alpha=0.1):
        super().__init__(num_classes=num_classes, feat_dim=feat_dim)

        self.fuse_f8 = GatedResidualFusion(
            decoder_ch=256, local_ch=local_f8_channels,
            init_alpha=fusion_init_alpha
        )
        self.fuse_f4 = GatedResidualFusion(
            decoder_ch=128, local_ch=local_f4_channels,
            init_alpha=fusion_init_alpha
        )

    def forward(self, dino_feats, local_feats, output_size=None):
        dino_low = dino_feats["dino_low"]
        dino_mid = dino_feats["dino_mid"]
        dino_high = dino_feats["dino_high"]

        # Multi-depth DINO semantic fusion: L4 / L8 / L12 by default.
        x = self.fusion(dino_low, dino_mid, dino_high)
        x = self.proj(x)

        # DINO patch grid is 1/16 for patch16.
        x = self.dec1(x)  # 1/16 -> 1/8
        x = self.fuse_f8(x, local_feats["cnn_f8"])

        x = self.dec2(x)  # 1/8 -> 1/4
        x = self.fuse_f4(x, local_feats["cnn_f4"])

        x = self.dec3(x)  # 1/4 -> 1/2
        x = self.dec4(x)  # 1/2 -> 1/1

        seg = self.seg_head(x)
        boundary = self.boundary_head(x)

        if output_size is not None:
            if seg.shape[-2:] != tuple(output_size):
                seg = F.interpolate(
                    seg, size=output_size,
                    mode="bilinear", align_corners=False
                )

            if boundary.shape[-2:] != tuple(output_size):
                boundary = F.interpolate(
                    boundary, size=output_size,
                    mode="bilinear", align_corners=False
                )

        return seg, boundary
