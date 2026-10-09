"""
DINOv3 + UNet++-style Dense Depth Fusion + Progressive Decoder.

Compared with the previous pseudo-pyramid UNet++ version, this model DOES NOT
resize the four DINO encoder hidden states into artificial p4/p8/p16/p32 encoder
features. All selected DINO features stay on the native patch-token grid
(e.g. 64x64 for a 1024x1024 input with patch_size=16).

Flow for 1024x1024 / ViT-B/16:
    input                  : [B,   3, 1024, 1024]
    DINO hidden states x 4 : [B, 768,   64,   64]
    channel projections    : [B, 128,   64,   64] x 4
    dense depth fusion     : [B, 128,   64,   64]
    decoder stage 1        : [B, 128,  128,  128]
    decoder stage 2        : [B,  96,  256,  256]
    decoder stage 3        : [B,  64,  512,  512]
    decoder stage 4        : [B,  32, 1024, 1024]
    segmentation logits    : [B,   C, 1024, 1024]

The UNet++ idea retained here is dense/nested connectivity across different
Transformer depths. Spatial upsampling begins only after the dense 64x64 bridge
has been formed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.boundary_head import BoundaryHead


def _group_norm(channels: int) -> nn.GroupNorm:
    """Choose a GroupNorm group count that divides channels."""
    for groups in (8, 4, 2):
        if channels >= groups and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Sequential):
    """Conv2d + GroupNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            _group_norm(out_channels),
            nn.GELU()
        )


class DenseConvBlock(nn.Module):
    """Two 3x3 convolutions used by each dense-fusion node."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(in_channels, out_channels, 3, 1),
            ConvGNAct(out_channels, out_channels, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DINOv3UNetPPEncoder(nn.Module):
    """Extract four DINOv3 hidden states on the native patch-token grid."""

    def __init__(self, model_name: str, trainable=False,
                 feature_layers: Sequence[int] = (-4, -3, -2, -1)):
        super().__init__()

        if len(feature_layers) != 4:
            raise ValueError(
                f"feature_layers must contain exactly 4 indices, got {feature_layers}"
            )

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )
        self.feature_layers = tuple(int(v) for v in feature_layers)
        self.trainable = bool(trainable)
        self.set_trainable(self.trainable)

    def set_trainable(self, trainable: bool) -> None:
        self.trainable = bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = self.trainable

    @staticmethod
    def _normalize_patch_size(patch_size) -> Tuple[int, int]:
        if isinstance(patch_size, int):
            return patch_size, patch_size
        if isinstance(patch_size, Iterable):
            patch_size = tuple(patch_size)
            if len(patch_size) != 2:
                raise ValueError(f"Unsupported patch_size={patch_size}")
            return int(patch_size[0]), int(patch_size[1])
        raise TypeError(f"Unsupported patch_size type: {type(patch_size)}")

    def _reshape_patch_tokens(self, feat: torch.Tensor,
                              input_size: Tuple[int, int]) -> torch.Tensor:
        """[B,N,C] -> [B,C,Hpatch,Wpatch], using correct DINOv3 token order."""
        if feat.ndim != 3:
            raise ValueError(f"Expected hidden state [B,N,C], got {feat.shape}")

        batch, _, channels = feat.shape
        num_register_tokens = int(
            getattr(self.backbone.config, "num_register_tokens", 0) or 0
        )

        # DINOv3 token order: [CLS] + register tokens + patch tokens
        feat = feat[:, 1 + num_register_tokens :, :]

        patch_h, patch_w = self._normalize_patch_size(
            getattr(self.backbone.config, "patch_size")
        )
        input_h, input_w = input_size

        if input_h % patch_h != 0 or input_w % patch_w != 0:
            raise ValueError(
                f"Input size {input_size} must be divisible by patch_size "
                f"{(patch_h, patch_w)}"
            )

        grid_h = input_h // patch_h
        grid_w = input_w // patch_w
        expected_tokens = grid_h * grid_w

        if feat.shape[1] != expected_tokens:
            raise RuntimeError(
                "Patch token count mismatch: "
                f"got {feat.shape[1]}, expected {expected_tokens} "
                f"for input={input_size}, patch_size={(patch_h, patch_w)}"
            )

        feat = feat.reshape(batch, grid_h, grid_w, channels)
        feat = feat.permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        num_hidden_states = len(hidden_states)
        selected = []
        for idx in self.feature_layers:
            resolved = idx if idx >= 0 else num_hidden_states + idx
            if resolved < 0 or resolved >= num_hidden_states:
                raise IndexError(
                    f"feature layer {idx} resolves to {resolved}, but "
                    f"len(hidden_states)={num_hidden_states}"
                )
            selected.append(hidden_states[resolved])

        input_size = (x.shape[-2], x.shape[-1])
        d0, d1, d2, d3 = [
            self._reshape_patch_tokens(feat, input_size) for feat in selected
        ]

        return {
            "dino_0": d0,
            "dino_1": d1,
            "dino_2": d2,
            "dino_3": d3,
        }


class ChannelProjection(nn.Module):
    """Project one DINO feature from dino_dim to a common dense width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            _group_norm(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


class DINODenseDepthFusion(nn.Module):
    """
    UNet++-style dense connectivity across Transformer DEPTH, not spatial scale.

    All tensors remain at the same native DINO patch-grid resolution.

        e0 ----- n0_1 ----- n0_2 ----- n0_3
         \\       /  \\       /          ^
          e1 --- n1_1 ----- n1_2 ---------|
           \\     /  \\       /
            e2 - n2_1 ---------------
             \\  /
              e3

    More explicitly:
        n0_1 = F(e0, e1)
        n1_1 = F(e1, e2)
        n2_1 = F(e2, e3)
        n0_2 = F(e0, n0_1, n1_1)
        n1_2 = F(e1, n1_1, n2_1)
        n0_3 = F(e0, n0_1, n0_2, n1_2)
    """

    def __init__(self, dino_dim: int = 768, dense_channels: int = 128):
        super().__init__()
        c = int(dense_channels)
        self.dense_channels = c

        self.proj0 = ChannelProjection(dino_dim, c)
        self.proj1 = ChannelProjection(dino_dim, c)
        self.proj2 = ChannelProjection(dino_dim, c)
        self.proj3 = ChannelProjection(dino_dim, c)

        # First dense column: 2*c -> c
        self.n0_1 = DenseConvBlock(c * 2, c)
        self.n1_1 = DenseConvBlock(c * 2, c)
        self.n2_1 = DenseConvBlock(c * 2, c)

        # Second dense column: 3*c -> c
        self.n0_2 = DenseConvBlock(c * 3, c)
        self.n1_2 = DenseConvBlock(c * 3, c)

        # Final dense node: 4*c -> c
        self.n0_3 = DenseConvBlock(c * 4, c)

    @staticmethod
    def _check_same_spatial(feats: Sequence[torch.Tensor]) -> None:
        sizes = {tuple(x.shape[-2:]) for x in feats}
        if len(sizes) != 1:
            raise RuntimeError(
                f"Dense depth fusion expects same spatial size, got {sorted(sizes)}"
            )

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        d0 = feats["dino_0"]
        d1 = feats["dino_1"]
        d2 = feats["dino_2"]
        d3 = feats["dino_3"]
        self._check_same_spatial((d0, d1, d2, d3))

        # 768 -> 128 (default), spatial size unchanged.
        e0 = self.proj0(d0)
        e1 = self.proj1(d1)
        e2 = self.proj2(d2)
        e3 = self.proj3(d3)

        # UNet++-style nested fusion, all on the same DINO grid.
        n0_1 = self.n0_1(torch.cat([e0, e1], dim=1))
        n1_1 = self.n1_1(torch.cat([e1, e2], dim=1))
        n2_1 = self.n2_1(torch.cat([e2, e3], dim=1))

        n0_2 = self.n0_2(torch.cat([e0, n0_1, n1_1], dim=1))
        n1_2 = self.n1_2(torch.cat([e1, n1_1, n2_1], dim=1))

        n0_3 = self.n0_3(torch.cat([e0, n0_1, n0_2, n1_2], dim=1))
        return n0_3


class ProgressiveUpBlock(nn.Module):
    """Bilinear x2 upsampling followed by learnable local refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = DenseConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor,
                target_size: Tuple[int, int] | None = None) -> torch.Tensor:
        if target_size is None:
            target_size = (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )
        return self.refine(x)


class ProgressiveDecoder(nn.Module):
    """
    Recover spatial resolution only AFTER dense DINO-depth fusion.

    Default channel flow:
        128@64^2 -> 128@128^2 -> 96@256^2 -> 64@512^2 -> 32@1024^2
    for a 1024 input with patch_size=16.
    """

    def __init__(self, in_channels: int = 128,
                 decoder_channels: Sequence[int] = (128, 96, 64, 32)):
        super().__init__()
        if len(decoder_channels) != 4:
            raise ValueError(
                "decoder_channels must contain exactly 4 values, "
                f"got {decoder_channels}"
            )

        c1, c2, c3, c4 = [int(v) for v in decoder_channels]
        self.out_channels = c4

        self.up1 = ProgressiveUpBlock(in_channels, c1)
        self.up2 = ProgressiveUpBlock(c1, c2)
        self.up3 = ProgressiveUpBlock(c2, c3)
        self.up4 = ProgressiveUpBlock(c3, c4)

    @staticmethod
    def _progressive_sizes(start_size: Tuple[int, int],
                           output_size: Tuple[int, int]) -> Tuple[Tuple[int, int], ...]:
        """
        Build 4 progressive target sizes.

        For normal patch16 inputs this is exactly x2/x2/x2/x2.
        The final stage is forced to output_size for defensive compatibility.
        """
        sh, sw = start_size
        oh, ow = output_size
        sizes = (
            (sh * 2, sw * 2),
            (sh * 4, sw * 4),
            (sh * 8, sw * 8),
            (oh, ow),
        )

        # The model is designed for patch16 DINO. This check catches accidental
        # use with incompatible spatial ratios instead of silently distorting it.
        expected_h = sh * 16
        expected_w = sw * 16
        if (expected_h, expected_w) != (oh, ow):
            raise ValueError(
                "ProgressiveDecoder expects output size to be 16x the native "
                f"DINO grid. Got start={start_size}, output={output_size}. "
                "For DINOv3 ViT-B/16 and 1024 input, expected 64 -> 1024."
            )
        return sizes

    def forward(self, x: torch.Tensor,
                output_size: Tuple[int, int]) -> torch.Tensor:
        sizes = self._progressive_sizes(tuple(x.shape[-2:]), output_size)
        x = self.up1(x, sizes[0])
        x = self.up2(x, sizes[1])
        x = self.up3(x, sizes[2])
        x = self.up4(x, sizes[3])
        return x


class DINOv3UNetPP(nn.Module):
    """
    End-to-end model with the same external interface as the previous UNet++
    experiment: forward(x) -> (seg, boundary).

    Existing train_unetpp.py / infer_unetpp.py can still instantiate this class.
    The old ``pyramid_channels`` config field is intentionally ignored so that
    existing config objects do not break; use ``dense_channels`` and
    ``decoder_channels`` for this architecture when you want to tune widths.
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        feature_layers = getattr(cfg, "feature_layers", (-4, -3, -2, -1))
        dino_dim = int(getattr(cfg, "dino_dim", 768))
        dense_channels = int(getattr(cfg, "dense_channels", 128))
        decoder_channels = tuple(
            int(v) for v in getattr(cfg, "decoder_channels", (128, 96, 64, 32))
        )

        self.encoder = DINOv3UNetPPEncoder(
            model_name=cfg.backbone_name,
            trainable=not cfg.freeze_backbone,
            feature_layers=feature_layers,
        )

        # No pseudo p4/p8/p16/p32 pyramid here.
        self.dense_fusion = DINODenseDepthFusion(
            dino_dim=dino_dim,
            dense_channels=dense_channels,
        )
        self.decoder = ProgressiveDecoder(
            in_channels=dense_channels,
            decoder_channels=decoder_channels,
        )

        head_channels = int(decoder_channels[-1])
        self.seg_head = nn.Conv2d(head_channels, cfg.num_classes, kernel_size=1)
        self.boundary_head = BoundaryHead(head_channels)

    def forward(self, x: torch.Tensor):
        output_size = (x.shape[-2], x.shape[-1])

        # 1) Four DINO depth features, all on the native token grid.
        dino_feats = self.encoder(x)

        # 2) 768 -> dense_channels and UNet++-style depth-wise dense fusion.
        #    No spatial resize occurs in this stage.
        bridge = self.dense_fusion(dino_feats)

        # 3) Spatial decoding starts here: 64 -> 128 -> 256 -> 512 -> 1024.
        full = self.decoder(bridge, output_size)

        seg = self.seg_head(full)
        boundary = self.boundary_head(full)

        if seg.shape[-2:] != output_size:
            seg = F.interpolate(
                seg,
                size=output_size,
                mode="bilinear",
                align_corners=False
            )
        if boundary.shape[-2:] != output_size:
            boundary = F.interpolate(
                boundary,
                size=output_size,
                mode="bilinear",
                align_corners=False
            )

        return seg, boundary


__all__ = [
    "DINOv3UNetPP",
    "DINOv3UNetPPEncoder",
    "DINODenseDepthFusion",
    "ProgressiveDecoder",
]
