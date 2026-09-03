"""DINOv3 encoder used by the small-target branch.

Important: DINOv3 token order in the HF implementation is
    [CLS] + [register tokens] + [patch tokens]
so patch tokens start at ``1 + num_register_tokens``.

Keeping a local encoder here prevents A-Small-v1 from accidentally inheriting a
legacy ``[:, 1:-4, :]`` slice from another experiment.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3EncoderSmall(nn.Module):
    def __init__(self, model_name: str, trainable: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.trainable = bool(trainable)
        self.set_trainable(self.trainable)

    def set_trainable(self, trainable: bool):
        self.trainable = bool(trainable)
        for p in self.backbone.parameters():
            p.requires_grad = self.trainable

    @staticmethod
    def _pair(value) -> Tuple[int, int]:
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    def _reshape_patch_tokens(self, feat: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        b, _, c = feat.shape

        num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
        feat = feat[:, 1 + num_register_tokens :, :]

        patch_size = getattr(self.backbone.config, "patch_size", 16)
        ph, pw = self._pair(patch_size)
        h = int(image_hw[0]) // ph
        w = int(image_hw[1]) // pw
        expected = h * w

        if feat.shape[1] != expected:
            raise RuntimeError(
                "DINOv3 patch-token 数量与输入尺寸不匹配: "
                f"tokens={feat.shape[1]}, expected={expected}, image_hw={image_hw}, "
                f"patch_size={(ph, pw)}, num_register_tokens={num_register_tokens}. "
                "请确认当前 backbone/token 提取方式。"
            )

        feat = feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hs = outputs.hidden_states
        image_hw = (int(x.shape[-2]), int(x.shape[-1]))

        return {
            "f4": self._reshape_patch_tokens(hs[-3], image_hw),
            "f8": self._reshape_patch_tokens(hs[-2], image_hw),
            "f16": self._reshape_patch_tokens(hs[-1], image_hw),
        }
