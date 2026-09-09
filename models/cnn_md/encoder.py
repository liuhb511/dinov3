"""DINOv3 Correct-token encoder using explicit multi-depth hidden states."""

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3EncoderMD(nn.Module):
    """
    DINOv3 ViT encoder.

    feature_layers means Transformer block depth, NOT spatial scale.
    For a 12-block ViT and HuggingFace hidden_states:
        hidden_states[0]  = embedding output
        hidden_states[4]  = block 4 output
        hidden_states[8]  = block 8 output
        hidden_states[12] = block 12 output

    All selected DINO features have the same patch-grid spatial resolution.
    """

    def __init__(
        self,
        model_name: str,
        trainable: bool = False,
        feature_layers=(4, 8, 12),
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.feature_layers = tuple(int(v) for v in feature_layers)
        if len(self.feature_layers) != 3:
            raise ValueError("feature_layers 必须正好包含 3 个 Transformer 层编号")

        num_hidden_layers = int(
            getattr(self.backbone.config, "num_hidden_layers", 0) or 0
        )
        if num_hidden_layers > 0:
            invalid = [
                v for v in self.feature_layers
                if v < 1 or v > num_hidden_layers
            ]
            if invalid:
                raise ValueError(
                    f"feature_layers={self.feature_layers} 超出有效范围 "
                    f"[1, {num_hidden_layers}]"
                )

        self.trainable = bool(trainable)
        self.set_trainable(self.trainable)

        print(
            "[DINOv3EncoderMD] feature_layers="
            f"{self.feature_layers}, num_hidden_layers={num_hidden_layers}"
        )

    def set_trainable(self, trainable: bool):
        self.trainable = bool(trainable)
        for p in self.backbone.parameters():
            p.requires_grad = self.trainable

    @staticmethod
    def _pair(value):
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        value = int(value)
        return value, value

    def _reshape_patch_tokens(self, feat: torch.Tensor, input_hw):
        b, _, c = feat.shape

        patch_h, patch_w = self._pair(
            getattr(self.backbone.config, "patch_size", 16)
        )
        input_h, input_w = int(input_hw[0]), int(input_hw[1])

        if input_h % patch_h != 0 or input_w % patch_w != 0:
            raise RuntimeError(
                f"输入尺寸 {input_hw} 不能被 patch_size={(patch_h, patch_w)} 整除"
            )

        expected_h = input_h // patch_h
        expected_w = input_w // patch_w
        expected_tokens = expected_h * expected_w

        # Correct DINOv3 order:
        # [CLS] + register tokens + patch tokens
        num_register_tokens = int(
            getattr(self.backbone.config, "num_register_tokens", 0) or 0
        )
        patch_tokens = feat[:, 1 + num_register_tokens:, :]

        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                "DINOv3 patch token 数量异常: "
                f"got={patch_tokens.shape[1]}, expected={expected_tokens}, "
                f"input={tuple(input_hw)}, patch_size={(patch_h, patch_w)}, "
                f"register_tokens={num_register_tokens}"
            )

        patch_tokens = patch_tokens.reshape(
            b, expected_h, expected_w, c
        ).permute(0, 3, 1, 2)

        return patch_tokens.contiguous()

    def forward(self, x: torch.Tensor):
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
        )
        hs = outputs.hidden_states

        max_index = max(self.feature_layers)
        if max_index >= len(hs):
            raise RuntimeError(
                f"hidden_states 数量={len(hs)}，无法读取 layer {max_index}"
            )

        low = hs[self.feature_layers[0]]
        mid = hs[self.feature_layers[1]]
        high = hs[self.feature_layers[2]]
        input_hw = x.shape[-2:]

        return {
            "dino_low": self._reshape_patch_tokens(low, input_hw),
            "dino_mid": self._reshape_patch_tokens(mid, input_hw),
            "dino_high": self._reshape_patch_tokens(high, input_hw),
        }
