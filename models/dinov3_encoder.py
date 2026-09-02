"""
models/dinov3_encoder.py

DINOv3 Multi-layer Feature Extractor
输出最后3层 hidden states 对应的 feature maps
"""
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(self, model_name: str, trainable=False):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=True
        )

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        )

        self.trainable = trainable
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_trainable(self, trainable: bool):
        """动态切换 backbone 是否可训练"""
        self.trainable = trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        Returns:
            feats = {
                "f4": low-level,
                "f8": mid-level,
                "f16": high-level
            }

        注意：
            f4 / f8 / f16 只是不同深度的 ViT hidden states，
            并不代表真正的 1/4、1/8、1/16 空间分辨率。

            三个 feature map 的空间分辨率相同，
            均由 DINOv3 patch size 决定。
        """
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True
        )

        hs = outputs.hidden_states

        # 取最后3层 hidden states
        f_low = hs[-3]
        f_mid = hs[-2]
        f_high = hs[-1]

        def reshape(feat):
            B, N, C = feat.shape

            # DINOv3 token 顺序：
            # [CLS] + register tokens + patch tokens
            num_register_tokens = (
                self.backbone.config.num_register_tokens
            )

            # 去掉 CLS + register tokens，只保留 patch tokens
            feat = feat[:, 1 + num_register_tokens:, :]

            num_patches = feat.shape[1]
            H = W = int(num_patches ** 0.5)

            assert H * W == num_patches, (
                f"Patch token 数量错误: "
                f"num_patches={num_patches}, "
                f"H={H}, W={W}"
            )

            feat = feat.reshape(B, H, W, C)
            feat = feat.permute(0, 3, 1, 2)

            return feat.contiguous()

        return {
            "f4": reshape(f_low),
            "f8": reshape(f_mid),
            "f16": reshape(f_high),
        }
