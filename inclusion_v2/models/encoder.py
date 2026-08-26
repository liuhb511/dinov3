"""
inclusion_v2/models/encoder.py

DINOv3 多层特征提取器（MVP-1 修复版）：
1. 修复特殊 token 提取：
   实际 token 顺序为 [CLS, register×N, patch×M]（register 在开头，不是结尾），
   因此必须使用 `feat[:, 1 + num_register_tokens:, :]`，
   而不是旧代码的 `feat[:, 1:-4, :]`（后者会把 register 当空间特征、丢右下角 patch）。
2. 特征层改为 L4/L8/L12（hidden_states[4]/[8]/[12]），替代旧的最后三层 L10/L11/L12。
3. 支持 freeze / unfreeze 两阶段训练。
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Encoder(nn.Module):
    def __init__(self, model_name: str = "dinov3_model", trainable: bool = False,
                 layers=(4, 8, 12)):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.config = self.backbone.config
        self.num_registers = int(getattr(self.config, "num_register_tokens", 0))
        self.layers = tuple(layers)

        self.trainable = trainable
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_trainable(self, trainable: bool):
        """两阶段训练：动态切换 backbone 是否可训练。"""
        self.trainable = trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        """
        x: [B, 3, H, W]（已归一化）
        Returns:
            feats = {"f1": 来自 hidden_states[L4], "f2": L8, "f3": L12}
            均为 [B, C, H/16, W/16]
        """
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
        )
        hs = outputs.hidden_states  # 13 项: [embeddings, L1..L12]

        feats = {}
        for idx, layer in enumerate(self.layers):
            feat = hs[layer]
            # 跳过 CLS + register tokens（它们在序列开头）
            feat = feat[:, 1 + self.num_registers:, :]
            B, N, C = feat.shape
            side = int(round(N ** 0.5))
            feat = feat.reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()
            feats[f"f{idx + 1}"] = feat

        return feats
