"""
inclusion_v2/losses/ohem.py

专家 head 的分类损失（支持 ignore_index=255）：

1. OhemCrossEntropy —— 在线困难样本挖掘：
   - 保留所有正样本像素
   - 背景像素只保留 loss 最大的一部分（min_kept 个）
   使空心点、不规则点、水渍边缘、倾斜条纹等自动成为高权重负样本，
   纯净金属背景不占据过多梯度。

2. FocalLoss —— 可选的 Focal-CE（gamma 默认 2.0）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_index: int = 255, min_kept: int = 100000):
        super().__init__()
        self.ignore_index = ignore_index
        self.min_kept = min_kept

    def forward(self, logits, target):
        """
        logits: [B, C, H, W]
        target: [B, H, W]，值 ∈ {0..C-1, ignore_index}
        """
        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1)                 # [B, C, N]
        target = target.reshape(B, -1)                    # [B, N]

        log_prob = F.log_softmax(logits, dim=1)
        target_clamped = target.clamp(min=0, max=C - 1)   # ignore(255) 不参与 gather
        ce = -log_prob.gather(1, target_clamped.unsqueeze(1)).squeeze(1)  # [B, N]

        valid = target != self.ignore_index
        pos = valid & (target > 0)

        keep = torch.zeros_like(valid, dtype=torch.bool)
        for b in range(B):
            valid_b = valid[b]
            if not valid_b.any():
                continue
            # 正样本全部保留
            pos_b = pos[b]
            keep_b = keep[b]
            keep_b[pos_b] = True

            bg_cand = valid_b & (~pos_b)
            n_bg = int(bg_cand.sum().item())
            if n_bg == 0:
                continue
            n_pos = int(pos_b.sum().item())
            k = max(0, self.min_kept - n_pos)
            k = min(k, n_bg)
            if k <= 0:
                continue
            if k == n_bg:
                keep_b[bg_cand] = True
            else:
                bg_ce = ce[b][bg_cand]
                _, top_idx = torch.topk(bg_ce, k=k)
                bg_positions = bg_cand.nonzero(as_tuple=True)[0]
                keep_b[bg_positions[top_idx]] = True

        loss = ce[keep].sum() / keep.sum().clamp(min=1)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        B, C, H, W = logits.shape
        logits = logits.reshape(B, C, -1)
        target = target.reshape(B, -1)

        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        tgt = target.clamp(min=0, max=C - 1)   # ignore(255) 不参与 gather
        onehot = F.one_hot(tgt, num_classes=C).permute(0, 2, 1).float()  # [B, C, N]

        pt = (prob * onehot).sum(dim=1)                                   # [B, N]
        ce = -log_prob.gather(1, tgt.unsqueeze(1)).squeeze(1)
        focal = ((1.0 - pt) ** self.gamma) * ce

        valid = target != self.ignore_index
        focal = focal * valid.float()
        denom = valid.sum().clamp(min=1)
        return focal.sum() / denom
