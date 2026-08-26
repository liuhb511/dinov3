"""
inclusion_v2/utils/output_fusion.py

Gate 与专家概率合成（Q7 软 Gate）：

    P(cls) = P_expert(cls) * P_gate(group)^α        (cls = 1..11)
    P(bg)  = max( gate_bg,
                  strip_bg * gate_strip^α,
                  point_bg * gate_point^α )

- gate_bg 不乘幂次，作为背景的权威概率（α=0 时各类退化为纯专家概率，
  背景退化为 max(gate_bg, strip_bg, point_bg)，行为一致）。
- 最终展示只保留夹杂物类别 {A,B,C,D,TINB/C,TIND}。
"""
import torch
import torch.nn.functional as F

from ..data.label_mapping import (
    NUM_CLASSES_UNIFIED,
    STRIP_HEAD_TO_UNIFIED,
    POINT_HEAD_TO_UNIFIED,
    INCLUSION_CLASSES,
)


def fuse_outputs(gate_logits, strip_logits, point_logits, alpha: float = 0.5):
    """
    gate_logits:  [B, 3, H, W]   (bg / strip / point)
    strip_logits: [B, 9, H, W]   (bg, A, B, C, TINB/C, TIND, HH, XW, XQL)
    point_logits: [B, 4, H, W]   (bg, D, HC, SZ)

    Returns:
        probs: [B, 12, H, W]  unified 12 类概率
    """
    B, _, H, W = gate_logits.shape
    gate = F.softmax(gate_logits, dim=1)
    strip = F.softmax(strip_logits, dim=1)
    point = F.softmax(point_logits, dim=1)

    probs = torch.zeros(B, NUM_CLASSES_UNIFIED, H, W,
                        dtype=gate_logits.dtype, device=gate_logits.device)

    g_strip = gate[:, 1].pow(alpha)
    g_point = gate[:, 2].pow(alpha)

    # 背景：三个来源取最大
    bg_sources = [
        gate[:, 0],
        strip[:, 0] * g_strip,
        point[:, 0] * g_point,
    ]
    probs[:, 0] = torch.maximum(torch.maximum(bg_sources[0], bg_sources[1]), bg_sources[2])

    # 条状类
    for head_i, uni in enumerate(STRIP_HEAD_TO_UNIFIED):
        if uni == 0:
            continue
        probs[:, uni] = strip[:, head_i] * g_strip

    # 点状类
    for head_i, uni in enumerate(POINT_HEAD_TO_UNIFIED):
        if uni == 0:
            continue
        probs[:, uni] = point[:, head_i] * g_point

    return probs


def fused_to_index_mask(probs, confidence_threshold: float = 0.0):
    """
    probs: [B, 12, H, W]
    返回:
        index_mask: [B, H, W]（类别 0..11；最大概率低于阈值置 0）
    """
    confidence, pred = torch.max(probs, dim=1)
    if confidence_threshold > 0:
        pred = pred.masked_fill(confidence < confidence_threshold, 0)
    return pred


def display_mask_from_pred(pred):
    """
    只保留夹杂物类别 {A,B,C,D,TINB/C,TIND}，其余（bg/噪声）置 0。
    pred: [..., H, W]（12 类索引）
    """
    import torch
    keep = torch.zeros_like(pred)
    for c in INCLUSION_CLASSES:
        keep[pred == c] = c
    return keep
