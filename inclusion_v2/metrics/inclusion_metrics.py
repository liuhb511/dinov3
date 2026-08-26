"""
inclusion_v2/metrics/inclusion_metrics.py

业务评估指标（12 类统一空间）：
- Inclusion Precision / Recall / F1（最高优先级，最终只展示夹杂物）
- 各类夹杂物 Precision / Recall（A/B/C/D/TINB-C/TIND）
- 关键误识别路径 FP Rate：
    HH→A/C 、HC→D 、SZ→D 、Bg→D
- ABC over-segmentation ratio 暂以"ABC 相对 GT 的面积比"近似（可选）

实现：累积 12×12 混淆矩阵，最后统一计算。
"""
import torch

from ..data.label_mapping import (
    NUM_CLASSES_UNIFIED,
    INCLUSION_CLASSES,
    INCLUSION_CLASS_NAMES,
)


class InclusionMetricsAccumulator:
    def __init__(self, num_classes: int = NUM_CLASSES_UNIFIED,
                 device: torch.device = torch.device("cpu")):
        self.num_classes = num_classes
        self.cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, pred, target):
        """
        pred:   [B, H, W]  12 类预测索引
        target: [B, H, W]  12 类 GT
        """
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        valid = (
            (target >= 0) & (target < self.num_classes)
            & (pred >= 0) & (pred < self.num_classes)
        )
        target = target[valid]
        pred = pred[valid]
        encoded = target * self.num_classes + pred
        self.cm += torch.bincount(encoded, minlength=self.num_classes * self.num_classes
                                  ).reshape(self.num_classes, self.num_classes)

    def compute(self, eps: float = 1e-6):
        """返回指标 dict（所有数值为 Python float）。"""
        cm = self.cm.float()
        tp = torch.diag(cm)
        col_sum = cm.sum(dim=0)   # 预测为该类的像素
        row_sum = cm.sum(dim=1)   # GT 为该类的像素

        per_cls_precision = tp / (col_sum + eps)
        per_cls_recall = tp / (row_sum + eps)
        per_cls_f1 = 2.0 * per_cls_precision * per_cls_recall / (
            per_cls_precision + per_cls_recall + eps)

        # ---------- 聚合夹杂物 ----------
        inc = torch.tensor(INCLUSION_CLASSES, device=cm.device)
        tp_inc = tp[inc].sum()
        fp_inc = (col_sum[inc].sum() - tp_inc)      # 预测为夹杂物但 GT 不是
        fn_inc = (row_sum[inc].sum() - tp_inc)      # GT 是夹杂物但预测不是
        prec_inc = tp_inc / (tp_inc + fp_inc + eps)
        rec_inc = tp_inc / (tp_inc + fn_inc + eps)
        f1_inc = 2.0 * prec_inc * rec_inc / (prec_inc + rec_inc + eps)

        metrics = {
            "inclusion_precision": float(prec_inc),
            "inclusion_recall": float(rec_inc),
            "inclusion_f1": float(f1_inc),
            "tp_inc_px": float(tp_inc),
            "fp_inc_px": float(fp_inc),
            "fn_inc_px": float(fn_inc),
        }

        # ---------- 各类夹杂物 ----------
        for i, name in zip(INCLUSION_CLASSES, INCLUSION_CLASS_NAMES):
            metrics[f"p_{name}"] = float(per_cls_precision[i])
            metrics[f"r_{name}"] = float(per_cls_recall[i])
            metrics[f"f1_{name}"] = float(per_cls_f1[i])

        # ---------- 关键误识别路径 ----------
        # HH→A/C : GT=HH(7) 被预测为 A(1) 或 C(3)
        hh_ac = cm[7, 1] + cm[7, 3]
        metrics["hh_to_ac_fp_rate"] = float(hh_ac / (row_sum[7] + eps))
        # HC→D   : GT=HC(10) 被预测为 D(4)
        metrics["hc_to_d_fp_rate"] = float(cm[10, 4] / (row_sum[10] + eps))
        # SZ→D   : GT=SZ(11) 被预测为 D(4)
        metrics["sz_to_d_fp_rate"] = float(cm[11, 4] / (row_sum[11] + eps))
        # Bg→D   : GT=bg(0) 被预测为 D(4)
        metrics["bg_to_d_fp_rate"] = float(cm[0, 4] / (row_sum[0] + eps))
        metrics["bg_to_d_fp_px"] = float(cm[0, 4])

        return metrics

    def reset(self):
        self.cm.zero_()
