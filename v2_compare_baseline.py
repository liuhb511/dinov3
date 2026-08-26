"""
v2_compare_baseline.py —— 新模型(MVP-1) vs 旧双模型(ABCTIN_1024_v2 + D_1024_v2)
══════════════════════════════════════════════════════
- 同一测试集、同一 1024 自适应重叠滑窗（sum/count 平均）
- 对比：业务 Inclusion Precision/Recall/F1 + 关键误检率 + 总推理时间
- 旧基线合并逻辑同 batch_infer_onnx.py：D 模型先上色，strip 模型非背景区域覆盖

如果测试目录同时有 GT mask（同目录 mask 子目录或同前缀 .png），会自动计算指标；
否则只统计推理时间。

所有配置在本文件顶部 COMPARE 字典中修改。
══════════════════════════════════════════════════════
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import math
import cv2
import numpy as np
import torch
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有对比配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
COMPARE = {
    # ========== 测试数据 ==========
    "test_image_dir": "F:/liuhaibo/datasets/test/JZW/HQL_0825/1",   # 测试图目录
    "test_mask_dir":  "",   # GT mask 目录（unified 12 类）；空 = 不计算指标

    # ========== 新模型（MVP-1）==========
    "new_checkpoint": "./checkpoints/inclusion_v2_mvp1/best_inclusion_f1.pth",
    "new_alpha":      0.5,

    # ========== 旧双模型（基线）==========
    "strip_checkpoint": "./checkpoints/ABCTIN_1024_v2/best_iou.pth",  # 9 类条状模型
    "point_checkpoint": "./checkpoints/D_1024_v2/best_iou.pth",       # 4 类点状模型

    # ========== 滑窗 ==========
    "crop_size": 1024,
    "stride":    0,        # 0=自适应重叠（首尾贴边）

    # ========== 输出 ==========
    "output_dir": "./output/compare_inclusion_v2",

    # ========== 设备 ==========
    "device": "cuda",
    "warmup_images": 3,
}

crop_size = COMPARE["crop_size"]
stride_len = COMPARE["stride"]

# 旧模型类别 -> unified 12 类映射
# strip 模型(9): 0=bg,1=A,2=B,3=C,4=HH,5=XW,6=XQL,7=TINBC,8=TIND
STRIP_OLD_TO_UNIFIED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 7, 5: 8, 6: 9, 7: 5, 8: 6}
# D 模型(4): 0=bg,1=D,2=HC,3=SZ
POINT_OLD_TO_UNIFIED = {0: 0, 1: 4, 2: 10, 3: 11}

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)
GRAY = (128, 128, 128)

# ╔══════════════════════════════════════════════════════════════╗
# ║                        模块导入                               ║
# ╚══════════════════════════════════════════════════════════════╝
from models.dinov3_segmentation import DINOv3Seg                       # 旧模型（只读）
from inclusion_v2.models import InclusionDualExpertNet                 # 新模型
from inclusion_v2.utils.output_fusion import fuse_outputs
from inclusion_v2.metrics import InclusionMetricsAccumulator


def _preprocess(images, device):
    images = images.astype(np.float32) / 255.0
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).half().to(device)
    return (batch - MEAN.to(device)) / STD.to(device)


def _adaptive_positions(length, window_size):
    if length <= window_size:
        return [0]
    span = length - window_size
    n = max(math.ceil(span / window_size) + 1, 2)
    step = span / (n - 1)
    xs = [round(i * step) for i in range(n - 1)] + [span]
    uniq = [xs[0]]
    for x in xs[1:]:
        if x - uniq[-1] >= 2:
            uniq.append(x)
        else:
            uniq[-1] = max(uniq[-1], x)
    return uniq


def _positions(length, window_size):
    return _adaptive_positions(length, window_size) if stride_len == 0 \
        else list(range(0, length, window_size))


def _sliding_probs(predict_fn, image, height, width):
    """通用滑窗：predict_fn(patch[B,3,ws,ws]) -> [12, ws, ws]。"""
    xs = _positions(width, crop_size)
    ys = _positions(height, crop_size)
    prob_sum = np.zeros((12, height, width), dtype=np.float64)
    prob_cnt = np.zeros((height, width), dtype=np.float32)
    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + crop_size, width)
            y2 = min(y1 + crop_size, height)
            patch = image[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]
            if ph != crop_size or pw != crop_size:
                padded = np.full((crop_size, crop_size, 3), GRAY, dtype=image.dtype)
                padded[:ph, :pw] = patch
                patch = padded
            probs = predict_fn(patch)               # [12, ws, ws]
            prob_sum[:, y1:y2, x1:x2] += probs[:, :ph, :pw]
            prob_cnt[y1:y2, x1:x2] += 1.0
    return (prob_sum / np.maximum(prob_cnt, 1.0)[None, :, :]).astype(np.float32)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        模型加载                               ║
# ╚══════════════════════════════════════════════════════════════╝
def load_new_model(weight_path, device):
    cfg = SimpleNamespace(
        backbone_name="dinov3_model", freeze_backbone=True,
        encoder_layers=(4, 8, 12), feat_dim=768, fusion_dim=512, decoder_dim=32,
    )
    model = InclusionDualExpertNet(cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def load_old_model(weight_path, num_classes, device):
    cfg = SimpleNamespace(
        backbone_name="dinov3_model", freeze_backbone=True, num_classes=num_classes,
    )
    model = DINOv3Seg(cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    for key in ("teacher", "student", "model"):
        if key in ckpt:
            model.load_state_dict(ckpt[key], strict=True)
            print(f"  old model({num_classes}cls) loaded from '{key}': {weight_path}")
            break
    else:
        model.load_state_dict(ckpt, strict=True)
        print(f"  old model({num_classes}cls) loaded (raw state_dict): {weight_path}")
    model.eval()
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║                        推理函数                               ║
# ╚══════════════════════════════════════════════════════════════╝
@torch.no_grad()
def predict_new(model, patch, device, alpha):
    """新模型：patch [H,W,3] -> [12,H,W] 融合概率。"""
    batch = _preprocess(np.expand_dims(patch, 0), device)
    out = model(batch)
    probs = fuse_outputs(out["gate"], out["strip"], out["point"], alpha=alpha)
    return probs.float().cpu().numpy()[0]


@torch.no_grad()
def predict_old(model, patch, device, to_unified_map, num_classes):
    """旧模型：patch [H,W,3] -> [12,H,W] 概率（映射到 unified 12 类）。"""
    batch = _preprocess(np.expand_dims(patch, 0), device)
    seg, _ = model(batch)
    probs = torch.softmax(seg, dim=1).float().cpu().numpy()[0]      # [C,H,W]
    out = np.zeros((12, probs.shape[1], probs.shape[2]), dtype=np.float32)
    for head_c, uni_c in to_unified_map.items():
        out[uni_c] += probs[head_c]
    return out


def merge_old_baseline(strip_probs, point_probs):
    """
    旧双模型合并（batch_infer_onnx 逻辑）：
    D 模型先上色，strip 模型非背景区域覆盖。
    strip_probs/point_probs: [12, H, W]
    """
    strip_pred = np.argmax(strip_probs, axis=0)
    point_pred = np.argmax(point_probs, axis=0)
    # 只有 strip/point 的 foreground 区域有效（各自映射后 >= 1 的类别）
    merged = np.zeros_like(strip_pred)
    # D 模型前景（D=4, HC=10, SZ=11）
    d_fg = (point_pred >= 1)
    merged[d_fg] = point_pred[d_fg]
    # strip 模型非背景覆盖
    s_fg = (strip_pred >= 1)
    merged[s_fg] = strip_pred[s_fg]
    return merged


def display_only(merged_pred):
    """只保留夹杂物类别。"""
    display = np.zeros_like(merged_pred)
    for c in (1, 2, 3, 4, 5, 6):
        display[merged_pred == c] = c
    return display


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    device = torch.device(COMPARE["device"] if torch.cuda.is_available() else "cpu")

    for tag, p in [("new", COMPARE["new_checkpoint"]),
                   ("strip", COMPARE["strip_checkpoint"]),
                   ("point", COMPARE["point_checkpoint"])]:
        if not os.path.exists(p):
            print(f"[{tag}] Checkpoint not found: {p}")
            exit(1)

    print("Loading models ...")
    model_new = load_new_model(COMPARE["new_checkpoint"], device)
    model_strip = load_old_model(COMPARE["strip_checkpoint"], 9, device)
    model_point = load_old_model(COMPARE["point_checkpoint"], 4, device)
    _ = MEAN.to(device)
    _ = STD.to(device)

    test_dir = COMPARE["test_image_dir"]
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        exit(1)

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    image_names = sorted(n for n in os.listdir(test_dir) if n.lower().endswith(valid_exts))
    if not image_names:
        print(f"No images found: {test_dir}")
        exit(1)

    has_gt = bool(COMPARE["test_mask_dir"]) and os.path.isdir(COMPARE["test_mask_dir"])
    acc_new = InclusionMetricsAccumulator()
    acc_old = InclusionMetricsAccumulator()

    out_dir = COMPARE["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    times_new, times_old = [], []
    print(f"Total images: {len(image_names)}  |  has_gt={has_gt}")

    for idx, name in enumerate(image_names):
        path = os.path.join(test_dir, name)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable: {path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # ---- 新模型 ----
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        probs_new = _sliding_probs(lambda p: predict_new(model_new, p, device, COMPARE["new_alpha"]),
                                   image, h, w)
        pred_new = display_only(np.argmax(probs_new, axis=0))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_new = (time.perf_counter() - t0) * 1000.0

        # ---- 旧双模型 ----
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        strip_probs = _sliding_probs(lambda p: predict_old(model_strip, p, device,
                                                           STRIP_OLD_TO_UNIFIED, 9), image, h, w)
        point_probs = _sliding_probs(lambda p: predict_old(model_point, p, device,
                                                           POINT_OLD_TO_UNIFIED, 4), image, h, w)
        merged_old = merge_old_baseline(strip_probs, point_probs)
        pred_old = display_only(merged_old)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_old = (time.perf_counter() - t0) * 1000.0

        if idx >= COMPARE["warmup_images"]:
            times_new.append(t_new)
            times_old.append(t_old)

        # 保存展示结果
        base = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(out_dir, f"new_{base}.png"), pred_new)
        cv2.imwrite(os.path.join(out_dir, f"old_{base}.png"), pred_old)

        # 指标
        if has_gt:
            gt_path = os.path.join(COMPARE["test_mask_dir"], base + ".png")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(COMPARE["test_mask_dir"], name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                acc_new.update(torch.from_numpy(pred_new.astype(np.int64)),
                               torch.from_numpy(gt.astype(np.int64)))
                acc_old.update(torch.from_numpy(pred_old.astype(np.int64)),
                               torch.from_numpy(gt.astype(np.int64)))

        print(f"[{idx + 1}/{len(image_names)}] {name}  new={t_new:.0f}ms  old={t_old:.0f}ms")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    if times_new and times_old:
        print(f"平均单图推理时间（不含预热）：")
        print(f"  新模型(单次DINO): {np.mean(times_new):.2f} ms")
        print(f"  旧双模型(串行)  : {np.mean(times_old):.2f} ms")
        print(f"  加速比          : {np.mean(times_old) / np.mean(times_new):.2f}×")
    else:
        print("预热图片数不足，未统计推理时间。")

    if has_gt:
        print("\n" + "=" * 70)
        m_new = acc_new.compute()
        m_old = acc_old.compute()
        keys = ["inclusion_precision", "inclusion_recall", "inclusion_f1",
                "hh_to_ac_fp_rate", "hc_to_d_fp_rate", "sz_to_d_fp_rate", "bg_to_d_fp_rate"]
        print(f"{'指标':<24}{'新模型':>12}{'旧双模型':>12}")
        for k in keys:
            print(f"{k:<24}{m_new[k]:>12.4f}{m_old[k]:>12.4f}")
        print("\n各类夹杂物 Precision/Recall：")
        for k in ["A", "B", "C", "D", "TINB/C", "TIND"]:
            print(f"  {k:<8} P: {m_new['p_' + k]:.4f}/{m_old['p_' + k]:.4f}   "
                  f"R: {m_new['r_' + k]:.4f}/{m_old['r_' + k]:.4f}   (new/old)")
    print("=" * 70)
    print("Done.")


