"""
v2_infer_inclusion.py —— 非金属夹杂物识别 MVP-1 推理入口
══════════════════════════════════════════════════════
- 1024 自适应首尾贴边重叠滑窗（stride=0 模式，逻辑同 infer_D_v2）
- 重叠区域采用 probability sum / count average（修复旧版覆盖式写入问题）
- 单次 DINOv3 前向；Gate 与专家概率软融合（P=expert×gate^α）
- 最终只输出夹杂物 {A,B,C,D,TINB/C,TIND}，噪声/背景置 0

所有配置在本文件顶部 INFER 字典中修改。
══════════════════════════════════════════════════════
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import time
import cv2
import numpy as np
import torch
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有推理配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    "test_image_dir": "F:/liuhaibo/datasets/test/JZW/HQL_0825/1",   # 改为你的测试图目录

    # ========== 模型 ==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,
    "encoder_layers":  (4, 8, 12),
    "feat_dim":        768,
    "fusion_dim":      512,
    "decoder_dim":     32,
    "checkpoint":      "./checkpoints/inclusion_v2_mvp1/best_inclusion_f1.pth",  # 权重路径

    # ========== 融合 ==========
    "fusion_alpha":     0.5,            # P=expert×gate^α

    # ========== 滑窗 ==========
    "crop_size":        1024,
    "stride":           0,              # 0=自适应重叠（首尾贴边），>0=固定步长
    "confidence_threshold": 0.0,        # 最大类别概率低于该值置背景0

    # ========== 输出 ==========
    "output_dir":       "./output/infer_inclusion_v2",
    "output_subdir":    "mask",
    "save_confidence":  False,          # 是否保存置信度图

    # ========== 设备 / 性能统计 ==========
    "device":           "cuda",
    "warmup_images":    3,
}

# ---- 模型 cfg ----
_model_cfg = SimpleNamespace(
    backbone_name   = INFER["backbone_name"],
    freeze_backbone = INFER["freeze_backbone"],
    encoder_layers  = INFER["encoder_layers"],
    feat_dim        = INFER["feat_dim"],
    fusion_dim      = INFER["fusion_dim"],
    decoder_dim     = INFER["decoder_dim"],
)

crop_size = INFER["crop_size"]
stride_len = INFER["stride"]
confidence_threshold = INFER["confidence_threshold"]
alpha = INFER["fusion_alpha"]

# ╔══════════════════════════════════════════════════════════════╗
# ║                        模块导入                               ║
# ╚══════════════════════════════════════════════════════════════╝
from inclusion_v2.models import InclusionDualExpertNet
from inclusion_v2.utils.output_fusion import fuse_outputs
from inclusion_v2.data.label_mapping import NUM_CLASSES_UNIFIED, INCLUSION_CLASSES


# FP16 预计算常量
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)
GRAY = (128, 128, 128)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        工具函数                               ║
# ╚══════════════════════════════════════════════════════════════╝
def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def print_model_info(model, checkpoint_path):
    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    checkpoint_bytes = os.path.getsize(checkpoint_path)
    print("=" * 70)
    print(f"参数量：{total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"模型参数大小：{model_bytes / (1024 ** 2):.2f} MB")
    print(f"权重文件大小：{checkpoint_bytes / (1024 ** 2):.2f} MB")
    print("=" * 70)


def load_model(weight_path, device):
    """加载 MVP-1 双专家模型（checkpoint 内 'model' 键）。"""
    model = InclusionDualExpertNet(_model_cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        info = ckpt
    else:
        # 兼容直接存 state_dict 的权重
        model.load_state_dict(ckpt, strict=True)
        info = {}
    model.eval()
    print(f"Loaded: {weight_path}")
    if info:
        vm = info.get("val_metrics", {})
        prec = vm.get("inclusion_precision", "?")
        print(f"  epoch={info.get('epoch', '?')}, inclusion_precision={prec}")
    return model


def _preprocess_batch(images, device):
    """images: [N,H,W,3] RGB -> [N,3,H,W] half 归一化。"""
    images = images.astype(np.float32) / 255.0
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).half().to(device)
    return (batch - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def _predict_probabilities(model, batch, alpha):
    """单次前向 + 软 Gate 融合 -> [N,12,H,W] float32。"""
    outputs = model(batch)
    probs = fuse_outputs(outputs["gate"], outputs["strip"], outputs["point"], alpha=alpha)
    return probs.float().cpu().numpy()


def _adaptive_positions(length, window_size):
    """
    自适应均匀分布：首窗口贴左上角，末窗口贴右下角，中间均匀分布，自动重叠。
    返回窗口起始位置列表。
    """
    if length <= window_size:
        return [0]
    span = length - window_size
    n = math.ceil(span / window_size) + 1
    n = max(n, 2)
    step = span / (n - 1)
    xs = [round(i * step) for i in range(n - 1)]
    xs.append(span)
    # 去重：浮点误差导致相邻两点差 < 2 时合并
    uniq = [xs[0]]
    for x in xs[1:]:
        if x - uniq[-1] >= 2:
            uniq.append(x)
        else:
            uniq[-1] = max(uniq[-1], x)
    return uniq


def _sliding_positions(length, window_size):
    if length <= 0 or window_size <= 0:
        raise ValueError("length/window_size 必须为正")
    return list(range(0, length, window_size))


def sliding_window_infer(model, image, device, window_size, stride, alpha):
    """
    双模式滑窗推理：
      stride > 0 -> 固定步长（窗口可能重叠/不重叠，边缘补灰）
      stride = 0 -> 自适应均匀分布（首尾贴边，窗口内容完整）
    重叠区域使用 probability sum / count average。
    返回 [12, H, W] float32 概率图。
    """
    height, width = image.shape[:2]

    if stride > 0:
        xs = _sliding_positions(width, window_size)
        ys = _sliding_positions(height, window_size)
    else:
        xs = _adaptive_positions(width, window_size)
        ys = _adaptive_positions(height, window_size)

    prob_sum = np.zeros((NUM_CLASSES_UNIFIED, height, width), dtype=np.float64)
    prob_cnt = np.zeros((height, width), dtype=np.float32)

    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + window_size, width)
            y2 = min(y1 + window_size, height)
            patch = image[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]

            if ph != window_size or pw != window_size:
                padded = np.full((window_size, window_size, 3), GRAY, dtype=image.dtype)
                padded[:ph, :pw] = patch
                patch = padded

            batch = _preprocess_batch(np.expand_dims(patch, axis=0), device)
            probs = _predict_probabilities(model, batch, alpha)[0]  # [12, ws, ws]

            prob_sum[:, y1:y2, x1:x2] += probs[:, :ph, :pw]
            prob_cnt[y1:y2, x1:x2] += 1.0

    probs = prob_sum / np.maximum(prob_cnt, 1.0)[None, :, :]
    return probs.astype(np.float32)


def infer_single(model, image_path, device):
    """
    单图推理：
      probs12: [H,W] float32（融合后 12 类概率，取最大作为置信度）
      index_mask: [H,W] uint8（12 类索引，阈值过滤后置 0）
      display_mask: [H,W] uint8（只保留夹杂物类别）
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    probs = sliding_window_infer(model, image, device, window_size=crop_size,
                                 stride=stride_len, alpha=alpha)

    confidence = np.max(probs, axis=0).astype(np.float32)
    pred = np.argmax(probs, axis=0).astype(np.uint8)
    if confidence_threshold > 0:
        pred[confidence < confidence_threshold] = 0
    display = np.zeros_like(pred)
    for c in INCLUSION_CLASSES:        # 只保留夹杂物类别
        display[pred == c] = c
    return pred, confidence, display


def save_result(display_mask, confidence, save_path, save_confidence=False):
    """保存只含夹杂物的展示 mask + 可选置信度图。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mask_path = save_path + ".png"
    if not cv2.imwrite(mask_path, display_mask):
        raise RuntimeError(f"Cannot save: {mask_path}")
    values, counts = np.unique(display_mask, return_counts=True)
    print(f"Saved: {mask_path} | classes={dict(zip(values.tolist(), counts.tolist()))}")
    if save_confidence:
        conf_img = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path + "_confidence.png", conf_img)


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    device = torch.device(INFER["device"] if torch.cuda.is_available() else "cpu")

    # --- 解析 checkpoint 路径 ---
    ckpt_path = INFER["checkpoint"]
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        exit(1)

    model = load_model(ckpt_path, device)
    _ = MEAN.to(device)
    _ = STD.to(device)

    mode_label = "adaptive(overlap)" if stride_len == 0 else f"fixed stride={stride_len}"
    print(f"crop_size={crop_size}  mode={mode_label}  alpha={alpha}  "
          f"confidence_threshold={confidence_threshold}")
    print_model_info(model, ckpt_path)

    # --- 测试目录 ---
    test_dir = INFER["test_image_dir"]
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        exit(1)

    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    image_names = sorted(n for n in os.listdir(test_dir) if n.lower().endswith(valid_exts))
    if not image_names:
        print(f"No images found in: {test_dir}")
        exit(1)

    print(f"Total images: {len(image_names)}")
    print("=" * 70)

    measured_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for idx, name in enumerate(image_names):
        path = os.path.join(test_dir, name)
        synchronize_device(device)
        t0 = time.perf_counter()
        pred, confidence, display = infer_single(model, path, device)
        synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= INFER["warmup_images"]:
            measured_times.append(elapsed_ms)

        save_result(display, confidence,
                    os.path.join(INFER["output_dir"], INFER["output_subdir"],
                                 os.path.splitext(name)[0]),
                    save_confidence=INFER["save_confidence"])

        warmup = " [预热]" if idx < INFER["warmup_images"] else ""
        print(f"[{idx + 1}/{len(image_names)}] {name}  Time: {elapsed_ms:.2f} ms{warmup}")

    # --- 统计 ---
    print("=" * 70)
    if measured_times:
        avg_ms = float(np.mean(measured_times))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print(f"有效统计图片数：{len(measured_times)}")
        print(f"平均单图推理时间：{avg_ms:.2f} ms")
        print(f"平均推理速度：{fps:.2f} FPS")
    else:
        print("有效统计图片数不足，请减少 warmup_images。")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU 峰值显存：{peak_mb:.2f} MB")

    print("=" * 70)
    print("Done.")



