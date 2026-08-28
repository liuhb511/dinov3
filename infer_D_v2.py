"""
infer_adaptive.py  ——  DINOv3 + DecoderV3 推理（双模式滑动窗口）
══════════════════════════════════════════════════════
所有配置在本文件顶部 INFER 字典中，零外部 config 依赖。

滑动窗口模式（通过 stride 控制）：
  stride > 0  → 固定步长模式（与 infer_new.py 相同）
                 窗口大小固定，步长 = stride
                 边缘不足补灰色

  stride = 0  → 自适应均匀分布模式
                 首窗口贴左上角，末窗口贴右下角
                 中间均匀分布，自动重叠
                 边缘窗口内容完整，无需补灰
══════════════════════════════════════════════════════
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import torch
import time
import math
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有推理配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/BG_test/testset",
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images",


    # "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/test_HQL",   #  哈汽轮 测试集
    # "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/test_BG",   #  宝钢 测试集
    # "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/20260630135631",   #  莱钢 测试集
    # "test_image_dir": "F:/liuhaibo/datasets/LG_JZW/LG_test",   #  莱钢 测试集


    # "test_image_dir": "F:/liuhaibo/datasets/test/JZW/DHTG/total",   #  东海特钢 测试集
    # "test_image_dir": "F:/liuhaibo/datasets/test/JZW/HQL_0825/1",   #  莱钢 测试集
    "test_image_dir": "F:/liuhaibo/datasets/test/HMS/test_images", 


    # ========== 模型结构（必须与训练时一致）==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,
    "num_classes":     5,            # 4(D类) / 5(ABCD) / 7(ABC) / 5(D+TIN-D) / 9(ABC+TIN-B+TIN-C) / 9(ABC+TIN)

    # ========== 权重加载 ==========
    "checkpoint_dir":  "./checkpoints/HMS_v2",             # 点状（D）类    
    # "checkpoint_dir":  "./checkpoints/ABCTIN1024",                # 条状（ABC + 氮化钛）
    # "checkpoint_dir":  "./checkpoints/ABCTIN1024_gray",          
    "checkpoint_name": "best_iou",                              # best_iou / best_dice / last / 完整路径
    "weight_source":   "auto",                                  # "teacher"(EMA) / "student" / "auto"

    # ========== 滑动窗口模式 ==========
    # stride > 0 → 固定步长（窗口不重叠）
    # stride = 0 → 自适应均匀分布（窗口有重叠，首尾贴边）
    "crop_size":         1024,                                   # 滑动窗口大小
    "stride":            0,                                     # 0=自适应模式, >0=固定步长
    "confidence_threshold": 0.0,                                # 最大类别概率低于该值时设为背景0

    # ========== 输出 ==========
    # "output_dir":     "./output/infer_BG",                    # 宝钢 输出路径
    # "output_dir":     "./output/infer_DHTG/DHTG16",                     # 哈汽轮 输出路径
    # "output_dir":     "./output/infer_LG/436_gray",                    # 莱钢 输出路径
    # "output_dir":     "./output/infer_HQL",                    # 莱钢 输出路径
    "output_dir":     "./output/infer_HMS",                    # 莱钢 输出路径
    
    # "output_dir":     "F:/liuhaibo/datasets/test/JZW/HQL_0825/output_dino",                    # 莱钢 输出路径

    "output_subdir":  "HMS_v2",                   # 输出子路径
    "save_confidence": False,                                                # 是否保存置信度图

    # ========== 设备 / 性能统计 ==========
    "device":        "cuda",
    "warmup_images": 3,
}

# ---- 构建模型构造函数所需的最小 cfg ----
_model_cfg = SimpleNamespace(
    backbone_name   = INFER["backbone_name"],
    freeze_backbone = INFER["freeze_backbone"],
    num_classes     = INFER["num_classes"],
)

# ---- 解包常用参数 ----
num_classes        = INFER["num_classes"]
crop_size          = INFER["crop_size"]
stride_len         = INFER["stride"]
confidence_threshold = INFER["confidence_threshold"]

if not 0.0 <= confidence_threshold <= 1.0:
    raise ValueError("confidence_threshold 必须在 0.0 到 1.0 之间")

# ╔══════════════════════════════════════════════════════════════╗
# ║                        模块导入                               ║
# ╚══════════════════════════════════════════════════════════════╝
from models.dinov3_segmentation import DINOv3Seg

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
    model_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    checkpoint_bytes = os.path.getsize(checkpoint_path)
    print("=" * 70)
    print(f"参数量：{total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"模型参数大小：{model_bytes / (1024 ** 2):.2f} MB")
    print(f"权重文件大小：{checkpoint_bytes / (1024 ** 2):.2f} MB")
    print("=" * 70)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        模型加载                               ║
# ╚══════════════════════════════════════════════════════════════╝
def load_model(weight_path, device, weight_source="teacher"):
    """加载推理模型。weight_source: teacher / student / auto"""
    model = DINOv3Seg(_model_cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)

    priority = (["teacher", "student", "model"] if weight_source in ("teacher", "auto")
                else ["student", "teacher", "model"])

    loaded_source = None
    for key in priority:
        if key in ckpt:
            loaded_source = key
            model.load_state_dict(ckpt[key], strict=True)
            break

    if loaded_source is None:
        raise KeyError(f"Checkpoint keys={list(ckpt.keys())}, expected one of {priority}")

    model.eval()
    print(f"Loaded: {weight_path}")
    print(f"  source={loaded_source}, epoch={ckpt.get('epoch','?')}, "
          f"IoU={ckpt.get('iou',0):.4f}, Dice={ckpt.get('dice',0):.4f}")
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║                       推理核心函数                             ║
# ╚══════════════════════════════════════════════════════════════╝
def _predict_probabilities(model, batch):
    """单次前向推理，返回各类别概率：[N,C,H,W]。"""
    with torch.no_grad():
        seg, _ = model(batch)
        probs = torch.softmax(seg, dim=1)
    return probs.float().cpu().numpy()


def _preprocess_batch(images, device):
    """images: [N,H,W,3] RGB"""
    images = images.astype(np.float32) / 255.0
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).half().to(device)
    return (batch - MEAN.to(device)) / STD.to(device)


# ============================================================
# 两种窗口位置生成策略
# ============================================================

def _sliding_positions(length, window_size):
    """固定步长模式：从0开始，步长=window_size，窗口不重叠。"""
    if length <= 0:  raise ValueError(f"length={length} <= 0")
    if window_size <= 0: raise ValueError(f"window_size={window_size} <= 0")
    return list(range(0, length, window_size))


def _adaptive_positions(length, window_size):
    """
    自适应均匀分布模式。

    规则：
    1. 第一个窗口起点 = 0，紧贴左/上边界
    2. 最后一个窗口终点 = length，紧贴右/下边界
    3. 中间窗口均匀分布，步长 ≤ window_size（保证重叠）
    4. 若 length ≤ window_size，只有一个窗口
    """
    if length <= window_size:
        return [0]

    span = length - window_size
    # 步长 ≤ window_size 所需的最少窗口数
    n = math.ceil(span / window_size) + 1
    n = max(n, 2)

    step = span / (n - 1)
    xs = [round(i * step) for i in range(n - 1)]
    xs.append(span)

    # 去重：浮点误差导致相邻两点差<2时合并
    uniq = [xs[0]]
    for x in xs[1:]:
        if x - uniq[-1] >= 2:
            uniq.append(x)
        else:
            uniq[-1] = max(uniq[-1], x)
    return uniq


def sliding_window_infer(model, image, device, window_size, stride):
    """
    双模式滑窗推理。

    stride > 0 → 固定步长（窗口不重叠，边缘补灰）
    stride = 0 → 自适应均匀分布（窗口重叠，首尾贴边）
    """
    height, width = image.shape[:2]

    # 根据 stride 选择窗口模式
    if stride > 0:
        xs = _sliding_positions(width,  window_size)
        ys = _sliding_positions(height, window_size)
        mode_name = "fixed stride"
    else:
        xs = _adaptive_positions(width,  window_size)
        ys = _adaptive_positions(height, window_size)
        mode_name = "adaptive"

    # print(f"  Mode: {mode_name}  |  "
    #       f"Grid: {len(ys)} rows x {len(xs)} cols  |  "
    #       f"Image: {width}x{height}")

    probability_map = np.zeros((num_classes, height, width), dtype=np.float32)

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
            probs = _predict_probabilities(model, batch)[0]

            assert probs.shape[0] == num_classes, f"channels: {probs.shape[0]} vs {num_classes}"
            probability_map[:, y1:y2, x1:x2] = probs[:, :ph, :pw]

    return probability_map


def infer_single(model, image_path, device):
    """单图推理 → [H,W] uint8 index_mask + [H,W] float32 confidence"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    oh, ow = image.shape[:2]

    probs = sliding_window_infer(model, image, device, window_size=crop_size, stride=stride_len)
    confidence = np.max(probs, axis=0).astype(np.float32)
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)

    # 最大类别概率低于阈值的像素作为不确定区域，统一设为背景类别0。
    index_mask[confidence < confidence_threshold] = 0
    return index_mask, confidence


def save_result(index_mask, confidence, save_path, save_confidence=False):
    """保存单通道类别索引 PNG"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    index_path = save_path + ".png"
    if not cv2.imwrite(index_path, index_mask):
        raise RuntimeError(f"Cannot save: {index_path}")
    values, counts = np.unique(index_mask, return_counts=True)
    # print(f"Saved: {index_path} | classes={dict(zip(values.tolist(), counts.tolist()))}")
    if save_confidence:
        conf_img = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path + "_confidence.png", conf_img)


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    device = torch.device(INFER["device"] if torch.cuda.is_available() else "cpu")

    # --- 解析 checkpoint 路径 ---
    ckpt_path = INFER["checkpoint_name"]
    sep = os.sep
    if sep not in ckpt_path and "/" not in ckpt_path and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(INFER["checkpoint_dir"], INFER["checkpoint_name"] + ".pth")
        if not os.path.exists(ckpt_path):
            for fallback in ("best_dice.pth", "best_iou.pth", "last.pth"):
                candidate = os.path.join(INFER["checkpoint_dir"], fallback)
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"  checkpoint_dir:  {INFER['checkpoint_dir']}")
        print(f"  checkpoint_name: {INFER['checkpoint_name']}")
        exit(1)

    model = load_model(ckpt_path, device, INFER["weight_source"])
    _ = MEAN.to(device); _ = STD.to(device)

    mode_label = "adaptive" if stride_len == 0 else f"fixed stride={stride_len}"
    print(f"num_classes={num_classes}  crop_size={crop_size}  "
          f"mode={mode_label}  confidence_threshold={confidence_threshold}")
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
        # print(f"Infer: {name}")

        synchronize_device(device)
        t0 = time.perf_counter()
        index_mask, confidence = infer_single(model, path, device)
        synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= INFER["warmup_images"]:
            measured_times.append(elapsed_ms)

        save_result(index_mask, confidence,
                    os.path.join(INFER["output_dir"], INFER["output_subdir"],
                                 os.path.splitext(name)[0]),
                    save_confidence=INFER["save_confidence"])

        warmup = " [预热]" if idx < INFER["warmup_images"] else ""
        print(f"[{idx + 1}/{len(image_names)}] Infer: {name} Time: {elapsed_ms:.2f} ms{warmup} save to: {INFER['output_subdir']}")

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