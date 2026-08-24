"""
infer_new.py  ——  DINOv3 + DecoderV3 推理
══════════════════════════════════════════════════════
所有配置在本文件顶部 INFER 字典中，不再依赖外部 config 文件。
FP16 + TTA 水平翻转 + 滑动窗口推理 + 形态学后处理
══════════════════════════════════════════════════════
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import torch
import time
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有推理配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/BG_test/testset", # 宝钢测试集
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images", # 哈汽轮测试集
    "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/20260630135631",  # 莱钢测试集

    # ========== 模型结构（必须与训练时一致）==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,
    "num_classes":     7,            # 4(D类) / 5(ABCD) / 7(ABC)

    # ========== 权重加载 ==========
    # "checkpoint_dir":  "./checkpoints/D_newdata",  # D类检测模型
    # "checkpoint_dir":  "./checkpoints/ABC_norotate",  # ABC类检测模型
    "checkpoint_dir":  "./checkpoints/BG_HQL_JZW_ABC",  # ABC类检测模型
    "checkpoint_name": "best_iou",   # best_iou / best_dice / last / 完整路径
    "weight_source":   "auto",    # "teacher"(EMA) / "student" / "auto"

    # ========== 推理参数 ==========
    "crop_size":         784,        # 滑动窗口大小
    "stride":            784,        # 步长（=crop_size 无重叠，<crop_size 有重叠）
    "infer_close_kernel": 0,         # 形态学闭运算核，0=禁用

    # ========== 输出 ==========
    # "output_dir":     "./output/infer_BG",  # 宝钢 输出路径
    # "output_dir":     "./output/infer_HQL",  # 哈汽轮 输出路径
    "output_dir":     "./output/infer_LG",  # 莱钢 输出路径
    "output_subdir":  "ABC_1",  # 输出子路径
    "save_confidence": False,

    # ========== 设备 / 性能统计 ==========
    "device":        "cuda",
    "warmup_images": 3,             # 预热图片数，不计入平均时间
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
infer_close_kernel = INFER["infer_close_kernel"]

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
# ║                      形态学后处理                              ║
# ╚══════════════════════════════════════════════════════════════╝
def enhance_connectivity(mask, num_classes, kernel_size=3):
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = mask.copy()
    original_foreground = mask > 0
    for class_id in range(1, num_classes):
        binary = (mask == class_id).astype(np.uint8)
        if not np.any(binary):
            continue
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        new_region = (closed > 0) & (~original_foreground) & (result == 0)
        result[new_region] = class_id
    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║                       推理核心函数                             ║
# ╚══════════════════════════════════════════════════════════════╝
def _predict_probabilities(model, batch):
    """TTA 水平翻转。Returns: [N,C,H,W]"""
    with torch.no_grad():
        seg, _ = model(batch)
        probs = torch.softmax(seg, dim=1)
        batch_flip = torch.flip(batch, dims=[3])
        seg_flip, _ = model(batch_flip)
        probs_flip = torch.softmax(seg_flip, dim=1)
        probs_flip = torch.flip(probs_flip, dims=[3])
        probs = (probs + probs_flip) / 2.0
    return probs.float().cpu().numpy()


def _preprocess_batch(images, device):
    """images: [N,H,W,3] RGB"""
    images = images.astype(np.float32) / 255.0
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).half().to(device)
    return (batch - MEAN.to(device)) / STD.to(device)


def _sliding_positions(length, window_size):
    if length <= 0:  raise ValueError(f"length={length} <= 0")
    if window_size <= 0: raise ValueError(f"window_size={window_size} <= 0")
    return list(range(0, length, window_size))


def sliding_window_infer(model, image, device, window_size, stride):
    """无缩放无重叠滑窗推理。"""
    if stride != window_size:
        raise ValueError(f"stride({stride}) != window_size({window_size})")

    height, width = image.shape[:2]
    xs = _sliding_positions(width,  window_size)
    ys = _sliding_positions(height, window_size)
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
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)
    index_mask = enhance_connectivity(index_mask, num_classes, infer_close_kernel)
    confidence  = np.max(probs, axis=0).astype(np.float32)
    return index_mask, confidence


def save_result(index_mask, confidence, save_path, save_confidence=False):
    """保存单通道类别索引 PNG"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    index_path = save_path + ".png"
    if not cv2.imwrite(index_path, index_mask):
        raise RuntimeError(f"Cannot save: {index_path}")
    values, counts = np.unique(index_mask, return_counts=True)
    print(f"Saved: {index_path} | classes={dict(zip(values.tolist(), counts.tolist()))}")
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

    print(f"num_classes={num_classes}  crop_size={crop_size}  "
          f"stride={stride_len}  close_kernel={infer_close_kernel}")
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
        print(f"Infer: {name}")

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
        print(f"Time: {elapsed_ms:.2f} ms{warmup}")

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
