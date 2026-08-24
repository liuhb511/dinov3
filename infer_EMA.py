"""
infer_EMA.py  ——  DINOv3 + DecoderV3 推理（EMA Teacher 优先）
══════════════════════════════════════════════════════
所有配置在本文件顶部 INFER 字典中，零外部 config 依赖。
默认优先加载 EMA Teacher 权重。
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
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/BG_test/testset",  # 宝钢测试集
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images",  # 哈汽轮测试集
    "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/20260630135631",  # 莱钢测试集

    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images_tile",  # 哈汽轮784*784测试集


    # ========== 模型结构（必须与训练时一致）==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,
    "num_classes":     7,            # 4(D) / 5(ABCD) / 7(ABC)

    # ========== 权重加载 ==========
    "checkpoint_dir":  "./checkpoints/ABC_EMA",
    "checkpoint_name": "best_iou",   # best_iou / best_dice / last / 完整路径
    "weight_source":   "teacher",    # "teacher"(EMA,默认) / "student" / "auto"

    # ========== 推理参数 ==========
    "crop_size":         784,        # 滑动窗口大小
    "stride":            784,        # 步长（=crop_size 无重叠，<crop_size 有重叠）
    "infer_close_kernel": 0,         # 形态学闭运算核，0=禁用

    # ========== 输出 ==========
    "output_dir":     "./output/infer_LG",  # 推理结果保存目录
    "output_subdir":  "ABC_EMA",  # 子目录名，便于区分不同配置
    "save_confidence": False,

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
infer_close_kernel = INFER["infer_close_kernel"]

# ╔══════════════════════════════════════════════════════════════╗
# ║                        模块导入                               ║
# ╚══════════════════════════════════════════════════════════════╝
from models.dinov3_segmentation import DINOv3Seg

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)
GRAY = (128, 128, 128)


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


def load_model(weight_path, device, weight_source="teacher"):
    """加载推理模型。默认加载 EMA Teacher，兼容 student / model"""
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


def enhance_connectivity(mask, num_classes, kernel_size=3):
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = mask.copy()
    fg = mask > 0
    for c in range(1, num_classes):
        binary = (mask == c).astype(np.uint8)
        if not np.any(binary): continue
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        new = (closed > 0) & (~fg) & (result == 0)
        result[new] = c
    return result


def _predict_probabilities(model, batch):
    """TTA 水平翻转"""
    with torch.no_grad():
        seg, _ = model(batch)
        probs = torch.softmax(seg, dim=1)
        bf = torch.flip(batch, dims=[3])
        sf, _ = model(bf)
        pf = torch.softmax(sf, dim=1)
        pf = torch.flip(pf, dims=[3])
        probs = (probs + pf) / 2.0
    return probs.float().cpu().numpy()


def _preprocess_batch(images, device):
    images = images.astype(np.float32) / 255.0
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).half().to(device)
    return (batch - MEAN.to(device)) / STD.to(device)


def _sliding_positions(length, window_size):
    if length <= 0: raise ValueError(f"length={length} <= 0")
    if window_size <= 0: raise ValueError(f"window_size={window_size} <= 0")
    return list(range(0, length, window_size))


def sliding_window_infer(model, image, device, window_size, stride):
    if stride != window_size:
        raise ValueError(f"stride({stride}) != window_size({window_size})")
    h, w = image.shape[:2]
    xs, ys = _sliding_positions(w, window_size), _sliding_positions(h, window_size)
    pmap = np.zeros((num_classes, h, w), dtype=np.float32)

    for y1 in ys:
        for x1 in xs:
            x2, y2 = min(x1 + window_size, w), min(y1 + window_size, h)
            patch = image[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]
            if ph != window_size or pw != window_size:
                pad = np.full((window_size, window_size, 3), GRAY, dtype=image.dtype)
                pad[:ph, :pw] = patch
                patch = pad
            batch = _preprocess_batch(np.expand_dims(patch, axis=0), device)
            probs = _predict_probabilities(model, batch)[0]
            pmap[:, y1:y2, x1:x2] = probs[:, :ph, :pw]
    return pmap


def infer_single(model, image_path, device):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None: raise RuntimeError(f"Cannot read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    probs = sliding_window_infer(model, image, device, crop_size, stride_len)
    mask = np.argmax(probs, axis=0).astype(np.uint8)
    mask = enhance_connectivity(mask, num_classes, infer_close_kernel)
    conf = np.max(probs, axis=0).astype(np.float32)
    return mask, conf


def save_result(index_mask, confidence, save_path, save_confidence=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ip = save_path + ".png"
    if not cv2.imwrite(ip, index_mask): raise RuntimeError(f"Cannot save: {ip}")
    vals, cnts = np.unique(index_mask, return_counts=True)
    print(f"Saved: {ip} | classes={dict(zip(vals.tolist(), cnts.tolist()))}")
    if save_confidence:
        ci = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path + "_confidence.png", ci)


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    device = torch.device(INFER["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = INFER["checkpoint_name"]
    sep = os.sep
    if sep not in ckpt_path and "/" not in ckpt_path and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(INFER["checkpoint_dir"], INFER["checkpoint_name"] + ".pth")
        if not os.path.exists(ckpt_path):
            for fb in ("best_dice.pth", "best_iou.pth", "last.pth"):
                c = os.path.join(INFER["checkpoint_dir"], fb)
                if os.path.exists(c): ckpt_path = c; break

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"  dir: {INFER['checkpoint_dir']}  name: {INFER['checkpoint_name']}")
        exit(1)

    model = load_model(ckpt_path, device, INFER["weight_source"])
    _ = MEAN.to(device); _ = STD.to(device)
    print(f"num_classes={num_classes} crop={crop_size} stride={stride_len} kernel={infer_close_kernel}")
    print_model_info(model, ckpt_path)

    test_dir = INFER["test_image_dir"]
    if not os.path.exists(test_dir): print(f"Not found: {test_dir}"); exit(1)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    names = sorted(n for n in os.listdir(test_dir) if n.lower().endswith(exts))
    if not names: print(f"No images in: {test_dir}"); exit(1)
    print(f"Total images: {len(names)}"); print("=" * 70)

    times = []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    for idx, name in enumerate(names):
        path = os.path.join(test_dir, name)
        print(f"Infer: {name}")
        synchronize_device(device)
        t0 = time.perf_counter()
        mask, conf = infer_single(model, path, device)
        synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1000.0

        if idx >= INFER["warmup_images"]: times.append(ms)
        save_result(mask, conf,
                    os.path.join(INFER["output_dir"], INFER["output_subdir"],
                                 os.path.splitext(name)[0]),
                    save_confidence=INFER["save_confidence"])
        warmup = " [预热]" if idx < INFER["warmup_images"] else ""
        print(f"Time: {ms:.2f} ms{warmup}")

    print("=" * 70)
    if times:
        am = float(np.mean(times))
        fps = 1000.0 / am if am > 0 else 0.0
        print(f"有效统计: {len(times)}张 | 平均: {am:.2f} ms | {fps:.2f} FPS")
    else:
        print("有效统计图片数不足")

    if device.type == "cuda":
        print(f"GPU 峰值显存: {torch.cuda.max_memory_allocated(device)/1024**2:.2f} MB")
    print("=" * 70)
    print("Done.")
