"""
infer_letterbox.py —— DINOv3 + DecoderV3 推理（与训练预处理一致）
训练/推理统一：原图保持宽高比 resize，使最长边=image_size，再用灰色填充到 image_size×image_size；预测后去除 padding，并恢复到原图尺寸。
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
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/BG_test/testset",
    # "test_image_dir": "F:/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images",
    # "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/test_HQL",
    # "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/test_BG",
    # "test_image_dir": "F:/liuhaibo/datasets/BG_HQL_JZW/20260630135631",
    # "test_image_dir": "F:/liuhaibo/datasets/LG_JZW/LG_test",
    # "test_image_dir": "F:/liuhaibo/datasets/test/JZW/DHTG/total",
    # "test_image_dir": "F:/liuhaibo/datasets/test/JZW/HQL_JZW_0810/1",
    "test_image_dir": "F:/liuhaibo/datasets/test/HMS/test_images",

    # ========== 模型结构（必须与训练时一致）==========
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "num_classes": 5,  # 4(D类) / 5(ABCD) / 7(ABC) / 5(D+TIN-D) / 9(ABC+TIN-B+TIN-C) / 9(ABC+TIN)

    # ========== 权重加载 ==========
    "checkpoint_dir": "./checkpoints/HMS",
    # "checkpoint_dir": "./checkpoints/ABCTIN1024",
    # "checkpoint_dir": "./checkpoints/ABCTIN1024_gray",
    "checkpoint_name": "best_iou",  # best_iou / best_dice / last / 完整路径
    "weight_source": "auto",        # teacher(EMA) / student / auto

    # ========== 输入预处理：必须与训练时 image_size 一致 ==========
    "image_size": 1024,
    "confidence_threshold": 0.4,

    # ========== 输出 ==========
    # "output_dir": "./output/infer_BG",
    # "output_dir": "./output/infer_DHTG/DHTG16",
    # "output_dir": "./output/infer_LG/436_gray",
    # "output_dir": "./output/infer_HQL",
    "output_dir": "F:/liuhaibo/datasets/test/HMS",
    "output_subdir": "test_masks",  # 子目录，保存 mask 的文件夹名
    "save_confidence": False,

    # ========== 设备 / 性能统计 ==========
    "device": "cuda",
    "warmup_images": 3,
}

_model_cfg = SimpleNamespace(backbone_name=INFER["backbone_name"], freeze_backbone=INFER["freeze_backbone"], num_classes=INFER["num_classes"])
num_classes = INFER["num_classes"]
image_size = INFER["image_size"]
confidence_threshold = INFER["confidence_threshold"]
if image_size <= 0: raise ValueError("image_size 必须 > 0")
if not 0.0 <= confidence_threshold <= 1.0: raise ValueError("confidence_threshold 必须在 0.0 到 1.0 之间")

from models.dinov3_segmentation import DINOv3Seg

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
GRAY = (128, 128, 128)


def synchronize_device(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def print_model_info(model, checkpoint_path):
    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    checkpoint_bytes = os.path.getsize(checkpoint_path)
    print("=" * 70)
    print(f"参数量：{total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"模型参数大小：{model_bytes / (1024 ** 2):.2f} MB")
    print(f"权重文件大小：{checkpoint_bytes / (1024 ** 2):.2f} MB")
    print("=" * 70)


def load_model(weight_path, device, weight_source="teacher"):
    """加载推理模型。weight_source: teacher / student / auto。"""
    model = DINOv3Seg(_model_cfg).to(device)
    if device.type == "cuda": model = model.half()
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    priority = ["teacher", "student", "model"] if weight_source in ("teacher", "auto") else ["student", "teacher", "model"]
    loaded_source = None
    for key in priority:
        if key in ckpt:
            loaded_source = key
            model.load_state_dict(ckpt[key], strict=True)
            break
    if loaded_source is None: raise KeyError(f"Checkpoint keys={list(ckpt.keys())}, expected one of {priority}")
    model.eval()
    print(f"Loaded: {weight_path}")
    print(f"  source={loaded_source}, epoch={ckpt.get('epoch','?')}, IoU={ckpt.get('iou',0):.4f}, Dice={ckpt.get('dice',0):.4f}")
    return model


def letterbox_resize(image, target_size):
    """与训练集一致：保持宽高比缩放，最长边到 target_size，灰色填充成正方形。返回处理后的图和逆变换所需信息。"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_h, pad_w = target_size - new_h, target_size - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=GRAY)
    meta = {"orig_h": h, "orig_w": w, "new_h": new_h, "new_w": new_w, "pad_top": pad_top, "pad_bottom": pad_bottom, "pad_left": pad_left, "pad_right": pad_right, "scale": scale}
    return padded, meta


def preprocess_image(image, device):
    """与训练时 A.Normalize(mean,std)+ToTensorV2 对齐，输入 RGB uint8 [H,W,3]。"""
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    batch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    if device.type == "cuda": batch = batch.half()
    return batch


@torch.no_grad()
def predict_probabilities(model, batch):
    """单次整图前向，返回 [C,H,W] float32 概率。"""
    seg, _ = model(batch)
    probs = torch.softmax(seg, dim=1)[0]
    return probs.float().cpu().numpy()


def restore_probabilities(probs, meta):
    """去除 letterbox padding，并把概率图恢复到原图尺寸。"""
    y1, x1 = meta["pad_top"], meta["pad_left"]
    y2, x2 = y1 + meta["new_h"], x1 + meta["new_w"]
    probs = probs[:, y1:y2, x1:x2]
    oh, ow = meta["orig_h"], meta["orig_w"]
    restored = np.empty((probs.shape[0], oh, ow), dtype=np.float32)
    for c in range(probs.shape[0]): restored[c] = cv2.resize(probs[c], (ow, oh), interpolation=cv2.INTER_LINEAR)
    return restored


def infer_single(model, image_path, device):
    """单图：原图 -> letterbox 1024 -> 模型 -> 去 padding -> 恢复原尺寸 -> mask/confidence。"""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None: raise RuntimeError(f"Cannot read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image, meta = letterbox_resize(image, image_size)
    batch = preprocess_image(input_image, device)
    probs = predict_probabilities(model, batch)
    if probs.shape[0] != num_classes: raise RuntimeError(f"输出通道数错误: {probs.shape[0]} vs num_classes={num_classes}")
    probs = restore_probabilities(probs, meta)
    confidence = np.max(probs, axis=0).astype(np.float32)
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)
    index_mask[confidence < confidence_threshold] = 0
    return index_mask, confidence, meta


def save_result(index_mask, confidence, save_path, save_confidence=False):
    """保存原图尺寸的单通道类别索引 PNG。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    index_path = save_path + ".png"
    if not cv2.imwrite(index_path, index_mask): raise RuntimeError(f"Cannot save: {index_path}")
    if save_confidence:
        conf_img = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
        if not cv2.imwrite(save_path + "_confidence.png", conf_img): raise RuntimeError(f"Cannot save: {save_path}_confidence.png")


if __name__ == "__main__":
    device = torch.device(INFER["device"] if torch.cuda.is_available() else "cpu")
    ckpt_path = INFER["checkpoint_name"]
    sep = os.sep
    if sep not in ckpt_path and "/" not in ckpt_path and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(INFER["checkpoint_dir"], INFER["checkpoint_name"] + ".pth")
        if not os.path.exists(ckpt_path):
            for fallback in ("best_dice.pth", "best_iou.pth", "last.pth"):
                candidate = os.path.join(INFER["checkpoint_dir"], fallback)
                if os.path.exists(candidate): ckpt_path = candidate; break
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print(f"  checkpoint_dir:  {INFER['checkpoint_dir']}")
        print(f"  checkpoint_name: {INFER['checkpoint_name']}")
        raise SystemExit(1)

    model = load_model(ckpt_path, device, INFER["weight_source"])
    print(f"num_classes={num_classes}  image_size={image_size}  preprocess=letterbox  confidence_threshold={confidence_threshold}")
    print_model_info(model, ckpt_path)

    test_dir = INFER["test_image_dir"]
    if not os.path.exists(test_dir): print(f"Test dir not found: {test_dir}"); raise SystemExit(1)
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    image_names = sorted(n for n in os.listdir(test_dir) if n.lower().endswith(valid_exts))
    if not image_names: print(f"No images found in: {test_dir}"); raise SystemExit(1)

    print(f"Total images: {len(image_names)}")
    print("=" * 70)
    measured_times = []
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

    for idx, name in enumerate(image_names):
        path = os.path.join(test_dir, name)
        synchronize_device(device)
        t0 = time.perf_counter()
        index_mask, confidence, meta = infer_single(model, path, device)
        synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if idx >= INFER["warmup_images"]: measured_times.append(elapsed_ms)
        save_base = os.path.join(INFER["output_dir"], INFER["output_subdir"], os.path.splitext(name)[0])
        save_result(index_mask, confidence, save_base, save_confidence=INFER["save_confidence"])
        warmup = " [预热]" if idx < INFER["warmup_images"] else ""
        print(f"[{idx + 1}/{len(image_names)}] Infer: {name}  original={meta['orig_w']}x{meta['orig_h']} -> resize={meta['new_w']}x{meta['new_h']} -> input={image_size}x{image_size}  pad=(L{meta['pad_left']},R{meta['pad_right']},T{meta['pad_top']},B{meta['pad_bottom']})  Time: {elapsed_ms:.2f} ms{warmup}  save to: {INFER['output_subdir']}")

    print("=" * 70)
    if measured_times:
        avg_ms = float(np.mean(measured_times))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print(f"有效统计图片数：{len(measured_times)}")
        print(f"平均单图推理时间：{avg_ms:.2f} ms")
        print(f"平均推理速度：{fps:.2f} FPS")
    else: print("有效统计图片数不足，请减少 warmup_images。")
    if device.type == "cuda": print(f"GPU 峰值显存：{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")
    print("=" * 70)
    print("Done.")