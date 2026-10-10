"""
infer_JZW_end2end.py —— DINOv3 夹杂物（JZW）单模型端到端推理

流程：
    1. 加载一个分割模型
    2. 整图 resize 至 crop_size × crop_size 后推理，恢复至原图尺寸
    3. 输出单通道类别索引 mask
    4. 将 mask 半透明叠加到原图并保存

类别：
    0: background
    1: A

注意：
    当前仅识别 A 1个前景类别，加上背景类别0，
    因此模型输出通道数 num_classes = 2。

可选参数：
    python infer_JZW_end2end.py --limit N
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import sys
import time
import torch
from types import SimpleNamespace

from models.dinov3_segmentation import DINOv3Seg


# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有推理配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    # "test_image_dir": "D:/lhb/datasets/testsets/JZW/JZW_test/JZW_test",
    # "test_image_dir": "D:/lhb/datasets/testsets/JZW/HQL_JZW_0810/2",
    "test_image_dir": "D:/lhb/datasets/testsets/DZ/SX1009/1",

    # ========== 单模型 ==========
    "checkpoint_dir":  "./checkpoints/SX1009",
    "checkpoint_name": "best_iou",   # best_iou / best_dice / last / 完整 .pth 路径

    # 0=background, 1=A
    "num_classes": 2,

    # ========== 模型结构（必须与训练时一致） ==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,
    "weight_source":   "auto",       # teacher(EMA) / student / auto

    # ========== 整图缩放 ==========
    "crop_size": 1024,               # 整图缩放至 crop_size × crop_size
    "confidence_threshold": 0.1,

    # ========== 输出 ==========
    # "output_dir": "D:/lhb/datasets/testsets/JZW/HQL_JZW_0810/result/2/3_JZW",
    # "output_dir": "D:/lhb/datasets/testsets/JZW/JZW_test/result_3/JZW_4_1",
    "output_dir": "D:/lhb/datasets/testsets/DZ/SX1009/result/Dino_2",
    "save_mask": True,               # 保存类别索引 mask
    "output_mode": "overlay",        # overlay / side_by_side

    # ========== 叠加显示 ==========
    "display_class_ids": {1},
    "alpha": 0.40,
    "draw_contours": True,
    "draw_labels": False,             # True=绘制类别标签；False=隐藏标签
    "contour_thickness": 0,          # >0 时绘制不透明轮廓
    "min_component_area": 2,
    "label_font_scale": 0.65,
    "label_font_thickness": 2,
    "label_padding": 4,

    # ========== 设备 / 性能统计 ==========
    "device": "cuda",
    "warmup_images": 3,
}


# BGR 颜色。可按项目需要自行调整。
CLASS_COLORS = {
    1: (255, 0,   0),    # A  蓝
    # 2: (0,   255, 0),    # B  绿
    # 3: (0,   0,   255),  # C  红
    # 4: (255, 0,   255),  # D  紫红

    # 5: (255, 255, 0),    # TIN-B/TIN-C  青
    # 6: (0,   165, 255),  # TIN-D        橙
}

CLASS_NAMES = {
    1: "A",
    # 2: "B",
    # 3: "C",
    # 4: "D",

    # 5: "TIN-B/TIN-C",
    # 6: "TIN-D",
}


num_classes          = int(INFER["num_classes"])
display_class_ids     = set(INFER["display_class_ids"])
alpha                 = float(INFER["alpha"])
draw_contours         = bool(INFER["draw_contours"])
draw_labels           = bool(INFER["draw_labels"])
contour_thickness     = int(INFER["contour_thickness"])
min_component_area    = int(INFER["min_component_area"])
label_font_scale      = float(INFER["label_font_scale"])
label_font_thickness  = int(INFER["label_font_thickness"])
label_padding         = int(INFER["label_padding"])
output_mode           = INFER["output_mode"]
crop_size             = int(INFER["crop_size"])
confidence_threshold  = float(INFER["confidence_threshold"])

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


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


def resolve_checkpoint(checkpoint_dir, checkpoint_name):
    """支持完整路径，或 checkpoint_dir 下的权重名称。"""
    ckpt_path = checkpoint_name

    if (
        os.sep not in ckpt_path
        and "/" not in ckpt_path
        and not os.path.exists(ckpt_path)
    ):
        filename = checkpoint_name
        if not filename.lower().endswith(".pth"):
            filename += ".pth"

        ckpt_path = os.path.join(checkpoint_dir, filename)

        if not os.path.exists(ckpt_path):
            for fallback in ("best_dice.pth", "best_iou.pth", "last.pth"):
                candidate = os.path.join(checkpoint_dir, fallback)
                if os.path.exists(candidate):
                    ckpt_path = candidate
                    break

    return ckpt_path


# ╔══════════════════════════════════════════════════════════════╗
# ║                        模型加载                               ║
# ╚══════════════════════════════════════════════════════════════╝
def load_model(weight_path, device, num_classes, weight_source="auto"):
    """加载单个推理模型。"""
    cfg = SimpleNamespace(
        backbone_name=INFER["backbone_name"],
        freeze_backbone=INFER["freeze_backbone"],
        num_classes=num_classes,
    )

    model = DINOv3Seg(cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)

    if weight_source == "student":
        priority = ["student", "teacher", "model"]
    else:
        priority = ["teacher", "student", "model"]

    loaded_source = None
    for key in priority:
        if key in ckpt:
            model.load_state_dict(ckpt[key], strict=True)
            loaded_source = key
            break

    if loaded_source is None:
        raise KeyError(
            f"Checkpoint keys={list(ckpt.keys())}, expected one of {priority}"
        )

    model.eval()

    print(f"Loaded: {weight_path}")
    print(
        f"  source={loaded_source}, epoch={ckpt.get('epoch', '?')}, "
        f"IoU={ckpt.get('iou', 0):.4f}, Dice={ckpt.get('dice', 0):.4f}"
    )
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║                       推理核心函数                             ║
# ╚══════════════════════════════════════════════════════════════╝
def _predict_probabilities(model, batch):
    """前向推理，返回 [N,C,H,W] 概率。"""
    with torch.no_grad():
        seg, _ = model(batch)
        probs = torch.softmax(seg, dim=1)

    return probs.float().cpu().numpy()


def _preprocess_batch(images, device):
    """images: [N,H,W,3] RGB"""
    images = images.astype(np.float32) / 255.0
    batch = torch.from_numpy(images).permute(0, 3, 1, 2).half().to(device)
    return (batch - MEAN.to(device)) / STD.to(device)


def infer_single(model, image_bgr, device):
    """整图缩放推理 -> 原图尺寸的 index mask + confidence。"""
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR
    )
    batch = _preprocess_batch(np.expand_dims(resized, axis=0), device)
    probs = _predict_probabilities(model, batch)[0]

    assert probs.shape[0] == num_classes, (
        f"模型输出通道数={probs.shape[0]}，配置 num_classes={num_classes}"
    )

    # 先将各类别概率恢复至原图尺寸，再确定类别，避免插值类别索引。
    probs = np.stack([
        cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
        for prob in probs
    ])
    confidence = np.max(probs, axis=0).astype(np.float32)
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)
    index_mask[confidence < confidence_threshold] = 0
    return index_mask, confidence


# ╔══════════════════════════════════════════════════════════════╗
# ║                     半透明叠加展示                             ║
# ╚══════════════════════════════════════════════════════════════╝
def choose_text_color(bgr_color):
    blue, green, red = bgr_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness >= 150 else (255, 255, 255)


def draw_class_contours_and_labels(image, mask):
    result = image.copy()
    image_height, image_width = result.shape[:2]

    for class_id in sorted(display_class_ids):
        bgr_color = CLASS_COLORS[class_id]
        class_name = CLASS_NAMES[class_id]

        binary_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_component_area:
                continue

            if draw_contours and contour_thickness > 0:
                cv2.drawContours(
                    result,
                    [contour],
                    contourIdx=-1,
                    color=bgr_color,
                    thickness=contour_thickness,
                    lineType=cv2.LINE_AA,
                )

            if not draw_labels:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                label_font_thickness,
            )

            label_width = text_width + label_padding * 2
            label_height = text_height + baseline + label_padding * 2

            label_x1 = max(0, min(x, image_width - label_width))
            if y >= label_height:
                label_y1 = y - label_height
            else:
                label_y1 = min(
                    y + height,
                    max(0, image_height - label_height),
                )

            label_x2 = min(image_width - 1, label_x1 + label_width)
            label_y2 = min(image_height - 1, label_y1 + label_height)

            cv2.rectangle(
                result,
                (label_x1, label_y1),
                (label_x2, label_y2),
                bgr_color,
                thickness=-1,
            )

            text_color = choose_text_color(bgr_color)
            text_x = label_x1 + label_padding
            text_y = label_y1 + label_padding + text_height

            cv2.putText(
                result,
                class_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                text_color,
                label_font_thickness,
                cv2.LINE_AA,
            )

    return result


def create_overlay(image, mask):
    """将显示类别的 mask 半透明叠加到原图。"""
    overlay = image.copy()

    for class_id in sorted(display_class_ids):
        class_region = mask == class_id
        if not np.any(class_region):
            continue

        color = np.asarray(CLASS_COLORS[class_id], dtype=np.float32)
        original = overlay[class_region].astype(np.float32)

        blended = original * (1.0 - alpha) + color * alpha
        overlay[class_region] = np.clip(blended, 0, 255).astype(np.uint8)

    if draw_contours or draw_labels:
        overlay = draw_class_contours_and_labels(overlay, mask)

    return overlay


def create_output_image(original_image, overlay_image):
    if output_mode == "overlay":
        return overlay_image

    if output_mode == "side_by_side":
        return np.hstack((original_image, overlay_image))

    raise ValueError(f"不支持的 output_mode：{output_mode}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        文件保存                               ║
# ╚══════════════════════════════════════════════════════════════╝
def save_image(output_path, image):
    """保存彩色图片，兼容中文路径。"""
    suffix = os.path.splitext(output_path)[1].lower()
    extension_map = {
        ".jpg": ".jpg",
        ".jpeg": ".jpg",
        ".png": ".png",
        ".bmp": ".bmp",
        ".tif": ".tif",
        ".tiff": ".tiff",
    }
    encode_extension = extension_map.get(suffix, ".png")

    encode_params = []
    if encode_extension == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    success, encoded_image = cv2.imencode(
        encode_extension, image, encode_params
    )
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")

    encoded_image.tofile(output_path)


def save_mask_png(output_path, mask):
    """
    保存 uint8 单通道类别索引 mask：
        0 background
        1 A
    """
    success, encoded_image = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError(f"Mask 编码失败：{output_path}")

    encoded_image.tofile(output_path)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        配置校验                               ║
# ╚══════════════════════════════════════════════════════════════╝
def validate_config():
    expected_foreground_ids = set(CLASS_NAMES.keys())
    expected_num_classes = len(expected_foreground_ids) + 1

    if num_classes != expected_num_classes:
        raise ValueError(
            f"num_classes 配置错误：当前定义了 {len(expected_foreground_ids)} 个前景类别，"
            f"加上背景后应为 {expected_num_classes} 类，实际配置为 {num_classes}。"
        )

    if set(CLASS_COLORS.keys()) != expected_foreground_ids:
        raise ValueError(
            "CLASS_COLORS 与 CLASS_NAMES 的类别编号不一致"
        )

    invalid_display = display_class_ids - expected_foreground_ids
    if invalid_display:
        raise ValueError(
            f"display_class_ids 中存在无效类别：{sorted(invalid_display)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在 0~1 之间")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold 必须在 0~1 之间")

    if crop_size <= 0:
        raise ValueError("crop_size 必须 > 0")

    if output_mode not in {"overlay", "side_by_side"}:
        raise ValueError(
            'output_mode 只能是 "overlay" 或 "side_by_side"'
        )
# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    validate_config()

    device = torch.device(
        INFER["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    # 单模型 checkpoint
    checkpoint_path = resolve_checkpoint(
        INFER["checkpoint_dir"],
        INFER["checkpoint_name"],
    )

    if not os.path.exists(checkpoint_path):
        print(f"模型权重不存在：{checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print("加载夹杂物（JZW）识别模型...")

    model = load_model(
        checkpoint_path,
        device,
        num_classes,
        INFER["weight_source"],
    )
    print_model_info(model, checkpoint_path)

    print(f"num_classes={num_classes}（背景1类 + 前景{num_classes - 1}类）")
    print(
        f"resize={crop_size}x{crop_size}  "
        f"confidence_threshold={confidence_threshold}"
    )
    print("=" * 70)

    test_dir = INFER["test_image_dir"]
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        sys.exit(1)

    image_names = sorted(
        name
        for name in os.listdir(test_dir)
        if name.lower().endswith(VALID_EXTS)
    )

    if not image_names:
        print(f"No images found in: {test_dir}")
        sys.exit(1)

    # 可选：--limit N
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        if len(sys.argv) < 3:
            raise ValueError("--limit 后必须提供图片数量")

        limit = int(sys.argv[2])
        image_names = image_names[:limit]
        print(f"[限制] 仅处理前 {limit} 张图片")

    output_dir = INFER["output_dir"]
    mask_dir = os.path.join(output_dir, "mask")
    overlay_dir = os.path.join(output_dir, "overlay")

    os.makedirs(overlay_dir, exist_ok=True)
    if INFER["save_mask"]:
        os.makedirs(mask_dir, exist_ok=True)

    print(f"Total images: {len(image_names)}")
    print(f"Mask目录：{mask_dir}")
    print(f"Overlay目录：{overlay_dir}")
    print("=" * 70)

    measured_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for idx, name in enumerate(image_names):
        image_path = os.path.join(test_dir, name)
        stem = os.path.splitext(name)[0]

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[跳过] 无法读取图片：{image_path}")
            continue

        synchronize_device(device)
        t0 = time.perf_counter()

        # ① 单模型推理 -> mask
        mask, _ = infer_single(model, image_bgr, device)

        # ② mask 直接叠加原图，不做任何 mask 合并/remap
        overlay = create_overlay(image_bgr, mask)
        output_image = create_output_image(image_bgr, overlay)

        synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= INFER["warmup_images"]:
            measured_times.append(elapsed_ms)

        # ③ 保存类别索引 mask
        if INFER["save_mask"]:
            save_mask_png(
                os.path.join(mask_dir, stem + ".png"),
                mask,
            )

        # ④ 保存原图叠加结果
        save_image(
            os.path.join(overlay_dir, name),
            output_image,
        )

        present_ids = [
            cid for cid in range(1, num_classes)
            if np.any(mask == cid)
        ]
        present_names = [
            CLASS_NAMES[cid] for cid in present_ids
        ]

        warmup = (
            " [预热]"
            if idx < INFER["warmup_images"]
            else ""
        )

        print(
            f"[{idx + 1}/{len(image_names)}] "
            f"Infer: {name}  "
            f"classes={present_names}  "
            f"Time: {elapsed_ms:.2f} ms{warmup}"
        )

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
        peak_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )
        print(f"GPU 峰值显存：{peak_mb:.2f} MB")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
