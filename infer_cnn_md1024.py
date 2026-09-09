"""
infer_cnn_md1024.py -- A-CNN-MD-v1 standalone end-to-end inference.

以旧版 infer_JZW_end2end.py 为参考：
1. checkpoint 只加载 ckpt['model']
2. Overlay / side_by_side / contour / label 行为与旧版一致
3. 加入旧版风格的配置检查
4. 保留新版 CNN-MD 模型与 batch 滑窗推理

Output class IDs:
0 BG, 1 A, 2 B, 3 C, 4 D, 5 TIN-B/TIN-C, 6 TIN-D
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from models.cnn_md.model import DINOv3SegCNNMD


# ╔══════════════════════════════════════════════════════════════╗
# ║                         推理配置                              ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    "test_image_dir": r"D:/lhb/datasets/testsets/JZW/HQL_JZW_0810/2",

    # ========== checkpoint ==========
    "checkpoint_dir": "./checkpoints/A_cnn_md_v2",
    "checkpoint_name": "best_iou",

    # ========== 模型结构（必须与训练时一致） ==========
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "num_classes": 7,
    "feature_layers": (4, 8, 12),
    "local_stem_channels": 16,
    "local_f4_channels": 32,
    "local_f8_channels": 64,
    "fusion_init_alpha": 0.1,

    # ========== 滑动窗口 ==========
    "crop_size": 1024,
    "stride": 0,                     # 0=自适应重叠；>0=固定步长
    "infer_batch_size": 6,
    "confidence_threshold": 0.1,

    # ========== 输出 ==========
    "output_dir": r"D:/lhb/datasets/testsets/JZW/HQL_JZW_0810/result/2/dino_cnn",
    "save_mask": True,
    "output_mode": "overlay",  # overlay / side_by_side

    # ========== 叠加显示：与旧版一致 ==========
    "display_class_ids": {1, 2, 3, 4, 5, 6},
    "alpha": 0.40,
    "draw_contours": True,
    "contour_thickness": 0,          # >0 时绘制不透明轮廓
    "min_component_area": 2,         # 仅影响显示，不修改预测 mask
    "label_font_scale": 0.65,
    "label_font_thickness": 2,
    "label_padding": 4,

    # ========== 设备 / 性能统计 ==========
    "device": "cuda",
    "warmup_images": 3,
}


CLASS_NAMES = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "TIN-B/TIN-C",
    6: "TIN-D",
}

# OpenCV BGR colors
CLASS_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (0, 165, 255),
}

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PAD_VALUE = 128


# ╔══════════════════════════════════════════════════════════════╗
# ║                      常用配置转变量                            ║
# ╚══════════════════════════════════════════════════════════════╝
num_classes = int(INFER["num_classes"])
display_class_ids = set(INFER["display_class_ids"])
alpha = float(INFER["alpha"])
draw_contours = bool(INFER["draw_contours"])
contour_thickness = int(INFER["contour_thickness"])
min_component_area = int(INFER["min_component_area"])
label_font_scale = float(INFER["label_font_scale"])
label_font_thickness = int(INFER["label_font_thickness"])
label_padding = int(INFER["label_padding"])
output_mode = INFER["output_mode"]

crop_size = int(INFER["crop_size"])
stride_len = int(INFER["stride"])
infer_batch_size = int(INFER["infer_batch_size"])
confidence_threshold = float(INFER["confidence_threshold"])

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        工具函数                               ║
# ╚══════════════════════════════════════════════════════════════╝
def sync(device):
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
    path = checkpoint_name

    if not os.path.isabs(path) and not os.path.exists(path):
        name = checkpoint_name if checkpoint_name.lower().endswith(".pth") else checkpoint_name + ".pth"
        path = os.path.join(checkpoint_dir, name)

    return path


def strip_module_prefix(state_dict):
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def read_image(path):
    """兼容 Windows 中文路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        模型加载                               ║
# ╚══════════════════════════════════════════════════════════════╝
def load_model(path, device):
    """
    CNN-MD checkpoint 只允许加载 ckpt['model']。
    不使用 teacher / student / state_dict 优先级。
    """
    cfg = SimpleNamespace(
        backbone_name=INFER["backbone_name"],
        freeze_backbone=INFER["freeze_backbone"],
        num_classes=INFER["num_classes"],
        feature_layers=INFER["feature_layers"],
        local_stem_channels=INFER["local_stem_channels"],
        local_f4_channels=INFER["local_f4_channels"],
        local_f8_channels=INFER["local_f8_channels"],
        fusion_init_alpha=INFER["fusion_init_alpha"],
    )

    model = DINOv3SegCNNMD(cfg).to(device)
    if device.type == "cuda":
        model = model.half()

    ckpt = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint 必须是 dict，实际为 {type(ckpt).__name__}")

    if "model" not in ckpt:
        raise KeyError(
            f"Checkpoint keys={list(ckpt.keys())}，未找到唯一允许的权重字段 'model'"
        )

    state = ckpt["model"]
    if not isinstance(state, dict):
        raise TypeError(f"ckpt['model'] 必须是 dict，实际为 {type(state).__name__}")

    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded: {path}")
    print(
        f"  source=model, epoch={ckpt.get('epoch', '?')}, "
        f"IoU={ckpt.get('iou', 0):.4f}, Dice={ckpt.get('dice', 0):.4f}"
    )
    return model


# ╔══════════════════════════════════════════════════════════════╗
# ║                       推理核心函数                             ║
# ╚══════════════════════════════════════════════════════════════╝
def adaptive_positions(length, window):
    """自适应均匀分布窗口：首尾贴边，中间均匀重叠。"""
    if length <= window:
        return [0]

    span = length - window
    count = max(math.ceil(span / window) + 1, 2)
    step = span / (count - 1)
    positions = [round(i * step) for i in range(count - 1)] + [span]

    # 与旧版一致：过近位置合并
    unique = [int(positions[0])]
    for pos in positions[1:]:
        pos = int(pos)
        if pos - unique[-1] >= 2:
            unique.append(pos)
        else:
            unique[-1] = max(unique[-1], pos)
    return unique


def fixed_positions(length, window, stride):
    """固定步长位置，并保证最后一个窗口贴住图像边缘。"""
    if length <= window:
        return [0]

    last = length - window
    positions = list(range(0, last + 1, stride))
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def preprocess_batch(patches, device):
    arr = np.stack(patches).astype(np.float32) / 255.0
    batch = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    batch = batch.to(dtype=dtype)

    mean = MEAN.to(device=device, dtype=dtype)
    std = STD.to(device=device, dtype=dtype)
    return (batch - mean) / std


@torch.inference_mode()
def predict_probabilities(model, batch):
    seg, _ = model(batch)
    return torch.softmax(seg, dim=1).float().cpu().numpy()


def sliding_window_infer(model, image_rgb, device):
    """
    batch 滑窗推理，返回 [C,H,W] 概率图。
    重叠区域做概率累加平均。
    """
    h, w = image_rgb.shape[:2]
    window = crop_size
    stride = stride_len
    batch_size = infer_batch_size
    ncls = num_classes

    xs = fixed_positions(w, window, stride) if stride > 0 else adaptive_positions(w, window)
    ys = fixed_positions(h, window, stride) if stride > 0 else adaptive_positions(h, window)

    prob_sum = np.zeros((ncls, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    patches = []
    infos = []

    for y1 in ys:
        for x1 in xs:
            y2 = min(y1 + window, h)
            x2 = min(x1 + window, w)

            patch = image_rgb[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]

            if ph != window or pw != window:
                padded = np.full((window, window, 3), PAD_VALUE, dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            patches.append(patch)
            infos.append((y1, y2, x1, x2, ph, pw))

    for start in range(0, len(patches), batch_size):
        bp = patches[start:start + batch_size]
        bi = infos[start:start + batch_size]

        batch = preprocess_batch(bp, device)
        probs = predict_probabilities(model, batch)

        if probs.ndim != 4:
            raise RuntimeError(f"模型输出概率维度错误：期望 [N,C,H,W]，实际 {probs.shape}")

        if probs.shape[1] != ncls:
            raise RuntimeError(
                f"模型输出通道数={probs.shape[1]}，配置 num_classes={ncls}"
            )

        for p, info in zip(probs, bi):
            y1, y2, x1, x2, ph, pw = info
            prob_sum[:, y1:y2, x1:x2] += p[:, :ph, :pw]
            count[y1:y2, x1:x2] += 1.0

    if np.any(count == 0):
        raise RuntimeError("存在滑窗未覆盖像素")

    return prob_sum / count[None, ...]


def infer_single(model, image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    probs = sliding_window_infer(model, image_rgb, device)

    confidence = np.max(probs, axis=0).astype(np.float32)
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)
    index_mask[confidence < confidence_threshold] = 0

    return index_mask, confidence


# ╔══════════════════════════════════════════════════════════════╗
# ║                  Overlay：与旧版一致                          ║
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

            if contour_thickness > 0:
                cv2.drawContours(
                    result,
                    [contour],
                    contourIdx=-1,
                    color=bgr_color,
                    thickness=contour_thickness,
                    lineType=cv2.LINE_AA,
                )

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
    overlay = image.copy()

    for class_id in sorted(display_class_ids):
        class_region = mask == class_id
        if not np.any(class_region):
            continue

        color = np.asarray(CLASS_COLORS[class_id], dtype=np.float32)
        original = overlay[class_region].astype(np.float32)
        blended = original * (1.0 - alpha) + color * alpha
        overlay[class_region] = np.clip(blended, 0, 255).astype(np.uint8)

    if draw_contours:
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
        encode_extension,
        image,
        encode_params,
    )
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")

    encoded_image.tofile(output_path)


def save_mask_png(output_path, mask):
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
        raise ValueError("CLASS_COLORS 与 CLASS_NAMES 的类别编号不一致")

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

    if stride_len < 0:
        raise ValueError("stride 必须 >= 0")

    if infer_batch_size <= 0:
        raise ValueError("infer_batch_size 必须 > 0")

    if output_mode not in {"overlay", "side_by_side"}:
        raise ValueError('output_mode 只能是 "overlay" 或 "side_by_side"')

    if contour_thickness < 0:
        raise ValueError("contour_thickness 必须 >= 0")

    if min_component_area < 0:
        raise ValueError("min_component_area 必须 >= 0")

    if label_font_scale <= 0:
        raise ValueError("label_font_scale 必须 > 0")

    if label_font_thickness <= 0:
        raise ValueError("label_font_thickness 必须 > 0")

    if label_padding < 0:
        raise ValueError("label_padding 必须 >= 0")

    if int(INFER["warmup_images"]) < 0:
        raise ValueError("warmup_images 必须 >= 0")

    feature_layers = tuple(INFER["feature_layers"])
    if len(feature_layers) != 3:
        raise ValueError("feature_layers 必须包含 3 个层索引")

    if any(not isinstance(v, int) for v in feature_layers):
        raise ValueError("feature_layers 必须全部为 int")

    for key in (
        "local_stem_channels",
        "local_f4_channels",
        "local_f8_channels",
    ):
        if int(INFER[key]) <= 0:
            raise ValueError(f"{key} 必须 > 0")

    fusion_init_alpha = float(INFER["fusion_init_alpha"])
    if not np.isfinite(fusion_init_alpha):
        raise ValueError("fusion_init_alpha 必须是有限数值")

    if not os.path.isdir(INFER["test_image_dir"]):
        raise FileNotFoundError(f"Test dir not found: {INFER['test_image_dir']}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    validate_config()

    device = torch.device(
        INFER["device"] if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = resolve_checkpoint(
        INFER["checkpoint_dir"],
        INFER["checkpoint_name"],
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"模型权重不存在：{checkpoint_path}")

    print("=" * 70)
    print("加载 A-CNN-MD-v1 夹杂物识别模型...")

    model = load_model(checkpoint_path, device)
    print_model_info(model, checkpoint_path)

    mode_label = "adaptive" if stride_len == 0 else f"fixed stride={stride_len}"

    print(f"num_classes={num_classes}（背景1类 + 前景{num_classes - 1}类）")
    print(
        f"feature_layers={INFER['feature_layers']}  "
        f"crop_size={crop_size}  "
        f"mode={mode_label}  "
        f"infer_batch_size={infer_batch_size}  "
        f"confidence_threshold={confidence_threshold}"
    )
    print("=" * 70)

    names = sorted(
        name
        for name in os.listdir(INFER["test_image_dir"])
        if name.lower().endswith(VALID_EXTS)
    )

    if not names:
        raise RuntimeError(f"No images found in: {INFER['test_image_dir']}")

    # 可选：--limit N
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        if len(sys.argv) < 3:
            raise ValueError("--limit 后必须提供图片数量")
        limit = int(sys.argv[2])
        if limit <= 0:
            raise ValueError("--limit N 中 N 必须 > 0")
        names = names[:limit]
        print(f"[限制] 仅处理前 {limit} 张图片")

    output_dir = INFER["output_dir"]
    mask_dir = os.path.join(output_dir, "mask")
    overlay_dir = os.path.join(output_dir, "overlay")

    os.makedirs(overlay_dir, exist_ok=True)
    if INFER["save_mask"]:
        os.makedirs(mask_dir, exist_ok=True)

    print(f"Total images: {len(names)}")
    if INFER["save_mask"]:
        print(f"Mask目录：{mask_dir}")
    print(f"Overlay目录：{overlay_dir}")
    print("=" * 70)

    measured_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for idx, name in enumerate(names):
        image_path = os.path.join(INFER["test_image_dir"], name)
        stem = os.path.splitext(name)[0]

        image_bgr = read_image(image_path)
        if image_bgr is None:
            print(f"[跳过] 无法读取图片：{image_path}")
            continue

        sync(device)
        t0 = time.perf_counter()

        # ① 模型推理 -> mask
        mask, _ = infer_single(model, image_bgr, device)

        # ② 与旧版一致：生成 overlay / side_by_side
        overlay = create_overlay(image_bgr, mask)
        output_image = create_output_image(image_bgr, overlay)

        sync(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= int(INFER["warmup_images"]):
            measured_times.append(elapsed_ms)

        # ③ 保存类别索引 mask
        if INFER["save_mask"]:
            save_mask_png(
                os.path.join(mask_dir, stem + ".png"),
                mask,
            )

        # ④ 与旧版一致：可视化结果沿用输入文件名/扩展名
        save_image(
            os.path.join(overlay_dir, name),
            output_image,
        )

        present_ids = [
            cid for cid in range(1, num_classes)
            if np.any(mask == cid)
        ]
        present_names = [CLASS_NAMES[cid] for cid in present_ids]

        warmup = " [预热]" if idx < int(INFER["warmup_images"]) else ""

        print(
            f"[{idx + 1}/{len(names)}] "
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
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU 峰值显存：{peak_mb:.2f} MB")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
