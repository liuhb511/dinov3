"""
infer_end2end.py —— DINOv3 双模型端到端推理（条状 + 点状 → 融合 → 叠加展示）

等价于依次执行（全程内存中完成，只需跑一次脚本）：
    1. infer_D_v2.py（条状模型 9 类）        → mask_strip
    2. infer_D_v2.py（点状模型 4 类）        → mask_point
    3. tools/merge_masks.py                 → mask_merged（11类，点状优先覆盖条状）
    4. tools/show_mask.py                   → 半透明叠加展示图（side_by_side / overlay）

所有配置集中在顶部 INFER 字典，零外部 config 依赖。

可选参数：
    python infer_end2end.py --limit N     # 仅处理测试集前 N 张图片（用于快速验证）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import cv2
import numpy as np
import torch
import time
import math
import sys
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有推理配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    "test_image_dir": "F:/liuhaibo/datasets/test/JZW/LG_JZW_0807/20260706152853",

    # ========== 条状模型（模型1，9类：A/B/C/HH/XW/XQL/TIN-B/TIN-C/TIN-D）==========
    "strip_checkpoint_dir":  "./checkpoints/ABCTIN1024",
    "strip_checkpoint_name": "ABCTIN_1024_slim",                    # best_iou / best_dice / last / 完整路径
    "strip_num_classes":     9,

    # ========== 点状模型（模型2，4类：D/HC/SZ）==========
    "point_checkpoint_dir":  "./checkpoints/D1024",
    "point_checkpoint_name": "best_iou",                    # best_iou / best_dice / last / 完整路径
    "point_num_classes":     4,

    # ========== 模型结构（两个模型共用，必须与训练时一致）==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,
    "weight_source":   "auto",                              # "teacher"(EMA) / "student" / "auto"

    # ========== 滑动窗口模式 ==========
    "crop_size":         1024,                              # 滑动窗口大小
    "stride":            0,                                 # 0=自适应模式(重叠), >0=固定步长
    "confidence_threshold": 0.1,                            # 最大类别概率低于该值时设为背景0

    # ========== 输出 ==========
    "output_dir": "F:/liuhaibo/datasets/output/LG/LG_JZW_0807/20260706152853",
    "save_intermediate_masks": True,                        # 是否顺带保存 mask_strip/mask_point/mask_merged
    "output_mode": "side_by_side",                          # side_by_side(原图|叠加) / overlay(仅叠加)

    # ========== 展示配置 ==========
    "num_classes_final":    11,                             # 融合后总类别数（含背景0）
    "display_class_ids":    {1, 2, 3, 4, 5, 6},             # 需要叠加显示的类别
    "alpha":                0.40,                           # 半透明叠加系数 0~1
    "draw_contours":        True,
    "contour_thickness":    0,                              # >0 时绘制不透明轮廓
    "min_component_area":   2,                              # 小于该面积的连通区域不画轮廓/标签
    "label_font_scale":     0.65,
    "label_font_thickness": 2,
    "label_padding":        4,

    # ========== 设备 / 性能统计 ==========
    "device":        "cuda",
    "warmup_images": 3,
}

# ---- 类别映射（与 tools/merge_masks.py 完全一致）----
# 条状模型(9类) → 融合mask(11类)
MAPPING_STRIP = {
    0: 0,  # 背景
    1: 1,  # A
    2: 2,  # B
    3: 3,  # C
    4: 7,  # HH
    5: 8,  # XW
    6: 9,  # XQL
    7: 5,  # TIN-B/TIN-C
    8: 6,  # TIN-D
}

# 点状模型(4类) → 融合mask(11类)
MAPPING_POINT = {
    0: 0,   # 背景
    1: 4,   # D
    2: 10,  # HC
    3: 11,  # SZ
}

# ---- 展示颜色 / 名称（与 tools/show_mask.py 完全一致，BGR 顺序）----
CLASS_COLORS = {
    1:  (255, 0,   0),      # A           蓝
    2:  (0,   255, 0),      # B           绿
    3:  (0,   0,   255),    # C           红
    4:  (255, 0,   255),    # D           紫红
    5:  (255, 255, 0),      # TIN-B/TIN-C 青
    6:  (0,   165, 255),    # TIN-D       橙
    7:  (203, 192, 255),    # HH          粉
    8:  (144, 238, 144),    # XW          浅绿
    9:  (0,   255, 255),    # XQL         黄
    10: (128, 0,   128),    # HC          深紫
    11: (255, 191, 0),      # SZ          深青蓝
}

CLASS_NAMES = {
    1:  "A",
    2:  "B",
    3:  "C",
    4:  "D",
    5:  "TIN-B/TIN-C",
    6:  "TIN-D",
    7:  "HH",
    8:  "XW",
    9:  "XQL",
    10: "HC",
    11: "SZ",
}

# ---- 解包常用参数 ----
num_classes_final     = INFER["num_classes_final"]
display_class_ids     = set(INFER["display_class_ids"])
alpha                 = float(INFER["alpha"])
draw_contours         = bool(INFER["draw_contours"])
contour_thickness     = int(INFER["contour_thickness"])
min_component_area    = int(INFER["min_component_area"])
label_font_scale      = float(INFER["label_font_scale"])
label_font_thickness  = int(INFER["label_font_thickness"])
label_padding         = int(INFER["label_padding"])
output_mode           = INFER["output_mode"]
crop_size             = INFER["crop_size"]
stride_len            = INFER["stride"]
confidence_threshold  = INFER["confidence_threshold"]

from models.dinov3_segmentation import DINOv3Seg

# FP16 预计算常量
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)
GRAY = (128, 128, 128)

valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


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


def resolve_checkpoint(checkpoint_dir, checkpoint_name):
    """解析 checkpoint 路径：支持完整路径 / 权重名(best_iou 等)。"""
    ckpt_path = checkpoint_name
    sep = os.sep
    if sep not in ckpt_path and "/" not in ckpt_path and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name + ".pth")
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
    """加载推理模型。weight_source: teacher / student / auto"""
    cfg = SimpleNamespace(
        backbone_name   = INFER["backbone_name"],
        freeze_backbone = INFER["freeze_backbone"],
        num_classes     = num_classes,
    )
    model = DINOv3Seg(cfg).half().to(device)
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


def _sliding_positions(length, window_size):
    """固定步长模式：从0开始，步长=window_size，窗口不重叠。"""
    if length <= 0:  raise ValueError(f"length={length} <= 0")
    if window_size <= 0: raise ValueError(f"window_size={window_size} <= 0")
    return list(range(0, length, window_size))


def _adaptive_positions(length, window_size):
    """
    自适应均匀分布模式（首窗口贴边，末窗口贴边，中间均匀重叠）。
    """
    if length <= window_size:
        return [0]

    span = length - window_size
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


def sliding_window_infer(model, image, device, window_size, stride, num_classes):
    """双模式滑窗推理 → [num_classes,H,W] float32 概率图。"""
    height, width = image.shape[:2]

    if stride > 0:
        xs = _sliding_positions(width,  window_size)
        ys = _sliding_positions(height, window_size)
    else:
        xs = _adaptive_positions(width,  window_size)
        ys = _adaptive_positions(height, window_size)

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


def infer_single(model, image_bgr, device, num_classes):
    """单图推理 → [H,W] uint8 index_mask + [H,W] float32 confidence"""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    probs = sliding_window_infer(model, image, device,
                                 window_size=crop_size, stride=stride_len, num_classes=num_classes)
    confidence = np.max(probs, axis=0).astype(np.float32)
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)

    # 最大类别概率低于阈值的像素作为不确定区域，统一设为背景类别0。
    index_mask[confidence < confidence_threshold] = 0
    return index_mask, confidence


# ╔══════════════════════════════════════════════════════════════╗
# ║                        Mask 融合                              ║
# ╚══════════════════════════════════════════════════════════════╝
def remap_mask(mask, mapping):
    """根据类别映射转换mask，未出现在mapping中的类别默认保持为背景0。"""
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for old_class, new_class in mapping.items():
        new_mask[mask == old_class] = new_class

    return new_mask


def merge_masks_in_memory(mask_strip, mask_point):
    """条状mask与点状mask融合（等价 tools/merge_masks.py）。点状非0区域优先覆盖条状。"""
    strip_new = remap_mask(mask_strip, MAPPING_STRIP)
    point_new = remap_mask(mask_point, MAPPING_POINT)

    merged = strip_new.copy()
    merged[point_new != 0] = point_new[point_new != 0]
    return merged


# ╔══════════════════════════════════════════════════════════════╗
# ║                     半透明叠加展示                             ║
# ╚══════════════════════════════════════════════════════════════╝
def choose_text_color(bgr_color):
    """根据标签背景亮度选择黑色或白色文字。"""
    blue, green, red = bgr_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness >= 150 else (255, 255, 255)


def draw_class_contours_and_labels(image, mask):
    """
    对每个类别的每个独立连通区域：
    1. 绘制与掩码颜色一致的不透明轮廓；
    2. 在目标附近绘制不透明类别标签。
    """
    result = image.copy()
    image_height, image_width = result.shape[:2]

    for class_id in sorted(display_class_ids):
        bgr_color = CLASS_COLORS[class_id]
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class_name = CLASS_NAMES[class_id]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_component_area:
                continue

            # 不透明轮廓，颜色与当前类别掩码一致
            if contour_thickness > 0:
                cv2.drawContours(result, [contour], contourIdx=-1, color=bgr_color,
                                 thickness=contour_thickness, lineType=cv2.LINE_AA)

            x, y, width, height = cv2.boundingRect(contour)
            label_text = class_name
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_font_thickness)

            label_width = text_width + label_padding * 2
            label_height = text_height + baseline + label_padding * 2

            # 优先将标签放在目标框上方。
            label_x1 = max(0, min(x, image_width - label_width))
            label_y1 = y - label_height if y >= label_height else min(y + height, image_height - label_height)
            label_x2 = min(image_width - 1, label_x1 + label_width)
            label_y2 = min(image_height - 1, label_y1 + label_height)

            # 类别色不透明标签底色
            cv2.rectangle(result, (label_x1, label_y1), (label_x2, label_y2), bgr_color, thickness=-1)

            text_color = choose_text_color(bgr_color)
            text_x = label_x1 + label_padding
            text_y = label_y1 + label_padding + text_height

            cv2.putText(result, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale, text_color, label_font_thickness, cv2.LINE_AA)

    return result


def create_overlay(image, mask):
    """根据类别索引 Mask 生成半透明叠加图，并绘制不透明轮廓和类别标签。"""
    overlay = image.copy()

    for class_id in sorted(display_class_ids):
        bgr_color = CLASS_COLORS[class_id]
        class_region = (mask == class_id)
        if not np.any(class_region):
            continue

        original_pixels = overlay[class_region].astype(np.float32)
        color_array = np.asarray(bgr_color, dtype=np.float32)
        blended_pixels = original_pixels * (1.0 - alpha) + color_array * alpha
        overlay[class_region] = np.clip(blended_pixels, 0, 255).astype(np.uint8)

    if draw_contours:
        overlay = draw_class_contours_and_labels(overlay, mask)

    return overlay


def create_output_image(original_image, overlay_image):
    """根据输出模式生成最终图片。"""
    if output_mode == "overlay":
        return overlay_image
    if output_mode == "side_by_side":
        return np.hstack((original_image, overlay_image))
    raise ValueError(f"不支持的输出模式：{output_mode}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        文件保存                               ║
# ╚══════════════════════════════════════════════════════════════╝
def save_image(output_path, image):
    """保存图片，并处理中文路径兼容问题。"""
    suffix = os.path.splitext(output_path)[1].lower()
    extension_map = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png", ".bmp": ".bmp",
                     ".tif": ".tif", ".tiff": ".tiff"}
    encode_extension = extension_map.get(suffix, ".png")

    encode_params = []
    if encode_extension == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    success, encoded_image = cv2.imencode(encode_extension, image, encode_params)
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")
    encoded_image.tofile(output_path)


def save_mask_png(output_path, mask):
    """保存单通道类别索引 PNG，兼容中文路径。"""
    success, encoded_image = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError(f"Mask 编码失败：{output_path}")
    encoded_image.tofile(output_path)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        配置校验                               ║
# ╚══════════════════════════════════════════════════════════════╝
def validate_config():
    if num_classes_final <= 0:
        raise ValueError("num_classes_final 必须大于0")

    if len(CLASS_COLORS) != num_classes_final:
        raise ValueError(f"类别数量与颜色数量不一致：num_classes_final={num_classes_final}，CLASS_COLORS数量={len(CLASS_COLORS)}")

    if len(CLASS_NAMES) != num_classes_final:
        raise ValueError(f"类别数量与类别名称数量不一致：num_classes_final={num_classes_final}，CLASS_NAMES数量={len(CLASS_NAMES)}")

    expected_class_ids = set(range(1, num_classes_final + 1))

    if set(CLASS_COLORS.keys()) != expected_class_ids:
        raise ValueError(f"CLASS_COLORS 的类别编号必须连续，应为：{sorted(expected_class_ids)}，实际为：{sorted(CLASS_COLORS.keys())}")

    if set(CLASS_NAMES.keys()) != expected_class_ids:
        raise ValueError(f"CLASS_NAMES 的类别编号必须连续，应为：{sorted(expected_class_ids)}，实际为：{sorted(CLASS_NAMES.keys())}")

    invalid_display = set(display_class_ids) - expected_class_ids
    if invalid_display:
        raise ValueError(f"display_class_ids 中存在无效类别编号：{sorted(invalid_display)}；可用类别为：{sorted(expected_class_ids)}")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在0.0到1.0之间")

    if output_mode not in {"overlay", "side_by_side"}:
        raise ValueError('output_mode 只能设置为 "overlay" 或 "side_by_side"')

    if crop_size <= 0:
        raise ValueError("crop_size 必须 > 0")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold 必须在 0.0 到 1.0 之间")

    all_mapped = set(MAPPING_STRIP.values()) | set(MAPPING_POINT.values())
    if all_mapped - ({0} | expected_class_ids):
        raise ValueError(f"映射表存在超出融合类别范围的目标类别：{sorted(all_mapped - ({0} | expected_class_ids))}")

    for cid, color in CLASS_COLORS.items():
        if len(color) != 3 or any(c < 0 or c > 255 for c in color):
            raise ValueError(f"类别 {cid} 的颜色必须为0~255的3通道值，当前为：{color}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    validate_config()

    device = torch.device(INFER["device"] if torch.cuda.is_available() else "cpu")
    _ = MEAN.to(device)
    _ = STD.to(device)

    # --- 解析 checkpoint 路径 ---
    strip_ckpt = resolve_checkpoint(INFER["strip_checkpoint_dir"], INFER["strip_checkpoint_name"])
    point_ckpt = resolve_checkpoint(INFER["point_checkpoint_dir"], INFER["point_checkpoint_name"])

    if not os.path.exists(strip_ckpt):
        print(f"条状模型权重不存在：{strip_ckpt}")
        exit(1)
    if not os.path.exists(point_ckpt):
        print(f"点状模型权重不存在：{point_ckpt}")
        exit(1)

    # --- 加载两个模型 ---
    print("=" * 70)
    print("加载条状模型（模型1）...")
    strip_model = load_model(strip_ckpt, device, INFER["strip_num_classes"], INFER["weight_source"])
    print_model_info(strip_model, strip_ckpt)

    print("加载点状模型（模型2）...")
    point_model = load_model(point_ckpt, device, INFER["point_num_classes"], INFER["weight_source"])
    print_model_info(point_model, point_ckpt)

    mode_label = "adaptive" if stride_len == 0 else f"fixed stride={stride_len}"
    print(f"条状 num_classes={INFER['strip_num_classes']}  点状 num_classes={INFER['point_num_classes']}")
    print(f"crop_size={crop_size}  mode={mode_label}  confidence_threshold={confidence_threshold}")
    print("=" * 70)

    # --- 测试目录 ---
    test_dir = INFER["test_image_dir"]
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        exit(1)

    image_names = sorted(n for n in os.listdir(test_dir) if n.lower().endswith(valid_exts))
    if not image_names:
        print(f"No images found in: {test_dir}")
        exit(1)

    # --- 可选：仅处理前 N 张（快速验证用） ---
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        limit = int(sys.argv[2])
        image_names = image_names[:limit]
        print(f"[限制] 仅处理前 {limit} 张图片")

    # --- 输出目录 ---
    overlay_dir = os.path.join(INFER["output_dir"], "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    if INFER["save_intermediate_masks"]:
        mask_strip_dir  = os.path.join(INFER["output_dir"], "mask_strip")
        mask_point_dir  = os.path.join(INFER["output_dir"], "mask_point")
        mask_merged_dir = os.path.join(INFER["output_dir"], "mask_merged")
        os.makedirs(mask_strip_dir,  exist_ok=True)
        os.makedirs(mask_point_dir,  exist_ok=True)
        os.makedirs(mask_merged_dir, exist_ok=True)

    print(f"Total images: {len(image_names)}")
    print(f"输出目录：{INFER['output_dir']}")
    print("=" * 70)

    measured_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for idx, name in enumerate(image_names):
        path = os.path.join(test_dir, name)
        stem = os.path.splitext(name)[0]

        synchronize_device(device)
        t0 = time.perf_counter()

        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[跳过] 无法读取图片：{path}")
            continue

        # ① 条状模型推理（9类）
        mask_strip, _ = infer_single(strip_model, image_bgr, device, INFER["strip_num_classes"])
        # ② 点状模型推理（4类）
        mask_point, _ = infer_single(point_model, image_bgr, device, INFER["point_num_classes"])
        # ③ 内存融合（11类，点状优先覆盖条状）
        mask_merged = merge_masks_in_memory(mask_strip, mask_point)
        # ④ 半透明叠加 + 轮廓/标签 + side_by_side
        overlay = create_overlay(image_bgr, mask_merged)
        output_image = create_output_image(image_bgr, overlay)

        synchronize_device(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= INFER["warmup_images"]:
            measured_times.append(elapsed_ms)

        # --- 保存 ---
        save_image(os.path.join(overlay_dir, name), output_image)
        if INFER["save_intermediate_masks"]:
            save_mask_png(os.path.join(mask_strip_dir,  stem + ".png"), mask_strip)
            save_mask_png(os.path.join(mask_point_dir,  stem + ".png"), mask_point)
            save_mask_png(os.path.join(mask_merged_dir, stem + ".png"), mask_merged)

        warmup = " [预热]" if idx < INFER["warmup_images"] else ""
        print(f"[{idx + 1}/{len(image_names)}] Infer: {name} Time: {elapsed_ms:.2f} ms{warmup}")

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


if __name__ == "__main__":
    main()
