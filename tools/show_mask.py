# -*- coding: utf-8 -*-
"""
单通道类别索引 Mask 与原图半透明叠加脚本。

功能：
1. 根据同名文件匹配原图和类别索引 Mask。
2. Mask 必须是单通道图，像素值为类别编号。
3. 支持自定义类别数量和类别显示颜色。
4. 支持两种输出形式：
   - overlay：只保存叠加图；
   - side_by_side：原图和叠加图左右拼接。
5. 输出文件名与原图保持一致。

示例：
原图：
    images/001.jpg
Mask：
    masks/001.png
输出：
    output/001.jpg

注意：
OpenCV 使用 BGR 颜色顺序。
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


# ============================================================
# 一、路径配置
# ============================================================

# 原图目录
IMAGE_DIR = Path(
    # r"F:/liuhaibo/datasets/BG_HQL_JZW/20260630135631"                                       #  莱钢 测试集原图
    # r"F:/liuhaibo/datasets/BG_HQL_JZW/test_HQL"                                             # 哈汽轮 测试集原图
    # r"F:/liuhaibo/datasets/BG_HQL_JZW/test_BG"                                              #  宝钢 测试集原图
    # r"F:/liuhaibo/datasets/LG_JZW/LG_test"                                                  # 莱钢2 测试集原图

    r"F:/liuhaibo/datasets/test/HMS/test_images"
    # r"F:/liuhaibo/datasets/test/JZW/HQL_0825/1/"
)


# 单通道类别索引 Mask 目录 —— 推理
# MASK_DIR = Path(r"F:/liuhaibo/datasets/test/JZW/HQL_0825/compare/new/")  # 哈汽轮 测试集融合掩码
MASK_DIR = Path(r"output/infer_HMS/HMS_v2/")  # 哈汽轮 测试集融合掩码

# 输出目录
OUTPUT_DIR = Path(r"output/overlay_results/HMS/HMS_v2/")  # 哈汽轮 测试集叠加结果


# ============================================================
# 二、类别和颜色配置
# ============================================================

# 前景类别数量，不包含背景类别0
# ABC：6， D：3， TIND：4， ABC+TINBC：8, ABCD+TINBCD：11
NUM_CLASSES = 4

# 类别编号 -> BGR显示颜色
#
# 注意：
# OpenCV 使用 BGR 顺序，而不是 RGB。
#
# 例如：
# 红色 RGB=(255, 0, 0)，在这里应写成 BGR=(0, 0, 255)
#
# 类别数量必须与这里配置的颜色数量一致。
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {

    1:  (255, 0,   0),      # A           蓝
    2:  (0,   255, 0),      # B           绿
    3:  (0,   0,   255),    # C           红
    4:  (255, 0,   255),    # D           紫红
    # 5:  (255, 255, 0),      # TIN-B/TIN-C 青
    # 6:  (0,   165, 255),    # TIN-D       橙
    # 7:  (203, 192, 255),    # HH          粉
    # 8:  (144, 238, 144),    # XW          浅绿
    # 9:  (0,   255, 255),    # XQL         黄
    # 10: (128, 0,   128),    # HC          深紫
    # 11: (255, 191, 0),      # SZ          深青蓝

    # 1: (155, 0, 128),
    # 2: (255, 0, 128),
    # 3: (155, 255, 128),

    # 4: (0, 255, 255),
    # 5: (255, 255, 0),
    # 6: (255, 128, 0),
    # 7: (0, 0, 255),
    # 8: (0, 255, 0),
}

# 类别编号 -> 显示名称
# 数量和类别编号必须与 CLASS_COLORS 保持一致。
CLASS_NAMES: Dict[int, str] = {
    # 1: "D",
    # 2: "HC",
    # 3: "SZ",
    # 4: "TIN-D",

    # 1: "A",
    # 2: "B",
    # 3: "C",
    # 4: "HH",
    # 5: "XW",
    # 6: "XQL",
    # 7: "TIN-B",
    # 8: "TIN-C",

    # 1: "A",
    # 2: "B",
    # 3: "C",
    # 4: "D",
    # 5: "HH",
    # 6: "XW",
    # 7: "XQL",
    # 8: "HC",
    # 9: "SZ",

    
    # 1: "A",
    # 2: "B",
    # 3: "C",
    # 4: "D",
    # 5: "TIN-B/TIN-C",
    # 6: "TIN-D",
    # 7: "HH",
    # 8: "XW",
    # 9: "XQL",
    # 10: "HC",
    # 11: "SZ",

    1: "hd_w",
    2: "hd_y",
    3: "hd_t",
    4: "red",

}

# 需要在结果中显示的类别编号。
# 只需修改这个集合；未列出的类别不会叠加颜色，也不会绘制轮廓和标签。
# 当前示例：显示 A、B、C、D 和 TIN。
DISPLAY_CLASS_IDS = {1, 2, 3, 4}

# 轮廓与标签显示配置``
DRAW_CONTOURS = True
CONTOUR_THICKNESS = 0
MIN_COMPONENT_AREA = 2
LABEL_FONT_SCALE = 0.65
LABEL_FONT_THICKNESS = 2
LABEL_PADDING = 4

# 背景类别编号
BACKGROUND_CLASS_ID = 0

# 半透明叠加系数
#
# 0.0：完全显示原图
# 1.0：完全显示类别颜色
ALPHA = 0.40


# ============================================================
# 三、输出模式配置
# ============================================================

# 可选：
#
# "overlay"
#   只输出半透明叠加图。
#
# "side_by_side"
#   左侧为原图，右侧为半透明叠加图。
OUTPUT_MODE = "overlay"


# ============================================================
# 四、其他配置
# ============================================================

# Mask 中出现未配置类别时的处理方式：
#
# "error"
#   直接报错并跳过该图片。
#
# "ignore"
#   忽略未配置类别，不进行颜色叠加。
UNKNOWN_CLASS_POLICY = "error"

# 支持的原图格式
SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# 支持的 Mask 格式
SUPPORTED_MASK_SUFFIXES = {
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def validate_config() -> None:
    """检查配置是否合法。"""

    if NUM_CLASSES <= 0:
        raise ValueError("NUM_CLASSES 必须大于0")

    if len(CLASS_COLORS) != NUM_CLASSES:
        raise ValueError(f"类别数量与颜色数量不一致：NUM_CLASSES={NUM_CLASSES}，CLASS_COLORS数量={len(CLASS_COLORS)}")

    if len(CLASS_NAMES) != NUM_CLASSES:
        raise ValueError(f"类别数量与类别名称数量不一致：NUM_CLASSES={NUM_CLASSES}，CLASS_NAMES数量={len(CLASS_NAMES)}")

    expected_class_ids = set(range(1, NUM_CLASSES + 1))
    actual_class_ids = set(CLASS_COLORS.keys())
    actual_name_ids = set(CLASS_NAMES.keys())

    if actual_class_ids != expected_class_ids:
        raise ValueError(f"CLASS_COLORS 的类别编号必须连续，应为：{sorted(expected_class_ids)}，实际为：{sorted(actual_class_ids)}")

    if actual_name_ids != expected_class_ids:
        raise ValueError(f"CLASS_NAMES 的类别编号必须连续，应为：{sorted(expected_class_ids)}，实际为：{sorted(actual_name_ids)}")

    invalid_display_class_ids = set(DISPLAY_CLASS_IDS) - expected_class_ids
    if invalid_display_class_ids:
        raise ValueError(f"DISPLAY_CLASS_IDS 中存在无效类别编号：{sorted(invalid_display_class_ids)}；可用类别为：{sorted(expected_class_ids)}")

    for class_id, color in CLASS_COLORS.items():
        if len(color) != 3:
            raise ValueError(f"类别 {class_id} 的颜色必须包含3个通道，当前颜色为：{color}")
        if any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(f"类别 {class_id} 的颜色值必须在0到255之间，当前颜色为：{color}")

    if not 0.0 <= ALPHA <= 1.0:
        raise ValueError("ALPHA 必须在0.0到1.0之间")

    if OUTPUT_MODE not in {"overlay", "side_by_side"}:
        raise ValueError('OUTPUT_MODE 只能设置为 "overlay" 或 "side_by_side"')

    if UNKNOWN_CLASS_POLICY not in {"error", "ignore"}:
        raise ValueError('UNKNOWN_CLASS_POLICY 只能设置为 "error" 或 "ignore"')


def build_mask_index(mask_dir: Path) -> Dict[str, Path]:
    """
    建立 Mask 文件索引。

    使用不带扩展名的文件名作为键，
    例如：
        001.png -> 001
    """

    mask_index: Dict[str, Path] = {}

    for mask_path in mask_dir.rglob("*"):
        if not mask_path.is_file():
            continue
        if mask_path.suffix.lower() not in SUPPORTED_MASK_SUFFIXES:
            continue

        key = mask_path.stem.lower()
        if key in mask_index:
            print(f"[警告] 发现重复名称的Mask：{mask_index[key]} 和 {mask_path}，将使用前者。")
            continue
        mask_index[key] = mask_path

    return mask_index


def read_index_mask(mask_path: Path, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    读取单通道类别索引 Mask。

    target_shape：
        (height, width)
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"无法读取Mask：{mask_path}")

    # 如果Mask被保存成三通道，但三个通道完全相同，则自动提取其中一个通道。
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        channel_0, channel_1, channel_2 = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
        if np.array_equal(channel_0, channel_1) and np.array_equal(channel_1, channel_2):
            mask = channel_0
        else:
            raise ValueError(f"Mask不是单通道类别索引图：{mask_path.name}")

    if mask.ndim != 2:
        raise ValueError(f"Mask维度不正确：{mask_path.name}，当前形状为：{mask.shape}")

    target_height, target_width = target_shape
    if mask.shape != (target_height, target_width):
        print(f"[提示] Mask尺寸与原图不一致，将使用最近邻插值调整：{mask.shape[::-1]} -> {(target_width, target_height)}")
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return mask.astype(np.int32)


def check_mask_classes(mask: np.ndarray, mask_name: str) -> None:
    """检查Mask中的类别值是否合法。"""

    actual_class_ids = set(np.unique(mask).tolist())
    valid_class_ids = {BACKGROUND_CLASS_ID, *CLASS_COLORS.keys()}
    unknown_class_ids = sorted(actual_class_ids - valid_class_ids)

    if not unknown_class_ids:
        return

    if UNKNOWN_CLASS_POLICY == "error":
        raise ValueError(f"Mask {mask_name} 中存在未配置类别：{unknown_class_ids}；允许类别为：{sorted(valid_class_ids)}")

    print(f"[提示] Mask {mask_name} 中存在未配置类别：{unknown_class_ids}，这些类别将被忽略。")


def choose_text_color(bgr_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """根据标签背景亮度选择黑色或白色文字。"""

    blue, green, red = bgr_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness >= 150 else (255, 255, 255)


def draw_class_contours_and_labels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    对每个类别的每个独立连通区域：
    1. 绘制与掩码颜色一致的不透明轮廓；
    2. 在目标附近绘制不透明类别标签。
    """

    result = image.copy()
    image_height, image_width = result.shape[:2]

    for class_id in sorted(DISPLAY_CLASS_IDS):
        bgr_color = CLASS_COLORS[class_id]
        binary_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class_name = CLASS_NAMES[class_id]

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_COMPONENT_AREA:
                continue

            # 不透明轮廓，颜色与当前类别掩码一致
            if CONTOUR_THICKNESS > 0:
                cv2.drawContours(result, [contour], contourIdx=-1, color=bgr_color, thickness=CONTOUR_THICKNESS, lineType=cv2.LINE_AA)

            x, y, width, height = cv2.boundingRect(contour)
            label_text = class_name
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_FONT_THICKNESS)

            label_width = text_width + LABEL_PADDING * 2
            label_height = text_height + baseline + LABEL_PADDING * 2

            # 优先将标签放在目标框上方。
            label_x1 = max(0, min(x, image_width - label_width))
            label_y1 = y - label_height if y >= label_height else min(y + height, image_height - label_height)

            label_x2 = min(image_width - 1, label_x1 + label_width)
            label_y2 = min(image_height - 1, label_y1 + label_height)

            # 类别色不透明标签底色
            cv2.rectangle(result, (label_x1, label_y1), (label_x2, label_y2), bgr_color, thickness=-1)

            text_color = choose_text_color(bgr_color)
            text_x = label_x1 + LABEL_PADDING
            text_y = label_y1 + LABEL_PADDING + text_height

            cv2.putText(result, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, text_color, LABEL_FONT_THICKNESS, cv2.LINE_AA)

    return result


def create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    根据类别索引 Mask 生成半透明叠加图，
    并绘制不透明轮廓和类别标签。
    """

    overlay = image.copy()

    for class_id in sorted(DISPLAY_CLASS_IDS):
        bgr_color = CLASS_COLORS[class_id]
        class_region = (mask == class_id)
        if not np.any(class_region):
            continue

        original_pixels = overlay[class_region].astype(np.float32)
        color_array = np.asarray(bgr_color, dtype=np.float32)
        blended_pixels = original_pixels * (1.0 - ALPHA) + color_array * ALPHA
        overlay[class_region] = np.clip(blended_pixels, 0, 255).astype(np.uint8)

    if DRAW_CONTOURS:
        overlay = draw_class_contours_and_labels(overlay, mask)

    return overlay


def create_output_image(original_image: np.ndarray, overlay_image: np.ndarray) -> np.ndarray:
    """根据输出模式生成最终图片。"""

    if OUTPUT_MODE == "overlay":
        return overlay_image
    if OUTPUT_MODE == "side_by_side":
        return np.hstack((original_image, overlay_image))
    raise ValueError(f"不支持的输出模式：{OUTPUT_MODE}")


def save_image(output_path: Path, image: np.ndarray) -> None:
    """保存图片，并处理中文路径兼容问题。"""

    suffix = output_path.suffix.lower()
    extension_map = {".jpg": ".jpg", ".jpeg": ".jpg", ".png": ".png", ".bmp": ".bmp", ".tif": ".tif", ".tiff": ".tiff"}
    encode_extension = extension_map.get(suffix, ".png")

    encode_params = []
    if encode_extension == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    success, encoded_image = cv2.imencode(encode_extension, image, encode_params)
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")
    encoded_image.tofile(str(output_path))


def process_dataset() -> None:
    """处理整个数据集。"""

    validate_config()

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"原图目录不存在：{IMAGE_DIR}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask目录不存在：{MASK_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_index = build_mask_index(MASK_DIR)

    image_paths = sorted(path for path in IMAGE_DIR.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)

    if not image_paths:
        raise RuntimeError(f"原图目录中没有找到支持的图片：{IMAGE_DIR}")

    print("=" * 80)
    print("单通道类别索引Mask半透明叠加")
    print("=" * 80)
    print(f"原图目录：{IMAGE_DIR}")
    print(f"Mask目录：{MASK_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"类别数量：{NUM_CLASSES}")
    print(f"类别颜色：{CLASS_COLORS}")
    print(f"类别名称：{CLASS_NAMES}")
    print(f"显示类别：{sorted(DISPLAY_CLASS_IDS)}")
    print(f"透明度：{ALPHA}")
    print(f"轮廓线宽：{CONTOUR_THICKNESS}")
    print(f"最小标注区域：{MIN_COMPONENT_AREA}")
    print(f"输出模式：{OUTPUT_MODE}")
    print(f"原图数量：{len(image_paths)}")
    print(f"Mask数量：{len(mask_index)}")
    print("=" * 80)

    success_count, skipped_count, item = 0, 0, 0
    total_items = len(image_paths)

    for image_path in image_paths:
        item += 1
        image_key = image_path.stem.lower()
        mask_path = mask_index.get(image_key)

        if mask_path is None:
            print(f"[跳过] 未找到同名Mask：{image_path.name}")
            skipped_count += 1
            continue

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"无法读取原图：{image_path}")

            height, width = image.shape[:2]
            mask = read_index_mask(mask_path, (height, width))
            check_mask_classes(mask, mask_path.name)

            overlay_image = create_overlay(image, mask)
            output_image = create_output_image(image, overlay_image)

            # 输出文件名和原图保持一致
            output_path = OUTPUT_DIR / image_path.name
            save_image(output_path, output_image)

            print(f"[完成] ({item}/{total_items}) {image_path.name} <-> {mask_path.name}")
            success_count += 1

        except Exception as error:
            print(f"[失败] ({item}/{total_items}) {image_path.name}：{error}")
            skipped_count += 1

    print("=" * 80)
    print(f"成功处理：{success_count}")
    print(f"跳过或失败：{skipped_count}")
    print(f"结果目录：{OUTPUT_DIR.resolve()}")
    print("=" * 80)


def main() -> None:
    process_dataset()


if __name__ == "__main__":
    main()