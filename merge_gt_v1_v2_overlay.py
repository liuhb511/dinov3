"""
merge_gt_v1_v2_overlay.py

功能：
    1. 读取原图 + GT / V1 / V2 三组类别索引 mask
    2. 使用相同 CLASS_COLORS 将三组 mask 分别半透明叠加到原图
    3. 按 GT | V1 | V2 从左到右拼接
    4. 每张子图左上角绘制 GT / V1 / V2 标签
    5. 每个分割连通区域标注类别名称    
    6. 按原文件 stem 匹配，不要求扩展名完全一致

mask 类别：
    0 background
    1 hd_w
    2 hd_y
    3 hd_t
    4 red
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# 配置区域
# ============================================================
IMAGE_DIR = Path(r"D:/lhb/datasets/testsets/HMS/testsets/test_images")
GT_DIR = Path(r"D:/lhb/datasets/testsets/HMS/testsets/test_GT")
V1_DIR = Path(r"D:/lhb/datasets/testsets/HMS/testsets/result/V1/test_masks")
V2_DIR = Path(r"D:/lhb/datasets/testsets/HMS/testsets/result/V2/mask")

OUTPUT_DIR = Path(r"D:/lhb/datasets/testsets/HMS/testsets/merge/GTV1V2")

# mask 半透明叠加系数
ALPHA = 0.40

# BGR 颜色
CLASS_COLORS = {
    1: (255, 0, 0),      # hd_w
    2: (0, 255, 255),    # hd_y
    3: (255, 0, 255),    # hd_t
    4: (0, 0, 255),      # red
}

# 子图标签
LABELS = ("GT", "V1", "V2")

# 标签显示（子图左上角）
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
FONT_SIZE = 36
LABEL_MARGIN = 15
LABEL_PADDING_X = 10
LABEL_PADDING_Y = 7
LABEL_RADIUS = 6

TEXT_COLOR = (255, 255, 255)      # RGB
LABEL_BG_COLOR = (0, 0, 0)        # RGB
LABEL_BG_ALPHA = 180

# 输出
JPG_QUALITY = 95
OUTPUT_SUFFIX = ".jpg"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


# ============================================================
# 文件读取
# ============================================================
def collect_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是文件夹：{folder}")

    images = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.stem in images:
            print(f"警告：{folder} 中存在重复 stem：{path.stem}，保留 {images[path.stem].name}")
            continue

        images[path.stem] = path

    return images


def read_bgr(path: Path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取原图：{path}")
    return image


def read_mask(path: Path):
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"无法读取 mask：{path}")

    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(
                f"mask 必须是单通道类别索引图：{path}，当前 shape={mask.shape}"
            )

    return mask.astype(np.uint8)


# ============================================================
# mask 叠加
# ============================================================
def overlay_mask(image_bgr, mask):
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"原图与 mask 尺寸不一致：image={image_bgr.shape[:2]}, mask={mask.shape[:2]}"
        )

    result = image_bgr.copy()

    for class_id, color in CLASS_COLORS.items():
        region = mask == class_id
        if not np.any(region):
            continue

        original = result[region].astype(np.float32)
        color_arr = np.asarray(color, dtype=np.float32)

        blended = original * (1.0 - ALPHA) + color_arr * ALPHA
        result[region] = np.clip(blended, 0, 255).astype(np.uint8)

    return result


# ============================================================
# 标签
# ============================================================
def load_font():
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        print(f"警告：无法加载字体 {FONT_PATH}，使用默认字体。")
        return ImageFont.load_default()


def draw_label_top_left(image_bgr, text, font):
    """
    在子图左上角绘制 GT / V1 / V2 标签。
    OpenCV BGR -> PIL RGB -> 绘制 -> BGR
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(image_pil, "RGBA")

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    box_width = text_width + LABEL_PADDING_X * 2
    box_height = text_height + LABEL_PADDING_Y * 2

    box_x1 = LABEL_MARGIN
    box_y1 = LABEL_MARGIN
    box_x2 = box_x1 + box_width
    box_y2 = box_y1 + box_height

    draw.rounded_rectangle(
        (box_x1, box_y1, box_x2, box_y2),
        radius=LABEL_RADIUS,
        fill=(*LABEL_BG_COLOR, LABEL_BG_ALPHA),
    )

    text_x = box_x1 + LABEL_PADDING_X - bbox[0]
    text_y = box_y1 + LABEL_PADDING_Y - bbox[1]

    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(*TEXT_COLOR, 255),
    )

    result_rgb = np.asarray(image_pil.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def draw_component_labels(image_bgr, mask, font):
    """
    为每个类别的每个连通区域绘制类别标签。

    标签格式：
        hd_w
        hd_y
        hd_t
        red

    标签优先放在连通域质心附近；如果质心不在区域内部，
    则使用距离变换寻找更靠区域内部的位置。
    """
    class_names = {
        1: "hd_w",
        2: "hd_y",
        3: "hd_t",
        4: "red",
    }

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).convert("RGBA")
    draw = ImageDraw.Draw(image_pil, "RGBA")

    for class_id, class_name in class_names.items():
        binary = (mask == class_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for component_id in range(1, num_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue

            component = labels == component_id
            cx, cy = centroids[component_id]
            x = int(round(cx))
            y = int(round(cy))

            # 质心可能落在凹形区域之外，此时寻找区域内部最深点。
            if (
                x < 0 or y < 0
                or x >= mask.shape[1] or y >= mask.shape[0]
                or not component[y, x]
            ):
                component_u8 = component.astype(np.uint8)
                dist = cv2.distanceTransform(component_u8, cv2.DIST_L2, 5)
                _, _, _, max_loc = cv2.minMaxLoc(dist)
                x, y = max_loc

            bbox = draw.textbbox((0, 0), class_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            pad_x = 6
            pad_y = 4
            box_width = text_width + pad_x * 2
            box_height = text_height + pad_y * 2

            box_x1 = x - box_width // 2
            box_y1 = y - box_height // 2
            box_x1 = max(0, min(box_x1, image_pil.width - box_width))
            box_y1 = max(0, min(box_y1, image_pil.height - box_height))
            box_x2 = box_x1 + box_width
            box_y2 = box_y1 + box_height

            draw.rounded_rectangle(
                (box_x1, box_y1, box_x2, box_y2),
                radius=4,
                fill=(0, 0, 0, 150),
            )

            text_x = box_x1 + pad_x - bbox[0]
            text_y = box_y1 + pad_y - bbox[1]

            draw.text(
                (text_x, text_y),
                class_name,
                font=font,
                fill=(255, 255, 255, 255),
            )

    result_rgb = np.asarray(image_pil.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# 拼接
# ============================================================
def merge_three(images):
    """
    GT | V1 | V2 左中右排列。
    三张图来自同一张原图，因此正常情况下尺寸完全一致。
    """
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]

    if len(set(heights)) != 1 or len(set(widths)) != 1:
        raise ValueError(f"三张叠加图尺寸不一致：{[(w, h) for w, h in zip(widths, heights)]}")

    return np.hstack(images)


def save_jpg(path: Path, image_bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY],
    )

    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")

    encoded.tofile(str(path))


# ============================================================
# 主函数
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_map = collect_images(IMAGE_DIR)
    gt_map = collect_images(GT_DIR)
    v1_map = collect_images(V1_DIR)
    v2_map = collect_images(V2_DIR)

    common_names = sorted(
        set(image_map)
        & set(gt_map)
        & set(v1_map)
        & set(v2_map)
    )

    print("=" * 80)
    print(f"原图：{len(image_map)}")
    print(f"GT：  {len(gt_map)}")
    print(f"V1：  {len(v1_map)}")
    print(f"V2：  {len(v2_map)}")
    print(f"四组共同样本：{len(common_names)}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 80)

    if not common_names:
        print("没有找到可匹配的共同文件。")
        return

    font = load_font()

    success_count = 0
    failed_count = 0

    for index, name in enumerate(common_names, start=1):
        try:
            image_bgr = read_bgr(image_map[name])

            gt_mask = read_mask(gt_map[name])
            v1_mask = read_mask(v1_map[name])
            v2_mask = read_mask(v2_map[name])

            if gt_mask.shape != v1_mask.shape or gt_mask.shape != v2_mask.shape:
                raise ValueError(
                    f"GT/V1/V2 mask 尺寸不一致："
                    f"GT={gt_mask.shape}, V1={v1_mask.shape}, V2={v2_mask.shape}"
                )

            if image_bgr.shape[:2] != gt_mask.shape:
                raise ValueError(
                    f"原图与 mask 尺寸不一致："
                    f"image={image_bgr.shape[:2]}, mask={gt_mask.shape}"
                )

            gt_overlay = overlay_mask(image_bgr, gt_mask)
            v1_overlay = overlay_mask(image_bgr, v1_mask)
            v2_overlay = overlay_mask(image_bgr, v2_mask)

            # 先给每个连通区域标注类别，再在每张子图左上角标注 GT / V1 / V2。
            gt_overlay = draw_component_labels(gt_overlay, gt_mask, font)
            v1_overlay = draw_component_labels(v1_overlay, v1_mask, font)
            v2_overlay = draw_component_labels(v2_overlay, v2_mask, font)

            gt_overlay = draw_label_top_left(gt_overlay, LABELS[0], font)
            v1_overlay = draw_label_top_left(v1_overlay, LABELS[1], font)
            v2_overlay = draw_label_top_left(v2_overlay, LABELS[2], font)

            merged = merge_three([gt_overlay, v1_overlay, v2_overlay])

            output_path = OUTPUT_DIR / f"{name}{OUTPUT_SUFFIX}"
            save_jpg(output_path, merged)

            success_count += 1
            print(f"[{index}/{len(common_names)}] OK：{output_path.name}")

        except Exception as error:
            failed_count += 1
            print(f"[{index}/{len(common_names)}] 失败：{name}，原因：{error}")

    print("\n" + "=" * 80)
    print("处理完成")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"输出目录：{OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
