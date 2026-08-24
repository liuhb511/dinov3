import os
from collections import defaultdict

import cv2
import numpy as np


# =========================
# 参数配置
# =========================

# 掩码文件夹
MASK_DIR = r"F:/liuhaibo/datasets/JZW/ABCTIN_1024/train/masks"

# 是否递归统计子文件夹
RECURSIVE = False

# 是否统计背景类别 0
COUNT_BACKGROUND = False

# 支持的图像扩展名
SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def get_mask_paths(mask_dir):
    """获取待统计的掩码文件路径。"""
    mask_paths = []

    if RECURSIVE:
        for root, _, filenames in os.walk(mask_dir):
            for filename in filenames:
                extension = os.path.splitext(filename)[1].lower()

                if extension in SUPPORTED_EXTENSIONS:
                    mask_paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(mask_dir):
            file_path = os.path.join(mask_dir, filename)

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(filename)[1].lower()

            if extension in SUPPORTED_EXTENSIONS:
                mask_paths.append(file_path)

    return sorted(mask_paths)


def count_mask_pixels(mask_dir):
    """统计文件夹中各索引类别的像素总数。"""
    if not os.path.isdir(mask_dir):
        raise NotADirectoryError(f"掩码文件夹不存在：{mask_dir}")

    class_pixel_counts = defaultdict(int)
    mask_paths = get_mask_paths(mask_dir)

    success_count = 0
    failed_count = 0
    total_pixels = 0

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"读取失败：{mask_path}")
            failed_count += 1
            continue

        # 索引掩码必须是单通道
        if mask.ndim != 2:
            print(
                f"跳过非单通道图像：{mask_path}，"
                f"实际形状：{mask.shape}"
            )
            failed_count += 1
            continue

        class_ids, pixel_counts = np.unique(mask, return_counts=True)

        for class_id, pixel_count in zip(class_ids, pixel_counts):
            class_id = int(class_id)
            pixel_count = int(pixel_count)

            if class_id == 0 and not COUNT_BACKGROUND:
                continue

            class_pixel_counts[class_id] += pixel_count
            total_pixels += pixel_count

        success_count += 1

    print("=" * 40)
    print(f"掩码文件夹：{mask_dir}")
    print(f"发现图像数量：{len(mask_paths)}")
    print(f"成功统计数量：{success_count}")
    print(f"读取或格式失败数量：{failed_count}")
    print(f"是否统计背景：{COUNT_BACKGROUND}")
    print("-" * 40)

    if not class_pixel_counts:
        print("没有统计到符合条件的像素。")
    else:
        for class_id in sorted(class_pixel_counts):
            pixel_count = class_pixel_counts[class_id]

            if total_pixels > 0:
                percentage = pixel_count / total_pixels * 100
            else:
                percentage = 0.0

            print(
                f"类别 {class_id:>3}："
                f"{pixel_count:>15,} 像素，"
                f"占比 {percentage:>8.4f}%"
            )

    print("-" * 40)
    print(f"统计像素总数：{total_pixels:,}")
    print("=" * 40)


if __name__ == "__main__":
    count_mask_pixels(MASK_DIR)