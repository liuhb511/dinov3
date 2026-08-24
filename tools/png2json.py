from pathlib import Path
import json

import cv2
import numpy as np


# ==================== 修改这里 ====================

# 单通道类别 mask 文件夹
MASK_FOLDER = Path(r"F:\liuhaibo\datasets\JZW\ABC_1024\train\masks")

# 对应原始图片文件夹
IMAGE_FOLDER = Path(r"F:\liuhaibo\datasets\JZW\ABC_1024\train\images")

# LabelMe JSON 输出文件夹
OUTPUT_FOLDER = Path(r"F:\liuhaibo\datasets\JZW\ABC_1024\train\json")


# 类别值 -> LabelMe 类别名称
CLASS_MAP = {
    1: "A",
    2: "B",
    3: "C",
    4: "HH",
    5: "XW",
    6: "XQL",
    7: "TIN-B/TIN-C",
    8: "TIN-D",
}

# CLASS_MAP = {
#     1: "A",
#     2: "B",
#     3: "C",
#     4: "D",
#     5: "TIN-B/TIN-C",
#     6: "TIN-D",
#     7: "HH",
#     8: "XW",
#     9: "XQL",
#     10: "HC",
#     11: "SZ",
# }


# 过滤面积过小的区域
MIN_AREA = 10

# polygon 轮廓简化程度
# 0 表示不简化
# 1.0 ~ 3.0 一般比较合适
APPROX_EPSILON = 1.0

# ======================================================


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
]


def find_image(mask_path: Path):
    """
    根据 mask 文件名寻找对应原图。

    例如：
    mask: 001.png
    image: 001.jpg
    """
    for ext in IMAGE_EXTENSIONS:
        image_path = IMAGE_FOLDER / f"{mask_path.stem}{ext}"

        if image_path.exists():
            return image_path

        # 兼容大写后缀
        image_path = IMAGE_FOLDER / f"{mask_path.stem}{ext.upper()}"

        if image_path.exists():
            return image_path

    return None


def contour_to_points(contour):
    """
    OpenCV contour 转 LabelMe points。
    """
    points = []

    for point in contour:
        x, y = point[0]

        points.append([
            float(x),
            float(y),
        ])

    return points


def mask_to_labelme(mask_path: Path, image_path: Path, json_path: Path, progress: int, total: int):
    # 按原始格式读取，避免类别值被改变
    mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
    )

    if mask is None:
        print(f"[失败] 无法读取 mask：{mask_path}")
        return False

    # 理论上应该是单通道
    # 如果意外是多通道，只取第一个通道
    if mask.ndim == 3:
        print(
            f"[警告] {mask_path.name} 不是单通道，"
            f"自动取第一个通道"
        )
        mask = mask[:, :, 0]

    height, width = mask.shape

    # 查看实际存在的类别值
    unique_values = np.unique(mask)

    # print(
    #     f"\n[处理] {mask_path.name} "
    #     f"尺寸={width}x{height} "
    #     f"类别={unique_values.tolist()}"
    # )

    shapes = []

    for class_id, class_name in CLASS_MAP.items():

        # 当前类别的二值 mask
        binary = np.where(
            mask == class_id,
            255,
            0,
        ).astype(np.uint8)

        # 如果当前类别完全不存在，跳过
        if not np.any(binary):
            continue

        # 提取每一个独立区域
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        valid_count = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_AREA:
                continue

            # 简化 polygon
            if APPROX_EPSILON > 0:
                contour = cv2.approxPolyDP(
                    contour,
                    APPROX_EPSILON,
                    True,
                )

            # polygon 至少需要 3 个点
            if len(contour) < 3:
                continue

            points = contour_to_points(contour)

            shape = {
                "label": class_name,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }

            shapes.append(shape)
            valid_count += 1

        # if valid_count > 0:
        #     print(
        #         f"  类别 {class_id}: "
        #         f"{class_name} -> {valid_count} 个区域"
        #     )

    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    with open(json_path, "w", encoding="utf-8",) as f:
        json.dump(
            labelme_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print( f"[完成 {progress}/{total}] {json_path.name}，共生成 {len(shapes)} 个 polygon")

    return True


def main():
    if not MASK_FOLDER.exists():
        raise FileNotFoundError(
            f"Mask 文件夹不存在：{MASK_FOLDER}"
        )

    if not IMAGE_FOLDER.exists():
        raise FileNotFoundError(
            f"原图文件夹不存在：{IMAGE_FOLDER}"
        )

    OUTPUT_FOLDER.mkdir( parents=True, exist_ok=True,)

    mask_files = sorted(MASK_FOLDER.glob("*.png"))

    success_count = 0
    skipped_count = 0
    index = 0

    for mask_path in mask_files:

        image_path = find_image(mask_path)

        if image_path is None:
            print(
                f"[跳过] 找不到对应原图："
                f"{mask_path.stem}"
            )
            skipped_count += 1
            continue

        json_path = (
            OUTPUT_FOLDER /
            f"{mask_path.stem}.json"
        )

        index += 1

        success = mask_to_labelme(mask_path, image_path, json_path, index, len(mask_files))

        if success:
            success_count += 1

    print("\n==============================")
    print("处理完成")
    print(f"Mask 总数：{len(mask_files)}")
    print(f"成功转换：{success_count}")
    print(f"跳过：{skipped_count}")
    print(f"JSON 输出：{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()