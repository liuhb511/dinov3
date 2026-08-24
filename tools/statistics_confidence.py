from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ============================================================
# 配置区域
# ============================================================

DATA_DIR = Path(r"output/infer_HQL/HQL_0730/ABC_1_confidence")
OUTPUT_TXT = DATA_DIR / "instance_confidence_statistics3.txt"

MASK_SUFFIXES = {".png", ".tif", ".tiff"}
CONFIDENCE_NAME_SUFFIX = "_confidence"
CONFIDENCE_SUFFIXES = [".png", ".tif", ".tiff", ".npy"]

TARGET_CLASSES = {
    1: "A",
    2: "B",
    3: "C",
}

# 与可视化脚本保持一致：cv2.contourArea(contour) < 10 时忽略
MIN_INSTANCE_AREA = 10

# 每个外轮廓视为一个实例
CONTOUR_RETRIEVAL_MODE = cv2.RETR_EXTERNAL
CONTOUR_APPROXIMATION_MODE = cv2.CHAIN_APPROX_SIMPLE

# 当前置信度 PNG 已确认是 uint8，范围 127~255
CONFIDENCE_IMAGE_SCALE: Optional[float] = 255.0


# ============================================================
# 图片读取
# ============================================================

def imread_unicode(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """支持中文路径的 OpenCV 图片读取。"""
    encoded_data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded_data, flags)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def to_single_channel(image: np.ndarray, image_path: Path) -> np.ndarray:
    """将实际内容相同的三通道图片转换为单通道。"""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] >= 3:
        channel_0, channel_1, channel_2 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        if np.array_equal(channel_0, channel_1) and np.array_equal(channel_0, channel_2):
            return channel_0
    raise ValueError(f"图片不是有效的单通道图：{image_path}，图片形状为 {image.shape}")


def load_mask(mask_path: Path) -> np.ndarray:
    """读取类别掩码。"""
    return to_single_channel(imread_unicode(mask_path), mask_path)


def normalize_confidence(confidence: np.ndarray, suffix: str) -> np.ndarray:
    """将置信度统一转换为百分数 0~100。"""
    confidence = confidence.astype(np.float32)
    finite_values = confidence[np.isfinite(confidence)]

    if finite_values.size == 0:
        raise ValueError("置信度图没有有效数值")

    min_value = float(finite_values.min())
    max_value = float(finite_values.max())

    if min_value < 0:
        raise ValueError(f"置信度图中存在负数，范围为 {min_value:.6f}~{max_value:.6f}")

    if suffix == ".npy":
        if max_value <= 1.0 + 1e-6:
            confidence *= 100.0
        elif max_value > 100.0 + 1e-6:
            raise ValueError(f"NPY 置信度范围无法识别：{min_value:.6f}~{max_value:.6f}")
    else:
        if CONFIDENCE_IMAGE_SCALE is not None:
            confidence = confidence / float(CONFIDENCE_IMAGE_SCALE) * 100.0
        elif max_value <= 1.0 + 1e-6:
            confidence *= 100.0
        elif max_value > 100.0 + 1e-6:
            raise ValueError(
                f"图片置信度范围无法识别：{min_value:.6f}~{max_value:.6f}；"
                f"请检查 CONFIDENCE_IMAGE_SCALE"
            )

    return np.clip(confidence, 0.0, 100.0)


def load_confidence(confidence_path: Path) -> np.ndarray:
    """读取置信度图并转换为 0~100。"""
    suffix = confidence_path.suffix.lower()
    confidence = np.load(confidence_path) if suffix == ".npy" else to_single_channel(
        imread_unicode(confidence_path), confidence_path
    )

    if confidence.ndim != 2:
        raise ValueError(f"置信度图不是二维数组：{confidence_path}，形状为 {confidence.shape}")

    return normalize_confidence(confidence, suffix)


# ============================================================
# 文件配对
# ============================================================

def find_confidence_file(mask_path: Path) -> Optional[Path]:
    """查找与掩码同目录、同主文件名的置信度文件。"""
    for suffix in CONFIDENCE_SUFFIXES:
        confidence_path = mask_path.parent / f"{mask_path.stem}{CONFIDENCE_NAME_SUFFIX}{suffix}"
        if confidence_path.exists():
            return confidence_path
    return None


def find_mask_files() -> list[Path]:
    """查找掩码图片，并排除所有 *_confidence 图片。"""
    return sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in MASK_SUFFIXES
        and not path.stem.endswith(CONFIDENCE_NAME_SUFFIX)
    )


# ============================================================
# 实例与置信度统计
# ============================================================

def get_confidence_level(confidence: float) -> str:
    """将置信度划分到指定区间。"""
    if confidence < 50:
        return "<50"
    if confidence < 60:
        return "50-60"
    if confidence < 70:
        return "60-70"
    if confidence < 80:
        return "70-80"
    if confidence < 90:
        return "80-90"
    return ">=90"


def extract_instances(mask: np.ndarray, confidence: np.ndarray, class_id: int) -> list[dict]:
    """
    使用外轮廓提取指定类别实例，并计算每个实例的平均置信度。

    与可视化逻辑保持一致：
    1. mask == class_id 后转成 0/255 二值图；
    2. 使用 RETR_EXTERNAL 和 CHAIN_APPROX_SIMPLE；
    3. 使用 cv2.contourArea(contour) < 10 过滤；
    4. 填充轮廓并计算该区域平均置信度。
    """
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    if not np.any(binary_mask):
        return []

    contours, _ = cv2.findContours(
        binary_mask,
        CONTOUR_RETRIEVAL_MODE,
        CONTOUR_APPROXIMATION_MODE,
    )

    instances = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < MIN_INSTANCE_AREA:
            continue

        contour_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)

        instance_region = contour_mask == 1
        pixel_confidences = confidence[instance_region]
        valid_confidences = pixel_confidences[np.isfinite(pixel_confidences)]

        if valid_confidences.size == 0:
            continue

        instances.append(
            {
                "instance_id": len(instances) + 1,
                "area": int(np.count_nonzero(instance_region)),
                "contour_area": contour_area,
                "mean_confidence": float(valid_confidences.mean()),
            }
        )

    return instances


def calculate_percentage(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total * 100.0


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    if not DATA_DIR.is_dir():
        raise NotADirectoryError(f"数据目录不存在：{DATA_DIR}")
    if not TARGET_CLASSES:
        raise ValueError("TARGET_CLASSES 不能为空")

    mask_files = find_mask_files()
    level_names = ["<50", "50-60", "60-70", "70-80", "80-90", ">=90"]

    total_level_statistics = {level: 0 for level in level_names}
    class_level_statistics = {
        class_id: {level: 0 for level in level_names}
        for class_id in TARGET_CLASSES
    }

    instance_records = []
    missing_confidence_files = []
    failed_files = []
    processed_images = 0
    total_instances = 0

    for mask_path in mask_files:
        confidence_path = find_confidence_file(mask_path)

        if confidence_path is None:
            missing_confidence_files.append(mask_path.name)
            print(f"[跳过] 未找到置信度图：{mask_path.name}")
            continue

        try:
            mask = load_mask(mask_path)
            confidence = load_confidence(confidence_path)

            if mask.shape != confidence.shape:
                raise ValueError(
                    f"掩码尺寸 {mask.shape} 与置信度图尺寸 {confidence.shape} 不一致"
                )

            image_instance_count = 0

            for class_id, class_name in TARGET_CLASSES.items():
                instances = extract_instances(mask, confidence, class_id)

                for instance in instances:
                    mean_confidence = instance["mean_confidence"]
                    level = get_confidence_level(mean_confidence)

                    total_instances += 1
                    image_instance_count += 1
                    total_level_statistics[level] += 1
                    class_level_statistics[class_id][level] += 1

                    instance_records.append(
                        {
                            "image_name": mask_path.name,
                            "class_id": class_id,
                            "class_name": class_name,
                            "instance_id": instance["instance_id"],
                            "area": instance["area"],
                            "contour_area": instance["contour_area"],
                            "mean_confidence": mean_confidence,
                            "level": level,
                        }
                    )

            processed_images += 1
            print(f"[完成] {mask_path.name}：{image_instance_count} 个实例")

        except Exception as error:
            failed_files.append(f"{mask_path.name}: {error}")
            print(f"[失败] {mask_path.name}：{error}")

    instance_records.sort(
        key=lambda record: (
            record["image_name"],
            record["class_id"],
            record["instance_id"],
        )
    )

    with OUTPUT_TXT.open("w", encoding="utf-8") as file:
        file.write("分割实例置信度统计\n")
        file.write("=" * 100 + "\n\n")

        file.write("一、统计配置\n")
        file.write("-" * 100 + "\n")
        file.write(f"数据目录：{DATA_DIR}\n")
        file.write(f"处理成功图片数量：{processed_images}\n")
        file.write(f"统计实例总数：{total_instances}\n")
        file.write(f"最小轮廓面积：{MIN_INSTANCE_AREA}\n")
        file.write("实例提取方式：cv2.findContours + RETR_EXTERNAL\n")
        file.write("置信度编码：uint8 / 255，再转换为百分数\n\n")

        file.write("统计类别：\n")
        for class_id, class_name in TARGET_CLASSES.items():
            file.write(f"  ID={class_id}, 类名={class_name}\n")

        file.write("\n二、实例详细结果\n")
        file.write("-" * 100 + "\n")
        file.write(
            f"{'图片名称':<28}"
            f"{'类别ID':<10}"
            f"{'类名':<18}"
            f"{'实例编号':<12}"
            f"{'像素数量':<12}"
            f"{'轮廓面积':<14}"
            f"{'平均置信度':<16}"
            f"{'等级':<10}\n"
        )
        file.write("-" * 100 + "\n")

        for record in instance_records:
            file.write(
                f"{record['image_name']:<28}"
                f"{record['class_id']:<10}"
                f"{record['class_name']:<18}"
                f"{record['instance_id']:<12}"
                f"{record['area']:<12}"
                f"{record['contour_area']:<14.2f}"
                f"{record['mean_confidence']:<16.2f}"
                f"{record['level']:<10}\n"
            )

        if not instance_records:
            file.write("没有找到符合条件的实例。\n")

        file.write("\n三、按类别分级统计\n")
        file.write("-" * 100 + "\n")

        for class_id, class_name in TARGET_CLASSES.items():
            class_statistics = class_level_statistics[class_id]
            class_total = sum(class_statistics.values())

            file.write(f"类别：{class_name} (ID={class_id})\n")
            file.write(f"实例总数：{class_total}\n")

            for level in level_names:
                count = class_statistics[level]
                percentage = calculate_percentage(count, class_total)
                file.write(f"  {level:<8}: {count:<8} 占比 {percentage:.2f}%\n")

            file.write("\n")

        file.write("四、最终分级统计\n")
        file.write("-" * 100 + "\n")

        for level in level_names:
            count = total_level_statistics[level]
            percentage = calculate_percentage(count, total_instances)
            file.write(f"{level:<8}: {count:<8} 占比 {percentage:.2f}%\n")

        if missing_confidence_files:
            file.write("\n五、缺少置信度图的掩码\n")
            file.write("-" * 100 + "\n")
            for filename in missing_confidence_files:
                file.write(f"{filename}\n")

        if failed_files:
            file.write("\n六、处理失败的文件\n")
            file.write("-" * 100 + "\n")
            for error_message in failed_files:
                file.write(f"{error_message}\n")

    print("\n统计完成")
    print(f"掩码图片数量：{len(mask_files)}")
    print(f"成功处理数量：{processed_images}")
    print(f"实例总数：{total_instances}")
    print(f"结果保存位置：{OUTPUT_TXT}")


if __name__ == "__main__":
    main()
