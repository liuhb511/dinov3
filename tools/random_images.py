import argparse
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".gif",
}


def remove_chinese(text):
    """移除字符串中的中文字符。"""
    return re.sub(r"[\u4e00-\u9fff]+", "", text)


def clean_name(name, fallback="unknown"):
    """移除中文，并清理首尾无意义字符。"""
    name = remove_chinese(name)
    name = name.strip(" _-.")
    return name if name else fallback


def collect_images(input_dir, output_dir, count=None, percentage=None, seed=None):
    """
    从 input_dir 下所有一级子文件夹中汇总图片，
    然后从全部图片中随机抽取指定数量或百分比，
    最终统一复制到 output_dir。

    输出图片命名格式：
        子文件夹名_原图片名
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹：{input_dir}")

    if count is None and percentage is None:
        raise ValueError("count 和 percentage 必须指定一个")

    if count is not None and percentage is not None:
        raise ValueError("count 和 percentage 只能指定一个")

    if count is not None and count < 0:
        raise ValueError("count 不能小于 0")

    if percentage is not None and not 0 <= percentage <= 100:
        raise ValueError("percentage 必须在 0~100 之间")

    if seed is not None:
        random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = []

    subfolders = [
        folder
        for folder in input_dir.iterdir()
        if folder.is_dir() and folder.resolve() != output_dir.resolve()
    ]

    if not subfolders:
        print("没有找到子文件夹。")
        return

    for subfolder in sorted(subfolders):
        images = [
            file
            for file in subfolder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(f"[扫描] {subfolder.name}: {len(images)} 张")

        for image_path in images:
            all_images.append((subfolder, image_path))

    total_images = len(all_images)

    if total_images == 0:
        print("没有找到任何图片。")
        return

    if count is not None:
        select_count = min(count, total_images)
    else:
        select_count = round(total_images * percentage / 100)

        if percentage > 0 and select_count == 0:
            select_count = 1

        select_count = min(select_count, total_images)

    print(f"\n共找到 {total_images} 张图片")
    print(f"随机抽取 {select_count} 张")

    selected_images = random.sample(all_images, select_count)

    for subfolder, image_path in selected_images:
        clean_folder_name = clean_name(subfolder.name, "folder")
        clean_image_stem = clean_name(image_path.stem, "image")
        extension = image_path.suffix.lower()

        new_name = f"{clean_folder_name}_{clean_image_stem}{extension}"
        destination = output_dir / new_name

        if destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            index = 1

            while destination.exists():
                destination = output_dir / f"{stem}_{index}{suffix}"
                index += 1

        shutil.copy2(image_path, destination)

    print("\n==============================")
    print("处理完成")
    print(f"原始图片总数：{total_images}")
    print(f"随机抽取数量：{select_count}")
    print(f"输出目录：{output_dir.resolve()}")
    print("==============================")


def main():
    parser = argparse.ArgumentParser(description="从多个子文件夹汇总图片后随机抽取")

    parser.add_argument("input_dir", help="输入文件夹")
    parser.add_argument("output_dir", help="输出文件夹")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-n",
        "--count",
        type=int,
        help="从全部图片中固定抽取多少张，例如 -n 300",
    )

    group.add_argument(
        "-p",
        "--percentage",
        type=float,
        help="从全部图片中按百分比抽取，例如 -p 20 表示 20%%",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，例如 --seed 42，可复现抽取结果",
    )

    args = parser.parse_args()

    collect_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        count=args.count,
        percentage=args.percentage,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()