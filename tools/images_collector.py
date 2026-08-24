from pathlib import Path
import shutil


# ==================== 在这里修改路径 ====================

# 给定目录
ROOT_FOLDER = Path(r"F:/liuhaibo/datasets/test/JZW/DHTG")

# 汇总后的图片保存目录
OUTPUT_FOLDER = ROOT_FOLDER / "total"

# ======================================================


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# 需要跳过的关键词（不区分大小写）
SKIP_KEYWORDS = {"stitching", "image"}


def get_unique_path(file_path: Path) -> Path:
    """
    如果目标文件已存在，自动添加序号，防止覆盖。
    """
    if not file_path.exists():
        return file_path

    index = 2

    while True:
        new_path = file_path.with_name(
            f"{file_path.stem}_{index}{file_path.suffix}"
        )

        if not new_path.exists():
            return new_path

        index += 1


def should_skip_file(filename: str) -> bool:
    """
    检查文件名是否包含需要跳过的关键词。
    """
    filename_lower = filename.lower()
    for keyword in SKIP_KEYWORDS:
        if keyword in filename_lower:
            return True
    return False


def collect_images():
    if not ROOT_FOLDER.exists():
        raise FileNotFoundError(f"目录不存在：{ROOT_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0

    # 遍历给定目录下的一级子文件夹
    for sub_folder in ROOT_FOLDER.iterdir():

        if not sub_folder.is_dir():
            continue

        # 跳过输出目录本身
        if sub_folder.resolve() == OUTPUT_FOLDER.resolve():
            continue

        # 获取文件夹名并去除所有空格
        folder_name = sub_folder.name.replace(" ", "")

        # 遍历该子文件夹中的文件
        for image_path in sub_folder.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # 如果文件名包含 "Stitching" 或 "Image"，跳过
            if should_skip_file(image_path.name):
                print(
                    f"[跳过] "
                    f"{sub_folder.name}/{image_path.name} "
                    f"(包含关键词：{image_path.name})"
                )
                skipped_count += 1
                continue

            # 新文件名：子文件夹名（去空格） + "_" + 原文件名
            new_filename = f"{folder_name}_{image_path.name}"

            output_path = OUTPUT_FOLDER / new_filename
            output_path = get_unique_path(output_path)

            shutil.copy2(image_path, output_path)

            copied_count += 1

            print(
                f"[已复制] "
                f"{sub_folder.name}/{image_path.name} "
                f"-> {output_path.name}"
            )

    print("\n汇总完成")
    print(f"共复制：{copied_count} 张图片")
    print(f"跳过：{skipped_count} 张图片")
    print(f"输出目录：{OUTPUT_FOLDER}")


if __name__ == "__main__":
    collect_images()