from pathlib import Path

# ==================== 在这里修改路径 ====================
ROOT_FOLDER = Path(r"D:/lhb/datasets/JZW_v4/ZDX")
# ======================================================

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".json"
}

def rename_files():
    if not ROOT_FOLDER.exists():
        raise FileNotFoundError(f"目录不存在：{ROOT_FOLDER}")

    if not ROOT_FOLDER.is_dir():
        raise NotADirectoryError(f"不是文件夹：{ROOT_FOLDER}")

    prefix = "Z_"
    renamed_count = 0
    skipped_count = 0

    # 递归遍历 ROOT_FOLDER 下的所有文件和子文件夹
    for file_path in ROOT_FOLDER.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        # 已经有指定前缀则跳过
        if file_path.name.startswith(prefix):
            print(f"[跳过] 已有前缀：{file_path}")
            skipped_count += 1
            continue

        new_name = f"{prefix}{file_path.name}"
        new_path = file_path.with_name(new_name)

        # 防止覆盖已有文件
        if new_path.exists():
            print(f"[跳过] 目标文件已存在：{new_path}")
            skipped_count += 1
            continue

        old_path = file_path
        file_path.rename(new_path)

        renamed_count += 1
        print(f"[已重命名] {old_path} -> {new_path}")

    print("\n处理完成")
    print(f"添加前缀：{prefix}")
    print(f"成功重命名：{renamed_count} 个文件")
    print(f"跳过：{skipped_count} 个文件")

if __name__ == "__main__":
    rename_files()
