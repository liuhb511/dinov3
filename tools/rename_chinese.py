from pathlib import Path


# ==================== 在这里修改路径 ====================

ROOT_FOLDER = Path(r"F:/liuhaibo/datasets/test/JZW/DHTG/total")

# ======================================================


# ==================== 中英文映射表 ====================
# 格式：中文 -> 英文
# 请根据实际情况补充
NAME_MAP = {
    "镀锌板": "duxinban",
    "马口铁": "matoutie",
    "硅钢": "guigang",
    "镀锡": "duxi",
    "新": "new",
    "板": "ban"
    # 在这里继续添加...
}
# ======================================================


def rename_files():
    """批量重命名文件，将中文替换为英文"""
    
    if not ROOT_FOLDER.exists():
        raise FileNotFoundError(f"目录不存在：{ROOT_FOLDER}")
    
    renamed_count = 0
    skipped_count = 0
    
    # 遍历所有文件
    for file_path in ROOT_FOLDER.rglob("*"):
        
        if not file_path.is_file():
            continue
        
        # 只处理图片文件
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        if file_path.suffix.lower() not in image_extensions:
            continue
        
        original_name = file_path.name
        
        # 检查是否包含中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in original_name)
        if not has_chinese:
            skipped_count += 1
            continue
        
        # 替换中文
        new_name = original_name
        for cn, en in NAME_MAP.items():
            new_name = new_name.replace(cn, en)
        
        if new_name == original_name:
            skipped_count += 1
            continue
        
        # 重命名
        new_path = file_path.with_name(new_name)
        
        # 如果目标已存在，添加序号
        counter = 1
        while new_path.exists():
            stem = new_path.stem
            suffix = new_path.suffix
            new_path = file_path.with_name(f"{stem}_{counter}{suffix}")
            counter += 1
        
        file_path.rename(new_path)
        renamed_count += 1
        print(f"[已重命名] {original_name} -> {new_path.name}")
    
    print(f"\n完成！成功：{renamed_count}，跳过：{skipped_count}")


if __name__ == "__main__":
    rename_files()