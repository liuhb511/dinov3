import os
import re
import shutil

input_dir = r"F:/liuhaibo/datasets/test/JZW/LG_JZW_0807"
output_dir = r"F:/liuhaibo/datasets/test/JZW/LG_JZW_0807_total"

os.makedirs(output_dir, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

for root, _, files in os.walk(input_dir):
    if root == input_dir:
        continue

    folder = os.path.basename(root)
    folder_clean = re.sub(r'[\u4e00-\u9fff]', '', folder)

    print(f"正在处理文件夹：{folder}")

    for file in files:
        name, ext = os.path.splitext(file)

        if ext.lower() not in img_exts:
            continue

        name = re.sub(r'[\u4e00-\u9fff]', '', name)

        new_name = f"{folder_clean}_{name}{ext}"
        src = os.path.join(root, file)
        dst = os.path.join(output_dir, new_name)

        shutil.copy2(src, dst)

print("处理完成")