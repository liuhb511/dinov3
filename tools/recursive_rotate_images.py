"""批量递归旋转图片文件夹中的所有图片文件（包括子文件夹），逆时针旋转 90 度，并原地保存。用于将纵向夹杂物图片旋转为横向。————哈汽轮项目，2026年9月9日"""
from pathlib import Path
from PIL import Image

# 指定根目录
folder = Path(r"D:/lhb/datasets/JZW_v3/JZW_h")

# 支持的图片格式
image_extensions = {".jpg", ".jpeg", ".png"}

success_count = 0
failed_count = 0

# 递归遍历根目录及所有子目录中的文件
for file in folder.rglob("*"):
    if not file.is_file():
        continue

    if file.suffix.lower() not in image_extensions:
        continue

    try:
        with Image.open(file) as img:
            # 逆时针旋转 90 度
            rotated = img.transpose(Image.Transpose.ROTATE_90)

            # JPG/JPEG 按指定质量重新保存
            if file.suffix.lower() in {".jpg", ".jpeg"}:
                rotated.save(file, format="JPEG", quality=95, subsampling=0)
            else:
                # PNG 等其它格式原地保存
                rotated.save(file)

        print(f"[成功] {file}")

        success_count += 1

    except Exception as e:
        print(f"[失败] {file}: {e}")
        failed_count += 1

print("\n处理完成")
print(f"成功旋转：{success_count} 张")
print(f"失败：{failed_count} 张")
