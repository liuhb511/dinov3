import os
import json
import base64
from io import BytesIO
from PIL import Image


def convert_to_rgb(image):
    """将图片转换为 RGB，透明区域使用白色背景。"""
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background

    if image.mode == "P":
        if "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")

    if image.mode != "RGB":
        return image.convert("RGB")

    return image


def image_to_base64(image):
    """将图片编码成 Labelme imageData 使用的 JPG base64。"""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def convert_png_to_jpg(png_path):
    """将单张 PNG 转换成 JPG，并修改对应的 Labelme JSON。"""
    directory = os.path.dirname(png_path)
    filename = os.path.basename(png_path)
    name, _ = os.path.splitext(filename)

    jpg_filename = f"{name}.jpg"
    jpg_path = os.path.join(directory, jpg_filename)
    json_path = os.path.join(directory, f"{name}.json")

    # 读取并转换图片
    with Image.open(png_path) as image:
        image = convert_to_rgb(image)
        width, height = image.size
        image.save(jpg_path, format="JPEG", quality=95)

        # 修改对应 JSON
        if os.path.isfile(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["imagePath"] = jpg_filename
            data["imageWidth"] = width
            data["imageHeight"] = height

            # 如果原 JSON 存在 imageData，则同步替换为 JPG 数据
            if data.get("imageData") is not None:
                data["imageData"] = image_to_base64(image)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"[OK] {filename} -> {jpg_filename} | JSON 已更新")
        else:
            print(f"[WARN] {filename} -> {jpg_filename} | 未找到对应 JSON")

    # JPG 成功生成后删除原 PNG
    os.remove(png_path)


def process_directory(input_dir):
    """处理目录下所有 PNG 文件。"""
    input_dir = os.path.abspath(input_dir)

    if not os.path.isdir(input_dir):
        raise ValueError(f"目录不存在: {input_dir}")

    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    success_count = 0
    error_count = 0

    print(f"目录: {input_dir}")
    print(f"发现 {len(png_files)} 个 PNG 文件")
    print("-" * 80)

    for filename in png_files:
        png_path = os.path.join(input_dir, filename)

        try:
            convert_png_to_jpg(png_path)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            error_count += 1

    print("-" * 80)
    print(f"处理完成 | 成功: {success_count} | 失败: {error_count}")


if __name__ == "__main__":
    input_dir = r"D:/lhb/datasets/DBXH/DHTG0921/labeled"
    process_directory(input_dir)