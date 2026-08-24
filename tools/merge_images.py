from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# 配置区域
# ============================================================

# 左侧图片文件夹
# LEFT_FOLDER = Path(r"F:/liuhaibo/datasets/test/HQL_JZW_0810/4")
LEFT_FOLDER = Path(r"F:\liuhaibo\datasets\output\unet\LG_JZW_0807\20260703151744\overlay_single_unet/")

# 右侧图片文件夹
RIGHT_FOLDER = Path(r"F:\liuhaibo\datasets\output\unet\LG_JZW_0807\20260703151744\overlay_single_dino/")

# 拼接结果输出文件夹
OUTPUT_FOLDER = Path(r"F:\liuhaibo\datasets\output\unet\LG_JZW_0807\20260703151744\/images_merge")

# 左右图片标签
LEFT_LABEL = "gray"
RIGHT_LABEL = "color"

# 字体文件。Windows 可使用：
# C:\Windows\Fonts\times.ttf
# C:\Windows\Fonts\arial.ttf
# 中文建议使用：
# C:\Windows\Fonts\msyh.ttc
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

# 字号
FONT_SIZE = 36

# 标签距离图片边缘的距离
LABEL_MARGIN = 15

# 标签文字内边距
LABEL_PADDING = 8

# JPG 保存质量
JPG_QUALITY = 95

# 标签文字颜色
TEXT_COLOR = (255, 255, 255)

# 标签背景颜色
BACKGROUND_COLOR = (0, 0, 0)

# 标签背景透明度，0 完全透明，255 完全不透明
BACKGROUND_ALPHA = 180

# 支持的图片格式
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def load_font(font_path: str, font_size: int):
    """加载指定字体，加载失败时使用默认字体。"""
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"警告：无法加载字体 {font_path}，将使用默认字体。")
        return ImageFont.load_default()


def collect_images(folder: Path) -> dict[str, Path]:
    """
    收集文件夹中的图片。

    使用不含扩展名的文件名作为匹配键，例如：
    001.jpg 和 001.png 会被认为名字相同。
    """
    images = {}

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            key = file_path.stem

            if key in images:
                print(
                    f"警告：文件夹 {folder} 中存在同名图片："
                    f"{images[key].name} 和 {file_path.name}，"
                    f"将使用 {images[key].name}"
                )
                continue

            images[key] = file_path

    return images


def resize_to_same_height(
    left_image: Image.Image,
    right_image: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """保持宽高比，将两张图片缩放到相同高度。"""
    target_height = max(left_image.height, right_image.height)

    def resize_image(image: Image.Image) -> Image.Image:
        if image.height == target_height:
            return image

        scale = target_height / image.height
        target_width = max(1, round(image.width * scale))

        return image.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )

    return resize_image(left_image), resize_image(right_image)


def draw_label(
    image: Image.Image,
    text: str,
    position: str,
    font
) -> None:
    """
    在图片上绘制标签。

    position:
        left  表示左上角
        right 表示右上角
    """
    draw = ImageDraw.Draw(image, "RGBA")

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    box_width = text_width + LABEL_PADDING * 2
    box_height = text_height + LABEL_PADDING * 2

    if position == "left":
        box_x1 = LABEL_MARGIN
        box_x2 = box_x1 + box_width
    elif position == "right":
        box_x2 = image.width - LABEL_MARGIN
        box_x1 = box_x2 - box_width
    else:
        raise ValueError("position 只能是 'left' 或 'right'")

    box_y1 = LABEL_MARGIN
    box_y2 = box_y1 + box_height

    draw.rounded_rectangle(
        (box_x1, box_y1, box_x2, box_y2),
        radius=6,
        fill=(*BACKGROUND_COLOR, BACKGROUND_ALPHA)
    )

    text_x = box_x1 + LABEL_PADDING - text_bbox[0]
    text_y = box_y1 + LABEL_PADDING - text_bbox[1]

    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(*TEXT_COLOR, 255)
    )


def merge_images(
    left_path: Path,
    right_path: Path,
    output_path: Path,
    font
) -> None:
    """读取、缩放、添加标签并左右拼接两张图片。"""
    with Image.open(left_path) as left_source:
        left_image = ImageOps.exif_transpose(left_source).convert("RGB")

    with Image.open(right_path) as right_source:
        right_image = ImageOps.exif_transpose(right_source).convert("RGB")

    left_image, right_image = resize_to_same_height(
        left_image,
        right_image
    )

    draw_label(
        image=left_image,
        text=LEFT_LABEL,
        position="left",
        font=font
    )

    draw_label(
        image=right_image,
        text=RIGHT_LABEL,
        position="left",
        font=font
    )

    merged_width = left_image.width + right_image.width
    merged_height = max(left_image.height, right_image.height)

    merged_image = Image.new(
        mode="RGB",
        size=(merged_width, merged_height),
        color=(255, 255, 255)
    )

    merged_image.paste(left_image, (0, 0))
    merged_image.paste(right_image, (left_image.width, 0))

    merged_image.save(
        output_path,
        format="JPEG",
        quality=JPG_QUALITY,
        subsampling=0,
        optimize=True
    )


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    left_images = collect_images(LEFT_FOLDER)
    right_images = collect_images(RIGHT_FOLDER)

    common_names = sorted(set(left_images) & set(right_images))

    if not common_names:
        print("没有找到名字相同的图片。")
        return

    only_left = sorted(set(left_images) - set(right_images))
    only_right = sorted(set(right_images) - set(left_images))

    if only_left:
        print(f"左侧文件夹中有 {len(only_left)} 张图片未找到对应右图：")
        for name in only_left:
            print(f"  {left_images[name].name}")

    if only_right:
        print(f"右侧文件夹中有 {len(only_right)} 张图片未找到对应左图：")
        for name in only_right:
            print(f"  {right_images[name].name}")

    font = load_font(FONT_PATH, FONT_SIZE)

    success_count = 0
    failed_count = 0

    for index, name in enumerate(common_names, start=1):
        left_path = left_images[name]
        right_path = right_images[name]
        output_path = OUTPUT_FOLDER / f"{name}.jpg"

        try:
            merge_images(
                left_path=left_path,
                right_path=right_path,
                output_path=output_path,
                font=font
            )

            success_count += 1
            print(
                f"merge images: [{index}/{len(common_names)}] merge ok: "
                f"{output_path.name}"
            )

        except Exception as error:
            failed_count += 1
            print(
                f"[{index}/{len(common_names)}] 处理失败："
                f"{name}，原因：{error}"
            )

    print("\n处理完成。")
    print(f"成功：{success_count} 张")
    print(f"失败：{failed_count} 张")
    print(f"输出目录：{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()