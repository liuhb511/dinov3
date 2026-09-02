from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ============================================================
# 配置区域
# ============================================================

IMAGE_FOLDERS = [
    Path(r"output/LG/compare/LG_0807/LG_JZW_0807_300_new/overlay/"),
    Path(r"output/LG/compare/LG_JZW_0807_300_2/overlay/"),
    Path(r"output/LG/compare/LG_0807/dino_inclusion/"),
    Path(r"output/LG/compare/LG_0807/unet/"),
]

IMAGE_LABELS = [
    "Dino_JZW",
    "Dino_2",
    "v2",
    "unet",
]

OUTPUT_FOLDER = Path(r"output/LG/compare/merge3/")

FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
FONT_SIZE = 36

LABEL_MARGIN = 15
LABEL_PADDING = 8

JPG_QUALITY = 95

TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
BACKGROUND_ALPHA = 180

CANVAS_COLOR = (255, 255, 255)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def load_font(font_path: str, font_size: int):
    """加载指定字体，失败时使用默认字体。"""
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"警告：无法加载字体 {font_path}，将使用默认字体。")
        return ImageFont.load_default()


def validate_config():
    """检查输入文件夹和标签配置是否合法。"""
    if len(IMAGE_FOLDERS) != 4:
        raise ValueError("IMAGE_FOLDERS 必须包含 4 个位置，不使用的位置请填写 None")

    if len(IMAGE_LABELS) != 4:
        raise ValueError("IMAGE_LABELS 必须包含 4 个位置")

    active_indices = [
        index
        for index, folder in enumerate(IMAGE_FOLDERS)
        if folder is not None
    ]

    if len(active_indices) < 2:
        raise ValueError("至少需要配置 2 个输入文件夹")

    labels = [IMAGE_LABELS[index].strip() for index in active_indices]
    has_label = [bool(label) for label in labels]

    if any(has_label) and not all(has_label):
        raise ValueError("标签配置错误：标签必须全部填写或全部留空，不允许只给部分图片设置标签")

    return active_indices, all(has_label)


def collect_images(folder: Path) -> dict[str, Path]:
    """收集文件夹中的图片，使用不含扩展名的文件名作为匹配键。"""
    images = {}

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"路径不是文件夹：{folder}")

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = file_path.stem

        if key in images:
            print(
                f"警告：文件夹 {folder} 中存在同名图片："
                f"{images[key].name} 和 {file_path.name}，将使用 {images[key].name}"
            )
            continue

        images[key] = file_path

    return images


def load_image(image_path: Path) -> Image.Image:
    """读取图片并处理 EXIF 方向。"""
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def normalize_images(images: list[Image.Image]) -> list[Image.Image]:
    """保持宽高比，将多张图片统一到相同画布尺寸。"""
    target_width = max(image.width for image in images)
    target_height = max(image.height for image in images)

    normalized = []

    for image in images:
        contained = ImageOps.contain(
            image,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
        )

        canvas = Image.new("RGB", (target_width, target_height), CANVAS_COLOR)

        x = (target_width - contained.width) // 2
        y = (target_height - contained.height) // 2

        canvas.paste(contained, (x, y))
        normalized.append(canvas)

    return normalized


def draw_label(image: Image.Image, text: str, font) -> None:
    """在图片左上角绘制标签。"""
    if not text:
        return

    draw = ImageDraw.Draw(image, "RGBA")
    text_bbox = draw.textbbox((0, 0), text, font=font)

    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    box_width = text_width + LABEL_PADDING * 2
    box_height = text_height + LABEL_PADDING * 2

    box_x1 = LABEL_MARGIN
    box_y1 = LABEL_MARGIN
    box_x2 = box_x1 + box_width
    box_y2 = box_y1 + box_height

    draw.rounded_rectangle(
        (box_x1, box_y1, box_x2, box_y2),
        radius=6,
        fill=(*BACKGROUND_COLOR, BACKGROUND_ALPHA),
    )

    text_x = box_x1 + LABEL_PADDING - text_bbox[0]
    text_y = box_y1 + LABEL_PADDING - text_bbox[1]

    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(*TEXT_COLOR, 255),
    )


def merge_two(images: list[Image.Image]) -> Image.Image:
    """两张图片左右排列。"""
    width = images[0].width
    height = images[0].height

    merged = Image.new("RGB", (width * 2, height), CANVAS_COLOR)

    merged.paste(images[0], (0, 0))
    merged.paste(images[1], (width, 0))

    return merged


def merge_three(images: list[Image.Image]) -> Image.Image:
    """三张图片左、中、右排列。"""
    width = images[0].width
    height = images[0].height

    merged = Image.new("RGB", (width * 3, height), CANVAS_COLOR)

    merged.paste(images[0], (0, 0))
    merged.paste(images[1], (width, 0))
    merged.paste(images[2], (width * 2, 0))

    return merged


def merge_four(images: list[Image.Image]) -> Image.Image:
    """四张图片按照 2×2 四宫格排列。"""
    width = images[0].width
    height = images[0].height

    merged = Image.new("RGB", (width * 2, height * 2), CANVAS_COLOR)

    merged.paste(images[0], (0, 0))
    merged.paste(images[1], (width, 0))
    merged.paste(images[2], (0, height))
    merged.paste(images[3], (width, height))

    return merged


def merge_images(image_paths, labels, output_path, font, show_labels):
    """读取图片、添加标签并进行拼接。"""
    images = [load_image(path) for path in image_paths]
    images = normalize_images(images)

    if show_labels:
        for image, label in zip(images, labels):
            draw_label(image, label, font)

    image_count = len(images)

    if image_count == 2:
        merged_image = merge_two(images)
    elif image_count == 3:
        merged_image = merge_three(images)
    elif image_count == 4:
        merged_image = merge_four(images)
    else:
        raise ValueError(f"只能拼接 2~4 张图片，当前为 {image_count} 张")

    merged_image.save(
        output_path,
        format="JPEG",
        quality=JPG_QUALITY,
        subsampling=0,
        optimize=True,
    )


def main():
    active_indices, show_labels = validate_config()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    image_maps = {}

    print("扫描输入文件夹：")

    for index in active_indices:
        folder = IMAGE_FOLDERS[index]
        images = collect_images(folder)
        image_maps[index] = images

        label = IMAGE_LABELS[index] if show_labels else "无标签"

        print(f"  [{index + 1}] {folder}: {len(images)} 张，标签：{label}")

    all_names = set()

    for images in image_maps.values():
        all_names.update(images.keys())

    all_names = sorted(all_names)

    if not all_names:
        print("没有找到任何图片。")
        return

    font = load_font(FONT_PATH, FONT_SIZE)

    success_count = 0
    failed_count = 0
    skipped_count = 0

    print(f"\n共发现 {len(all_names)} 个不同图片名称。\n")

    for index, name in enumerate(all_names, start=1):
        image_paths = []
        labels = []

        for folder_index in active_indices:
            image_path = image_maps[folder_index].get(name)

            if image_path is None:
                continue

            image_paths.append(image_path)
            labels.append(IMAGE_LABELS[folder_index])

        image_count = len(image_paths)

        if image_count < 2:
            skipped_count += 1
            print(f"[{index}/{len(all_names)}] 跳过：{name}，只找到 {image_count} 张图片")
            continue

        output_path = OUTPUT_FOLDER / f"{name}.jpg"

        try:
            merge_images(image_paths, labels, output_path, font, show_labels)

            success_count += 1
            print(f"[{index}/{len(all_names)}] merge ok: {output_path.name} ({image_count} images)")

        except Exception as error:
            failed_count += 1
            print(f"[{index}/{len(all_names)}] 处理失败：{name}，原因：{error}")

    print("\n==============================")
    print("处理完成")
    print(f"成功：{success_count}")
    print(f"跳过：{skipped_count}")
    print(f"失败：{failed_count}")
    print(f"输出目录：{OUTPUT_FOLDER.resolve()}")
    print("==============================")


if __name__ == "__main__":
    main()