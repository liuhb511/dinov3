from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_FOLDERS = [
    Path(r"D:/lhb/datasets/testsets/HMS/test/test01/"),
    Path(r"D:/lhb/datasets/testsets/HMS/test/result/overlay/"),
    Path(r"D:/lhb/datasets/testsets/HMS/test/output_lab/"),
    None,
]

IMAGE_LABELS = ["GT", "V1", "V2", None]

OUTPUT_FOLDER = Path(r"D:/lhb/datasets/testsets/HMS/test/merge/GTV1V2/")

# "horizontal"：左右拼接
# "vertical"：上下拼接
CONCAT_DIRECTION = "vertical"

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
    try:
        return ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"警告：无法加载字体 {font_path}，将使用默认字体。")
        return ImageFont.load_default()


def validate_config():
    if len(IMAGE_FOLDERS) != 4:
        raise ValueError("IMAGE_FOLDERS 必须包含 4 个位置")

    if len(IMAGE_LABELS) != 4:
        raise ValueError("IMAGE_LABELS 必须包含 4 个位置")

    if CONCAT_DIRECTION not in {"horizontal", "vertical"}:
        raise ValueError('CONCAT_DIRECTION 只能是 "horizontal" 或 "vertical"')

    active_indices = [
        i for i, folder in enumerate(IMAGE_FOLDERS)
        if folder is not None
    ]

    if len(active_indices) < 2:
        raise ValueError("至少需要配置 2 个输入文件夹")

    labels = [
        IMAGE_LABELS[i].strip() if IMAGE_LABELS[i] else ""
        for i in active_indices
    ]

    has_label = [bool(label) for label in labels]

    if any(has_label) and not all(has_label):
        raise ValueError("标签必须全部填写或全部留空")

    return active_indices, all(has_label)


def collect_images(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    images = {}

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        key = file_path.stem

        if key in images:
            print(
                f"警告：{folder} 中存在同名图片："
                f"{images[key].name} 和 {file_path.name}，"
                f"保留 {images[key].name}"
            )
            continue

        images[key] = file_path

    return images


def load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def normalize_images(images: list[Image.Image]) -> list[Image.Image]:
    target_width = max(image.width for image in images)
    target_height = max(image.height for image in images)

    normalized = []

    for image in images:
        contained = ImageOps.contain(
            image, (target_width, target_height), method=Image.Resampling.LANCZOS
        )

        canvas = Image.new("RGB", (target_width, target_height), CANVAS_COLOR)

        x = (target_width - contained.width) // 2
        y = (target_height - contained.height) // 2

        canvas.paste(contained, (x, y))
        normalized.append(canvas)

    return normalized


def draw_label(image: Image.Image, text: str, font) -> None:
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

    draw.text((text_x, text_y), text, font=font, fill=(*TEXT_COLOR, 255))


def merge_horizontal(images: list[Image.Image]) -> Image.Image:
    width = images[0].width
    height = images[0].height

    merged = Image.new("RGB", (width * len(images), height), CANVAS_COLOR)

    for index, image in enumerate(images):
        merged.paste(image, (width * index, 0))

    return merged


def merge_vertical(images: list[Image.Image]) -> Image.Image:
    width = images[0].width
    height = images[0].height

    merged = Image.new("RGB", (width, height * len(images)), CANVAS_COLOR)

    for index, image in enumerate(images):
        merged.paste(image, (0, height * index))

    return merged


def merge_images(image_paths, labels, output_path, font, show_labels):
    images = [load_image(path) for path in image_paths]
    images = normalize_images(images)

    if show_labels:
        for image, label in zip(images, labels):
            draw_label(image, label, font)

    image_count = len(images)

    if image_count < 2 or image_count > 4:
        raise ValueError(f"只能拼接 2~4 张图片，当前为 {image_count} 张")

    if CONCAT_DIRECTION == "horizontal":
        merged_image = merge_horizontal(images)
    else:
        merged_image = merge_vertical(images)

    merged_image.save(
        output_path, format="JPEG", quality=JPG_QUALITY, subsampling=0, optimize=True
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

    direction_text = "水平（左 → 右）" if CONCAT_DIRECTION == "horizontal" else "垂直（上 → 下）"

    print(f"\n共发现 {len(all_names)} 个不同图片名称。")
    print(f"当前拼接方式：{direction_text}\n")

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
            print(
                f"[{index}/{len(all_names)}] "
                f"跳过：{name}，只找到 {image_count} 张图片"
            )
            continue

        output_path = OUTPUT_FOLDER / f"{name}.jpg"

        try:
            merge_images(image_paths, labels, output_path, font, show_labels)

            success_count += 1

            print(
                f"[{index}/{len(all_names)}] "
                f"merge ok: {output_path.name} "
                f"({image_count} images)"
            )

        except Exception as error:
            failed_count += 1
            print(
                f"[{index}/{len(all_names)}] "
                f"处理失败：{name}，原因：{error}"
            )

    print("\n==============================")
    print("处理完成")
    print(f"拼接方式：{direction_text}")
    print(f"成功：{success_count}")
    print(f"跳过：{skipped_count}")
    print(f"失败：{failed_count}")
    print(f"输出目录：{OUTPUT_FOLDER.resolve()}")
    print("==============================")


if __name__ == "__main__":
    main()
