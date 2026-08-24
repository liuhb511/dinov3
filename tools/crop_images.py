#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Labelme 简易裁剪脚本

支持两种裁剪方式：

1. stride 模式
   指定裁剪尺寸和步长。
   边缘不足一个完整裁剪窗口时直接舍弃。

2. count 模式
   指定横向裁剪数量和纵向裁剪数量。
   根据原图尺寸自动计算步长，使第一张贴左/上边，最后一张贴右/下边。

支持 Labelme:
    polygon
    circle

完整 circle 保持 circle；
被裁断的 circle 转为 polygon。
"""

import copy
import json
import math
from pathlib import Path

import cv2
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box


# ============================ 用户配置 ============================

IMAGE_DIR = Path(r"F:/dataset/images")
JSON_DIR = Path(r"F:/dataset/json")
OUTPUT_DIR = Path(r"F:/dataset/crops")

CROP_WIDTH = 1024
CROP_HEIGHT = 1024

# "stride"：指定步长
# "count" ：指定横纵裁剪数量
MODE = "stride"

# stride 模式参数
STRIDE_X = 512
STRIDE_Y = 512

# count 模式参数
COUNT_X = 4
COUNT_Y = 3

MIN_POLYGON_AREA = 1.0
CIRCLE_SEGMENTS = 128
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================ Labelme 几何处理 ============================

def get_shape_type(shape):
    return str(shape.get("shape_type", "polygon") or "polygon").strip().lower()


def valid_points(points, minimum):
    if not isinstance(points, list) or len(points) < minimum:
        return False

    try:
        return all(math.isfinite(float(p[0])) and math.isfinite(float(p[1])) for p in points)
    except (TypeError, ValueError, IndexError):
        return False


def polygon_from_points(points):
    try:
        geometry = Polygon([(float(p[0]), float(p[1])) for p in points])

        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        if geometry.is_empty:
            return None

        if isinstance(geometry, Polygon):
            return geometry

        if isinstance(geometry, MultiPolygon):
            return max(geometry.geoms, key=lambda g: g.area)

    except Exception:
        pass

    return None


def circle_to_polygon(points):
    try:
        cx, cy = float(points[0][0]), float(points[0][1])
        px, py = float(points[1][0]), float(points[1][1])
        radius = math.hypot(px - cx, py - cy)

        if radius <= 0:
            return None

        coords = []
        for i in range(CIRCLE_SEGMENTS):
            angle = 2 * math.pi * i / CIRCLE_SEGMENTS
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            coords.append((x, y))

        return Polygon(coords)

    except Exception:
        return None


def shape_to_geometry(shape):
    shape_type = get_shape_type(shape)
    points = shape.get("points", [])

    if shape_type == "polygon" and valid_points(points, 3):
        return polygon_from_points(points)

    if shape_type == "circle" and len(points) == 2 and valid_points(points, 2):
        return circle_to_polygon(points)

    return None


def extract_polygons(geometry):
    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        result = []
        for item in geometry.geoms:
            result.extend(extract_polygons(item))
        return result

    return []


def geometry_to_labelme_polygons(geometry, source_shape, crop_x, crop_y):
    result = []

    for polygon in extract_polygons(geometry):
        if polygon.area < MIN_POLYGON_AREA:
            continue

        coords = list(polygon.exterior.coords)

        if coords and coords[0] == coords[-1]:
            coords = coords[:-1]

        if len(coords) < 3:
            continue

        shape = copy.deepcopy(source_shape)
        shape["shape_type"] = "polygon"
        shape["points"] = [[float(x) - crop_x, float(y) - crop_y] for x, y in coords]
        result.append(shape)

    return result


def crop_shape(shape, crop_x, crop_y, crop_w, crop_h):
    geometry = shape_to_geometry(shape)

    if geometry is None or geometry.is_empty:
        return []

    crop_box = box(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
    clipped = geometry.intersection(crop_box)

    if clipped.is_empty or clipped.area < MIN_POLYGON_AREA:
        return []

    # 完整包含
    if crop_box.covers(geometry):
        if get_shape_type(shape) == "circle":
            new_shape = copy.deepcopy(shape)
            new_shape["points"] = [
                [float(p[0]) - crop_x, float(p[1]) - crop_y]
                for p in shape["points"]
            ]
            return [new_shape]

        return geometry_to_labelme_polygons(geometry, shape, crop_x, crop_y)

    # 被裁断，统一转 polygon
    return geometry_to_labelme_polygons(clipped, shape, crop_x, crop_y)


# ============================ 裁剪窗口生成 ============================

def generate_stride_positions(image_size, crop_size, stride):
    """
    固定步长。
    最后不足一个完整窗口的部分直接舍弃。
    """

    if stride <= 0:
        raise ValueError("步长必须大于 0")

    if image_size < crop_size:
        return []

    return list(range(0, image_size - crop_size + 1, stride))


def generate_count_positions(image_size, crop_size, count):
    """
    指定裁剪数量，自适应步长。

    第一张从 0 开始，
    最后一张贴到图像末端，
    中间位置均匀分布。
    """

    if count <= 0:
        raise ValueError("裁剪数量必须大于 0")

    if image_size < crop_size:
        return []

    if count == 1:
        return [0]

    max_start = image_size - crop_size
    step = max_start / (count - 1)

    positions = [round(i * step) for i in range(count)]
    positions[0] = 0
    positions[-1] = max_start

    return positions


def generate_windows(width, height):
    if MODE == "stride":
        xs = generate_stride_positions(width, CROP_WIDTH, STRIDE_X)
        ys = generate_stride_positions(height, CROP_HEIGHT, STRIDE_Y)

    elif MODE == "count":
        xs = generate_count_positions(width, CROP_WIDTH, COUNT_X)
        ys = generate_count_positions(height, CROP_HEIGHT, COUNT_Y)

    else:
        raise ValueError(f"未知裁剪模式：{MODE}")

    return [(x, y, CROP_WIDTH, CROP_HEIGHT) for y in ys for x in xs]


# ============================ 文件处理 ============================

def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)


def find_image(json_path, data):
    image_path = data.get("imagePath")

    if image_path:
        candidate = IMAGE_DIR / Path(str(image_path)).name
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGE_DIR / f"{json_path.stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def save_labelme_json(path, source_data, image_name, shapes):
    data = copy.deepcopy(source_data)
    data["imagePath"] = image_name
    data["imageData"] = None
    data["imageWidth"] = CROP_WIDTH
    data["imageHeight"] = CROP_HEIGHT
    data["shapes"] = shapes

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def process_one(json_path, output_dir):
    data = load_json(json_path)
    shapes = data.get("shapes", [])

    image_path = find_image(json_path, data)

    if image_path is None:
        print(f"[跳过] 找不到对应图片：{json_path.name}")
        return 0

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"[跳过] 图片读取失败：{image_path}")
        return 0

    height, width = image.shape[:2]

    if width < CROP_WIDTH or height < CROP_HEIGHT:
        print(
            f"[跳过] {image_path.name} 尺寸为 {width}×{height}，"
            f"小于裁剪尺寸 {CROP_WIDTH}×{CROP_HEIGHT}"
        )
        return 0

    windows = generate_windows(width, height)
    saved = 0

    for index, (x, y, crop_w, crop_h) in enumerate(windows):
        cropped = image[y:y + crop_h, x:x + crop_w].copy()

        if cropped.shape[:2] != (crop_h, crop_w):
            continue

        cropped_shapes = []

        for shape in shapes:
            cropped_shapes.extend(crop_shape(shape, x, y, crop_w, crop_h))

        stem = f"{json_path.stem}_crop_{index:03d}_x{x}_y{y}"
        image_name = f"{stem}.jpg"
        json_name = f"{stem}.json"

        image_out = output_dir / image_name
        json_out = output_dir / json_name

        if not cv2.imwrite(str(image_out), cropped):
            print(f"[失败] 图片保存失败：{image_out}")
            continue

        save_labelme_json(json_out, data, image_name, cropped_shapes)
        saved += 1

    return saved


# ============================ 主程序 ============================

def main():
    output_dir = OUTPUT_DIR / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not JSON_DIR.is_dir():
        raise NotADirectoryError(f"JSON_DIR 不存在：{JSON_DIR}")

    if not IMAGE_DIR.is_dir():
        raise NotADirectoryError(f"IMAGE_DIR 不存在：{IMAGE_DIR}")

    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"没有找到 JSON：{JSON_DIR}")
        return

    print(f"裁剪模式：{MODE}")
    print(f"裁剪尺寸：{CROP_WIDTH}×{CROP_HEIGHT}")

    if MODE == "stride":
        print(f"固定步长：X={STRIDE_X}, Y={STRIDE_Y}")
    else:
        print(f"指定数量：横向={COUNT_X}, 纵向={COUNT_Y}")

    total = 0

    for index, json_path in enumerate(json_files, 1):
        try:
            count = process_one(json_path, output_dir)
            total += count
            print(f"[{index}/{len(json_files)}] {json_path.name}：{count} 张")
        except Exception as exc:
            print(f"[错误] {json_path.name}：{exc}")

    print(f"处理完成，共生成 {total} 张裁剪图")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()