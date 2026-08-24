#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Labelme 语义分割数据智能离线裁剪脚本

功能：
1. 不 resize，不 letterbox；
2. 将原图离线裁剪为固定 784×784；
3. 支持 Labelme：
   - polygon：至少 3 个点；
   - circle：2 个点，points[0] 为圆心，points[1] 为圆周点；
4. 自动生成候选裁剪窗口并评分；
5. 优先保留标注较多、完整目标较多、空白较少的区域；
6. 尽量避免重复裁剪；
7. polygon 被边界切开时，使用 Shapely 裁切后继续保存；
8. circle 完整落入裁剪框时保留为 circle；
9. circle 被边界切开时，不写入裁剪 JSON，并对候选窗口施加惩罚；
10. 输出目录仅包含：
    - images：裁剪图像和对应的 Labelme JSON；
    - logs：manifest.csv 和 warning.log。

依赖：
    pip install opencv-python numpy shapely

说明：
- 所有参数均在“用户配置区”修改；
- 若原图尺寸小于 784×784，可使用反射填充；
- 默认每张原图最多裁剪 MAX_CROPS_PER_IMAGE 张。
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box


# ============================================================
# 用户配置
# ============================================================

# 原图目录
IMAGE_DIR = Path(r"F:/liuhaibo/datasets/BG_HQL_JZW/data_V2_trainval")

# Labelme JSON 目录
JSON_DIR = Path(r"F:/liuhaibo/datasets/BG_HQL_JZW/data_V2_trainval")

# 输出根目录
OUTPUT_DIR = Path(r"F:/liuhaibo/datasets/BG_HQL_JZW/dataset_crops_1024_v2")

# 是否递归扫描 JSON
RECURSIVE = True

# 裁剪尺寸
CROP_SIZE = 1024

# 每张原图最多输出多少张裁剪图
MAX_CROPS_PER_IMAGE = 4

# 候选窗口步长
# 越小候选越多，速度越慢，但更容易找到更好的区域
WINDOW_STRIDE = 98

# 目标中心周围额外生成候选窗口
USE_OBJECT_CENTER_CANDIDATES = True

# 裁剪图之间允许的最大重叠比例
# overlap = intersection_area / min(area_a, area_b)
MAX_CROP_OVERLAP = 0

# 候选窗口中最少前景像素数
MIN_FOREGROUND_PIXELS = 30

# 是否允许对小于裁剪尺寸的图像做少量填充
ALLOW_PADDING = True

# 图像后缀搜索顺序
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

# 候选窗口评分权重
SCORE_FOREGROUND_RATIO = 2.0
SCORE_COMPLETE_SHAPE = 4.0
SCORE_CUT_POLYGON = -3.0
SCORE_CUT_CIRCLE = -6.0
SCORE_OBJECT_DIVERSITY = 1.5
SCORE_EMPTY_RATIO = -0.5

# polygon 裁切后最小有效面积
MIN_POLYGON_AREA = 1.0


# ============================================================
# 数据结构
# ============================================================

@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    size: int

    @property
    def x2(self) -> int:
        return self.x + self.size

    @property
    def y2(self) -> int:
        return self.y + self.size

    @property
    def area(self) -> int:
        return self.size * self.size


@dataclass
class CandidateScore:
    window: CropWindow
    score: float
    foreground_pixels: int
    foreground_ratio: float
    complete_shapes: int
    cut_polygons: int
    cut_circles: int
    labels_count: int


# ============================================================
# 基础工具
# ============================================================

def log_warning(message: str, warning_lines: List[str]) -> None:
    print(f"[警告] {message}")
    warning_lines.append(message)


def load_json(json_path: Path) -> Dict[str, Any]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with json_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON 根节点不是对象")

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError("JSON 中 shapes 不是列表")

    return data


def find_image_path(
    json_path: Path,
    json_data: Dict[str, Any],
) -> Optional[Path]:
    image_path_field = json_data.get("imagePath")

    if image_path_field:
        candidate = Path(str(image_path_field))

        if candidate.is_absolute() and candidate.exists():
            return candidate

        local_candidate = json_path.parent / candidate
        if local_candidate.exists():
            return local_candidate

        root_candidate = IMAGE_DIR / candidate.name
        if root_candidate.exists():
            return root_candidate

    stem = json_path.stem

    for ext in IMAGE_EXTENSIONS:
        candidate = IMAGE_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    for ext in IMAGE_EXTENSIONS:
        matches = list(IMAGE_DIR.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]

    return None


def ensure_output_dirs() -> Dict[str, Path]:
    dirs = {
        "images": OUTPUT_DIR / "images",
        "logs": OUTPUT_DIR / "logs",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def is_valid_point(point: Any) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False

    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return False

    return math.isfinite(x) and math.isfinite(y)


def validate_points(points: Any, minimum_count: int) -> bool:
    return (
        isinstance(points, list)
        and len(points) >= minimum_count
        and all(is_valid_point(point) for point in points)
    )


# ============================================================
# Labelme shape 几何处理
# ============================================================

def get_shape_type(shape: Dict[str, Any]) -> str:
    return str(shape.get("shape_type", "polygon") or "polygon").strip().lower()


def circle_parameters(
    points: Sequence[Sequence[float]],
) -> Tuple[float, float, float]:
    center_x = float(points[0][0])
    center_y = float(points[0][1])
    edge_x = float(points[1][0])
    edge_y = float(points[1][1])
    radius = math.hypot(edge_x - center_x, edge_y - center_y)

    return center_x, center_y, radius


def circle_fully_inside_crop(
    points: Sequence[Sequence[float]],
    window: CropWindow,
) -> bool:
    center_x, center_y, radius = circle_parameters(points)

    return (
        center_x - radius >= window.x
        and center_x + radius <= window.x2
        and center_y - radius >= window.y
        and center_y + radius <= window.y2
    )


def circle_intersects_crop(
    points: Sequence[Sequence[float]],
    window: CropWindow,
) -> bool:
    center_x, center_y, radius = circle_parameters(points)

    nearest_x = min(max(center_x, window.x), window.x2)
    nearest_y = min(max(center_y, window.y), window.y2)

    distance_squared = (
        (center_x - nearest_x) ** 2
        + (center_y - nearest_y) ** 2
    )

    return distance_squared <= radius ** 2


def get_circle_crop_status(
    points: Sequence[Sequence[float]],
    window: CropWindow,
) -> str:
    if circle_fully_inside_crop(points, window):
        return "complete"

    if circle_intersects_crop(points, window):
        return "cut"

    return "outside"


def polygon_from_points(
    points: Sequence[Sequence[float]],
) -> Optional[Polygon]:
    try:
        polygon = Polygon([
            (float(point[0]), float(point[1]))
            for point in points
        ])

        if polygon.is_empty:
            return None

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.is_empty:
            return None

        if isinstance(polygon, Polygon):
            return polygon

        if isinstance(polygon, MultiPolygon):
            return max(polygon.geoms, key=lambda geom: geom.area)

    except Exception:
        return None

    return None


def extract_polygons_from_geometry(geometry: Any) -> List[Polygon]:
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        polygons: List[Polygon] = []

        for geom in geometry.geoms:
            polygons.extend(extract_polygons_from_geometry(geom))

        return polygons

    return []


def translate_circle_to_crop(
    shape: Dict[str, Any],
    window: CropWindow,
) -> Dict[str, Any]:
    new_shape = copy.deepcopy(shape)
    new_shape["shape_type"] = "circle"
    new_shape["points"] = [
        [
            float(shape["points"][0][0]) - window.x,
            float(shape["points"][0][1]) - window.y,
        ],
        [
            float(shape["points"][1][0]) - window.x,
            float(shape["points"][1][1]) - window.y,
        ],
    ]

    return new_shape


def clip_polygon_to_crop(
    shape: Dict[str, Any],
    window: CropWindow,
) -> List[Dict[str, Any]]:
    polygon = polygon_from_points(shape.get("points", []))

    if polygon is None:
        return []

    crop_box = box(window.x, window.y, window.x2, window.y2)
    clipped = polygon.intersection(crop_box)
    result_shapes: List[Dict[str, Any]] = []

    for clipped_polygon in extract_polygons_from_geometry(clipped):
        if clipped_polygon.area < MIN_POLYGON_AREA:
            continue

        coords = list(clipped_polygon.exterior.coords)

        # Shapely 的首尾点重复，Labelme polygon 不需要最后一个重复点
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]

        if len(coords) < 3:
            continue

        new_shape = copy.deepcopy(shape)
        new_shape["shape_type"] = "polygon"
        new_shape["points"] = [
            [
                float(x) - window.x,
                float(y) - window.y,
            ]
            for x, y in coords
        ]

        result_shapes.append(new_shape)

    return result_shapes


def crop_shape_to_window(
    shape: Dict[str, Any],
    window: CropWindow,
    json_path: Path,
    warning_lines: List[str],
) -> List[Dict[str, Any]]:
    shape_type = get_shape_type(shape)
    points = shape.get("points", [])
    label = str(shape.get("label", "") or "")

    if shape_type == "circle":
        if not validate_points(points, 2) or len(points) != 2:
            point_count = len(points) if isinstance(points, list) else "invalid"
            log_warning(
                f"跳过异常 circle：{json_path}，"
                f"label={label}，points={point_count}",
                warning_lines,
            )
            return []

        if get_circle_crop_status(points, window) == "complete":
            return [translate_circle_to_crop(shape, window)]

        return []

    if shape_type == "polygon":
        if not validate_points(points, 3):
            point_count = len(points) if isinstance(points, list) else "invalid"
            log_warning(
                f"跳过异常 polygon：{json_path}，"
                f"label={label}，points={point_count}",
                warning_lines,
            )
            return []

        return clip_polygon_to_crop(shape, window)

    log_warning(
        f"暂不支持 shape_type={shape_type}：{json_path}，label={label}",
        warning_lines,
    )

    return []


# ============================================================
# 填充与裁剪
# ============================================================

def pad_image_if_needed(
    image: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]

    pad_bottom = max(0, CROP_SIZE - height)
    pad_right = max(0, CROP_SIZE - width)

    if pad_bottom == 0 and pad_right == 0:
        return image, 0, 0

    if not ALLOW_PADDING:
        raise ValueError(
            f"图像尺寸 {width}×{height} 小于裁剪尺寸 {CROP_SIZE}，"
            "且未允许填充"
        )

    padded_image = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded_image, pad_right, pad_bottom


def crop_image(
    image: np.ndarray,
    window: CropWindow,
) -> np.ndarray:
    return image[
        window.y:window.y2,
        window.x:window.x2,
    ].copy()


# ============================================================
# 候选窗口生成
# ============================================================

def clamp_window_start(value: float, limit: int) -> int:
    max_start = max(0, limit - CROP_SIZE)
    return int(min(max(round(value), 0), max_start))


def generate_grid_starts(length: int) -> List[int]:
    if length <= CROP_SIZE:
        return [0]

    starts = list(range(0, length - CROP_SIZE + 1, WINDOW_STRIDE))
    last_start = length - CROP_SIZE

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def get_shape_center(
    shape: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    shape_type = get_shape_type(shape)
    points = shape.get("points", [])

    if shape_type == "circle":
        if not validate_points(points, 2) or len(points) != 2:
            return None

        return float(points[0][0]), float(points[0][1])

    if shape_type == "polygon":
        if not validate_points(points, 3):
            return None

        point_array = np.asarray(points, dtype=np.float64)

        return (
            float(point_array[:, 0].mean()),
            float(point_array[:, 1].mean()),
        )

    return None


def generate_candidate_windows(
    image_width: int,
    image_height: int,
    shapes: List[Dict[str, Any]],
) -> List[CropWindow]:
    candidates = set()

    x_starts = generate_grid_starts(image_width)
    y_starts = generate_grid_starts(image_height)

    for y in y_starts:
        for x in x_starts:
            candidates.add((x, y))

    if USE_OBJECT_CENTER_CANDIDATES:
        for shape in shapes:
            center = get_shape_center(shape)

            if center is None:
                continue

            center_x, center_y = center
            x = clamp_window_start(
                center_x - CROP_SIZE / 2,
                image_width,
            )
            y = clamp_window_start(
                center_y - CROP_SIZE / 2,
                image_height,
            )
            candidates.add((x, y))

    return [
        CropWindow(x=x, y=y, size=CROP_SIZE)
        for x, y in sorted(candidates)
    ]


# ============================================================
# 候选评分
# ============================================================

def score_candidate(
    window: CropWindow,
    shapes: List[Dict[str, Any]],
) -> CandidateScore:
    """
    根据标注几何关系评分，仅使用标注几何信息。

    前景面积通过 shape 与裁剪框的几何交集估算：
    - polygon：使用 Shapely 精确求交面积；
    - circle：完整圆使用圆面积；被切圆只用于惩罚，
      不计入最终有效前景面积。
    """
    complete_shapes = 0
    cut_polygons = 0
    cut_circles = 0
    labels = set()
    foreground_area = 0.0
    crop_box = box(window.x, window.y, window.x2, window.y2)

    for shape in shapes:
        label = str(shape.get("label", "") or "")
        shape_type = get_shape_type(shape)
        points = shape.get("points", [])

        if shape_type == "circle":
            if not validate_points(points, 2) or len(points) != 2:
                continue

            status = get_circle_crop_status(points, window)
            _, _, radius = circle_parameters(points)

            if status == "complete":
                complete_shapes += 1
                labels.add(label)
                foreground_area += math.pi * radius * radius
            elif status == "cut":
                cut_circles += 1
                labels.add(label)

        elif shape_type == "polygon":
            if not validate_points(points, 3):
                continue

            polygon = polygon_from_points(points)

            if polygon is None or not polygon.intersects(crop_box):
                continue

            intersection = polygon.intersection(crop_box)
            intersection_area = float(intersection.area)

            if crop_box.covers(polygon):
                complete_shapes += 1
            else:
                cut_polygons += 1

            if intersection_area > 0:
                labels.add(label)
                foreground_area += intersection_area

    foreground_pixels = int(round(min(foreground_area, window.area)))
    foreground_ratio = foreground_pixels / float(window.area)
    empty_ratio = 1.0 - foreground_ratio

    score = (
        SCORE_FOREGROUND_RATIO * foreground_ratio
        + SCORE_COMPLETE_SHAPE * complete_shapes
        + SCORE_CUT_POLYGON * cut_polygons
        + SCORE_CUT_CIRCLE * cut_circles
        + SCORE_OBJECT_DIVERSITY * len(labels)
        + SCORE_EMPTY_RATIO * empty_ratio
    )

    return CandidateScore(
        window=window,
        score=float(score),
        foreground_pixels=foreground_pixels,
        foreground_ratio=float(foreground_ratio),
        complete_shapes=complete_shapes,
        cut_polygons=cut_polygons,
        cut_circles=cut_circles,
        labels_count=len(labels),
    )


def window_overlap_ratio(
    a: CropWindow,
    b: CropWindow,
) -> float:
    intersection_width = max(
        0,
        min(a.x2, b.x2) - max(a.x, b.x),
    )
    intersection_height = max(
        0,
        min(a.y2, b.y2) - max(a.y, b.y),
    )
    intersection_area = intersection_width * intersection_height

    if intersection_area <= 0:
        return 0.0

    return intersection_area / float(min(a.area, b.area))


def select_best_candidates(
    scored_candidates: List[CandidateScore],
) -> List[CandidateScore]:
    valid_candidates = [
        candidate
        for candidate in scored_candidates
        if candidate.foreground_pixels >= MIN_FOREGROUND_PIXELS
    ]

    valid_candidates.sort(
        key=lambda item: (
            item.score,
            item.complete_shapes,
            item.foreground_pixels,
            -item.cut_circles,
            -item.cut_polygons,
        ),
        reverse=True,
    )

    selected: List[CandidateScore] = []

    for candidate in valid_candidates:
        too_much_overlap = any(
            window_overlap_ratio(candidate.window, chosen.window)
            > MAX_CROP_OVERLAP
            for chosen in selected
        )

        if too_much_overlap:
            continue

        selected.append(candidate)

        if len(selected) >= MAX_CROPS_PER_IMAGE:
            break

    return selected


# ============================================================
# 输出 Labelme JSON
# ============================================================

def make_cropped_json(
    original_json: Dict[str, Any],
    cropped_shapes: List[Dict[str, Any]],
    output_image_name: str,
) -> Dict[str, Any]:
    result = copy.deepcopy(original_json)

    result["imagePath"] = output_image_name
    result["imageHeight"] = CROP_SIZE
    result["imageWidth"] = CROP_SIZE
    result["imageData"] = None
    result["shapes"] = cropped_shapes

    return result


def save_cropped_json(
    output_json_path: Path,
    cropped_json: Dict[str, Any],
) -> None:
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(cropped_json, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ============================================================
# 单文件处理
# ============================================================

def process_one_json(
    json_path: Path,
    output_dirs: Dict[str, Path],
    manifest_rows: List[Dict[str, Any]],
    warning_lines: List[str],
) -> int:
    json_data = load_json(json_path)
    shapes = json_data.get("shapes", [])

    image_path = find_image_path(json_path, json_data)

    if image_path is None:
        log_warning(f"找不到对应图像：{json_path}", warning_lines)
        return 0

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        log_warning(f"图像读取失败：{image_path}", warning_lines)
        return 0

    image, pad_right, pad_bottom = pad_image_if_needed(image)
    height, width = image.shape[:2]

    candidate_windows = generate_candidate_windows(
        image_width=width,
        image_height=height,
        shapes=shapes,
    )

    scored_candidates = [
        score_candidate(
            window=window,
            shapes=shapes,
        )
        for window in candidate_windows
    ]

    selected_candidates = select_best_candidates(scored_candidates)
    saved_count = 0

    for crop_index, candidate in enumerate(selected_candidates):
        window = candidate.window
        cropped_image = crop_image(image, window)

        if cropped_image.shape[:2] != (CROP_SIZE, CROP_SIZE):
            log_warning(
                f"裁剪尺寸异常，跳过：{json_path}，window={window}",
                warning_lines,
            )
            continue

        cropped_shapes: List[Dict[str, Any]] = []

        for shape in shapes:
            cropped_shapes.extend(
                crop_shape_to_window(
                    shape=shape,
                    window=window,
                    json_path=json_path,
                    warning_lines=warning_lines,
                )
            )

        output_stem = (
            f"{json_path.stem}_crop_{crop_index:02d}_"
            f"x{window.x}_y{window.y}"
        )
        output_image_name = f"{output_stem}.jpg"
        output_json_name = f"{output_stem}.json"

        output_image_path = output_dirs["images"] / output_image_name
        output_json_path = output_dirs["images"] / output_json_name

        image_saved = cv2.imwrite(
            str(output_image_path),
            cropped_image,
        )

        if not image_saved:
            log_warning(
                f"裁剪图保存失败：{output_image_path}",
                warning_lines,
            )
            continue

        cropped_json = make_cropped_json(
            original_json=json_data,
            cropped_shapes=cropped_shapes,
            output_image_name=output_image_name,
        )
        save_cropped_json(output_json_path, cropped_json)

        manifest_rows.append(
            {
                "source_json": str(json_path),
                "source_image": str(image_path),
                "output_image": str(output_image_path),
                "output_json": str(output_json_path),
                "crop_index": crop_index,
                "crop_x": window.x,
                "crop_y": window.y,
                "crop_size": window.size,
                "score": candidate.score,
                "foreground_pixels": candidate.foreground_pixels,
                "foreground_ratio": candidate.foreground_ratio,
                "complete_shapes": candidate.complete_shapes,
                "cut_polygons": candidate.cut_polygons,
                "cut_circles": candidate.cut_circles,
                "labels_count": candidate.labels_count,
                "saved_shapes": len(cropped_shapes),
                "pad_right": pad_right,
                "pad_bottom": pad_bottom,
            }
        )

        saved_count += 1

    return saved_count


# ============================================================
# 日志输出
# ============================================================

def write_manifest(
    manifest_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    fieldnames = [
        "source_json",
        "source_image",
        "output_image",
        "output_json",
        "crop_index",
        "crop_x",
        "crop_y",
        "crop_size",
        "score",
        "foreground_pixels",
        "foreground_ratio",
        "complete_shapes",
        "cut_polygons",
        "cut_circles",
        "labels_count",
        "saved_shapes",
        "pad_right",
        "pad_bottom",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_warning_log(
    warning_path: Path,
    warning_lines: List[str],
) -> None:
    if warning_lines:
        warning_path.write_text(
            "\n".join(warning_lines) + "\n",
            encoding="utf-8",
        )
    else:
        warning_path.write_text(
            "无警告。\n",
            encoding="utf-8",
        )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    if not JSON_DIR.is_dir():
        raise NotADirectoryError(
            f"JSON_DIR 不存在：{JSON_DIR.resolve()}"
        )

    if not IMAGE_DIR.is_dir():
        raise NotADirectoryError(
            f"IMAGE_DIR 不存在：{IMAGE_DIR.resolve()}"
        )

    output_dirs = ensure_output_dirs()

    if RECURSIVE:
        json_files = sorted(JSON_DIR.rglob("*.json"))
    else:
        json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"未找到 JSON：{JSON_DIR.resolve()}")
        return

    manifest_rows: List[Dict[str, Any]] = []
    warning_lines: List[str] = []

    total_crops = 0
    success_files = 0
    failed_files = 0

    print("=" * 80)
    print("Labelme 智能离线裁剪")
    print("=" * 80)
    print(f"图像目录：{IMAGE_DIR.resolve()}")
    print(f"JSON目录：{JSON_DIR.resolve()}")
    print(f"输出目录：{OUTPUT_DIR.resolve()}")
    print(f"裁剪尺寸：{CROP_SIZE}×{CROP_SIZE}")
    print(f"JSON数量：{len(json_files)}")
    print()

    for index, json_path in enumerate(json_files, start=1):
        try:
            crop_count = process_one_json(
                json_path=json_path,
                output_dirs=output_dirs,
                manifest_rows=manifest_rows,
                warning_lines=warning_lines,
            )

            total_crops += crop_count

            if crop_count > 0:
                success_files += 1
                print(
                    f"[{index}/{len(json_files)}] "
                    f"完成：{json_path.name}，输出 {crop_count} 张"
                )
            else:
                print(
                    f"[{index}/{len(json_files)}] "
                    f"无有效裁剪：{json_path.name}"
                )

        except Exception as exc:
            failed_files += 1
            log_warning(
                f"处理失败：{json_path}，原因：{exc}",
                warning_lines,
            )

    manifest_path = output_dirs["logs"] / "manifest.csv"
    warning_path = output_dirs["logs"] / "warning.log"

    write_manifest(manifest_path, manifest_rows)
    write_warning_log(warning_path, warning_lines)

    print()
    print("=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"JSON总数：{len(json_files)}")
    print(f"成功输出文件数：{success_files}")
    print(f"失败文件数：{failed_files}")
    print(f"裁剪图总数：{total_crops}")
    print(f"images：{output_dirs['images'].resolve()}")
    print(f"manifest：{manifest_path.resolve()}")
    print(f"warning：{warning_path.resolve()}")


if __name__ == "__main__":
    main()