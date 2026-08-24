#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Labelme 智能离线裁剪：固定窗口、低前景重复、位置平衡、D 类截断治理。"""

from __future__ import annotations
import copy, csv, hashlib, json, math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple
import cv2
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

# ============================ 用户配置 ============================
IMAGE_DIR = Path(r"F:/liuhaibo/datasets/JZW_v3/TIN")
JSON_DIR = Path(r"F:/liuhaibo/datasets/JZW_v3/TIN")
OUTPUT_DIR = Path(r"F:/liuhaibo/datasets/JZW_v3/crop_1024/TIN")
RECURSIVE = False                       # 是否递归扫描子目录
CROP_SIZE = 1024                        # 裁剪窗口边长，可改为 784
MAX_CROPS_PER_IMAGE = 4                 # 每张原图最多保存窗口数
WINDOW_STRIDE = 98                      # 网格候选窗口的滑动步长
USE_OBJECT_POSITION_CANDIDATES = True   # 是否生成目标位置锚点候选
TARGET_RELATIVE_POSITIONS = (           # 目标在窗口九宫格中的期望位置
    (0.25, 0.25), (0.50, 0.25), (0.75, 0.25),
    (0.25, 0.50), (0.50, 0.50), (0.75, 0.50),
    (0.25, 0.75), (0.50, 0.75), (0.75, 0.75),
)
MIN_FOREGROUND_PIXELS = 30              # 候选窗口最少有效前景像素
MIN_POLYGON_AREA = 1.0                  # 裁剪后多边形最小有效面积
CIRCLE_POLYGON_SEGMENTS = 128           # 圆转多边形的圆周采样点数
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]  # 图像后缀
MAX_FOREGROUND_REPEAT = 0.10            # 最大前景重复率（硬限制）
MAX_WINDOW_OVERLAP = 0.30               # 最大窗口重叠率（硬限制）
MAX_SAMPLES_PER_INSTANCE = 1            # 单个标注实例的默认采样次数
MIN_PREFERRED_INSTANCE_RETAIN = 0.90    # 优先视为有效覆盖的实例保留比例
D_JSON_RETAIN_THRESHOLD = 0.50          # 截断 D 严格大于该比例才写 JSON
D_HIGH_RETAIN_THRESHOLD = 0.65          # D 高保留比例分界
D_SMALL_FRAGMENT_THRESHOLD = 0.35       # D 小残片比例分界
SCORE_FOREGROUND_RATIO = 2.0            # 前景比例评分权重
SCORE_COMPLETE_SHAPE = 4.0              # 完整目标加分
SCORE_CUT_POLYGON = -3.0                # 非 D 截断多边形惩罚
SCORE_CUT_CIRCLE = -6.0                 # 非 D 截断圆惩罚
SCORE_OBJECT_DIVERSITY = 1.5            # 标签多样性加分
SCORE_CUT_D_OVER_65 = -1.0              # 截断 D 保留率不低于 65% 的惩罚
SCORE_CUT_D_35_TO_65 = -3.0             # 截断 D 保留率 35%～65% 的惩罚
SCORE_CUT_D_UNDER_35 = -0.5             # 截断 D 保留率低于 35% 的惩罚
TIE_BREAK_SEED = 2026                   # 同分候选稳定随机种子


@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    size: int
    @property
    def x2(self): return self.x + self.size
    @property
    def y2(self): return self.y + self.size
    @property
    def area(self): return self.size * self.size


@dataclass(frozen=True)
class Placement:
    instance_id: str
    label: str
    position_bin: int
    retained_ratio: float
    complete: bool


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
    labels: Set[str] = field(default_factory=set)
    placements: List[Placement] = field(default_factory=list)
    foreground_geometry: Any = None
    cut_d_over_65: int = 0
    cut_d_35_to_65: int = 0
    cut_d_under_35: int = 0
    saved_cut_d: int = 0
    ignored_cut_d: int = 0
    stable_tie_value: int = 0


@dataclass
class DynamicInfo:
    new_instance_ids: Set[str]
    position_gain: float
    has_new_position: bool
    foreground_repeat: float
    new_foreground_ratio: float
    window_repeat: float


@dataclass
class SelectionState:
    instance_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    position_counts: DefaultDict[str, List[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0] * 9)
    )


def normalize_label(value): return str(value or "").strip().upper()
def get_shape_type(shape): return str(shape.get("shape_type", "polygon") or "polygon").strip().lower()


def log_warning(message, warning_lines):
    print(f"[警告] {message}")
    warning_lines.append(message)


def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f: data = json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f: data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("shapes", []), list):
        raise ValueError("JSON 根节点或 shapes 格式错误")
    return data


def find_image_path(json_path, data):
    value = data.get("imagePath")
    if value:
        candidate = Path(str(value))
        for path in (candidate, json_path.parent / candidate, IMAGE_DIR / candidate.name):
            if path.exists(): return path
    for ext in IMAGE_EXTENSIONS:
        path = IMAGE_DIR / f"{json_path.stem}{ext}"
        if path.exists(): return path
    return None


def valid_points(points, minimum):
    if not isinstance(points, list) or len(points) < minimum: return False
    try: return all(math.isfinite(float(p[0])) and math.isfinite(float(p[1])) for p in points)
    except (TypeError, ValueError, IndexError): return False


def circle_parameters(points):
    cx, cy = float(points[0][0]), float(points[0][1])
    return cx, cy, math.hypot(float(points[1][0]) - cx, float(points[1][1]) - cy)


def circle_to_polygon(points):
    try:
        cx, cy, radius = circle_parameters(points)
        if radius <= 0: return None
        return Polygon([
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_POLYGON_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_POLYGON_SEGMENTS))
            for i in range(CIRCLE_POLYGON_SEGMENTS)
        ])
    except Exception: return None


def polygon_from_points(points):
    try:
        geometry = Polygon([(float(p[0]), float(p[1])) for p in points])
        if not geometry.is_valid: geometry = geometry.buffer(0)
        if geometry.is_empty: return None
        if isinstance(geometry, Polygon): return geometry
        if isinstance(geometry, MultiPolygon): return max(geometry.geoms, key=lambda g: g.area)
    except Exception: pass
    return None


def shape_to_geometry(shape):
    points, kind = shape.get("points", []), get_shape_type(shape)
    if kind == "circle" and valid_points(points, 2) and len(points) == 2:
        return circle_to_polygon(points)
    if kind == "polygon" and valid_points(points, 3):
        return polygon_from_points(points)
    return None


def extract_polygons(geometry):
    if geometry is None or geometry.is_empty: return []
    if isinstance(geometry, Polygon): return [geometry]
    if isinstance(geometry, MultiPolygon): return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        result = []
        for item in geometry.geoms: result.extend(extract_polygons(item))
        return result
    return []


def geometry_to_labelme_polygons(geometry, shape, window):
    result = []
    for polygon in extract_polygons(geometry):
        if polygon.area < MIN_POLYGON_AREA: continue
        coords = list(polygon.exterior.coords)
        if coords and coords[0] == coords[-1]: coords = coords[:-1]
        if len(coords) < 3: continue
        item = copy.deepcopy(shape)
        item["shape_type"] = "polygon"
        item["points"] = [[float(x) - window.x, float(y) - window.y] for x, y in coords]
        result.append(item)
    return result


def crop_shape(shape, window, json_path, warnings):
    geometry = shape_to_geometry(shape)
    if geometry is None:
        log_warning(f"跳过异常 shape：{json_path}，label={normalize_label(shape.get('label'))}", warnings)
        return []
    crop_box = box(window.x, window.y, window.x2, window.y2)
    clipped = geometry.intersection(crop_box)
    if clipped.is_empty or clipped.area < MIN_POLYGON_AREA: return []
    if crop_box.covers(geometry):
        if get_shape_type(shape) == "circle":
            item = copy.deepcopy(shape)
            item["points"] = [[float(p[0]) - window.x, float(p[1]) - window.y] for p in shape["points"]]
            return [item]
        return geometry_to_labelme_polygons(geometry, shape, window)
    ratio = clipped.area / geometry.area
    if normalize_label(shape.get("label")) == "D" and ratio <= D_JSON_RETAIN_THRESHOLD:
        return []
    return geometry_to_labelme_polygons(clipped, shape, window)


def geometry_position_bin(geometry, window):
    center = geometry.centroid
    u = min(max((center.x - window.x) / window.size, 0.0), 1.0)
    v = min(max((center.y - window.y) / window.size, 0.0), 1.0)
    return min(int(v * 3), 2) * 3 + min(int(u * 3), 2)


def clamp_start(value, limit): return int(min(max(round(value), 0), limit - CROP_SIZE))


def grid_starts(length):
    if length < CROP_SIZE: raise ValueError(f"图像边长 {length} 小于裁剪尺寸 {CROP_SIZE}")
    if length == CROP_SIZE: return [0]
    result = list(range(0, length - CROP_SIZE + 1, WINDOW_STRIDE))
    if result[-1] != length - CROP_SIZE: result.append(length - CROP_SIZE)
    return result


def shape_center(shape):
    geometry = shape_to_geometry(shape)
    if geometry is None: return None
    x1, y1, x2, y2 = geometry.bounds
    return (x1 + x2) / 2, (y1 + y2) / 2


def generate_windows(width, height, shapes):
    starts = {(x, y) for y in grid_starts(height) for x in grid_starts(width)}
    if USE_OBJECT_POSITION_CANDIDATES:
        for shape in shapes:
            center = shape_center(shape)
            if center is None: continue
            for rx, ry in TARGET_RELATIVE_POSITIONS:
                starts.add((clamp_start(center[0] - rx * CROP_SIZE, width),
                            clamp_start(center[1] - ry * CROP_SIZE, height)))
    return [CropWindow(x, y, CROP_SIZE) for x, y in sorted(starts)]


def stable_tie(json_path, window):
    text = f"{json_path.as_posix()}|{window.x}|{window.y}|{window.size}|{TIE_BREAK_SEED}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def score_candidate(window, shapes, json_path):
    crop_box = box(window.x, window.y, window.x2, window.y2)
    parts, placements, labels = [], [], set()
    complete = cut_polygon = cut_circle = 0
    d65 = d3565 = d35 = saved_d = ignored_d = 0
    d_penalty = 0.0
    for index, shape in enumerate(shapes):
        geometry = shape_to_geometry(shape)
        if geometry is None or geometry.area <= 0: continue
        clipped = geometry.intersection(crop_box)
        if clipped.is_empty or clipped.area < MIN_POLYGON_AREA: continue
        label = normalize_label(shape.get("label"))
        is_complete = crop_box.covers(geometry)
        ratio = min(max(clipped.area / geometry.area, 0.0), 1.0)
        effective = clipped
        if is_complete:
            complete += 1
        elif label == "D":
            if ratio >= D_HIGH_RETAIN_THRESHOLD: d65 += 1; d_penalty += SCORE_CUT_D_OVER_65
            elif ratio >= D_SMALL_FRAGMENT_THRESHOLD: d3565 += 1; d_penalty += SCORE_CUT_D_35_TO_65
            else: d35 += 1; d_penalty += SCORE_CUT_D_UNDER_35
            if ratio > D_JSON_RETAIN_THRESHOLD: saved_d += 1
            else: ignored_d += 1; effective = None
        elif get_shape_type(shape) == "circle": cut_circle += 1
        else: cut_polygon += 1
        if effective is None: continue
        labels.add(label); parts.append(effective)
        placements.append(Placement(
            f"{json_path.as_posix()}#shape_{index}", label,
            geometry_position_bin(effective, window), ratio, is_complete
        ))
    foreground = unary_union(parts) if parts else GeometryCollection()
    area = min(foreground.area, window.area)
    ratio = area / window.area
    score = (SCORE_FOREGROUND_RATIO * ratio + SCORE_COMPLETE_SHAPE * complete
             + SCORE_CUT_POLYGON * cut_polygon + SCORE_CUT_CIRCLE * cut_circle
             + SCORE_OBJECT_DIVERSITY * len(labels) + d_penalty)
    return CandidateScore(window, score, round(area), ratio, complete, cut_polygon,
                          cut_circle, len(labels), labels, placements, foreground,
                          d65, d3565, d35, saved_d, ignored_d, stable_tie(json_path, window))


def window_overlap(a, b):
    width = max(0, min(a.x2, b.x2) - max(a.x, b.x))
    height = max(0, min(a.y2, b.y2) - max(a.y, b.y))
    return width * height / a.area


def dynamic_info(candidate, selected, covered_foreground, state, coverage_thresholds):
    # 只有达到该实例最佳可用保留水平的 placement 才算“覆盖实例”。
    # 普通目标通常要求 >=90%；若目标本身大于窗口，则采用其候选中的最佳保留率。
    covered_placements = [
        p for p in candidate.placements
        if p.retained_ratio + 1e-9 >= coverage_thresholds.get(p.instance_id, 1.0)
    ]
    ids = {p.instance_id for p in covered_placements}
    new_ids = {i for i in ids if state.instance_counts[i] < MAX_SAMPLES_PER_INSTANCE}
    gains, new_position = [], False
    for p in candidate.placements:
        count = state.position_counts[p.label][p.position_bin]
        gains.append(1 / math.sqrt(count + 1)); new_position |= count == 0
    position_gain = 0 if not gains else 0.7 * sum(gains) / len(gains) + 0.3 * max(gains)
    if covered_foreground.is_empty or candidate.foreground_geometry.is_empty:
        repeat = 0.0
    else:
        repeat = candidate.foreground_geometry.intersection(covered_foreground).area / candidate.foreground_geometry.area
    overlap = max((window_overlap(candidate.window, item.window) for item in selected), default=0.0)
    return DynamicInfo(new_ids, position_gain, new_position, repeat, 1 - repeat, overlap)


def select_candidates(candidates, state):
    remaining = [c for c in candidates if c.foreground_pixels >= MIN_FOREGROUND_PIXELS]
    best_retained = {}
    for candidate in remaining:
        for placement in candidate.placements:
            best_retained[placement.instance_id] = max(
                best_retained.get(placement.instance_id, 0.0),
                placement.retained_ratio,
            )
    coverage_thresholds = {
        instance_id: min(MIN_PREFERRED_INSTANCE_RETAIN, retained_ratio)
        for instance_id, retained_ratio in best_retained.items()
    }
    selected, result, covered = [], [], GeometryCollection()
    while remaining and len(selected) < MAX_CROPS_PER_IMAGE:
        best = None
        for candidate in remaining:
            info = dynamic_info(candidate, selected, covered, state, coverage_thresholds)
            # 每张入选图必须带来新实例；重复率和窗口重叠均为硬限制。
            if not info.new_instance_ids: continue
            if info.foreground_repeat > MAX_FOREGROUND_REPEAT: continue
            if info.window_repeat > MAX_WINDOW_OVERLAP: continue

            new_placements = [
                p for p in candidate.placements
                if p.instance_id in info.new_instance_ids
            ]
            new_complete_count = sum(p.complete for p in new_placements)
            mean_new_retained = (
                sum(p.retained_ratio for p in new_placements) / len(new_placements)
                if new_placements else 0.0
            )
            key = (
                len(info.new_instance_ids),          # 尽量覆盖更多新实例
                new_complete_count,                  # 新实例尽量完整
                round(mean_new_retained, 8),         # 截断时保留比例尽量高
                round(info.new_foreground_ratio, 8), # 新增前景尽量多
                -round(info.foreground_repeat, 8),   # 前景重复尽量少
                -round(info.window_repeat, 8),       # 窗口重叠尽量少
                round(candidate.score, 6),           # 原评分作为辅助
                round(info.position_gain, 8),        # 位置平衡作为次级条件
                candidate.stable_tie_value,
            )
            if best is None or key > best[0]: best = (key, candidate, info)
        if best is None: break
        _, candidate, info = best
        selected.append(candidate); result.append((candidate, info)); remaining.remove(candidate)
        covered = unary_union([covered, candidate.foreground_geometry])
        for instance_id in info.new_instance_ids:
            state.instance_counts[instance_id] += 1
        for p in candidate.placements:
            state.position_counts[p.label][p.position_bin] += 1
    return result


def ensure_dirs():
    result = {"images": OUTPUT_DIR / "images", "logs": OUTPUT_DIR / "logs"}
    for path in result.values(): path.mkdir(parents=True, exist_ok=True)
    return result


def save_json(path, source, shapes, image_name):
    data = copy.deepcopy(source)
    data.update(imagePath=image_name, imageHeight=CROP_SIZE, imageWidth=CROP_SIZE, imageData=None, shapes=shapes)
    with path.open("w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2); f.write("\n")


def process_one(json_path, dirs, rows, warnings, state):
    data = load_json(json_path); shapes = data.get("shapes", [])
    image_path = find_image_path(json_path, data)
    if image_path is None: log_warning(f"找不到对应图像：{json_path}", warnings); return 0
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None: log_warning(f"图像读取失败：{image_path}", warnings); return 0
    height, width = image.shape[:2]
    if width < CROP_SIZE or height < CROP_SIZE:
        raise ValueError(f"图像 {width}×{height} 小于窗口，当前标准不允许 padding")
    scored = [score_candidate(w, shapes, json_path) for w in generate_windows(width, height, shapes)]
    selected = select_candidates(scored, state); saved = 0
    for crop_index, (candidate, info) in enumerate(selected):
        w = candidate.window; cropped = image[w.y:w.y2, w.x:w.x2].copy()
        if cropped.shape[:2] != (CROP_SIZE, CROP_SIZE): continue
        cropped_shapes = []
        for shape in shapes: cropped_shapes.extend(crop_shape(shape, w, json_path, warnings))
        stem = f"{json_path.stem}_crop_{crop_index:02d}_x{w.x}_y{w.y}"
        image_name, json_name = f"{stem}.jpg", f"{stem}.json"
        image_out, json_out = dirs["images"] / image_name, dirs["images"] / json_name
        if not cv2.imwrite(str(image_out), cropped): log_warning(f"图像保存失败：{image_out}", warnings); continue
        save_json(json_out, data, cropped_shapes, image_name)
        placements = candidate.placements
        reasons = [f"new_instances={len(info.new_instance_ids)}", f"new_fg={info.new_foreground_ratio:.4f}"]
        if info.has_new_position: reasons.append("new_position")
        rows.append({
            "source_json": str(json_path), "source_image": str(image_path),
            "output_image": str(image_out), "output_json": str(json_out),
            "crop_index": crop_index, "crop_x": w.x, "crop_y": w.y, "crop_size": w.size,
            "score": candidate.score, "foreground_pixels": candidate.foreground_pixels,
            "foreground_ratio": candidate.foreground_ratio, "complete_shapes": candidate.complete_shapes,
            "cut_polygons": candidate.cut_polygons, "cut_circles": candidate.cut_circles,
            "labels_count": candidate.labels_count, "labels": "|".join(sorted(candidate.labels)),
            "instance_ids": "|".join(sorted({p.instance_id for p in placements})),
            "new_instance_count": len(info.new_instance_ids),
            "position_bins": "|".join(f"{p.label}:{p.position_bin}" for p in placements),
            "position_gain": info.position_gain, "foreground_repeat": info.foreground_repeat,
            "new_foreground_ratio": info.new_foreground_ratio, "window_repeat": info.window_repeat,
            "cut_d_over_65": candidate.cut_d_over_65, "cut_d_35_to_65": candidate.cut_d_35_to_65,
            "cut_d_under_35": candidate.cut_d_under_35, "saved_cut_d": candidate.saved_cut_d,
            "ignored_cut_d": candidate.ignored_cut_d, "saved_shapes": len(cropped_shapes),
            "selection_reason": ";".join(reasons),
        }); saved += 1
    return saved


FIELDS = ["source_json", "source_image", "output_image", "output_json", "crop_index",
          "crop_x", "crop_y", "crop_size", "score", "foreground_pixels", "foreground_ratio",
          "complete_shapes", "cut_polygons", "cut_circles", "labels_count", "labels",
          "instance_ids", "new_instance_count", "position_bins", "position_gain",
          "foreground_repeat", "new_foreground_ratio", "window_repeat", "cut_d_over_65",
          "cut_d_35_to_65", "cut_d_under_35", "saved_cut_d", "ignored_cut_d",
          "saved_shapes", "selection_reason"]


def main():
    if not JSON_DIR.is_dir(): raise NotADirectoryError(f"JSON_DIR 不存在：{JSON_DIR.resolve()}")
    if not IMAGE_DIR.is_dir(): raise NotADirectoryError(f"IMAGE_DIR 不存在：{IMAGE_DIR.resolve()}")
    dirs = ensure_dirs(); json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files: print(f"未找到 JSON：{JSON_DIR.resolve()}"); return
    rows, warnings, state = [], [], SelectionState(); total = success = failed = 0
    print(f"Labelme 智能裁剪：{len(json_files)} 个 JSON，窗口 {CROP_SIZE}×{CROP_SIZE}")
    for index, path in enumerate(json_files, 1):
        try:
            count = process_one(path, dirs, rows, warnings, state); total += count
            success += count > 0; print(f"[{index}/{len(json_files)}] {path.name}：{count} 张")
        except Exception as exc: failed += 1; log_warning(f"处理失败：{path}，原因：{exc}", warnings)
    with (dirs["logs"] / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    with (dirs["logs"] / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["label"] + [f"position_{i}" for i in range(9)] + ["total"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for label in sorted(state.position_counts):
            counts = state.position_counts[label]
            row = {"label": label, "total": sum(counts)}
            row.update({f"position_{i}": n for i, n in enumerate(counts)}); writer.writerow(row)
    (dirs["logs"] / "warning.log").write_text("\n".join(warnings) + "\n" if warnings else "无警告。\n", encoding="utf-8")
    print(f"完成：成功源文件 {success}，失败 {failed}，裁剪图 {total}")


if __name__ == "__main__": main()