#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修正 Labelme JSON 标注类型。

处理规则：
    shape_type == "polygon"
    且 points 恰好有 2 个
    则将 shape_type 改为 "circle"

特点：
1. 所有参数都在脚本顶部设置；
2. 直接在原 JSON 文件上修改；
3. 不创建备份；
4. 不按类别筛选；
5. 只保留两种模式：
   - DRY_RUN = True：预演，只检查，不修改；
   - DRY_RUN = False：实际修改；
6. 输出 TXT 日志和 CSV 修改明细。
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# 用户配置
# ============================================================

# Labelme JSON 文件所在目录。
JSON_DIR = Path(r"F:/liuhaibo/datasets/JZW_v3/data_V3_trainval")

# 是否递归扫描子目录。
RECURSIVE = False

# True：预演，只统计，不修改 JSON。
# False：实际修改原 JSON。
DRY_RUN = True

# 日志输出目录。
LOG_DIR = Path(r"./logs")

# 日志文件名前缀。
LOG_PREFIX = "polygon_to_circle"


# ============================================================
# 核心函数
# ============================================================

def get_json_files(json_dir: Path) -> List[Path]:
    """获取所有待处理的 JSON 文件。"""
    if RECURSIVE:
        return sorted(json_dir.rglob("*.json"))
    return sorted(json_dir.glob("*.json"))


def load_json(json_path: Path) -> Dict[str, Any]:
    """读取 Labelme JSON，兼容普通 UTF-8 和 UTF-8 BOM。"""
    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except UnicodeDecodeError:
        with json_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("JSON 根节点不是对象")

    shapes = data.get("shapes")
    if shapes is None:
        data["shapes"] = []
    elif not isinstance(shapes, list):
        raise ValueError("shapes 字段不是列表")

    return data


def save_json(json_path: Path, data: Dict[str, Any]) -> None:
    """
    安全写回原 JSON。

    先写入临时文件，成功后再替换原文件，
    避免写入过程中断导致原 JSON 损坏。
    """
    temp_path = json_path.with_suffix(json_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    temp_path.replace(json_path)


def is_valid_point(point: Any) -> bool:
    """判断是否为有效二维坐标点。"""
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False

    try:
        x = float(point[0])
        y = float(point[1])
    except (TypeError, ValueError):
        return False

    return math.isfinite(x) and math.isfinite(y)


def is_target_shape(shape: Any) -> bool:
    """
    判断是否需要转换。

    必须满足：
    1. shape 是字典；
    2. shape_type 为 polygon；
    3. points 恰好有两个有效点；
    4. 两个点不完全重合。
    """
    if not isinstance(shape, dict):
        return False

    shape_type = str(shape.get("shape_type", "") or "").strip().lower()
    if shape_type != "polygon":
        return False

    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 2:
        return False

    if not all(is_valid_point(point) for point in points):
        return False

    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[1][0]), float(points[1][1])

    # 两点完全重合时圆半径为 0，不做转换。
    if x1 == x2 and y1 == y2:
        return False

    return True


def calculate_radius(points: List[List[float]]) -> float:
    """计算 Labelme 圆的半径。"""
    x1, y1 = float(points[0][0]), float(points[0][1])
    x2, y2 = float(points[1][0]), float(points[1][1])
    return math.hypot(x2 - x1, y2 - y1)


def process_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    检查并处理一个 JSON。

    返回该文件中的修改明细。
    """
    data = load_json(json_path)
    changes: List[Dict[str, Any]] = []

    for shape_index, shape in enumerate(data.get("shapes", [])):
        if not is_target_shape(shape):
            continue

        points = shape["points"]
        label = str(shape.get("label", "") or "")

        changes.append(
            {
                "json_file": str(json_path),
                "shape_index": shape_index,
                "label": label,
                "old_shape_type": "polygon",
                "new_shape_type": "circle",
                "center_x": float(points[0][0]),
                "center_y": float(points[0][1]),
                "edge_x": float(points[1][0]),
                "edge_y": float(points[1][1]),
                "radius": calculate_radius(points),
            }
        )

        if not DRY_RUN:
            shape["shape_type"] = "circle"

    if changes and not DRY_RUN:
        save_json(json_path, data)

    return changes


def write_csv_log(csv_path: Path, changes: List[Dict[str, Any]]) -> None:
    """输出 CSV 修改明细。"""
    fieldnames = [
        "json_file",
        "shape_index",
        "label",
        "old_shape_type",
        "new_shape_type",
        "center_x",
        "center_y",
        "edge_x",
        "edge_y",
        "radius",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changes)


def write_text_log(
    txt_path: Path,
    json_files: List[Path],
    changed_files: int,
    changes: List[Dict[str, Any]],
    failed_files: List[Dict[str, str]],
) -> None:
    """输出文本处理日志。"""
    mode = "预演模式，不修改 JSON" if DRY_RUN else "实际修改模式"

    lines = [
        "=" * 80,
        "Labelme 两点 polygon 转 circle 处理日志",
        "=" * 80,
        f"执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"运行模式：{mode}",
        f"JSON 目录：{JSON_DIR.resolve()}",
        f"递归扫描：{RECURSIVE}",
        f"扫描 JSON 数量：{len(json_files)}",
        f"命中 JSON 数量：{changed_files}",
        f"命中 shape 数量：{len(changes)}",
        f"失败 JSON 数量：{len(failed_files)}",
        "",
        "=" * 80,
        "修改明细",
        "=" * 80,
    ]

    if changes:
        for item in changes:
            lines.extend(
                [
                    f"文件：{item['json_file']}",
                    f"shape_index：{item['shape_index']}",
                    f"label：{item['label']}",
                    "类型：polygon -> circle",
                    (
                        f"圆心：({item['center_x']:.6f}, "
                        f"{item['center_y']:.6f})"
                    ),
                    (
                        f"圆周点：({item['edge_x']:.6f}, "
                        f"{item['edge_y']:.6f})"
                    ),
                    f"半径：{item['radius']:.6f}",
                    "-" * 80,
                ]
            )
    else:
        lines.append("未发现需要修改的 shape。")

    lines.extend(
        [
            "",
            "=" * 80,
            "失败文件",
            "=" * 80,
        ]
    )

    if failed_files:
        for item in failed_files:
            lines.append(f"{item['json_file']}：{item['error']}")
    else:
        lines.append("无")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """程序入口。"""
    if not JSON_DIR.is_dir():
        raise NotADirectoryError(f"JSON 目录不存在：{JSON_DIR.resolve()}")

    json_files = get_json_files(JSON_DIR)
    if not json_files:
        print(f"没有找到 JSON 文件：{JSON_DIR.resolve()}")
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_name = "dry_run" if DRY_RUN else "modified"

    txt_log_path = LOG_DIR / f"{LOG_PREFIX}_{mode_name}_{timestamp}.txt"
    csv_log_path = LOG_DIR / f"{LOG_PREFIX}_{mode_name}_{timestamp}.csv"

    all_changes: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, str]] = []
    changed_files = 0

    print("=" * 70)
    print("Labelme 两点 polygon 转 circle")
    print("=" * 70)
    print(f"JSON 目录：{JSON_DIR.resolve()}")
    print(f"运行模式：{'预演' if DRY_RUN else '实际修改'}")
    print(f"JSON 数量：{len(json_files)}")
    print()

    for index, json_path in enumerate(json_files, start=1):
        try:
            changes = process_json(json_path)

            if changes:
                changed_files += 1
                all_changes.extend(changes)

                action = "发现" if DRY_RUN else "已修改"
                print(
                    f"[{index}/{len(json_files)}] "
                    f"{action} {len(changes)} 个：{json_path}"
                )

        except Exception as exc:
            failed_files.append(
                {
                    "json_file": str(json_path),
                    "error": str(exc),
                }
            )
            print(
                f"[{index}/{len(json_files)}] 失败："
                f"{json_path}，原因：{exc}"
            )

    write_csv_log(csv_log_path, all_changes)
    write_text_log(
        txt_path=txt_log_path,
        json_files=json_files,
        changed_files=changed_files,
        changes=all_changes,
        failed_files=failed_files,
    )

    print()
    print("=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"扫描 JSON：{len(json_files)}")
    print(f"命中 JSON：{changed_files}")
    print(f"命中 shape：{len(all_changes)}")
    print(f"失败 JSON：{len(failed_files)}")
    print(f"文本日志：{txt_log_path.resolve()}")
    print(f"CSV 日志：{csv_log_path.resolve()}")

    if DRY_RUN:
        print("当前为预演模式，JSON 未被修改。")
        print("确认日志无误后，将 DRY_RUN 改为 False 再运行。")
    else:
        print("当前为实际修改模式，原 JSON 已直接更新。")


if __name__ == "__main__":
    main()
