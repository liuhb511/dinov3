#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import cv2
import numpy as np


# ============================ 用户配置 ============================

JSON_DIR = Path(r"D:/lhb/datasets/AFe/SGWC_AFe-0911/json")
OUTPUT_DIR = Path(r"D:/lhb/datasets/AFe/SGWC_AFe-0911/masks")

RECURSIVE = False  # 是否递归处理子目录

CLASS_MAP = {
    "tie": 1,
}


# ============================ 基础函数 ============================

def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)


def normalize_label(label):
    return str(label or "").strip().upper()


def draw_polygon(mask, points, class_id):
    if not isinstance(points, list) or len(points) < 3:
        return

    try:
        pts = np.array(points, dtype=np.float32)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], class_id)
    except Exception:
        return


def draw_circle(mask, points, class_id):
    if not isinstance(points, list) or len(points) != 2:
        return

    try:
        cx, cy = float(points[0][0]), float(points[0][1])
        px, py = float(points[1][0]), float(points[1][1])

        radius = int(round(np.hypot(px - cx, py - cy)))
        center = (int(round(cx)), int(round(cy)))

        if radius > 0:
            cv2.circle(mask, center, radius, class_id, thickness=-1)

    except Exception:
        return


def draw_shape(mask, shape, class_id):
    shape_type = str(shape.get("shape_type", "polygon") or "polygon").strip().lower()
    points = shape.get("points", [])

    if shape_type == "polygon":
        draw_polygon(mask, points, class_id)
    elif shape_type == "circle":
        draw_circle(mask, points, class_id)


# ============================ 单个 JSON 处理 ============================

def json_to_mask(json_path, output_path):
    data = load_json(json_path)

    height = int(data.get("imageHeight", 0))
    width = int(data.get("imageWidth", 0))

    if width <= 0 or height <= 0:
        raise ValueError(f"JSON 中 imageWidth/imageHeight 无效：{json_path}")

    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = normalize_label(shape.get("label"))
        class_id = CLASS_MAP.get(label, 0)

        # 不在映射表里的类别保持为背景 0
        if class_id == 0:
            continue

        draw_shape(mask, shape, class_id)

    if not cv2.imwrite(str(output_path), mask):
        raise IOError(f"PNG 保存失败：{output_path}")


# ============================ 主程序 ============================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if RECURSIVE:
        json_files = sorted(JSON_DIR.rglob("*.json"))
    else:
        json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"没有找到 JSON：{JSON_DIR}")
        return

    success = 0
    failed = 0

    for index, json_path in enumerate(json_files, 1):
        try:
            if RECURSIVE:
                relative_path = json_path.relative_to(JSON_DIR)
                output_path = OUTPUT_DIR / relative_path.with_suffix(".png")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_path = OUTPUT_DIR / f"{json_path.stem}.png"

            json_to_mask(json_path, output_path)

            success += 1
            print(f"[{index}/{len(json_files)}] {json_path.name} -> {output_path.name}")

        except Exception as exc:
            failed += 1
            print(f"[错误] {json_path.name}：{exc}")

    print(f"完成：成功 {success}，失败 {failed}")


if __name__ == "__main__":
    main()