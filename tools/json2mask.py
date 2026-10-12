
import json
from pathlib import Path

import cv2
import numpy as np


# ====================== 用户配置 ======================

JSON_DIR = Path(r"D:\lhb\datasets\AFe\total")
OUTPUT_DIR = Path(r"D:\lhb\datasets\AFe\masks")

RECURSIVE = False  # True：递归处理子目录

# ====================================================


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def draw_shape(mask, shape):
    points = shape.get("points", [])
    shape_type = shape.get("shape_type", "polygon")

    if not points:
        return

    pts = np.round(np.array(points, dtype=np.float32)).astype(np.int32)

    if shape_type == "polygon":
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)

    elif shape_type == "rectangle":
        if len(pts) == 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cv2.rectangle(
                mask,
                (min(x1, x2), min(y1, y2)),
                (max(x1, x2), max(y1, y2)),
                1,
                thickness=-1
            )

    elif shape_type == "circle":
        if len(pts) == 2:
            center = tuple(pts[0])
            radius = int(np.linalg.norm(pts[1] - pts[0]))
            cv2.circle(mask, center, radius, 1, thickness=-1)

    elif shape_type == "linestrip":
        if len(pts) >= 2:
            cv2.polylines(mask, [pts], False, 1, thickness=1)

    elif shape_type == "line":
        if len(pts) == 2:
            cv2.line(mask, tuple(pts[0]), tuple(pts[1]), 1, thickness=1)

    elif shape_type == "point":
        cv2.circle(mask, tuple(pts[0]), 1, 1, thickness=-1)


def json_to_mask(json_path, output_path):
    data = load_json(json_path)

    height = int(data["imageHeight"])
    width = int(data["imageWidth"])

    # 初始化背景为 0
    mask = np.zeros((height, width), dtype=np.uint8)

    # 所有标注区域统一为前景 1
    for shape in data.get("shapes", []):
        draw_shape(mask, shape)

    # Unicode 路径兼容保存
    success, encoded = cv2.imencode(".png", mask)

    if not success:
        raise IOError(f"PNG 编码失败：{output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded.tobytes())

    return np.unique(mask).tolist()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if RECURSIVE:
        json_files = sorted(JSON_DIR.rglob("*.json"))
    else:
        json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print(f"未找到 JSON 文件：{JSON_DIR}")
        return

    success_count = 0
    failed_count = 0

    for index, json_path in enumerate(json_files, 1):
        try:
            if RECURSIVE:
                relative = json_path.relative_to(JSON_DIR)
                output_path = OUTPUT_DIR / relative.with_suffix(".png")
            else:
                output_path = OUTPUT_DIR / (json_path.stem + ".png")

            unique_values = json_to_mask(json_path, output_path)

            success_count += 1

            print(
                f"[{index}/{len(json_files)}] "
                f"{json_path.name} -> {output_path.name} "
                f"像素值：{unique_values}"
            )

        except Exception as e:
            failed_count += 1
            print(f"[错误] {json_path.name}：{e}")

    print("\n========== 转换完成 ==========")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")


if __name__ == "__main__":
    main()
