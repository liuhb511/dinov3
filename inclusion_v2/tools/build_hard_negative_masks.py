# -*- coding: utf-8 -*-
"""
build_hard_negative_masks.py

读取人工审核后的 candidates.csv，生成独立的 hard-negative mask。

默认只使用：
    review_label == 0  -> CONFIRMED_NEGATIVE

不会修改原始 GT mask。

输出：
    hard_negative_masks/
      <image_stem>.png

mask 格式：
    与训练 mask 同尺寸（默认 1024x1024）
    uint8
    0 = 非 hard-negative
    1 = 已确认 hard-negative

同时输出：
    confirmed_hard_negatives.csv
    missed_inclusions.csv
    known_distractors.csv
    uncertain.csv

说明：
- component_mask_path 保存的是精确 bbox component mask，
  所以构建时不是把整个 bbox 填成 1，而是把模型实际误检的 component 精确贴回。
- 默认不把 KNOWN_DISTRACTOR 自动加入 hard-negative mask。
  如果后续训练只用于 Strip-inclusion rejection，可把
  include_known_distractors_in_mask 改成 True。
"""

import csv
from pathlib import Path

import cv2
import numpy as np


CONFIG = {
    "candidate_csv": "./hard_negative_mining/candidates.csv",

    # 用原训练 mask 获取标准输出尺寸与文件名
    "train_mask_dir": r"F:/liuhaibo/datasets/inclusion_unified/train/masks",

    "output_mask_dir": "./hard_negative_mining/hard_negative_masks",
    "output_manifest_dir": "./hard_negative_mining",

    # MVP 第一版：只把明确 confirmed negative 用于 mask。
    "include_known_distractors_in_mask": False,

    # 若希望给精确 component 边缘轻微扩张，可设为 1/2。
    # 第一版建议 0，避免把邻近真实夹杂物吃进去。
    "dilate_pixels": 0,
}


LABEL_NAME = {
    "0": "CONFIRMED_NEGATIVE",
    "1": "MISSED_INCLUSION",
    "2": "KNOWN_DISTRACTOR",
    "3": "UNCERTAIN",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return

    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_train_mask(image_name):
    stem = Path(image_name).stem
    p = Path(CONFIG["train_mask_dir"]) / f"{stem}.png"
    if not p.exists():
        raise FileNotFoundError(f"找不到训练 mask：{p}")
    return p


def candidate_is_used(row):
    label = str(row.get("review_label", "")).strip()
    if label == "0":
        return True
    if label == "2" and CONFIG["include_known_distractors_in_mask"]:
        return True
    return False


def main():
    candidate_csv = Path(CONFIG["candidate_csv"])
    if not candidate_csv.exists():
        raise FileNotFoundError(
            f"找不到 {candidate_csv}，请先完成 mining + review"
        )

    rows = read_rows(candidate_csv)
    if not rows:
        raise RuntimeError("candidates.csv 为空。")

    manifest_dir = Path(CONFIG["output_manifest_dir"])
    out_mask_dir = Path(CONFIG["output_mask_dir"])
    ensure_dir(manifest_dir)
    ensure_dir(out_mask_dir)

    groups = {
        "0": [],
        "1": [],
        "2": [],
        "3": [],
        "": [],
    }
    for row in rows:
        label = str(row.get("review_label", "")).strip()
        groups.setdefault(label, []).append(row)

    write_csv(
        manifest_dir / "confirmed_hard_negatives.csv",
        groups.get("0", []),
    )
    write_csv(
        manifest_dir / "missed_inclusions.csv",
        groups.get("1", []),
    )
    write_csv(
        manifest_dir / "known_distractors.csv",
        groups.get("2", []),
    )
    write_csv(
        manifest_dir / "uncertain.csv",
        groups.get("3", []),
    )
    write_csv(
        manifest_dir / "unreviewed.csv",
        groups.get("", []),
    )

    selected = [row for row in rows if candidate_is_used(row)]

    if not selected:
        print("没有可用于 hard-negative mask 的已确认候选。")
        print("请先运行 review_hard_negatives.py。")
        return

    by_image = {}
    for row in selected:
        by_image.setdefault(row["image_name"], []).append(row)

    kernel = None
    d = int(CONFIG["dilate_pixels"])
    if d > 0:
        k = 2 * d + 1
        kernel = np.ones((k, k), dtype=np.uint8)

    total_pixels = 0

    for image_name, image_rows in by_image.items():
        train_mask_path = find_train_mask(image_name)
        gt = cv2.imread(str(train_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise RuntimeError(f"无法读取：{train_mask_path}")

        hn = np.zeros(gt.shape[:2], dtype=np.uint8)

        for row in image_rows:
            x = int(row["x"])
            y = int(row["y"])
            w = int(row["w"])
            h = int(row["h"])

            component_path = Path(row["component_mask_path"])
            component = cv2.imread(
                str(component_path),
                cv2.IMREAD_GRAYSCALE,
            )
            if component is None:
                raise RuntimeError(
                    f"无法读取 component mask：{component_path}"
                )

            if component.shape[:2] != (h, w):
                raise ValueError(
                    f"{row['candidate_id']} component mask 尺寸 "
                    f"{component.shape[:2]} != bbox {(h, w)}"
                )

            binary = (component > 0).astype(np.uint8)

            if kernel is not None:
                binary = cv2.dilate(binary, kernel, iterations=1)

            # 防止 bbox 越界
            y2 = min(hn.shape[0], y + h)
            x2 = min(hn.shape[1], x + w)
            hh = y2 - y
            ww = x2 - x

            if hh <= 0 or ww <= 0:
                continue

            # 极重要：只允许原 GT 为 background 的 pixel 成为 hard negative。
            # 防止 dilation 或标注更新后覆盖已有真实类别。
            target_region = hn[y:y2, x:x2]
            gt_region = gt[y:y2, x:x2]
            src = binary[:hh, :ww]

            valid = (src > 0) & (gt_region == 0)
            target_region[valid] = 1

        save_path = out_mask_dir / f"{Path(image_name).stem}.png"
        cv2.imwrite(str(save_path), hn)
        total_pixels += int((hn > 0).sum())

    print("Hard-negative masks 构建完成。")
    print(f"使用候选数：{len(selected)}")
    print(f"涉及训练图：{len(by_image)}")
    print(f"hard-negative pixels：{total_pixels}")
    print(f"输出目录：{out_mask_dir}")
    print("\n原始训练 GT 没有被修改。")


if __name__ == "__main__":
    main()
