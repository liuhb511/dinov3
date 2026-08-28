# -*- coding: utf-8 -*-
"""
review_hard_negatives.py

逐张人工审核 mine_hard_negatives.py 生成的 candidates.csv。

按键：
    0 = CONFIRMED_NEGATIVE
        明确不是夹杂物：划痕、抛光痕、污渍、伪影、其他明确非夹杂物

    1 = MISSED_INCLUSION
        实际是真实夹杂物，只是原 GT 漏标
        绝对不能作为 hard negative

    2 = KNOWN_DISTRACTOR
        明确是 HH / XW / XQL 等已知干扰物，但原 GT 漏标
        默认先单独保存，不自动当成 background GT

    3 = UNCERTAIN
        不能确定，后续不用

其他按键：
    S = skip / 暂时跳过
    B = back / 返回上一条
    Q / ESC = 保存并退出

特点：
- 每次按键后立即保存 CSV，不怕中途退出。
- 再次运行会自动跳到第一条未审核候选。
- 窗口左边显示 raw crop，右边显示 overlay crop。
"""

import csv
from pathlib import Path

import cv2
import numpy as np


CONFIG = {
    "candidate_csv": "./hard_negative_mining/candidates.csv",
    "window_name": "Hard Negative Review",
    "max_display_height": 900,
    "max_display_width": 1700,
}


LABELS = {
    "0": ("0", "CONFIRMED_NEGATIVE"),
    "1": ("1", "MISSED_INCLUSION"),
    "2": ("2", "KNOWN_DISTRACTOR"),
    "3": ("3", "UNCERTAIN"),
}


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    if not rows:
        return

    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resize_to_fit(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
    if scale >= 1.0:
        return img
    return cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def pad_to_same_height(a, b):
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)

    def pad(img, target_h):
        dh = target_h - img.shape[0]
        if dh <= 0:
            return img
        top = dh // 2
        bottom = dh - top
        return cv2.copyMakeBorder(
            img, top, bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    return pad(a, h), pad(b, h)


def build_display(row, index, total):
    raw = cv2.imread(row["raw_crop_path"], cv2.IMREAD_COLOR)
    overlay = cv2.imread(row["overlay_path"], cv2.IMREAD_COLOR)

    if raw is None:
        raise RuntimeError(f"无法读取 {row['raw_crop_path']}")
    if overlay is None:
        raise RuntimeError(f"无法读取 {row['overlay_path']}")

    raw, overlay = pad_to_same_height(raw, overlay)
    panel = np.hstack([raw, overlay])

    header_h = 105
    header = np.zeros(
        (header_h, panel.shape[1], 3),
        dtype=np.uint8,
    )

    lines = [
        (
            f"[{index + 1}/{total}] {row['candidate_id']}  "
            f"Pred={row['pred_class_name']}  "
            f"area={row['area_px']}  "
            f"AR={float(row['aspect_ratio']):.2f}"
        ),
        f"Image: {row['image_name']}",
        (
            "0=NEGATIVE   1=MISSED_INCLUSION   "
            "2=KNOWN_DISTRACTOR   3=UNCERTAIN   "
            "S=SKIP   B=BACK   Q=QUIT"
        ),
    ]

    ys = [24, 50, 82]
    for text, y in zip(lines, ys):
        cv2.putText(
            header,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    display = np.vstack([header, panel])
    return resize_to_fit(
        display,
        CONFIG["max_display_width"],
        CONFIG["max_display_height"],
    )


def first_unreviewed(rows):
    for i, row in enumerate(rows):
        if str(row.get("review_label", "")).strip() == "":
            return i
    return len(rows)


def main():
    csv_path = Path(CONFIG["candidate_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}，请先运行 mine_hard_negatives.py"
        )

    rows = read_rows(csv_path)
    if not rows:
        print("candidates.csv 为空。")
        return

    idx = first_unreviewed(rows)
    if idx >= len(rows):
        print("全部候选均已审核。")
        return

    print(f"从第 {idx + 1}/{len(rows)} 个候选开始审核。")

    cv2.namedWindow(CONFIG["window_name"], cv2.WINDOW_NORMAL)

    while 0 <= idx < len(rows):
        row = rows[idx]
        display = build_display(row, idx, len(rows))
        cv2.imshow(CONFIG["window_name"], display)

        key = cv2.waitKey(0) & 0xFF

        # 0/1/2/3
        ch = chr(key) if 0 <= key < 128 else ""

        if ch in LABELS:
            label, name = LABELS[ch]
            row["review_label"] = label
            row["review_name"] = name
            # note 由用户后续可在 CSV 手工填写；先不阻塞审核速度
            save_rows(csv_path, rows)
            idx += 1

        elif ch.lower() == "s":
            idx += 1

        elif ch.lower() == "b":
            idx = max(0, idx - 1)

        elif ch.lower() == "q" or key == 27:
            save_rows(csv_path, rows)
            print("审核进度已保存。")
            break

    else:
        save_rows(csv_path, rows)
        print("全部候选已审核完成。")

    cv2.destroyAllWindows()

    counts = {}
    for row in rows:
        name = row.get("review_name", "").strip() or "UNREVIEWED"
        counts[name] = counts.get(name, 0) + 1

    print("\n当前审核统计：")
    for name, n in counts.items():
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
