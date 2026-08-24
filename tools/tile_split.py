#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图片等尺寸切分脚本

功能：
- 将给定目录下的图片（同级、不递归）按指定尺寸平铺裁剪
- 边缘剩余像素不足一个 tile 时直接舍弃
- 保持原图格式，输出到指定目录
- 每张图打印一行进度日志（图片名 + 裁剪张数）

用法：
    直接修改下方 SOURCE_DIR / OUTPUT_DIR / TILE_SIZE 后运行即可。
    也可通过命令行传参：
        python tile_split.py <source_dir> <output_dir> [tile_size]
"""

import argparse
import os
import sys
from pathlib import Path

import cv2 as cv

# ============================================================
# 用户配置（直接运行时使用）
# ============================================================

SOURCE_DIR = Path(r"/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images")
OUTPUT_DIR = Path(r"/liuhaibo/unet-pytorch-main/trains/VOCdevkit_BG_HQL_JZW_D/HQL_testcollected/images_tile")
TILE_SIZE = 784

# ============================================================


def tile_image(image_path: Path, output_dir: Path, tile_size: int) -> int:
    """将单张图裁切成 tile_size × tile_size 的小块，返回生成张数。"""
    img = cv.imread(str(image_path))
    if img is None:
        print(f"[跳过] 读取失败: {image_path.name}")
        return 0

    h, w = img.shape[:2]
    if h < tile_size or w < tile_size:
        print(f"[跳过] 尺寸不足 {tile_size}×{tile_size}: {image_path.name} ({w}×{h})")
        return 0

    stem = image_path.stem                    # 无扩展名
    ext = image_path.suffix                   # 含点，如 .jpg .png
    count = 0
    rows = h // tile_size
    cols = w // tile_size

    for ri in range(rows):
        for ci in range(cols):
            y1 = ri * tile_size
            y2 = y1 + tile_size
            x1 = ci * tile_size
            x2 = x1 + tile_size

            tile = img[y1:y2, x1:x2]
            out_name = f"{stem}_tile_{ri}_{ci}{ext}"
            out_path = output_dir / out_name
            cv.imwrite(str(out_path), tile)
            count += 1

    return count


def run(source_dir: Path, output_dir: Path, tile_size: int):
    if not source_dir.is_dir():
        print(f"源目录不存在: {source_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片（常见格式）
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    if not images:
        print(f"未找到图片文件: {source_dir}")
        sys.exit(1)

    total_files = 0
    total_tiles = 0

    print(f"源目录:   {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"裁剪尺寸: {tile_size}×{tile_size}")
    print(f"图片总数: {len(images)}")
    print("=" * 50)

    for img_path in images:
        n = tile_image(img_path, output_dir, tile_size)
        print(f"{img_path.name}  →  {n} 张")
        if n > 0:
            total_files += 1
            total_tiles += n

    print("=" * 50)
    print(f"完成！有效图片 {total_files} 张，共裁剪 {total_tiles} 张 tile。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片等尺寸平铺裁剪")
    parser.add_argument("source_dir", nargs="?", type=Path, default=None,
                        help="源图片目录")
    parser.add_argument("output_dir", nargs="?", type=Path, default=None,
                        help="输出目录")
    parser.add_argument("tile_size", nargs="?", type=int, default=None,
                        help="裁剪尺寸（正方形边长，默认784）")

    args = parser.parse_args()

    src = args.source_dir or SOURCE_DIR
    out = args.output_dir or OUTPUT_DIR
    size = args.tile_size or TILE_SIZE

    run(src, out, size)
