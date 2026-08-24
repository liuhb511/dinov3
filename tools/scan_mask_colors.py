from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def scan_mask_colors(mask_dir: Path) -> None:
    """
    遍历文件夹中的所有 PNG 掩码，统计所有像素颜色。

    支持：
        3 通道 RGB/BGR PNG
        4 通道 RGBA/BGRA PNG
        单通道 PNG

    参数：
        mask_dir：真实 mask 所在文件夹
    """

    if not mask_dir.exists():
        raise FileNotFoundError(
            f"文件夹不存在：{mask_dir}"
        )

    png_paths = sorted(
        path
        for path in mask_dir.rglob("*.png")
        if path.is_file()
    )

    if not png_paths:
        raise RuntimeError(
            f"没有找到 PNG 图片：{mask_dir}"
        )

    # 统计所有图片中的颜色总数
    total_color_count = defaultdict(int)

    # 记录每种颜色出现在哪些图片中
    color_files = defaultdict(set)

    print("=" * 80)
    print(f"PNG 图片数量：{len(png_paths)}")
    print("=" * 80)

    for mask_path in png_paths:
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            print(f"[读取失败] {mask_path}")
            continue

        print(f"\n图片：{mask_path.name}")
        print(f"尺寸：{mask.shape}")
        print(f"数据类型：{mask.dtype}")

        # --------------------------------------------------
        # 单通道图片
        # --------------------------------------------------
        if mask.ndim == 2:
            values, counts = np.unique(
                mask,
                return_counts=True,
            )

            print("类型：单通道图片")
            print("像素值：")

            for value, count in zip(values, counts):
                value_int = int(value)

                print(
                    f"  像素值={value_int:3d}，"
                    f"像素数量={int(count)}"
                )

                key = ("gray", value_int)
                total_color_count[key] += int(count)
                color_files[key].add(mask_path.name)

        # --------------------------------------------------
        # 三通道或四通道图片
        # --------------------------------------------------
        elif mask.ndim == 3:
            channel_count = mask.shape[2]

            if channel_count == 4:
                print("类型：四通道 BGRA 图片")

                # 保留 alpha 通道参与统计
                pixels = mask.reshape(-1, 4)

                colors, counts = np.unique(
                    pixels,
                    axis=0,
                    return_counts=True,
                )

                for color, count in zip(colors, counts):
                    b, g, r, a = map(int, color)

                    print(
                        f"  BGRA=({b:3d}, {g:3d}, {r:3d}, {a:3d})，"
                        f"RGBA=({r:3d}, {g:3d}, {b:3d}, {a:3d})，"
                        f"像素数量={int(count)}"
                    )

                    key = ("bgra", b, g, r, a)
                    total_color_count[key] += int(count)
                    color_files[key].add(mask_path.name)

            elif channel_count == 3:
                print("类型：三通道 BGR 图片")

                pixels = mask.reshape(-1, 3)

                colors, counts = np.unique(
                    pixels,
                    axis=0,
                    return_counts=True,
                )

                for color, count in zip(colors, counts):
                    b, g, r = map(int, color)

                    hex_color = f"#{r:02X}{g:02X}{b:02X}"

                    print(
                        f"  BGR=({b:3d}, {g:3d}, {r:3d})，"
                        f"RGB=({r:3d}, {g:3d}, {b:3d})，"
                        f"HEX={hex_color}，"
                        f"像素数量={int(count)}"
                    )

                    key = ("bgr", b, g, r)
                    total_color_count[key] += int(count)
                    color_files[key].add(mask_path.name)

            else:
                print(
                    f"[跳过] 不支持的通道数量：{channel_count}"
                )

        else:
            print(
                f"[跳过] 不支持的图片维度：{mask.ndim}"
            )

    print("\n")
    print("=" * 80)
    print("所有图片的颜色汇总")
    print("=" * 80)

    # 按像素数量从多到少排序
    sorted_colors = sorted(
        total_color_count.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for key, count in sorted_colors:
        color_type = key[0]
        files = sorted(color_files[key])

        if color_type == "gray":
            value = key[1]

            print(
                f"单通道值={value}，"
                f"总像素数={count}，"
                f"出现图片数={len(files)}"
            )

        elif color_type == "bgr":
            _, b, g, r = key
            hex_color = f"#{r:02X}{g:02X}{b:02X}"

            print(
                f"BGR=({b}, {g}, {r})，"
                f"RGB=({r}, {g}, {b})，"
                f"HEX={hex_color}，"
                f"总像素数={count}，"
                f"出现图片数={len(files)}"
            )

        elif color_type == "bgra":
            _, b, g, r, a = key

            print(
                f"BGRA=({b}, {g}, {r}, {a})，"
                f"RGBA=({r}, {g}, {b}, {a})，"
                f"总像素数={count}，"
                f"出现图片数={len(files)}"
            )

        print(
            "  图片：",
            ", ".join(files[:20]),
        )

        if len(files) > 20:
            print(
                f"  其余 {len(files) - 20} 张图片未显示"
            )

    print("\n统计完成。")


if __name__ == "__main__":
    # 修改成真实彩色 mask 所在文件夹
    mask_dir = Path(
        r"F:/liuhaibo/datasets/BG_HQL_JZW/dataset_crops_784/total/train/masks"
    )

    scan_mask_colors(mask_dir)