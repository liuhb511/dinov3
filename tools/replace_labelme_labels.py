import json
import os
from collections import Counter
from datetime import datetime


# 标签替换规则
LABEL_MAPPING = {
    "yangpin": "XQY",
}


def replace_labelme_labels(
    json_directory: str,
    output_log: str = "label_replace_log.txt"
) -> None:
    """
    递归遍历目录中的 Labelme JSON 文件，并原地修改指定标签。

    替换规则：
        S -> A
        D类-球状氧化物 -> D
        C类-硅酸盐 -> C

    参数：
        json_directory:
            Labelme JSON 文件所在目录。

        output_log:
            修改日志文件路径。
    """

    json_directory = os.path.abspath(json_directory)
    output_log = os.path.abspath(output_log)

    scanned_json_count = 0
    modified_json_count = 0
    modified_shape_count = 0

    replacement_counter = Counter()
    failed_files = []
    modified_files = []

    for root, _, files in os.walk(json_directory):
        for filename in files:
            if not filename.lower().endswith(".json"):
                continue

            json_path = os.path.join(root, filename)
            scanned_json_count += 1

            try:
                with open(json_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                shapes = data.get("shapes", [])

                if not isinstance(shapes, list):
                    failed_files.append(
                        (json_path, "shapes 字段不是列表")
                    )
                    continue

                current_file_counter = Counter()

                for shape in shapes:
                    if not isinstance(shape, dict):
                        continue

                    old_label = shape.get("label")

                    if not isinstance(old_label, str):
                        continue

                    # 去除标签前后可能存在的空格
                    normalized_label = old_label.strip()

                    if normalized_label in LABEL_MAPPING:
                        new_label = LABEL_MAPPING[normalized_label]

                        shape["label"] = new_label

                        replacement_key = (
                            f"{normalized_label} -> {new_label}"
                        )

                        replacement_counter[replacement_key] += 1
                        current_file_counter[replacement_key] += 1
                        modified_shape_count += 1

                # 当前 JSON 有标签发生修改时才覆盖文件
                if current_file_counter:
                    with open(json_path, "w", encoding="utf-8") as file:
                        json.dump(
                            data,
                            file,
                            ensure_ascii=False,
                            indent=2
                        )

                    modified_json_count += 1

                    modified_files.append({
                        "json_path": json_path,
                        "changes": current_file_counter,
                    })

                    print(f"[已修改] {json_path}")

                    for change, count in current_file_counter.items():
                        print(f"    {change}: {count}")

            except json.JSONDecodeError as error:
                failed_files.append(
                    (json_path, f"JSON 格式错误：{error}")
                )

            except UnicodeDecodeError as error:
                failed_files.append(
                    (json_path, f"文件编码错误：{error}")
                )

            except OSError as error:
                failed_files.append(
                    (json_path, f"文件读写错误：{error}")
                )

            except Exception as error:
                failed_files.append(
                    (json_path, f"未知错误：{error}")
                )

    lines = []

    lines.append("=" * 80)
    lines.append("Labelme 标签批量修改日志")
    lines.append(f"修改时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"扫描目录：{json_directory}")
    lines.append("=" * 80)
    lines.append(f"扫描 JSON 文件数：{scanned_json_count}")
    lines.append(f"修改 JSON 文件数：{modified_json_count}")
    lines.append(f"修改标注总数：{modified_shape_count}")
    lines.append(f"失败文件数：{len(failed_files)}")
    lines.append("")

    lines.append("标签替换统计：")

    if replacement_counter:
        for replacement, count in replacement_counter.items():
            lines.append(f"  {replacement:<30}: {count}")
    else:
        lines.append("  未找到需要替换的标签。")

    if modified_files:
        lines.append("")
        lines.append("=" * 80)
        lines.append("修改文件明细")
        lines.append("=" * 80)

        for index, item in enumerate(modified_files, start=1):
            lines.append("")
            lines.append(f"[{index}] {item['json_path']}")

            for replacement, count in item["changes"].items():
                lines.append(f"    {replacement}: {count}")

    if failed_files:
        lines.append("")
        lines.append("=" * 80)
        lines.append("处理失败文件")
        lines.append("=" * 80)

        for json_path, error_message in failed_files:
            lines.append("")
            lines.append(f"文件：{json_path}")
            lines.append(f"错误：{error_message}")

    report = "\n".join(lines)

    print("\n" + report)

    with open(output_log, "w", encoding="utf-8") as file:
        file.write(report)

    print(f"\n修改日志已保存到：{output_log}")


if __name__ == "__main__":
    # 修改为你的 Labelme JSON 标注文件目录
    JSON_DIRECTORY = r"D:/lhb/datasets/DBXH/DHTG0921/labeled"

    # 修改日志保存位置
    OUTPUT_LOG = r"./logs/label_replace_log.txt"

    replace_labelme_labels(
        json_directory=JSON_DIRECTORY,
        output_log=OUTPUT_LOG
    ) 