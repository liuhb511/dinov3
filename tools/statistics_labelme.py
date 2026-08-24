import os
import json
from collections import Counter
from datetime import datetime


def statistics_labelme(json_dir, output_txt="label_statistics.txt"):
    """
    统计Labelme标注类别数量

    Parameters
    ----------
    json_dir : str
        Labelme json所在目录
    output_txt : str
        输出统计日志
    """

    label_counter = Counter()
    json_count = 0
    error_files = []

    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if not file.lower().endswith(".json"):
                continue

            json_path = os.path.join(root, file)
            json_count += 1

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                shapes = data.get("shapes", [])

                for shape in shapes:
                    label = shape.get("label")
                    if label is not None:
                        label_counter[label] += 1

            except Exception as e:
                error_files.append((json_path, str(e)))

    # 按数量降序排序
    sorted_labels = sorted(label_counter.items(),
                           key=lambda x: x[1],
                           reverse=True)

    # 打印结果
    print("=" * 60)
    print(f"统计时间：{datetime.now()}")
    print(f"JSON文件数量：{json_count}")
    print(f"标记种类数量：{len(label_counter)}")
    print("=" * 60)

    for label, count in sorted_labels:
        print(f"{label:<20} : {count}")

    if error_files:
        print("\n读取失败文件：")
        for f, err in error_files:
            print(f"{f}\n  {err}")

    # 保存日志
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"统计时间：{datetime.now()}\n")
        f.write(f"JSON文件数量：{json_count}\n")
        f.write(f"标记种类数量：{len(label_counter)}\n")
        f.write("=" * 60 + "\n")

        for label, count in sorted_labels:
            f.write(f"{label:<20} : {count}\n")

        if error_files:
            f.write("\n读取失败文件：\n")
            for file, err in error_files:
                f.write(f"{file}\n")
                f.write(f"  {err}\n")

    print("\n统计结果已保存到：", output_txt)


if __name__ == "__main__":
    # 修改为你的Labelme标注目录
    json_directory = r"D:\liuhaibo\data\temp\LG_labeled"

    # 输出日志名称
    output_log = "./tools/label_statistics_BGHQL.txt"

    statistics_labelme(json_directory, output_log)