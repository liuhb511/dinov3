from pathlib import Path
import argparse
import shutil


def find_jpgs(folder):
    """
    递归查找目录中的所有 jpg 文件。
    返回：
        {
            "a.jpg": [Path(...), Path(...)],
            "b.jpg": [Path(...)]
        }
    """
    result = {}

    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() == ".jpg":
            result.setdefault(path.name, []).append(path)

    return result


def main(folder1, folder2, output_folder):
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_folder = Path(output_folder)

    jpgs1 = find_jpgs(folder1)
    jpgs2 = find_jpgs(folder2)

    names1 = set(jpgs1.keys())
    names2 = set(jpgs2.keys())

    only_in_1 = sorted(names1 - names2)

    output_folder.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    for name in only_in_1:
        for src_path in jpgs1[name]:
            # 保留目录1中的相对目录结构
            relative_path = src_path.relative_to(folder1)
            dst_path = output_folder / relative_path

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_path, dst_path)

            print(f"复制: {src_path} -> {dst_path}")
            copied_count += 1

    print("\n处理完成")
    print(f"目录1独有文件名数量: {len(only_in_1)}")
    print(f"实际复制文件数量: {copied_count}")
    print(f"输出目录: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将目录1中存在、目录2中不存在的 JPG 文件复制到新文件夹"
    )

    parser.add_argument("folder1", help="第一个文件夹路径")
    parser.add_argument("folder2", help="第二个文件夹路径")
    parser.add_argument("output", help="输出文件夹路径")

    args = parser.parse_args()

    main(args.folder1, args.folder2, args.output)