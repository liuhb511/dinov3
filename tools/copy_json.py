from pathlib import Path
import shutil
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_json_index(source_folder):
    source_folder = Path(source_folder)
    json_index = {}

    for path in source_folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".json":
            json_index[path.stem] = path

    for subfolder in source_folder.iterdir():
        if subfolder.is_dir():
            for path in subfolder.iterdir():
                if path.is_file() and path.suffix.lower() == ".json":
                    json_index[path.stem] = path

    return json_index

def main(source_folder, target_folder):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)

    if not source_folder.is_dir():
        print(f"源文件夹不存在: {source_folder}")
        return

    if not target_folder.is_dir():
        print(f"目标文件夹不存在: {target_folder}")
        return

    json_index = build_json_index(source_folder)
    copied_count = 0
    missing_count = 0

    for image_path in target_folder.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        json_path = json_index.get(image_path.stem)

        if json_path is None:
            print(f"[未找到 JSON] {image_path.name}")
            missing_count += 1
            continue

        target_json = target_folder / json_path.name
        shutil.copy2(json_path, target_json)
        print(f"[复制] {json_path} -> {target_json}")
        copied_count += 1

    print(f"\n处理完成，成功复制: {copied_count}，未找到: {missing_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据目标图片复制对应 JSON")
    parser.add_argument("--source", default=r"D:/lhb/datasets/JZW_v4/process/JZW_V4_1024/images", help="JSON 源目录")
    parser.add_argument("--target", default=r"D:/lhb/datasets/JZW_v4/process/ZDX/images", help="目标图片文件夹")
    args = parser.parse_args()

    main(args.source, args.target)
