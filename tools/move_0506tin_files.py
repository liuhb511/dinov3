import os
import shutil


def move_files_by_prefix(source_root, target_root, prefix="0506tin"):
    """
    从 source_root/train 和 source_root/val 中查找文件名以 prefix 开头的文件，
    将 images 和 masks 中匹配的文件移动到 target_root，并保留原目录结构。

    只处理以下四个目录：
        train/images
        train/masks
        val/images
        val/masks

    """
    source_root = os.path.abspath(source_root)
    target_root = os.path.abspath(target_root)

    if not os.path.isdir(source_root):
        raise NotADirectoryError("源目录不存在：" + source_root)

    if source_root == target_root:
        raise ValueError("目标目录不能与源目录相同")

    prefix_lower = prefix.lower()
    dataset_splits = ("train", "val")
    data_types = ("images", "masks")

    total_moved = 0
    total_skipped = 0

    for split in dataset_splits:
        for data_type in data_types:
            source_dir = os.path.join(source_root, split, data_type)
            target_dir = os.path.join(target_root, split, data_type)

            if not os.path.isdir(source_dir):
                print("目录不存在，已跳过：", source_dir)
                continue

            os.makedirs(target_dir, exist_ok=True)

            moved_count = 0
            skipped_count = 0

            for filename in sorted(os.listdir(source_dir)):
                source_path = os.path.join(source_dir, filename)

                if not os.path.isfile(source_path):
                    continue

                if not filename.lower().startswith(prefix_lower):
                    continue

                target_path = os.path.join(target_dir, filename)

                if os.path.exists(target_path):
                    print("目标文件已存在，未移动：", target_path)
                    skipped_count += 1
                    total_skipped += 1
                    continue

                shutil.move(source_path, target_path)
                print("已移动：", source_path, "->", target_path)

                moved_count += 1
                total_moved += 1

            print(
                split + "/" + data_type,
                "处理完成，移动",
                moved_count,
                "个文件，跳过",
                skipped_count,
                "个文件"
            )

    print("--------------------------------")
    print("全部处理完成")
    print("成功移动文件数量：", total_moved)
    print("因目标文件已存在而跳过的数量：", total_skipped)
    print("文件保存目录：", target_root)


if __name__ == "__main__":
    # AAAAA 目录路径
    source_root = r"F:/liuhaibo/datasets/BG_HQL_JZW/dataset_crops_1024/ABCTIN"

    # 移出的文件保存路径
    target_root = r"F:/liuhaibo/datasets/BG_HQL_JZW/dataset_crops_1024/ABCTIN/0506tin_files"

    move_files_by_prefix(source_root, target_root, prefix="0506tin")
