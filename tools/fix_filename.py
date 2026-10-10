from pathlib import Path

# ====================== 用户配置 ======================

TARGET_DIR = Path(r"D:\lhb\datasets\AFe\SGWC_AFe-0911\SegmentationClass")  # 目标目录

RECURSIVE = False  # True：包含子文件夹；False：仅当前目录

OLD_CHAR = "α"
NEW_CHAR = "A"

# ====================================================


def main():
    if not TARGET_DIR.is_dir():
        print(f"目录不存在：{TARGET_DIR}")
        return

    files = (
        TARGET_DIR.rglob("*")
        if RECURSIVE
        else TARGET_DIR.iterdir()
    )

    success = 0
    skipped = 0
    failed = 0

    for file_path in files:
        if not file_path.is_file():
            continue

        if OLD_CHAR not in file_path.name:
            skipped += 1
            continue

        new_name = file_path.name.replace(OLD_CHAR, NEW_CHAR)
        new_path = file_path.with_name(new_name)

        try:
            if new_path.exists():
                print(f"[跳过] 目标文件已存在：{new_path.name}")
                skipped += 1
                continue

            file_path.rename(new_path)
            success += 1

            print(f"[成功] {file_path.name} -> {new_name}")

        except Exception as e:
            failed += 1
            print(f"[失败] {file_path.name}：{e}")

    print("\n========== 处理完成 ==========")
    print(f"成功重命名：{success}")
    print(f"跳过文件：{skipped}")
    print(f"失败文件：{failed}")


if __name__ == "__main__":
    main()