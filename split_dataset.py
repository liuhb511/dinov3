import os
import random
import shutil
from pathlib import Path


# ==========================================================
# 参数配置
# ==========================================================

# 原始数据路径
images_dir = r"D:/lhb/datasets/HMS/JPEGImages/"
masks_dir = r"D:/lhb/datasets/HMS/SegmentationClass/"

# 输出数据集路径
output_dir = r"D:/lhb/datasets/HMS/"

# 数据划分比例
train_ratio = 0.8
val_ratio = 0.2
test_ratio = 0.0     # 设置为0表示不划分测试集


# 随机种子
random_seed = 42


# 支持格式
image_extensions = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
]


# ==========================================================
# 工具函数
# ==========================================================

def create_dir(path):
    """
    创建目录
    """
    os.makedirs(path, exist_ok=True)



def get_images(folder):
    """
    获取所有图片路径
    """
    files = []

    for f in Path(folder).iterdir():

        if f.suffix.lower() in image_extensions:
            files.append(f)

    return sorted(files)



def copy_file(src, dst):
    """
    文件复制
    """
    shutil.copy2(src, dst)



# ==========================================================
# 数据划分
# ==========================================================

def split_dataset():

    random.seed(random_seed)


    # --------------------------
    # 检查比例
    # --------------------------

    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"比例错误: train+val+test={total_ratio}"
        )


    # --------------------------
    # 创建输出目录
    # --------------------------

    train_img_dir = Path(output_dir) / "train/images"
    train_mask_dir = Path(output_dir) / "train/masks"

    val_img_dir = Path(output_dir) / "val/images"
    val_mask_dir = Path(output_dir) / "val/masks"


    create_dir(train_img_dir)
    create_dir(train_mask_dir)

    create_dir(val_img_dir)
    create_dir(val_mask_dir)


    # test 可选

    if test_ratio > 0:

        test_img_dir = Path(output_dir) / "test/images"
        create_dir(test_img_dir)



    # --------------------------
    # 获取图片
    # --------------------------

    images = get_images(images_dir)


    if len(images) == 0:
        raise RuntimeError(
            "没有找到图片"
        )


    print(f"发现图片数量: {len(images)}")


    # --------------------------
    # 检查mask
    # --------------------------

    samples = []
    mask_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    for img in images:
        mask_path = None

        for ext in mask_extensions:
            candidate = Path(masks_dir) / f"{img.stem}{ext}"

            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"找不到原图 {img.name} 对应的 mask"
            )


        samples.append(
            (
                img,
                mask_path
            )
        )


    # --------------------------
    # 随机打乱
    # --------------------------

    random.shuffle(samples)



    n_total = len(samples)

    n_train = int(
        n_total * train_ratio
    )

    n_val = int(
        n_total * val_ratio
    )


    train_samples = samples[:n_train]

    val_samples = samples[
        n_train:n_train+n_val
    ]


    if test_ratio > 0:

        test_samples = samples[
            n_train+n_val:
        ]

    else:

        test_samples = []



    # --------------------------
    # 复制训练集
    # --------------------------

    print("复制训练集...")

    for img, mask in train_samples:

        copy_file(
            img,
            train_img_dir / img.name
        )

        copy_file(
            mask,
            train_mask_dir / mask.name
        )



    # --------------------------
    # 复制验证集
    # --------------------------

    print("复制验证集...")

    for img, mask in val_samples:

        copy_file(
            img,
            val_img_dir / img.name
        )

        copy_file(
            mask,
            val_mask_dir / mask.name
        )



    # --------------------------
    # 复制测试集
    # --------------------------

    if test_ratio > 0:

        print("复制测试集...")

        for img, mask in test_samples:

            copy_file(
                img,
                test_img_dir / img.name
            )



    # --------------------------
    # 输出统计
    # --------------------------

    print("\n数据划分完成")
    print("---------------------")

    print(
        f"Train : {len(train_samples)}"
    )

    print(
        f"Val   : {len(val_samples)}"
    )

    print(
        f"Test  : {len(test_samples)}"
    )


# ==========================================================
# main
# ==========================================================

if __name__ == "__main__":

    split_dataset()