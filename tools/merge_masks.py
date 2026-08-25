import cv2
import numpy as np
import os


def remap_mask(mask, mapping):
    """
    根据类别映射转换mask。
    未出现在mapping中的类别默认保持为背景0。
    """
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for old_class, new_class in mapping.items():
        new_mask[mask == old_class] = new_class

    return new_mask


def merge_single_mask(mask1_path, mask2_path, save_path, mapping1, mapping2):
    """
    融合一组同名mask。
    第二组mask中的非0区域优先覆盖第一组mask。
    """
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    if mask1 is None:
        print("第一组mask读取失败，已跳过：", mask1_path)
        return False

    if mask2 is None:
        print("第二组mask读取失败，已跳过：", mask2_path)
        return False

    if mask1.shape != mask2.shape:
        print("两个mask尺寸不一致，已跳过：", os.path.basename(mask1_path))
        print("第一组尺寸：", mask1.shape)
        print("第二组尺寸：", mask2.shape)
        return False

    mask1_new = remap_mask(mask1, mapping1)
    mask2_new = remap_mask(mask2, mapping2)

    # 第二组优先级高于第一组，第二组中的非0区域覆盖第一组
    result = mask1_new.copy()
    result[mask2_new != 0] = mask2_new[mask2_new != 0]

    save_result = cv2.imwrite(save_path, result)

    if not save_result:
        print("融合结果保存失败：", save_path)
        return False

    print("融合完成：", save_path)
    return True


def merge_mask_folders(mask1_dir, mask2_dir, output_dir):
    """
    遍历两个掩码文件夹，将同名mask进行融合。

    参数：
        mask1_dir：第一组mask文件夹
        mask2_dir：第二组mask文件夹
        output_dir：融合结果保存文件夹
    """
    if not os.path.isdir(mask1_dir):
        raise NotADirectoryError("第一组mask文件夹不存在：" + mask1_dir)

    if not os.path.isdir(mask2_dir):
        raise NotADirectoryError("第二组mask文件夹不存在：" + mask2_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 第一组类别映射
    mapping1 = {
        0: 0,  # 背景
        1: 1,  # A
        2: 2,  # B
        3: 3,  # C
        4: 7,  # HH
        5: 8,  # XW
        6: 9,  # XQL
        7: 5,  # TIN-B/TIN-C
        8: 6,  # TIN-D
    }

    # 第二组类别映射
    mapping2 = {
        0: 0,   # 背景
        1: 4,   # D
        2: 10,  # HC
        3: 11,  # SZ
    }

    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    total_count = 0
    success_count = 0
    missing_count = 0
    failed_count = 0

    mask1_files = sorted(os.listdir(mask1_dir))

    for filename in mask1_files:
        mask1_path = os.path.join(mask1_dir, filename)

        # 跳过文件夹
        if not os.path.isfile(mask1_path):
            continue

        extension = os.path.splitext(filename)[1].lower()

        # 只处理图像文件
        if extension not in supported_extensions:
            continue

        total_count += 1

        mask2_path = os.path.join(mask2_dir, filename)
        save_path = os.path.join(output_dir, filename)

        # 第二个文件夹中不存在同名文件
        if not os.path.isfile(mask2_path):
            print("第二组文件夹中找不到同名mask，已跳过：", filename)
            missing_count += 1
            continue

        success = merge_single_mask(mask1_path, mask2_path, save_path, mapping1, mapping2)

        if success:
            success_count += 1
        else:
            failed_count += 1

    print("------------------------------")
    print("处理完成")
    print("第一组mask数量：", total_count)
    print("成功融合数量：", success_count)
    print("缺少同名mask数量：", missing_count)
    print("处理失败数量：", failed_count)
    print("输出文件夹：", output_dir)


if __name__ == "__main__":
    # 第一组掩码文件夹
    mask1_dir = "F:/liuhaibo/datasets/test/JZW/HQL_0825/output_unet/mask_ABC/"
    # mask1_dir = "./output/infer_LG/744_gray/ABCTIN1024_gray/"

    # 第二组掩码文件夹
    mask2_dir = "F:/liuhaibo/datasets/test/JZW/HQL_0825/output_unet/mask_D/"
    # mask2_dir = "./output/infer_LG/744_gray/D1024_gray/"

    # 融合后的掩码保存文件夹
    output_dir = "F:/liuhaibo/datasets/test/JZW/HQL_0825/output_unet/masks_merge/"
    # output_dir = "./output/merge/744_gray/"

    merge_mask_folders(mask1_dir, mask2_dir, output_dir)