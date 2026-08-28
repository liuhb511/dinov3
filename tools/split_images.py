import os
import random
import shutil
from pathlib import Path

def select_and_move_images(src_dir, dst_dir, mode="num", select_value=500, seed=42):
    """
    从源目录随机挑选图片并移动到目标目录
    
    Args:
        src_dir: 原始数据目录路径
        dst_dir: 输出目录路径
        mode: 选择模式，"num" 按数量选择，"ratio" 按比例选择
        select_value: num模式表示选择数量；ratio模式下表示百分比（如20表示20%）
        seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(seed)
    
    # 转换为Path对象便于处理
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # 检查源目录是否存在
    if not src_path.exists():
        print(f"错误：源目录不存在 - {src_dir}")
        return
    
    # 创建目标目录
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件（支持常见图片格式）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    all_images = [f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not all_images:
        print(f"警告：在 {src_dir} 中未找到图片文件")
        return
    
    total_count = len(all_images)
    print(f"找到 {total_count} 个图片文件")
    
    # 计算要选择的图片数量
    if mode == "num":
        select_count = min(select_value, total_count)  # 不能超过总数
        print(f"按数量模式：选择 {select_count} 张图片")
    elif mode == "ratio":
        # 支持百分比数字（如20）或小数（如0.2）
        if select_value <= 1:
            ratio = select_value
        else:
            ratio = select_value / 100.0
        ratio = min(ratio, 1.0)  # 不能超过100%
        select_count = int(total_count * ratio)
        print(f"按比例模式：选择 {ratio*100:.1f}%（{select_count} 张图片）")
    else:
        print(f"错误：无效的模式 '{mode}'，请使用 'num' 或 'ratio'")
        return
    
    if select_count == 0:
        print("没有图片需要移动")
        return
    
    # 随机选择图片
    selected_images = random.sample(all_images, select_count)
    
    # 移动图片
    moved_count = 0
    for img_path in selected_images:
        try:
            # 构建目标路径
            dst_file_path = dst_path / img_path.name
            
            # 如果目标文件已存在，添加序号避免覆盖
            if dst_file_path.exists():
                base_name = img_path.stem
                extension = img_path.suffix
                counter = 1
                while dst_file_path.exists():
                    new_name = f"{base_name}_{counter}{extension}"
                    dst_file_path = dst_path / new_name
                    counter += 1
                print(f"警告：目标文件已存在，重命名为 {dst_file_path.name}")
            
            # 移动文件
            shutil.move(str(img_path), str(dst_file_path))
            moved_count += 1
        except Exception as e:
            print(f"移动文件 {img_path.name} 失败: {e}")
    
    print(f"成功移动 {moved_count} 张图片到 {dst_dir}")
    print(f"剩余 {total_count - moved_count} 张图片在源目录")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    src_dir = r"F:/liuhaibo/datasets/test/JZW/LG_JZW_0807_total"       # 原始数据目录
    dst_dir = r"F:/liuhaibo/datasets/test/JZW/LG_JZW_0807_test"        # 测试集输出目录
    
    mode = "num"              # 可选: "num" 或 "ratio"
    select_value = 500        # num模式表示数量；ratio模式下0.2或20都表示20%
    seed = 42                 # 随机种子
    
    # 执行移动操作
    select_and_move_images(src_dir, dst_dir, mode, select_value, seed)