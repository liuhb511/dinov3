"""
inclusion_v2/data/transforms.py

MVP-1 增强策略（见方案文档 §8）：
- 删除 ElasticTransform（避免破坏 A/C 依赖的端部/宽度形态）
- 降低 Brightness/Contrast 幅度（A/C 首要依据是颜色/灰度）
- 降低 GaussNoise 概率/幅度（避免污染小 D、灰尘、点状内部结构）
- 保留 Horizontal/Vertical Flip（不产生任意倾斜角）
- 不做任意角度旋转（倾斜方向是条状背景噪声的有效先验）

所有 mask 类 target（gate/strip/point/mask）作为 additional_targets 同步变换，
mask 使用最近邻插值保证类别值不变。
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# 需要与 mask 同步变换的 target key
TARGET_KEYS = ["mask", "gate", "strip", "point"]


def _targets():
    return {k: "mask" for k in TARGET_KEYS}


def get_train_transform():
    """训练增强：几何（轻量）+ 弱颜色 → Normalize → Tensor。"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 轻量几何：微小平移/缩放（保留 A/C 宽度与端部形态）
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            border_mode=0,          # 常量填充，mask 用 0（背景）
            p=0.5,
        ),

        # 弱亮度/对比度（原为 ±0.15，降为 ±0.08）
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit=0.08,
            p=0.3,
        ),

        # 低噪声（原为 (0.01,0.05)/p=0.3，降为 (0.005,0.02)/p=0.15）
        A.GaussNoise(
            std_range=(0.005, 0.02),
            p=0.15,
        ),

        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ], additional_targets=_targets())


def get_val_transform():
    """验证/测试：仅 Normalize。"""
    return A.Compose([
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),
    ], additional_targets=_targets())
