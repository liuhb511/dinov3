"""Small-target friendly augmentation that preserves physical object scale."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_small_train_transform():
    """No random scale / elastic warp.

    The original A-model transform can still be used for a strict augmentation
    ablation. This transform is the recommended Small-v1 default because
    D/TIND size thresholds are defined in micrometers.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])
