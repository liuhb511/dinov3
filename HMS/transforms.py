import albumentations as A
import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

def letterbox_resize(image, mask, image_size=1024):
    h, w = image.shape[:2]
    scale = image_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_h = image_size - new_h
    pad_w = image_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    if mask is not None:
        mask = cv2.copyMakeBorder(
            mask, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
    return image, mask

def get_train_augmentation():
    """
    Important:
    LAB a* is calculated AFTER this augmentation, so RGB and a* always describe
    the same augmented image.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.05, 0.05),
            rotate=(-12, 12),
            shear=(-4, 4),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(128, 128, 128),
            fill_mask=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.20,
            p=0.5
        ),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.20),
    ])

def get_val_augmentation():
    return None

def rgb_to_lab_a(rgb_uint8):
    """
    OpenCV LAB a channel is encoded in [0,255], neutral near 128.
    Return a float32 tensor-like array normalized approximately to [-1,1].
    """
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    a = lab[..., 1].astype(np.float32)
    a = (a - 128.0) / 127.0
    return np.clip(a, -1.0, 1.0)

def rgb_to_tensor(rgb_uint8):
    x = rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()
    return x

def a_to_tensor(a_float):
    return torch.from_numpy(a_float[None, ...]).float()
