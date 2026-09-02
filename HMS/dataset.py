import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import (
    letterbox_resize,
    rgb_to_lab_a,
    rgb_to_tensor,
    a_to_tensor,
)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

class HMSDataset(Dataset):
    """
    Returns:
        rgb:  [3,H,W], ImageNet-normalized RGB
        lab_a:[1,H,W], LAB a* normalized around [-1,1]
        mask: [H,W], int64 class-index mask
        name: filename

    LAB a* is computed from the image AFTER augmentation.
    """
    def __init__(
        self,
        image_dir,
        mask_dir=None,
        augmentation=None,
        image_size=1024,
    ):
        self.image_dir = str(image_dir)
        self.mask_dir = None if mask_dir is None else str(mask_dir)
        self.augmentation = augmentation
        self.image_size = int(image_size)
        self.image_list = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        self.has_mask = self.mask_dir is not None

    def __len__(self):
        return len(self.image_list)

    def _load_mask(self, img_name):
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Mask not found/unreadable: {mask_path}")
        return mask.astype(np.uint8)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask = self._load_mask(img_name) if self.has_mask else None
        rgb, mask = letterbox_resize(rgb, mask, self.image_size)

        if self.augmentation is not None:
            if mask is not None:
                out = self.augmentation(image=rgb, mask=mask)
                rgb, mask = out["image"], out["mask"]
            else:
                out = self.augmentation(image=rgb)
                rgb = out["image"]

        # KEY: calculate a* only after brightness/contrast/color-related augmentation.
        lab_a = rgb_to_lab_a(rgb)

        rgb_t = rgb_to_tensor(rgb)
        a_t = a_to_tensor(lab_a)

        if mask is None:
            return rgb_t, a_t, img_name

        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        return rgb_t, a_t, mask_t, img_name
