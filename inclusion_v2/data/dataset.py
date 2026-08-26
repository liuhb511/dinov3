"""
inclusion_v2/data/dataset.py

读取用户提供的全类别（12 类 + bg）unified mask 数据集，
在线生成三个监督 target：
    gate  : [B,H,W]  3 类   bg=0 / strip=1 / point=2
    strip : [B,H,W]  9 类   bg+A/B/C/TINB-C/TIND/HH/XW/XQL；D/HC/SZ -> IGNORE
    point : [B,H,W]  4 类   bg+D/HC/SZ；A/B/C/TINB-C/TIND/HH/XW/XQL -> IGNORE
    mask  : [B,H,W]  12 类  原始 unified mask（供 boundary 等使用）

数据目录约定：
    <root>/train/images  +  <root>/train/masks
    <root>/val/images    +  <root>/val/masks
mask 像素值必须为 unified 编码 0..11。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .label_mapping import (
    CLASS_NAMES,
    NUM_CLASSES_UNIFIED,
    gate_target_tensor,
    strip_target_tensor,
    point_target_tensor,
)

_TENSOR_CACHE = {}


def _get_tensors():
    if "gate" not in _TENSOR_CACHE:
        _TENSOR_CACHE["gate"] = gate_target_tensor()
        _TENSOR_CACHE["strip"] = strip_target_tensor()
        _TENSOR_CACHE["point"] = point_target_tensor()
    return _TENSOR_CACHE["gate"], _TENSOR_CACHE["strip"], _TENSOR_CACHE["point"]


def quick_validate_labels(mask_dir, num_files=64, num_classes=NUM_CLASSES_UNIFIED):
    """
    启动时抽查 mask 类别值，确保都在 0..num_classes-1 内。
    返回实际出现的类别计数 Counter（dict）。
    """
    from collections import Counter
    files = sorted(f for f in os.listdir(mask_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")))
    if not files:
        raise RuntimeError(f"mask 目录为空: {mask_dir}")
    seen = Counter()
    invalid = set()
    for name in files[:num_files]:
        path = os.path.join(mask_dir, name)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"无法读取 mask: {path}")
        uniq = np.unique(m).tolist()
        for v in uniq:
            seen[int(v)] += 1
            if int(v) < 0 or int(v) >= num_classes:
                invalid.add(int(v))
    if invalid:
        raise ValueError(
            f"mask 存在非 unified 编码的类别值 {sorted(invalid)}（应 ∈ 0..{num_classes - 1}）。"
            f"请确认数据集使用统一编码: {CLASS_NAMES}"
        )
    return dict(seen)


class InclusionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_list = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        )
        if not self.image_list:
            raise RuntimeError(f"image 目录为空: {image_dir}")

    def __len__(self):
        return len(self.image_list)

    def _load(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"无法读取图像: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"无法读取 mask: {mask_path}")
        return image, mask, img_name

    def __getitem__(self, idx):
        image, mask, img_name = self._load(idx)

        gate_t, strip_t, point_t = _get_tensors()
        # 查表生成三个 target（先转 torch int64 索引，避免 numpy 高级索引歧义）
        mask_t = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))
        gate = gate_t[mask_t]       # [H,W] long
        strip = strip_t[mask_t]
        point = point_t[mask_t]

        if self.transform is not None:
            aug = self.transform(
                image=image,
                mask=mask,                       # 原始 uint8 mask（boundary 用）同步变换
                gate=gate.to(torch.uint8).numpy(),
                strip=strip.to(torch.uint8).numpy(),
                point=point.to(torch.uint8).numpy(),
            )
            image = aug["image"]
            mask = aug["mask"]
            gate = aug["gate"]
            strip = aug["strip"]
            point = aug["point"]

        targets = {
            "gate": torch.as_tensor(np.asarray(gate), dtype=torch.long),
            "strip": torch.as_tensor(np.asarray(strip), dtype=torch.long),
            "point": torch.as_tensor(np.asarray(point), dtype=torch.long),
            "mask": torch.as_tensor(np.asarray(mask), dtype=torch.long),
        }
        return image, targets, img_name
