import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import read_image


def letterbox(image, masks, size):
    h, w = image.shape[:2]
    scale = size/max(h, w)
    nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
    top, left = (size-nh)//2, (size-nw)//2
    canvas = np.full((size, size, 3), 128, np.uint8)
    canvas[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
    result = []
    for i, mask in enumerate(masks):
        fill = 255 if i == 0 else 1 if i == 2 else 0
        target = np.full((size, size), fill, np.uint8)
        target[top:top+nh, left:left+nw] = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        result.append(target)
    return canvas, result


class DDataset(Dataset):
    def __init__(self, rows, size=1024, augment=True):
        if not rows or size <= 0:
            raise ValueError("Nonempty dataset and positive size required")
        self.rows, self.size, self.augment = rows, size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_image(row["image_path"])[..., ::-1].copy()
        gt = read_image(row["gt_path"], True)
        negative = read_image(row["negative_path"], True) if row.get("negative_path") else np.zeros(gt.shape, np.uint8)
        ignored = read_image(row["ignore_path"], True) if row.get("ignore_path") else (gt == 255).astype(np.uint8)
        if any(mask.shape != image.shape[:2] for mask in (gt, negative, ignored)):
            raise ValueError("Supervision dimensions differ")
        if set(np.unique(gt).tolist()) - {0, 1, 255}:
            raise ValueError("GT must contain 0/1/255")
        image, masks = letterbox(image, [gt, negative, ignored], self.size)
        # Apply geometry once to all arrays, preserving the native point outline.
        if self.augment:
            for axis in (0, 1):
                if random.random() < .5:
                    image = np.flip(image, axis)
                    masks = [np.flip(mask, axis) for mask in masks]
            turns = random.randrange(4)
            image = np.rot90(image, turns)
            masks = [np.rot90(mask, turns) for mask in masks]
            if random.random() < .5:
                image = np.clip(image.astype(np.float32)*random.uniform(.9, 1.1)+random.uniform(-10, 10), 0, 255)
        gt, negative, ignored = [np.ascontiguousarray(mask) for mask in masks]
        ignored = (ignored > 0) | (gt == 255)
        negative = (negative > 0) & ~ignored
        if (negative & (gt == 1)).any():
            raise ValueError("Hard-negative overlaps D")
        gt[ignored] = 255
        image = np.ascontiguousarray(image, dtype=np.float32)/255
        image = (image-np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
        return (torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(gt.astype(np.int64)),
                torch.from_numpy(negative), row["image_id"])
