"""Patch-level oversampling for tiny D and small TIND without changing the dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from .targets import equivalent_diameter_um


def _mask_path(mask_dir: str, image_name: str) -> str:
    stem = os.path.splitext(image_name)[0]
    p = os.path.join(mask_dir, stem + ".png")
    if os.path.exists(p):
        return p
    p2 = os.path.join(mask_dir, image_name)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"找不到 mask: {image_name} under {mask_dir}")


def _component_sizes(mask: np.ndarray, class_id: int, um_per_px: float) -> List[float]:
    binary = (mask == class_id).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > 0:
            out.append(equivalent_diameter_um(area, um_per_px))
    return out


def inspect_small_target_dataset(dataset, mask_dir: str, cfg) -> Tuple[List[float], Dict]:
    """Return one sampling weight per dataset item and a readable summary."""
    um_per_px = float(getattr(cfg, "um_per_px", 0.5448))
    d_class_id = int(getattr(cfg, "d_class_id", 4))
    tind_class_id = int(getattr(cfg, "tind_class_id", 6))

    w_d_le3 = float(getattr(cfg, "sample_weight_d_le3", 4.0))
    w_d_3_4 = float(getattr(cfg, "sample_weight_d_3_4", 3.0))
    w_d_4_5 = float(getattr(cfg, "sample_weight_d_4_5", 1.5))
    w_tind = float(getattr(cfg, "sample_weight_tind_small", 1.5))

    weights: List[float] = []
    summary = {
        "num_images": len(dataset),
        "images_with_d_le3": 0,
        "images_with_d_3_4": 0,
        "images_with_d_4_5": 0,
        "images_with_small_tind": 0,
        "normal_weight_images": 0,
    }

    for image_name in dataset.image_list:
        p = _mask_path(mask_dir, image_name)
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Cannot read mask: {p}")

        d_sizes = _component_sizes(m, d_class_id, um_per_px)
        tind_sizes = _component_sizes(m, tind_class_id, um_per_px)

        weight = 1.0
        has_le3 = any(s <= 3.0 for s in d_sizes)
        has_3_4 = any(3.0 < s <= 4.0 for s in d_sizes)
        has_4_5 = any(4.0 < s <= 5.0 for s in d_sizes)
        has_small_tind = any(s <= 5.0 for s in tind_sizes)

        if has_le3:
            weight = max(weight, w_d_le3)
            summary["images_with_d_le3"] += 1
        if has_3_4:
            weight = max(weight, w_d_3_4)
            summary["images_with_d_3_4"] += 1
        if has_4_5:
            weight = max(weight, w_d_4_5)
            summary["images_with_d_4_5"] += 1
        if has_small_tind:
            weight = max(weight, w_tind)
            summary["images_with_small_tind"] += 1
        if weight == 1.0:
            summary["normal_weight_images"] += 1

        weights.append(weight)

    summary["mean_sampling_weight"] = float(np.mean(weights)) if weights else 0.0
    summary["max_sampling_weight"] = float(np.max(weights)) if weights else 0.0
    return weights, summary


def build_small_target_sampler(dataset, mask_dir: str, cfg, summary_json: str | None = None):
    weights, summary = inspect_small_target_dataset(dataset, mask_dir, cfg)
    if summary_json:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    factor = float(getattr(cfg, "sampler_epoch_factor", 1.0))
    num_samples = max(1, int(round(len(weights) * factor)))
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )
    return sampler, summary
