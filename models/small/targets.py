"""Build size-aware main-loss weights and auxiliary labels from 7-class masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


@dataclass
class SmallTargetStats:
    d_le3: int = 0
    d_3_4: int = 0
    d_4_5: int = 0
    d_gt5: int = 0
    tind_small: int = 0
    tind_other: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


def normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4:
        if mask.shape[1] != 1:
            raise ValueError(f"mask 必须是 [B,1,H,W] 或 [B,H,W]，实际 {tuple(mask.shape)}")
        mask = mask[:, 0]
    if mask.dim() != 3:
        raise ValueError(f"mask 必须是 [B,1,H,W] 或 [B,H,W]，实际 {tuple(mask.shape)}")
    return mask.long()


def equivalent_diameter_um(area_px: int, um_per_px: float) -> float:
    if area_px <= 0:
        return 0.0
    diameter_px = 2.0 * math.sqrt(float(area_px) / math.pi)
    return diameter_px * float(um_per_px)


def _disk(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    k = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


class SmallTargetTargetBuilder:
    """CPU target builder.

    Default 7-class scheme:
      0 BG, 1 A, 2 B, 3 C, 4 D, 5 TINBC, 6 TIND

    Main segmentation GT is never changed. Only an additional pixel-weight map
    and two auxiliary binary targets are produced.
    """

    def __init__(self, cfg):
        self.um_per_px = float(getattr(cfg, "um_per_px", 0.5448))
        self.d_class_id = int(getattr(cfg, "d_class_id", 4))
        self.tind_class_id = int(getattr(cfg, "tind_class_id", 6))

        self.d_w_le3 = float(getattr(cfg, "d_weight_le3", 4.0))
        self.d_w_3_4 = float(getattr(cfg, "d_weight_3_4", 3.0))
        self.d_w_4_5 = float(getattr(cfg, "d_weight_4_5", 1.5))
        self.tind_small_weight = float(getattr(cfg, "tind_small_weight", 1.5))

        self.d_aux_max_um = float(getattr(cfg, "d_aux_max_um", 5.0))
        self.tind_aux_max_um = float(getattr(cfg, "tind_aux_max_um", 5.0))
        self.d_dilate_px = int(getattr(cfg, "d_aux_dilate_px", 2))
        self.tind_dilate_px = int(getattr(cfg, "tind_aux_dilate_px", 1))
        self.d_ring_px = int(getattr(cfg, "d_aux_ring_px", 5))
        self.tind_ring_px = int(getattr(cfg, "tind_aux_ring_px", 3))

        self.aux_bg_weight = float(getattr(cfg, "aux_bg_weight", 0.03))
        self.aux_ring_weight = float(getattr(cfg, "aux_ring_weight", 0.25))
        self.aux_pos_weight = float(getattr(cfg, "aux_pos_weight", 1.0))

    @staticmethod
    def _components(binary: np.ndarray):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        for comp_id in range(1, n):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            yield labels == comp_id, area

    def _write_aux_component(
        self,
        cls_mask: np.ndarray,
        full_mask: np.ndarray,
        comp: np.ndarray,
        aux_target: np.ndarray,
        aux_weight: np.ndarray,
        channel: int,
        dilate_px: int,
        ring_px: int,
    ):
        # Positive auxiliary label can expand into background, but never overwrite
        # another true inclusion class.
        comp_u8 = comp.astype(np.uint8)
        positive = cv2.dilate(comp_u8, _disk(dilate_px), iterations=1) > 0
        allowed_positive = (full_mask == 0) | cls_mask
        positive &= allowed_positive

        aux_target[channel, positive] = 1.0
        aux_weight[channel, positive] = np.maximum(
            aux_weight[channel, positive], self.aux_pos_weight
        )

        # Local background ring: explicitly teach "object center/strip vs nearby BG".
        outer = cv2.dilate(comp_u8, _disk(ring_px), iterations=1) > 0
        ring = outer & (~positive) & (full_mask == 0)
        aux_weight[channel, ring] = np.maximum(
            aux_weight[channel, ring], self.aux_ring_weight
        )

    def build(self, masks: torch.Tensor):
        masks = normalize_mask_tensor(masks).detach().cpu()
        b, h, w = masks.shape

        pixel_weights = np.ones((b, h, w), dtype=np.float32)
        aux_targets = np.zeros((b, 2, h, w), dtype=np.float32)
        aux_weights = np.full((b, 2, h, w), self.aux_bg_weight, dtype=np.float32)
        stats = SmallTargetStats()

        for bi in range(b):
            m = masks[bi].numpy().astype(np.uint8, copy=False)

            d_mask = m == self.d_class_id
            for comp, area in self._components(d_mask):
                size_um = equivalent_diameter_um(area, self.um_per_px)
                if size_um <= 3.0:
                    pixel_weights[bi, comp] = np.maximum(pixel_weights[bi, comp], self.d_w_le3)
                    stats.d_le3 += 1
                elif size_um <= 4.0:
                    pixel_weights[bi, comp] = np.maximum(pixel_weights[bi, comp], self.d_w_3_4)
                    stats.d_3_4 += 1
                elif size_um <= 5.0:
                    pixel_weights[bi, comp] = np.maximum(pixel_weights[bi, comp], self.d_w_4_5)
                    stats.d_4_5 += 1
                else:
                    stats.d_gt5 += 1

                if size_um <= self.d_aux_max_um:
                    self._write_aux_component(
                        d_mask, m, comp,
                        aux_targets[bi], aux_weights[bi],
                        channel=0,
                        dilate_px=self.d_dilate_px,
                        ring_px=self.d_ring_px,
                    )

            tind_mask = m == self.tind_class_id
            for comp, area in self._components(tind_mask):
                size_um = equivalent_diameter_um(area, self.um_per_px)
                if size_um <= self.tind_aux_max_um:
                    pixel_weights[bi, comp] = np.maximum(
                        pixel_weights[bi, comp], self.tind_small_weight
                    )
                    stats.tind_small += 1
                    self._write_aux_component(
                        tind_mask, m, comp,
                        aux_targets[bi], aux_weights[bi],
                        channel=1,
                        dilate_px=self.tind_dilate_px,
                        ring_px=self.tind_ring_px,
                    )
                else:
                    stats.tind_other += 1

        return (
            torch.from_numpy(pixel_weights),
            torch.from_numpy(aux_targets),
            torch.from_numpy(aux_weights),
            stats.as_dict(),
        )
