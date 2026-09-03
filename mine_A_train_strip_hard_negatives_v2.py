# -*- coding: utf-8 -*-
"""
mine_A_train_strip_hard_negatives_v2.py

用途
----
使用当前 A 模型（Simple-7 + Correct Token）完整推理一遍训练集，挖掘需要人工确认的
“条状夹杂物假阳性候选”，用于后续 verified hard-negative 训练。

A 模型类别（Simple-7）
---------------------
0 = Background
1 = A 类夹杂物
2 = B 类夹杂物
3 = C 类夹杂物
4 = D 类夹杂物
5 = TIN-B/TIN-C
6 = TIN-D

默认重点挖掘：B / C / TIN-B/TIN-C（2,3,5）。
原因：统一验证实验中，A 模型剩余的大面积条状 FP 主要集中在 B/C/TINBC，且大量为
细长、竖向、普通背景纹理。若希望扩大范围，可把 candidate_classes 改为 (1,2,3,5,6)。

核心安全规则
------------
1) 必须先在“完整 pred == class”上做连通域；
2) 只要整个预测连通域与任意真实夹杂物 GT(1..6) 有重叠，就整块排除；
3) 默认再排除距离真实夹杂物 <=3 px 的候选，避免把边缘过分割当作负样本；
4) component.png 是精确候选区域，bbox/crop 仅用于人工查看；
5) 本脚本只在 TRAIN SET 上挖掘，不要把验证集人工审核结果回灌训练。

人工审核标签建议
----------------
0 = confirmed_negative：确认不是夹杂物，可作为 Background hard negative
1 = missed_inclusion：实际是真夹杂物，GT 漏标；禁止作为负样本
2 = known_distractor：明确是 HH/XW/XQL/HC/SZ 等干扰物；对 Simple-7 仍作为 Background
3 = uncertain：不确定；禁止作为负样本

后续训练可使用 review_label in {0, 2}，禁止使用 {1, 3}。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


# ============================================================
# CONFIG：只需要优先检查这里
# ============================================================
CONFIG = {
    # A 模型训练集。按你当前 JZW 数据目录推定；如本机不同只改这两行。
    "image_dir": r"D:/lhb/datasets/JZW_v3/JZW/train/images",
    "mask_dir":  r"D:/lhb/datasets/JZW_v3/JZW/train/masks",

    # 当前 A = Simple-7-Correct checkpoint
    "checkpoint": r"./checkpoints/JZW_2/best_iou.pth",
    "backbone_name": r"dinov3_model",
    "num_classes": 7,

    # A 必须使用 correct token。脚本内置 encoder，不依赖当前 models/dinov3_encoder.py。
    "token_mode": "correct",
    "weight_source": "auto",  # auto / teacher / student

    "device": "cuda",
    "use_amp": True,
    "expected_hw": (1024, 1024),

    # --------------------------------------------------------
    # 条状 hard-negative 候选
    # --------------------------------------------------------
    # 第一轮建议先看 B/C/TINBC：这是当前 A 大 FP 最集中的类别。
    "candidate_classes": (2, 3, 5),
    # 若后续想把所有 strip 类都扫一遍：
    # "candidate_classes": (1, 2, 3, 5, 6),

    # 人工优先看较有价值的 component。
    "min_component_area_px": 64,
    "min_max_side_px": 16,

    # 与任何真实 inclusion GT 接触的完整预测 component 直接排除。
    "require_zero_inclusion_overlap": True,

    # 再排除 GT inclusion 周边 N 像素，避免边缘 over-segmentation。
    "exclude_inclusion_margin_px": 3,

    # 若训练 GT 是 unified12，7..11 可自动提示为 known distractor；
    # 若训练 GT 是 Simple-7，则这项只是保持兼容，不会自动判断。
    "known_distractor_suggest_fraction": 0.50,

    # crop 人工查看上下文。
    "crop_padding": 80,

    # 每类人工审核上限。按面积降序，优先保留大 FP。
    # None = 全部保存。
    "max_candidates_per_class": 500,
    "rank_by_area": True,

    # 对大面积候选再额外标记优先级，便于 CSV 排序/筛选。
    "priority_area_px": 256,
    "priority_aspect_ratio": 3.0,

    "output_dir": r"./hard_negative_A_train_strip_v2",
}


CLASS_NAMES = ["BG", "A", "B", "C", "D", "TINBC", "TIND"]
INCLUSION_CLASSES = (1, 2, 3, 4, 5, 6)
KNOWN_DISTRACTOR_NAMES = {
    7: "HH",
    8: "XW",
    9: "XQL",
    10: "HC",
    11: "SZ",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# 基础工具
# ============================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在：{image_dir}")
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise RuntimeError(f"没有找到训练图片：{image_dir}")
    return images


def find_mask_path(mask_dir: Path, image_path: Path) -> Path:
    direct = mask_dir / f"{image_path.stem}.png"
    if direct.exists():
        return direct
    for p in mask_dir.glob(f"{image_path.stem}.*"):
        if p.suffix.lower() in (".png", ".bmp", ".tif", ".tiff"):
            return p
    raise FileNotFoundError(f"找不到 {image_path.name} 对应 mask：{mask_dir}")


def validate_mask(mask: np.ndarray, path: Path):
    u = np.unique(mask)
    bad = u[(u < 0) | (u > 11)]
    if len(bad):
        raise ValueError(f"GT mask 存在非法类别 {bad.tolist()}：{path}")


def strip_module_prefix(state: Dict[str, torch.Tensor]):
    if state and all(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def select_state_dict(ckpt, weight_source: str = "auto"):
    """兼容普通 checkpoint 与 teacher/student/EMA checkpoint。"""
    if not isinstance(ckpt, dict):
        return ckpt, "raw"
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, "raw_state_dict"

    if weight_source == "teacher":
        priority = ["teacher", "student", "model", "state_dict", "model_state_dict"]
    elif weight_source == "student":
        priority = ["student", "teacher", "model", "state_dict", "model_state_dict"]
    else:
        priority = ["teacher", "student", "model", "state_dict", "model_state_dict"]

    for key in priority:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key], key
    raise ValueError(f"无法识别 checkpoint 格式，keys={list(ckpt.keys())[:30]}")


# ============================================================
# A 模型：显式 Correct Token，避免受仓库当前 encoder 源码影响
# ============================================================
class DINOv3EncoderCompat(nn.Module):
    """state_dict key 与原 encoder.backbone.* 兼容，但 token 逻辑显式固定。"""

    def __init__(self, model_name: str, token_mode: str = "correct"):
        super().__init__()
        if token_mode not in ("correct", "legacy"):
            raise ValueError("token_mode 必须是 correct 或 legacy")
        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.token_mode = token_mode
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _take_patch_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        if self.token_mode == "legacy":
            return feat[:, 1:-4, :]
        nreg = int(getattr(self.backbone.config, "num_register_tokens", 4))
        return feat[:, 1 + nreg:, :]

    def _reshape(self, feat: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        feat = self._take_patch_tokens(feat)
        b, n, c = feat.shape

        patch = getattr(self.backbone.config, "patch_size", 16)
        if isinstance(patch, (tuple, list)):
            ph, pw = int(patch[0]), int(patch[1])
        else:
            ph = pw = int(patch)

        ih, iw = input_hw
        h, w = ih // ph, iw // pw
        if h * w != n:
            side = int(round(n ** 0.5))
            if side * side != n:
                raise RuntimeError(f"patch token 数 {n} 无法 reshape")
            h = w = side

        return feat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        out = self.backbone(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        hw = (int(x.shape[-2]), int(x.shape[-1]))
        return {
            "f4": self._reshape(hs[-3], hw),
            "f8": self._reshape(hs[-2], hw),
            "f16": self._reshape(hs[-1], hw),
        }


class Simple7Correct(nn.Module):
    def __init__(self):
        super().__init__()
        from models.decoder_v3 import DecoderV3
        self.encoder = DINOv3EncoderCompat(
            CONFIG["backbone_name"], token_mode=CONFIG["token_mode"]
        )
        self.decoder = DecoderV3(num_classes=CONFIG["num_classes"], feat_dim=768)

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        return self.decoder(feats, output_size=x.shape[2:])


def load_model(device: torch.device):
    if CONFIG["token_mode"] != "correct":
        print("[WARN] 当前脚本用于 A 模型，正常应设置 token_mode='correct'")

    model = Simple7Correct().to(device)
    ckpt_path = Path(CONFIG["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state, source = select_state_dict(ckpt, CONFIG.get("weight_source", "auto"))
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded A checkpoint : {ckpt_path}")
    print(f"Loaded weight source: {source}")
    print(f"Token mode          : {CONFIG['token_mode']}")
    return model


# ============================================================
# 推理与连通域
# ============================================================
def preprocess_image(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    return x.to(device, non_blocking=True)


@torch.no_grad()
def infer_one(model: nn.Module, image_rgb: np.ndarray, device: torch.device):
    x = preprocess_image(image_rgb, device)
    amp = bool(CONFIG["use_amp"] and device.type == "cuda")
    if amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            seg, _ = model(x)
    else:
        seg, _ = model(x)

    prob = torch.softmax(seg.float(), dim=1)
    conf, pred = torch.max(prob, dim=1)
    return (
        pred[0].detach().cpu().numpy().astype(np.uint8),
        conf[0].detach().cpu().numpy().astype(np.float32),
    )


def connected_components(binary: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.asarray(binary, dtype=np.uint8), connectivity=8
    )
    comps = []
    for cid in range(1, n):
        x, y, w, h, area = stats[cid].tolist()
        cx, cy = centroids[cid].tolist()
        comps.append({
            "component_id": int(cid),
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area_px": int(area), "cx": float(cx), "cy": float(cy),
        })
    return labels, comps


def inclusion_mask(gt: np.ndarray) -> np.ndarray:
    return np.isin(gt, np.asarray(INCLUSION_CLASSES, dtype=np.uint8))


def build_exclusion_zone(gt: np.ndarray, margin_px: int) -> np.ndarray:
    fg = inclusion_mask(gt).astype(np.uint8)
    if margin_px <= 0:
        return fg.astype(bool)
    k = 2 * int(margin_px) + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(fg, kernel, iterations=1).astype(bool)


def known_distractor_stats(gt: np.ndarray, region: np.ndarray):
    total = int(region.sum())
    if total == 0:
        return 0, "", 0, 0.0
    counts = {}
    for cls_id, name in KNOWN_DISTRACTOR_NAMES.items():
        n = int(np.logical_and(region, gt == cls_id).sum())
        if n > 0:
            counts[cls_id] = n
    if not counts:
        return 0, "", 0, 0.0
    cls_id = max(counts, key=counts.get)
    px = counts[cls_id]
    return cls_id, KNOWN_DISTRACTOR_NAMES[cls_id], px, float(px / total)


def make_priority(area: int, aspect_ratio: float, orientation: str) -> str:
    # P0：大面积 + 明显细长/竖向，优先人工确认；P1：大面积；P2：普通候选。
    if (
        area >= int(CONFIG["priority_area_px"])
        and aspect_ratio >= float(CONFIG["priority_aspect_ratio"])
        and orientation in ("vertical", "horizontal")
    ):
        return "P0"
    if area >= int(CONFIG["priority_area_px"]):
        return "P1"
    return "P2"


# ============================================================
# 挖掘
# ============================================================
def collect_candidates(model: nn.Module, device: torch.device):
    image_dir = Path(CONFIG["image_dir"])
    mask_dir = Path(CONFIG["mask_dir"])
    images = list_images(image_dir)

    rows = []
    reject_overlap = 0
    reject_margin = 0

    for i, image_path in enumerate(images, 1):
        mask_path = find_mask_path(mask_dir, image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            raise RuntimeError(f"读取失败：{image_path} / {mask_path}")
        if bgr.shape[:2] != gt.shape[:2]:
            raise ValueError(f"image/mask 尺寸不一致：{image_path.name}")
        if CONFIG["expected_hw"] is not None and tuple(gt.shape[:2]) != tuple(CONFIG["expected_hw"]):
            raise ValueError(
                f"{image_path.name}: shape={gt.shape[:2]} != expected={CONFIG['expected_hw']}"
            )
        validate_mask(gt, mask_path)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred, conf = infer_one(model, rgb, device)
        inc_gt = inclusion_mask(gt)
        exclusion = build_exclusion_zone(gt, int(CONFIG["exclude_inclusion_margin_px"]))

        for cls_id in CONFIG["candidate_classes"]:
            if cls_id <= 0 or cls_id >= CONFIG["num_classes"]:
                raise ValueError(f"candidate class {cls_id} 非法")

            # 关键：必须在完整 pred==cls 上做连通域。
            labels, comps = connected_components(pred == cls_id)
            for comp in comps:
                region = labels == comp["component_id"]
                pred_area = int(region.sum())
                if pred_area <= 0:
                    continue

                overlap_px = int(np.logical_and(region, inc_gt).sum())
                if CONFIG["require_zero_inclusion_overlap"] and overlap_px > 0:
                    reject_overlap += 1
                    continue

                candidate_region = region & (~inc_gt)
                area = int(candidate_region.sum())
                if area < int(CONFIG["min_component_area_px"]):
                    continue
                if max(comp["w"], comp["h"]) < int(CONFIG["min_max_side_px"]):
                    continue

                near_px = int(np.logical_and(candidate_region, exclusion).sum())
                if near_px > 0:
                    reject_margin += 1
                    continue

                short_side = max(1, min(comp["w"], comp["h"]))
                long_side = max(comp["w"], comp["h"])
                ar = float(long_side / short_side)
                orientation = (
                    "vertical" if comp["h"] >= 2 * comp["w"]
                    else "horizontal" if comp["w"] >= 2 * comp["h"]
                    else "compact"
                )

                kd_id, kd_name, kd_px, kd_frac = known_distractor_stats(gt, candidate_region)
                suggested = "2" if kd_frac >= float(CONFIG["known_distractor_suggest_fraction"]) else ""
                priority = make_priority(area, ar, orientation)

                rows.append({
                    "image_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "gt_mask_path": str(mask_path.resolve()),
                    "pred_class_id": int(cls_id),
                    "pred_class_name": CLASS_NAMES[cls_id],
                    "component_id": comp["component_id"],
                    "area_px": area,
                    "pred_component_area_px": pred_area,
                    "inclusion_gt_overlap_px": overlap_px,
                    "known_distractor_majority_id": kd_id,
                    "known_distractor_majority_name": kd_name,
                    "known_distractor_overlap_px": kd_px,
                    "known_distractor_overlap_fraction": kd_frac,
                    "suggested_review_label": suggested,
                    "priority": priority,
                    "x": comp["x"], "y": comp["y"], "w": comp["w"], "h": comp["h"],
                    "aspect_ratio": ar,
                    "orientation": orientation,
                    "mean_confidence": float(conf[candidate_region].mean()),
                    "max_confidence": float(conf[candidate_region].max()),
                })

        if i % 20 == 0 or i == len(images):
            print(
                f"[Mining] {i}/{len(images)} | raw={len(rows)} | "
                f"reject_overlap={reject_overlap} | reject_margin={reject_margin}"
            )

    print(f"[Filter] reject_overlap={reject_overlap}, reject_margin={reject_margin}")
    return rows


def rank_and_limit(rows: list[dict]):
    # 每类分别限制数量；优先 P0/P1，再按面积/置信度。
    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    grouped = {c: [] for c in CONFIG["candidate_classes"]}
    for r in rows:
        grouped[r["pred_class_id"]].append(r)

    final = []
    for cls_id, group in grouped.items():
        if CONFIG["rank_by_area"]:
            group.sort(
                key=lambda r: (
                    priority_order[r["priority"]],
                    -r["area_px"],
                    -r["mean_confidence"],
                    r["image_name"],
                )
            )
        max_n = CONFIG["max_candidates_per_class"]
        if max_n is not None:
            group = group[: int(max_n)]
        final.extend(group)

    final.sort(
        key=lambda r: (
            priority_order[r["priority"]],
            -r["area_px"],
            r["pred_class_id"],
            r["image_name"],
        )
    )
    for idx, r in enumerate(final, 1):
        r["candidate_id"] = f"A_HN_{idx:06d}"
    return final


# ============================================================
# 保存人工审核素材
# ============================================================
def save_assets(model: nn.Module, device: torch.device, rows: list[dict]):
    out_root = Path(CONFIG["output_dir"])
    ensure_dir(out_root)

    by_image = {}
    for r in rows:
        by_image.setdefault(r["image_name"], []).append(r)

    for i, (image_name, image_rows) in enumerate(by_image.items(), 1):
        image_path = Path(CONFIG["image_dir"]) / image_name
        mask_path = find_mask_path(Path(CONFIG["mask_dir"]), image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred, _ = infer_one(model, rgb, device)

        inc_gt = inclusion_mask(gt)
        exclusion = build_exclusion_zone(gt, int(CONFIG["exclude_inclusion_margin_px"]))
        H, W = gt.shape

        cc_cache = {
            cls_id: connected_components(pred == cls_id)
            for cls_id in set(r["pred_class_id"] for r in image_rows)
        }

        for row in image_rows:
            labels, comps = cc_cache[row["pred_class_id"]]
            # 重新定位 component，避免第一次 mining 与保存阶段 ID 不一致风险。
            best = None
            best_score = None
            for c in comps:
                score = (
                    abs(c["x"] - row["x"]) + abs(c["y"] - row["y"])
                    + abs(c["w"] - row["w"]) + abs(c["h"] - row["h"])
                    + abs(c["area_px"] - row["pred_component_area_px"])
                    / max(1, row["pred_component_area_px"])
                )
                if best_score is None or score < best_score:
                    best, best_score = c, score
            if best is None:
                continue

            region = labels == best["component_id"]
            if CONFIG["require_zero_inclusion_overlap"] and np.logical_and(region, inc_gt).any():
                continue
            candidate_region = region & (~inc_gt)
            if np.logical_and(candidate_region, exclusion).any():
                continue

            x, y, w, h = best["x"], best["y"], best["w"], best["h"]
            pad = int(CONFIG["crop_padding"])
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(W, x+w+pad), min(H, y+h+pad)

            raw = bgr[y1:y2, x1:x2].copy()
            overlay = raw.copy()
            crop_region = candidate_region[y1:y2, x1:x2]

            layer = overlay.copy()
            layer[crop_region] = (255, 255, 255)
            overlay = cv2.addWeighted(overlay, 0.72, layer, 0.28, 0.0)
            cv2.rectangle(
                overlay,
                (x-x1, y-y1),
                (x-x1+w-1, y-y1+h-1),
                (0, 0, 255), 2,
            )
            text = (
                f"{row['candidate_id']} {row['priority']} "
                f"Pred={row['pred_class_name']} area={row['area_px']} "
                f"AR={row['aspect_ratio']:.1f} conf={row['mean_confidence']:.2f}"
            )
            cv2.putText(
                overlay, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 255), 1, cv2.LINE_AA,
            )

            # 按优先级/类别分目录，P0 可以最先人工审核。
            cls_dir = out_root / row["priority"] / row["pred_class_name"]
            ensure_dir(cls_dir)
            raw_path = cls_dir / f"{row['candidate_id']}_raw.jpg"
            overlay_path = cls_dir / f"{row['candidate_id']}_overlay.jpg"
            comp_path = cls_dir / f"{row['candidate_id']}_component.png"

            cv2.imwrite(str(raw_path), raw)
            cv2.imwrite(str(overlay_path), overlay)
            exact = candidate_region[y:y+h, x:x+w].astype(np.uint8) * 255
            cv2.imwrite(str(comp_path), exact)

            row.update({
                "component_id": best["component_id"],
                "x": x, "y": y, "w": w, "h": h,
                "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                "raw_crop_path": str(raw_path.resolve()),
                "overlay_path": str(overlay_path.resolve()),
                "component_mask_path": str(comp_path.resolve()),
                "review_label": "",
                "review_name": "",
                "review_note": "",
            })

        if i % 10 == 0 or i == len(by_image):
            print(f"[Saving] {i}/{len(by_image)} selected images")

    return rows


def write_candidates_csv(rows: list[dict]):
    out_root = Path(CONFIG["output_dir"])
    ensure_dir(out_root)
    csv_path = out_root / "candidates.csv"
    fields = [
        "candidate_id", "priority", "image_name", "image_path", "gt_mask_path",
        "pred_class_id", "pred_class_name", "component_id",
        "area_px", "pred_component_area_px",
        "inclusion_gt_overlap_px",
        "known_distractor_majority_id", "known_distractor_majority_name",
        "known_distractor_overlap_px", "known_distractor_overlap_fraction",
        "suggested_review_label",
        "x", "y", "w", "h", "aspect_ratio", "orientation",
        "mean_confidence", "max_confidence",
        "crop_x1", "crop_y1", "crop_x2", "crop_y2",
        "overlay_path", "raw_crop_path", "component_mask_path",
        "review_label", "review_name", "review_note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_summary(rows: list[dict]):
    out = Path(CONFIG["output_dir"]) / "mining_summary.csv"
    summary = []
    for cls_id in CONFIG["candidate_classes"]:
        rr = [r for r in rows if r["pred_class_id"] == cls_id]
        summary.append({
            "pred_class_id": cls_id,
            "pred_class_name": CLASS_NAMES[cls_id],
            "candidates": len(rr),
            "P0": sum(r["priority"] == "P0" for r in rr),
            "P1": sum(r["priority"] == "P1" for r in rr),
            "P2": sum(r["priority"] == "P2" for r in rr),
            "area_sum_px": sum(r["area_px"] for r in rr),
            "area_max_px": max((r["area_px"] for r in rr), default=0),
        })
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pred_class_id", "pred_class_name", "candidates", "P0", "P1", "P2", "area_sum_px", "area_max_px"],
        )
        writer.writeheader()
        writer.writerows(summary)
    return out


def main():
    device = torch.device(
        CONFIG["device"] if not str(CONFIG["device"]).startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    print("=" * 90)
    print("A 模型训练集 Strip Hard-Negative Mining")
    print(f"Device      : {device}")
    print(f"Image dir   : {CONFIG['image_dir']}")
    print(f"Mask dir    : {CONFIG['mask_dir']}")
    print(f"Checkpoint  : {CONFIG['checkpoint']}")
    print(f"Token mode  : {CONFIG['token_mode']}")
    print(f"Classes     : {CONFIG['candidate_classes']}")
    print(f"Output      : {CONFIG['output_dir']}")
    print("=" * 90)

    ensure_dir(CONFIG["output_dir"])
    model = load_model(device)

    print("\n[1/3] 推理训练集并挖完整预测连通域...")
    rows = collect_candidates(model, device)
    print(f"Raw candidates: {len(rows)}")

    print("\n[2/3] 按优先级/面积排序并限额...")
    rows = rank_and_limit(rows)
    for cls_id in CONFIG["candidate_classes"]:
        rr = [r for r in rows if r["pred_class_id"] == cls_id]
        print(
            f"  {CLASS_NAMES[cls_id]:>6}: {len(rr):4d} | "
            f"P0={sum(r['priority']=='P0' for r in rr):3d} "
            f"P1={sum(r['priority']=='P1' for r in rr):3d} "
            f"P2={sum(r['priority']=='P2' for r in rr):3d}"
        )

    print("\n[3/3] 保存人工审核 crop + 精确 component mask...")
    rows = save_assets(model, device, rows)
    csv_path = write_candidates_csv(rows)
    summary_path = write_summary(rows)

    print("\n完成。")
    print(f"候选 CSV : {csv_path}")
    print(f"统计 CSV : {summary_path}")
    print(f"素材目录 : {CONFIG['output_dir']}/P0, P1, P2")
    print("\n建议审核顺序：P0 -> P1 -> P2。")
    print("review_label: 0=确认负样本, 1=GT漏标夹杂物, 2=已知干扰物, 3=不确定")
    print("后续 Hard Negative 训练只使用 review_label 0 和 2。")
    print("\n如果继续用之前的网页审核器：")
    print("  python review_hard_negatives_web.py \\")
    print(f"      --csv {csv_path} --host 127.0.0.1 --port 7860")


if __name__ == "__main__":
    main()