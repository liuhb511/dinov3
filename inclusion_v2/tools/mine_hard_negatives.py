# -*- coding: utf-8 -*-
"""
mine_hard_negatives.py

从训练集自动挖掘大型 BG -> A/B/C apparent false positives。

原则：
- 只在 TRAIN SET 上运行，不要在验证集上挖出来再用于训练。
- 原 GT mask 不做任何修改。
- 仅生成候选、裁图、精确 component mask 与 CSV，供人工审核。
- 默认候选条件：
    GT == background(0)
    final_pred in {A(1), B(2), C(3)}
    connected component area >= 64 px
    max(width, height) >= 16 px

数据格式：
    train/images/*
    train/masks/*.png
mask 为单通道索引 PNG，0..11，默认 1024x1024。

输出：
    hard_negative_mining/
      candidates.csv
      A/
        HN_000001_overlay.jpg
        HN_000001_raw.jpg
        HN_000001_component.png
      B/
      C/

component.png 是该候选的精确二值 component mask，仅 bbox 大小：
    0 = 非候选
    255 = candidate component
后续 build_hard_negative_masks.py 会把确认后的 component 精确贴回 1024x1024。
"""

import os
import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT: {PROJECT_ROOT}")


# ============================================================
# CONFIG：按你本机路径修改
# ============================================================
CONFIG = {
    "train_image_dir": r"F:/liuhaibo/datasets/JZW_v3/JZW_ALL/total/train/images",
    "train_mask_dir":  r"F:/liuhaibo/datasets/JZW_v3/JZW_ALL/total/train/masks",

    # 推荐用当前较激进、容易暴露误检的 checkpoint，例如 epoch64
    "checkpoint": r"./checkpoints/inclusion_v2_mvp1/best_inclusion_f1.pth",

    # 必须与训练时模型结构一致
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "encoder_layers": (4, 8, 12),
    "feat_dim": 768,
    "fusion_dim": 512,
    "decoder_dim": 32,

    "fusion_alpha": 0.5,
    "confidence_threshold": 0.0,

    "device": "cuda",
    "use_amp": True,
    "expected_hw": (1024, 1024),

    # 只挖 BG -> A/B/C
    "candidate_classes": (1, 2, 3),

    # 去掉太小、人工价值低的碎片
    "min_component_area_px": 64,
    "min_max_side_px": 16,

    # V2: hard negative 必须是“完整预测连通域”层面的纯背景误检。
    # 这样可避免真实 B/C 已正确标注并正确预测，只是预测向背景延伸，
    # 旧脚本却把延伸部分单独挖出来并用大 bbox 包住真实夹杂物。
    "require_zero_gt_foreground_overlap": True,

    # 排除距离任意已标注前景 <= N 像素的候选。
    # 主要用于过滤真实夹杂物边缘的轻微过分割/粘连。
    "exclude_gt_margin_px": 3,

    # crop 上下文
    "crop_padding": 64,

    # 每个类别最多保存多少个候选。
    # None = 全部保存；第一次建议 300~500/类即可。
    "max_candidates_per_class": 500,

    # 按 component 面积从大到小截取 max_candidates_per_class
    "rank_by_area": True,

    "output_dir": "./hard_negative_mining",
}


CLASS_NAMES = [
    "bg", "A", "B", "C", "D", "TINB/TINC", "TIND",
    "HH", "XW", "XQL", "HC", "SZ",
]
NUM_CLASSES = 12
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_mask_path(mask_dir, image_name):
    stem = Path(image_name).stem
    candidates = (
        Path(mask_dir) / f"{stem}.png",
        Path(mask_dir) / image_name,
    )
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"找不到 {image_name} 对应 mask")


def validate_mask(mask, path):
    uniq = np.unique(mask)
    bad = uniq[(uniq < 0) | (uniq >= NUM_CLASSES)]
    if len(bad):
        raise ValueError(f"mask 存在非法类别值 {bad.tolist()}：{path}")


def build_model_cfg():
    return SimpleNamespace(
        backbone_name=CONFIG["backbone_name"],
        freeze_backbone=CONFIG["freeze_backbone"],
        encoder_layers=CONFIG["encoder_layers"],
        feat_dim=CONFIG["feat_dim"],
        fusion_dim=CONFIG["fusion_dim"],
        decoder_dim=CONFIG["decoder_dim"],
    )


def load_model(device):
    from inclusion_v2.models.model import InclusionDualExpertNet

    model = InclusionDualExpertNet(build_model_cfg()).to(device)
    ckpt = torch.load(CONFIG["checkpoint"], map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        epoch = ckpt.get("epoch", None)
    else:
        state = ckpt
        epoch = None

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {CONFIG['checkpoint']}, epoch={epoch}")
    return model


def preprocess_image(image_rgb, device):
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    return x.to(device, non_blocking=True)


@torch.no_grad()
def infer_final(model, image_rgb, device):
    from inclusion_v2.utils.output_fusion import fuse_outputs

    x = preprocess_image(image_rgb, device)
    amp_enabled = CONFIG["use_amp"] and device.type == "cuda"

    if amp_enabled:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x)
            fused = fuse_outputs(
                outputs["gate"],
                outputs["strip"],
                outputs["point"],
                alpha=CONFIG["fusion_alpha"],
            )
    else:
        outputs = model(x)
        fused = fuse_outputs(
            outputs["gate"],
            outputs["strip"],
            outputs["point"],
            alpha=CONFIG["fusion_alpha"],
        )

    confidence, pred = torch.max(fused, dim=1)

    if CONFIG["confidence_threshold"] > 0:
        pred = pred.masked_fill(
            confidence < CONFIG["confidence_threshold"],
            0,
        )

    pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)
    conf_np = confidence[0].float().detach().cpu().numpy()
    return pred_np, conf_np


def connected_components(binary):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.asarray(binary, dtype=np.uint8),
        connectivity=8,
    )

    comps = []
    for cid in range(1, n):
        x, y, w, h, area = stats[cid].tolist()
        cx, cy = centroids[cid].tolist()
        comps.append({
            "component_id": int(cid),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area_px": int(area),
            "cx": float(cx),
            "cy": float(cy),
        })
    return labels, comps


def build_gt_exclusion_zone(gt, margin_px):
    """返回 GT 前景及其周围 margin_px 像素的排除区域。"""
    fg = (gt != 0).astype(np.uint8)
    if margin_px is None or margin_px <= 0:
        return fg.astype(bool)

    k = int(margin_px) * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(fg, kernel, iterations=1).astype(bool)


def collect_candidates(model, device):
    image_dir = Path(CONFIG["train_image_dir"])
    mask_dir = Path(CONFIG["train_mask_dir"])

    names = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not names:
        raise RuntimeError(f"训练图像目录为空：{image_dir}")

    candidates = []
    rejected_overlap = 0
    rejected_near_gt = 0

    for idx, image_name in enumerate(names, start=1):
        image_path = image_dir / image_name
        mask_path = find_mask_path(mask_dir, image_name)

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if bgr is None:
            raise RuntimeError(f"无法读取 image：{image_path}")
        if gt is None:
            raise RuntimeError(f"无法读取 mask：{mask_path}")
        if bgr.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"image/mask 尺寸不一致：{image_name}, "
                f"{bgr.shape[:2]} vs {gt.shape[:2]}"
            )

        expected = CONFIG["expected_hw"]
        if expected is not None and tuple(gt.shape[:2]) != tuple(expected):
            raise ValueError(
                f"{image_name} 尺寸为 {gt.shape[:2]}，预期 {expected}"
            )

        validate_mask(gt, mask_path)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred, confidence = infer_final(model, rgb, device)

        exclusion_zone = build_gt_exclusion_zone(
            gt, CONFIG.get("exclude_gt_margin_px", 0)
        )

        for cls_id in CONFIG["candidate_classes"]:
            # V2 核心：
            # 先对完整预测类别做连通域，而不是先做 (gt == 0) & pred。
            pred_binary = (pred == cls_id)
            labels, comps = connected_components(pred_binary)

            for comp in comps:
                region = labels == comp["component_id"]
                pred_area = int(region.sum())
                if pred_area <= 0:
                    continue

                gt_fg_overlap_px = int((region & (gt != 0)).sum())
                same_class_overlap_px = int((region & (gt == cls_id)).sum())

                # 只要完整预测块碰到任何已标注前景，就不作为 hard negative。
                if (
                    CONFIG.get("require_zero_gt_foreground_overlap", True)
                    and gt_fg_overlap_px > 0
                ):
                    rejected_overlap += 1
                    continue

                hn_region = region & (gt == 0)
                hn_area = int(hn_region.sum())

                if hn_area < CONFIG["min_component_area_px"]:
                    continue
                if max(comp["w"], comp["h"]) < CONFIG["min_max_side_px"]:
                    continue

                # 再排除紧贴 GT 前景的候选，避免边界外扩被当成负样本。
                near_gt_px = int((hn_region & exclusion_zone).sum())
                if near_gt_px > 0:
                    rejected_near_gt += 1
                    continue

                mean_conf = float(confidence[hn_region].mean()) if hn_region.any() else 0.0
                max_conf = float(confidence[hn_region].max()) if hn_region.any() else 0.0

                short_side = max(1, min(comp["w"], comp["h"]))
                long_side = max(comp["w"], comp["h"])
                aspect_ratio = float(long_side / short_side)
                orientation = (
                    "vertical" if comp["h"] >= 2 * comp["w"]
                    else "horizontal" if comp["w"] >= 2 * comp["h"]
                    else "compact"
                )

                candidates.append({
                    "image_name": image_name,
                    "pred_class_id": cls_id,
                    "pred_class_name": CLASS_NAMES[cls_id],
                    "component_id": comp["component_id"],
                    "area_px": hn_area,
                    "pred_component_area_px": pred_area,
                    "gt_foreground_overlap_px": gt_fg_overlap_px,
                    "gt_foreground_overlap_fraction": float(
                        gt_fg_overlap_px / max(1, pred_area)
                    ),
                    "same_class_overlap_px": same_class_overlap_px,
                    "same_class_overlap_fraction": float(
                        same_class_overlap_px / max(1, pred_area)
                    ),
                    "gt_exclusion_margin_px": int(
                        CONFIG.get("exclude_gt_margin_px", 0)
                    ),
                    "x": comp["x"],
                    "y": comp["y"],
                    "w": comp["w"],
                    "h": comp["h"],
                    "aspect_ratio": aspect_ratio,
                    "orientation": orientation,
                    "mean_confidence": mean_conf,
                    "max_confidence": max_conf,
                })

        if idx % 20 == 0 or idx == len(names):
            print(
                f"[Mining] {idx}/{len(names)} images, "
                f"raw candidates={len(candidates)}, "
                f"reject_overlap={rejected_overlap}, "
                f"reject_near_gt={rejected_near_gt}"
            )

    print(
        "[Mining filter summary] "
        f"reject_overlap={rejected_overlap}, "
        f"reject_near_gt={rejected_near_gt}"
    )
    return candidates


def rank_and_limit(candidates):
    grouped = {c: [] for c in CONFIG["candidate_classes"]}
    for row in candidates:
        grouped[row["pred_class_id"]].append(row)

    final_rows = []
    for cls_id, rows in grouped.items():
        if CONFIG["rank_by_area"]:
            rows.sort(key=lambda r: r["area_px"], reverse=True)

        max_n = CONFIG["max_candidates_per_class"]
        if max_n is not None:
            rows = rows[:max_n]

        final_rows.extend(rows)

    # 最后仍按类别 + 面积排序，便于人工审核
    final_rows.sort(
        key=lambda r: (
            r["pred_class_id"],
            -r["area_px"],
            r["image_name"],
        )
    )

    for i, row in enumerate(final_rows, start=1):
        row["candidate_id"] = f"HN_{i:06d}"

    return final_rows


def save_candidate_assets(model, device, rows):
    """
    为避免保存全图预测，第二遍只处理真正被选中的图。
    每张图仅推理一次，然后保存：
      overlay crop
      raw crop
      exact component mask bbox crop
    """
    out_root = Path(CONFIG["output_dir"])
    ensure_dir(out_root)

    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_name"], []).append(row)

    total = len(by_image)

    for idx, (image_name, image_rows) in enumerate(by_image.items(), start=1):
        image_path = Path(CONFIG["train_image_dir"]) / image_name
        mask_path = find_mask_path(CONFIG["train_mask_dir"], image_name)

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred, confidence = infer_final(model, rgb, device)

        h_img, w_img = gt.shape[:2]

        # V2: 第二遍同样使用完整预测连通域
        cc_cache = {}
        for cls_id in set(r["pred_class_id"] for r in image_rows):
            binary = (pred == cls_id)
            cc_cache[cls_id] = connected_components(binary)

        for row in image_rows:
            cls_id = row["pred_class_id"]
            labels, comps = cc_cache[cls_id]

            # 原始 component_id 在第二遍理论上应一致；
            # 为鲁棒性，优先用 bbox/area 最近邻重新匹配。
            best = None
            best_score = None
            for comp in comps:
                if comp["area_px"] < CONFIG["min_component_area_px"]:
                    continue
                score = (
                    abs(comp["x"] - row["x"])
                    + abs(comp["y"] - row["y"])
                    + abs(comp["w"] - row["w"])
                    + abs(comp["h"] - row["h"])
                    + abs(
                        comp["area_px"] - row.get("pred_component_area_px", row["area_px"])
                    ) / max(1, row.get("pred_component_area_px", row["area_px"]))
                )
                if best_score is None or score < best_score:
                    best = comp
                    best_score = score

            if best is None:
                print(f"[Warning] 无法重新定位候选 {row['candidate_id']}")
                continue

            row["component_id"] = best["component_id"]
            row["x"], row["y"] = best["x"], best["y"]
            row["w"], row["h"] = best["w"], best["h"]

            pred_component_region = labels == best["component_id"]

            # 第二遍再次做安全检查
            gt_fg_overlap_px = int((pred_component_region & (gt != 0)).sum())
            if (
                CONFIG.get("require_zero_gt_foreground_overlap", True)
                and gt_fg_overlap_px > 0
            ):
                print(
                    f"[Warning] {row['candidate_id']} 与 GT 前景重叠 "
                    f"{gt_fg_overlap_px}px，跳过"
                )
                continue

            component_region = pred_component_region & (gt == 0)
            exclusion_zone = build_gt_exclusion_zone(
                gt, CONFIG.get("exclude_gt_margin_px", 0)
            )
            if int((component_region & exclusion_zone).sum()) > 0:
                print(
                    f"[Warning] {row['candidate_id']} 距离 GT 前景过近，跳过"
                )
                continue

            pred_area = int(pred_component_region.sum())
            hn_area = int(component_region.sum())
            same_class_overlap_px = int(
                (pred_component_region & (gt == cls_id)).sum()
            )

            row["area_px"] = hn_area
            row["pred_component_area_px"] = pred_area
            row["gt_foreground_overlap_px"] = gt_fg_overlap_px
            row["gt_foreground_overlap_fraction"] = float(
                gt_fg_overlap_px / max(1, pred_area)
            )
            row["same_class_overlap_px"] = same_class_overlap_px
            row["same_class_overlap_fraction"] = float(
                same_class_overlap_px / max(1, pred_area)
            )
            row["gt_exclusion_margin_px"] = int(
                CONFIG.get("exclude_gt_margin_px", 0)
            )

            x, y, w, h = best["x"], best["y"], best["w"], best["h"]
            pad = CONFIG["crop_padding"]

            crop_x1 = max(0, x - pad)
            crop_y1 = max(0, y - pad)
            crop_x2 = min(w_img, x + w + pad)
            crop_y2 = min(h_img, y + h + pad)

            raw_crop = bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            overlay_crop = raw_crop.copy()

            # 组件相对于 crop 的 mask
            crop_region = component_region[crop_y1:crop_y2, crop_x1:crop_x2]

            # 半透明填充：不用依赖类别颜色，统一白色遮罩 + 红框
            layer = overlay_crop.copy()
            layer[crop_region] = (255, 255, 255)
            overlay_crop = cv2.addWeighted(
                overlay_crop, 0.72, layer, 0.28, 0.0
            )

            bx1 = x - crop_x1
            by1 = y - crop_y1
            bx2 = bx1 + w - 1
            by2 = by1 + h - 1
            cv2.rectangle(
                overlay_crop,
                (bx1, by1),
                (bx2, by2),
                (0, 0, 255),
                2,
            )

            title = (
                f"{row['candidate_id']} Pred={row['pred_class_name']} "
                f"area={row['area_px']} ar={row['aspect_ratio']:.2f}"
            )
            cv2.putText(
                overlay_crop,
                title,
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            cls_dir = out_root / row["pred_class_name"]
            ensure_dir(cls_dir)

            overlay_path = cls_dir / f"{row['candidate_id']}_overlay.jpg"
            raw_path = cls_dir / f"{row['candidate_id']}_raw.jpg"

            # 精确 component mask 只保存 bbox 区域，后续贴回全图。
            bbox_component = (
                component_region[y:y+h, x:x+w].astype(np.uint8) * 255
            )
            component_path = cls_dir / f"{row['candidate_id']}_component.png"

            cv2.imwrite(str(overlay_path), overlay_crop)
            cv2.imwrite(str(raw_path), raw_crop)
            cv2.imwrite(str(component_path), bbox_component)

            row["crop_x1"] = crop_x1
            row["crop_y1"] = crop_y1
            row["crop_x2"] = crop_x2
            row["crop_y2"] = crop_y2
            row["overlay_path"] = str(overlay_path)
            row["raw_crop_path"] = str(raw_path)
            row["component_mask_path"] = str(component_path)
            row["review_label"] = ""
            row["review_name"] = ""
            row["review_note"] = ""

        if idx % 10 == 0 or idx == total:
            print(f"[Saving] {idx}/{total} selected images")

    return rows


def write_csv(rows):
    path = Path(CONFIG["output_dir"]) / "candidates.csv"
    fields = [
        "candidate_id",
        "image_name",
        "pred_class_id",
        "pred_class_name",
        "component_id",
        "area_px",
        "pred_component_area_px",
        "gt_foreground_overlap_px",
        "gt_foreground_overlap_fraction",
        "same_class_overlap_px",
        "same_class_overlap_fraction",
        "gt_exclusion_margin_px",
        "x", "y", "w", "h",
        "aspect_ratio",
        "orientation",
        "mean_confidence",
        "max_confidence",
        "crop_x1", "crop_y1", "crop_x2", "crop_y2",
        "overlay_path",
        "raw_crop_path",
        "component_mask_path",
        "review_label",
        "review_name",
        "review_note",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCandidate CSV: {path}")
    return path


def main():
    device = torch.device(
        CONFIG["device"]
        if CONFIG["device"] == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    ensure_dir(CONFIG["output_dir"])
    model = load_model(device)

    print("\n[1/3] Mining raw BG->A/B/C candidates...")
    candidates = collect_candidates(model, device)
    print(f"Raw candidates: {len(candidates)}")

    print("\n[2/3] Ranking and limiting...")
    candidates = rank_and_limit(candidates)
    for cls_id in CONFIG["candidate_classes"]:
        n = sum(r["pred_class_id"] == cls_id for r in candidates)
        print(f"  {CLASS_NAMES[cls_id]}: {n}")

    print("\n[3/3] Saving crops and exact component masks...")
    rows = save_candidate_assets(model, device, candidates)
    write_csv(rows)

    print("\n完成。下一步运行：")
    print("  python review_hard_negatives.py")
    print("\n审核规则：")
    print("  0 = CONFIRMED_NEGATIVE")
    print("  1 = MISSED_INCLUSION")
    print("  2 = KNOWN_DISTRACTOR")
    print("  3 = UNCERTAIN")


if __name__ == "__main__":
    main()