# -*- coding: utf-8 -*-
"""
evaluate_inclusion_diagnostics.py

独立诊断评估脚本：只读取验证集与 checkpoint，不修改训练/网络代码。

验证集格式：
    <val_root>/images/*
    <val_root>/masks/*.png
mask 为 1024x1024 单通道索引 PNG，类别 0..11。

统一类别：
0=bg, 1=A, 2=B, 3=C, 4=D, 5=TINB/TINC, 6=TIND,
7=HH, 8=XW, 9=XQL, 10=HC, 11=SZ

输出重点：
- 12x12 confusion matrix
- Inclusion pixel P/R/F1 与每类 pixel P/R/F1
- TIND Gate->Strip / Strip Expert->TIND / Final->TIND 的 pixel/object recall
- D 按等效直径分组的 object recall
- A/B/C/TINB/TINC/TIND object consistency（量化 AAACCCAAAA）
- Coverage / Purity（量化不完整与面积外扩）
- Background->Strip / Background->D 的 apparent FP component
- 可选保存 apparent FP crops 供人工复核

注意：只有 semantic mask，没有 instance id，所以 object 指标用 connected components 近似。
同类目标如果相互接触，会被合并为一个 component。
"""

import os
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict

import cv2
import numpy as np
import torch


CONFIG = {
    "val_image_dir": r"F:/liuhaibo/datasets/JZW_v3/JZW_ALL/total/val/images",
    "val_mask_dir":  r"F:/liuhaibo/datasets/JZW_v3/JZW_ALL/total/val/masks",

    "checkpoints": {
        "epoch62": r"./checkpoints/inclusion_v2_mvp1/best_inclusion_precision.pth",
        "epoch64": r"./checkpoints/inclusion_v2_mvp1/best_inclusion_f1.pth",
    },

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

    "um_per_pixel": 0.5448,
    "object_coverage_threshold": 0.30,
    "fragment_min_ratio": 0.05,
    "d_size_bins_um": (5.0, 10.0),

    "fp_min_area_px": 2,
    "save_fp_crops": True,
    "max_fp_crops_each_type": 100,
    "fp_crop_padding": 48,

    "output_dir": "./diagnostics_output",
}


CLASS_NAMES = [
    "bg", "A", "B", "C", "D", "TINB/TINC", "TIND",
    "HH", "XW", "XQL", "HC", "SZ",
]
NUM_CLASSES = 12
INCLUSION_CLASSES = (1, 2, 3, 4, 5, 6)
STRIP_INCLUSION_CLASSES = (1, 2, 3, 5, 6)

A, B, C, D, TINBC, TIND, HH, XW, XQL, HC, SZ = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
)

STRIP_HEAD_TO_UNIFIED = np.asarray([0, 1, 2, 3, 5, 6, 7, 8, 9], dtype=np.uint8)
POINT_HEAD_TO_UNIFIED = np.asarray([0, 4, 10, 11], dtype=np.uint8)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def f1_score(p, r):
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def isin_np(arr, values):
    return np.isin(arr, np.asarray(values))


def find_mask_path(mask_dir, image_name):
    stem = Path(image_name).stem
    for p in (Path(mask_dir) / f"{stem}.png", Path(mask_dir) / image_name):
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"找不到 {image_name} 对应 mask")


def validate_mask(mask, mask_path):
    uniq = np.unique(mask)
    bad = uniq[(uniq < 0) | (uniq >= NUM_CLASSES)]
    if len(bad):
        raise ValueError(f"mask 存在非法类别值 {bad.tolist()}：{mask_path}")


def build_model_cfg():
    return SimpleNamespace(
        backbone_name=CONFIG["backbone_name"],
        freeze_backbone=CONFIG["freeze_backbone"],
        encoder_layers=CONFIG["encoder_layers"],
        feat_dim=CONFIG["feat_dim"],
        fusion_dim=CONFIG["fusion_dim"],
        decoder_dim=CONFIG["decoder_dim"],
    )


def load_model(checkpoint_path, device):
    from inclusion_v2.models import InclusionDualExpertNet

    model = InclusionDualExpertNet(build_model_cfg()).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        epoch = ckpt.get("epoch")
    else:
        state = ckpt
        epoch = None
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, epoch


def preprocess_image(image_rgb, device):
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    return x.to(device, non_blocking=True)


@torch.no_grad()
def infer_one(model, image_rgb, device):
    from inclusion_v2.utils.output_fusion import fuse_outputs

    x = preprocess_image(image_rgb, device)
    amp_enabled = CONFIG["use_amp"] and device.type == "cuda"

    if amp_enabled:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(x)
            fused = fuse_outputs(
                outputs["gate"], outputs["strip"], outputs["point"],
                alpha=CONFIG["fusion_alpha"],
            )
    else:
        outputs = model(x)
        fused = fuse_outputs(
            outputs["gate"], outputs["strip"], outputs["point"],
            alpha=CONFIG["fusion_alpha"],
        )

    confidence, pred = torch.max(fused, dim=1)
    if CONFIG["confidence_threshold"] > 0:
        pred = pred.masked_fill(confidence < CONFIG["confidence_threshold"], 0)

    gate_pred = torch.argmax(outputs["gate"], dim=1)
    strip_local = torch.argmax(outputs["strip"], dim=1)
    point_local = torch.argmax(outputs["point"], dim=1)

    final_pred = pred[0].cpu().numpy().astype(np.uint8)
    gate_np = gate_pred[0].cpu().numpy().astype(np.uint8)
    strip_local_np = strip_local[0].cpu().numpy().astype(np.uint8)
    point_local_np = point_local[0].cpu().numpy().astype(np.uint8)

    strip_uni = STRIP_HEAD_TO_UNIFIED[strip_local_np]
    point_uni = POINT_HEAD_TO_UNIFIED[point_local_np]

    return final_pred, gate_np, strip_uni, point_uni


def update_confusion(cm, gt, pred):
    idx = gt.reshape(-1).astype(np.int64) * NUM_CLASSES + pred.reshape(-1).astype(np.int64)
    cm += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def pixel_metrics_from_cm(cm):
    rows = []
    for c in range(NUM_CLASSES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        rows.append({
            "class_id": c,
            "class_name": CLASS_NAMES[c],
            "precision": p,
            "recall": r,
            "f1": f1_score(p, r),
            "tp_px": tp,
            "fp_px": fp,
            "fn_px": fn,
        })
    return rows


def binary_inclusion_counts(gt, pred):
    gt_inc = isin_np(gt, INCLUSION_CLASSES)
    pred_inc = isin_np(pred, INCLUSION_CLASSES)
    tp = int(np.logical_and(gt_inc, pred_inc).sum())
    fp = int(np.logical_and(~gt_inc, pred_inc).sum())
    fn = int(np.logical_and(gt_inc, ~pred_inc).sum())
    return tp, fp, fn


def connected_components(binary):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        np.asarray(binary, dtype=np.uint8), connectivity=8
    )
    comps = []
    for cid in range(1, n):
        x, y, w, h, area = stats[cid].tolist()
        cx, cy = centroids[cid].tolist()
        comps.append({
            "component_id": cid,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area), "cx": float(cx), "cy": float(cy),
        })
    return labels, comps


def gt_components_for_class(gt, cls_id):
    return connected_components(gt == cls_id)


def equivalent_diameter_px(area_px):
    return 2.0 * math.sqrt(float(area_px) / math.pi)


def classify_d_size(d_um):
    t1, t2 = CONFIG["d_size_bins_um"]
    if d_um <= t1:
        return f"small_<={t1:g}um"
    if d_um <= t2:
        return f"medium_{t1:g}-{t2:g}um"
    return f"large_>{t2:g}um"


def evaluate_tind_objects(image_name, gt, gate_pred, strip_pred_uni, final_pred):
    rows = []
    gt_labels, comps = gt_components_for_class(gt, TIND)
    thr = CONFIG["object_coverage_threshold"]

    for comp in comps:
        region = gt_labels == comp["component_id"]
        area = comp["area"]
        gate_cov = safe_div(np.logical_and(region, gate_pred == 1).sum(), area)
        expert_cov = safe_div(np.logical_and(region, strip_pred_uni == TIND).sum(), area)
        final_cov = safe_div(np.logical_and(region, final_pred == TIND).sum(), area)
        dpx = equivalent_diameter_px(area)
        rows.append({
            "image": image_name,
            "component_id": comp["component_id"],
            "area_px": area,
            "eq_diameter_px": dpx,
            "eq_diameter_um": dpx * CONFIG["um_per_pixel"],
            "gate_strip_coverage": gate_cov,
            "strip_expert_tind_coverage": expert_cov,
            "final_tind_coverage": final_cov,
            "gate_strip_detected": int(gate_cov >= thr),
            "strip_expert_tind_detected": int(expert_cov >= thr),
            "final_tind_detected": int(final_cov >= thr),
        })
    return rows


def evaluate_d_objects(image_name, gt, final_pred):
    rows = []
    gt_labels, comps = gt_components_for_class(gt, D)
    thr = CONFIG["object_coverage_threshold"]
    for comp in comps:
        region = gt_labels == comp["component_id"]
        area = comp["area"]
        dpx = equivalent_diameter_px(area)
        dum = dpx * CONFIG["um_per_pixel"]
        cov = safe_div(np.logical_and(region, final_pred == D).sum(), area)
        rows.append({
            "image": image_name,
            "component_id": comp["component_id"],
            "area_px": area,
            "eq_diameter_px": dpx,
            "eq_diameter_um": dum,
            "size_group": classify_d_size(dum),
            "d_class_coverage": cov,
            "detected": int(cov >= thr),
        })
    return rows


def evaluate_inclusion_objects(image_name, gt, final_pred):
    rows = []
    pred_inc_labels, pred_inc_comps = connected_components(isin_np(final_pred, INCLUSION_CLASSES))
    pred_area_map = {c["component_id"]: c["area"] for c in pred_inc_comps}

    for cls_id in INCLUSION_CLASSES:
        gt_labels, comps = gt_components_for_class(gt, cls_id)
        for comp in comps:
            region = gt_labels == comp["component_id"]
            gt_area = comp["area"]
            pred_inside = final_pred[region]
            pred_inc_inside = pred_inside[np.isin(pred_inside, INCLUSION_CLASSES)]

            inclusion_cov = safe_div(pred_inc_inside.size, gt_area)
            correct_cov = safe_div((pred_inside == cls_id).sum(), gt_area)

            if pred_inc_inside.size:
                counts = np.bincount(pred_inc_inside.astype(np.int64), minlength=NUM_CLASSES)
                inc_counts = {c: int(counts[c]) for c in INCLUSION_CLASSES if counts[c] > 0}
                dominant_cls = max(inc_counts, key=inc_counts.get)
                dominant_ratio = safe_div(inc_counts[dominant_cls], pred_inc_inside.size)
                correct_ratio = safe_div(inc_counts.get(cls_id, 0), pred_inc_inside.size)
                fragments = [
                    c for c, n in inc_counts.items()
                    if safe_div(n, pred_inc_inside.size) >= CONFIG["fragment_min_ratio"]
                ]
            else:
                dominant_cls = 0
                dominant_ratio = 0.0
                correct_ratio = 0.0
                fragments = []

            pred_ids = pred_inc_labels[region]
            pred_ids = pred_ids[pred_ids > 0]
            purity = 0.0
            matched_id = 0
            matched_inter = 0
            matched_area = 0
            if pred_ids.size:
                ids, nums = np.unique(pred_ids, return_counts=True)
                k = int(np.argmax(nums))
                matched_id = int(ids[k])
                matched_inter = int(nums[k])
                matched_area = int(pred_area_map.get(matched_id, 0))
                purity = safe_div(matched_inter, matched_area)

            rows.append({
                "image": image_name,
                "gt_class_id": cls_id,
                "gt_class_name": CLASS_NAMES[cls_id],
                "gt_component_id": comp["component_id"],
                "gt_area_px": gt_area,
                "inclusion_coverage": inclusion_cov,
                "correct_class_coverage": correct_cov,
                "dominant_pred_class_id": dominant_cls,
                "dominant_pred_class_name": CLASS_NAMES[dominant_cls],
                "dominant_ratio": dominant_ratio,
                "correct_class_ratio": correct_ratio,
                "fragment_class_count": len(fragments),
                "fragment_classes": "|".join(CLASS_NAMES[c] for c in fragments),
                "matched_pred_component_id": matched_id,
                "matched_intersection_px": matched_inter,
                "matched_pred_area_px": matched_area,
                "purity": purity,
            })
    return rows


def extract_apparent_fp_components(image_name, gt, final_pred):
    definitions = {
        "bg_to_strip_inclusion": (gt == 0) & isin_np(final_pred, STRIP_INCLUSION_CLASSES),
        "bg_to_D": (gt == 0) & (final_pred == D),
    }
    rows = []
    for fp_type, binary in definitions.items():
        labels, comps = connected_components(binary)
        for comp in comps:
            if comp["area"] < CONFIG["fp_min_area_px"]:
                continue
            region = labels == comp["component_id"]
            pred_vals = final_pred[region]
            counts = np.bincount(pred_vals.astype(np.int64), minlength=NUM_CLASSES)
            maj = int(np.argmax(counts))
            rows.append({
                "image": image_name,
                "fp_type": fp_type,
                "component_id": comp["component_id"],
                "area_px": comp["area"],
                "x": comp["x"], "y": comp["y"], "w": comp["w"], "h": comp["h"],
                "majority_pred_class_id": maj,
                "majority_pred_class_name": CLASS_NAMES[maj],
            })
    return rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        if not rows:
            path.write_text("", encoding="utf-8-sig")
            return
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_confusion(cm, out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["GT\\Pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name] + cm[i].tolist())

    denom = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm.astype(np.float64), denom, out=np.zeros_like(cm, dtype=np.float64), where=denom > 0)
    with open(out_dir / "confusion_matrix_normalized.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["GT\\Pred"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            w.writerow([name] + [f"{v:.6f}" for v in norm[i]])

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(norm)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Pred")
        ax.set_ylabel("GT")
        ax.set_title("Normalized Confusion Matrix")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if norm[i, j] >= 0.01:
                    ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=180)
        plt.close(fig)
    except Exception as e:
        print(f"[Warning] confusion matrix png 保存失败: {e}")


def save_fp_crops(fp_rows, image_dir, out_dir):
    if not CONFIG["save_fp_crops"]:
        return
    by_type = defaultdict(list)
    for r in fp_rows:
        by_type[r["fp_type"]].append(r)

    pad = CONFIG["fp_crop_padding"]
    for fp_type, rows in by_type.items():
        save_dir = Path(out_dir) / "fp_crops" / fp_type
        ensure_dir(save_dir)
        rows = sorted(rows, key=lambda r: r["area_px"], reverse=True)[:CONFIG["max_fp_crops_each_type"]]
        for rank, r in enumerate(rows, 1):
            img = cv2.imread(str(Path(image_dir) / r["image"]))
            if img is None:
                continue
            H, W = img.shape[:2]
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(W, x+w+pad), min(H, y+h+pad)
            crop = img[y1:y2, x1:x2].copy()
            bx1, by1 = x-x1, y-y1
            bx2, by2 = bx1+w, by1+h
            cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0,0,255), 1)
            txt = f"{r['majority_pred_class_name']} area={r['area_px']}"
            cv2.putText(crop, txt, (max(0,bx1), max(14,by1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
            name = f"{rank:03d}_{Path(r['image']).stem}_{r['majority_pred_class_name']}_a{r['area_px']}.jpg"
            cv2.imwrite(str(save_dir / name), crop)


def summarize_tind(rows):
    n = len(rows)
    if not n:
        return {"count":0, "gate_strip_object_recall":0.0, "strip_expert_tind_object_recall":0.0, "final_tind_object_recall":0.0}
    return {
        "count": n,
        "gate_strip_object_recall": safe_div(sum(r["gate_strip_detected"] for r in rows), n),
        "strip_expert_tind_object_recall": safe_div(sum(r["strip_expert_tind_detected"] for r in rows), n),
        "final_tind_object_recall": safe_div(sum(r["final_tind_detected"] for r in rows), n),
        "mean_gate_strip_coverage": float(np.mean([r["gate_strip_coverage"] for r in rows])),
        "mean_strip_expert_tind_coverage": float(np.mean([r["strip_expert_tind_coverage"] for r in rows])),
        "mean_final_tind_coverage": float(np.mean([r["final_tind_coverage"] for r in rows])),
    }


def summarize_d(rows):
    t1, t2 = CONFIG["d_size_bins_um"]
    names = [f"small_<={t1:g}um", f"medium_{t1:g}-{t2:g}um", f"large_>{t2:g}um"]
    result = {}
    for name in names:
        rs = [r for r in rows if r["size_group"] == name]
        result[name] = {
            "count": len(rs),
            "detected": int(sum(r["detected"] for r in rs)),
            "object_recall": safe_div(sum(r["detected"] for r in rs), len(rs)),
            "mean_class_coverage": float(np.mean([r["d_class_coverage"] for r in rs])) if rs else 0.0,
        }
    return result


def summarize_objects(rows):
    result = {}
    for cls_id in INCLUSION_CLASSES:
        rs = [r for r in rows if r["gt_class_id"] == cls_id]
        if not rs:
            result[CLASS_NAMES[cls_id]] = {"count":0}
            continue
        dominant = np.asarray([r["dominant_ratio"] for r in rs])
        frag = np.asarray([r["fragment_class_count"] for r in rs])
        result[CLASS_NAMES[cls_id]] = {
            "count": len(rs),
            "mean_coverage": float(np.mean([r["inclusion_coverage"] for r in rs])),
            "mean_purity": float(np.mean([r["purity"] for r in rs])),
            "mean_correct_class_coverage": float(np.mean([r["correct_class_coverage"] for r in rs])),
            "mean_dominant_ratio": float(np.mean(dominant)),
            "mean_correct_class_ratio": float(np.mean([r["correct_class_ratio"] for r in rs])),
            "pct_consistency_lt_0_8": float(np.mean(dominant < 0.8)),
            "pct_multi_class_fragment": float(np.mean(frag >= 2)),
        }
    return result


def confusion_rate(cm, gt_cls, pred_classes):
    denom = int(cm[gt_cls, :].sum())
    num = int(sum(cm[gt_cls, c] for c in pred_classes))
    return safe_div(num, denom), num, denom


def evaluate_checkpoint(tag, checkpoint_path):
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() or CONFIG["device"] == "cpu" else "cpu")
    out_dir = Path(CONFIG["output_dir"]) / tag
    ensure_dir(out_dir)

    print("\n" + "="*80)
    print(f"Evaluating {tag}: {checkpoint_path}")
    print("="*80)

    model, ckpt_epoch = load_model(checkpoint_path, device)

    image_dir = Path(CONFIG["val_image_dir"])
    mask_dir = Path(CONFIG["val_mask_dir"])
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not names:
        raise RuntimeError(f"验证集为空: {image_dir}")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    inc_tp = inc_fp = inc_fn = 0
    tind_total_px = tind_gate_px = tind_expert_px = tind_final_px = 0
    tind_rows, d_rows, object_rows, fp_rows, image_rows = [], [], [], [], []

    for i, name in enumerate(names, 1):
        img_path = image_dir / name
        mask_path = find_mask_path(mask_dir, name)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            raise RuntimeError(f"读取失败: {name}")
        if bgr.shape[:2] != gt.shape[:2]:
            raise ValueError(f"image/mask 尺寸不一致: {name}")
        if CONFIG["expected_hw"] is not None and tuple(gt.shape[:2]) != tuple(CONFIG["expected_hw"]):
            raise ValueError(f"尺寸不是 {CONFIG['expected_hw']}: {name} -> {gt.shape[:2]}")
        validate_mask(gt, mask_path)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        final_pred, gate_pred, strip_pred_uni, point_pred_uni = infer_one(model, rgb, device)

        update_confusion(cm, gt, final_pred)
        tp, fp, fn = binary_inclusion_counts(gt, final_pred)
        inc_tp += tp; inc_fp += fp; inc_fn += fn

        tind_region = gt == TIND
        tind_total_px += int(tind_region.sum())
        tind_gate_px += int(np.logical_and(tind_region, gate_pred == 1).sum())
        tind_expert_px += int(np.logical_and(tind_region, strip_pred_uni == TIND).sum())
        tind_final_px += int(np.logical_and(tind_region, final_pred == TIND).sum())

        tind_rows.extend(evaluate_tind_objects(name, gt, gate_pred, strip_pred_uni, final_pred))
        d_rows.extend(evaluate_d_objects(name, gt, final_pred))
        this_obj = evaluate_inclusion_objects(name, gt, final_pred)
        object_rows.extend(this_obj)
        this_fp = extract_apparent_fp_components(name, gt, final_pred)
        fp_rows.extend(this_fp)

        p = safe_div(tp, tp+fp); r = safe_div(tp, tp+fn)
        image_rows.append({
            "image": name,
            "inclusion_precision": p,
            "inclusion_recall": r,
            "inclusion_f1": f1_score(p,r),
            "gt_inclusion_objects": len(this_obj),
            "apparent_bg_strip_fp_components": sum(x["fp_type"]=="bg_to_strip_inclusion" for x in this_fp),
            "apparent_bg_d_fp_components": sum(x["fp_type"]=="bg_to_D" for x in this_fp),
        })

        if i % 10 == 0 or i == len(names):
            print(f"[{tag}] {i}/{len(names)} {name}")

    pixel_rows = pixel_metrics_from_cm(cm)
    inc_p = safe_div(inc_tp, inc_tp+inc_fp)
    inc_r = safe_div(inc_tp, inc_tp+inc_fn)
    inc_f1 = f1_score(inc_p, inc_r)

    tind_summary = summarize_tind(tind_rows)
    d_summary = summarize_d(d_rows)
    obj_summary = summarize_objects(object_rows)

    hh_ac_rate, hh_ac_px, hh_total = confusion_rate(cm, HH, (A,C))
    hc_d_rate, hc_d_px, hc_total = confusion_rate(cm, HC, (D,))
    sz_d_rate, sz_d_px, sz_total = confusion_rate(cm, SZ, (D,))
    bg_strip_rate, bg_strip_px, bg_total = confusion_rate(cm, 0, STRIP_INCLUSION_CLASSES)
    bg_d_rate, bg_d_px, _ = confusion_rate(cm, 0, (D,))

    fp_summary = {}
    for t in ("bg_to_strip_inclusion", "bg_to_D"):
        rs = [r for r in fp_rows if r["fp_type"] == t]
        fp_summary[t] = {
            "component_count": len(rs),
            "total_area_px": int(sum(r["area_px"] for r in rs)),
            "mean_area_px": float(np.mean([r["area_px"] for r in rs])) if rs else 0.0,
            "max_area_px": int(max((r["area_px"] for r in rs), default=0)),
        }

    summary = {
        "checkpoint_tag": tag,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ckpt_epoch,
        "num_images": len(names),
        "pixel_inclusion": {
            "precision": inc_p, "recall": inc_r, "f1": inc_f1,
            "tp_px": int(inc_tp), "fp_px": int(inc_fp), "fn_px": int(inc_fn),
        },
        "tind_pixel_chain": {
            "gt_tind_pixels": int(tind_total_px),
            "gate_strip_pixel_recall": safe_div(tind_gate_px, tind_total_px),
            "strip_expert_tind_pixel_recall": safe_div(tind_expert_px, tind_total_px),
            "final_tind_pixel_recall": safe_div(tind_final_px, tind_total_px),
        },
        "tind_object_chain": tind_summary,
        "d_object_recall_by_size": d_summary,
        "object_metrics_by_class": obj_summary,
        "directional_errors": {
            "HH_to_A_or_C": {"rate":hh_ac_rate, "error_px":hh_ac_px, "gt_px":hh_total},
            "HC_to_D": {"rate":hc_d_rate, "error_px":hc_d_px, "gt_px":hc_total},
            "SZ_to_D": {"rate":sz_d_rate, "error_px":sz_d_px, "gt_px":sz_total},
            "BG_to_strip_inclusion_apparent": {"rate":bg_strip_rate, "error_px":bg_strip_px, "gt_bg_px":bg_total},
            "BG_to_D_apparent": {"rate":bg_d_rate, "error_px":bg_d_px, "gt_bg_px":bg_total},
        },
        "apparent_fp_components": fp_summary,
        "notes": [
            "Object 指标基于 semantic mask connected components。",
            "背景存在漏标时，BG->inclusion 是 apparent FP 上界，不等于真实 FP。",
        ],
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(out_dir / "class_pixel_metrics.csv", pixel_rows)
    write_csv(out_dir / "tind_objects.csv", tind_rows)
    write_csv(out_dir / "d_objects.csv", d_rows)
    write_csv(out_dir / "object_metrics.csv", object_rows)
    write_csv(out_dir / "image_metrics.csv", image_rows)
    write_csv(out_dir / "apparent_fp_components.csv", fp_rows)
    save_confusion(cm, out_dir)
    save_fp_crops(fp_rows, image_dir, out_dir)

    print(f"Inclusion P/R/F1 = {inc_p:.4f}/{inc_r:.4f}/{inc_f1:.4f}")
    print("TIND Object Recall: ", tind_summary)
    print("D size recall: ", d_summary)
    print(f"HH->A/C={hh_ac_rate:.6f}, HC->D={hc_d_rate:.6f}, SZ->D={sz_d_rate:.6f}")
    print(f"BG->Strip apparent={bg_strip_rate:.8f}, BG->D apparent={bg_d_rate:.8f}")

    return summary, pixel_rows


def flatten_comparison(summary, pixel_rows):
    per_class = {r["class_name"]: r for r in pixel_rows}
    tind = summary["tind_object_chain"]
    dsize = summary["d_object_recall_by_size"]
    obj = summary["object_metrics_by_class"]
    t1, t2 = CONFIG["d_size_bins_um"]
    small = f"small_<={t1:g}um"
    medium = f"medium_{t1:g}-{t2:g}um"
    large = f"large_>{t2:g}um"

    def objv(cls, key):
        return obj.get(cls, {}).get(key, 0.0)

    return {
        "checkpoint": summary["checkpoint_tag"],
        "epoch": summary["checkpoint_epoch"],
        "inclusion_precision": summary["pixel_inclusion"]["precision"],
        "inclusion_recall": summary["pixel_inclusion"]["recall"],
        "inclusion_f1": summary["pixel_inclusion"]["f1"],
        "TIND_pixel_recall": per_class["TIND"]["recall"],
        "TIND_gate_strip_object_recall": tind.get("gate_strip_object_recall",0.0),
        "TIND_expert_object_recall": tind.get("strip_expert_tind_object_recall",0.0),
        "TIND_final_object_recall": tind.get("final_tind_object_recall",0.0),
        "D_pixel_recall": per_class["D"]["recall"],
        "D_small_object_recall": dsize[small]["object_recall"],
        "D_medium_object_recall": dsize[medium]["object_recall"],
        "D_large_object_recall": dsize[large]["object_recall"],
        "A_mean_consistency": objv("A","mean_dominant_ratio"),
        "C_mean_consistency": objv("C","mean_dominant_ratio"),
        "A_pct_consistency_lt_0_8": objv("A","pct_consistency_lt_0_8"),
        "C_pct_consistency_lt_0_8": objv("C","pct_consistency_lt_0_8"),
        "A_mean_coverage": objv("A","mean_coverage"),
        "A_mean_purity": objv("A","mean_purity"),
        "B_mean_coverage": objv("B","mean_coverage"),
        "B_mean_purity": objv("B","mean_purity"),
        "C_mean_coverage": objv("C","mean_coverage"),
        "C_mean_purity": objv("C","mean_purity"),
        "HH_to_AC_rate": summary["directional_errors"]["HH_to_A_or_C"]["rate"],
        "HC_to_D_rate": summary["directional_errors"]["HC_to_D"]["rate"],
        "SZ_to_D_rate": summary["directional_errors"]["SZ_to_D"]["rate"],
        "BG_to_strip_apparent_rate": summary["directional_errors"]["BG_to_strip_inclusion_apparent"]["rate"],
        "BG_to_D_apparent_rate": summary["directional_errors"]["BG_to_D_apparent"]["rate"],
        "BG_to_strip_apparent_components": summary["apparent_fp_components"]["bg_to_strip_inclusion"]["component_count"],
        "BG_to_D_apparent_components": summary["apparent_fp_components"]["bg_to_D"]["component_count"],
    }


def main():
    ensure_dir(CONFIG["output_dir"])
    comparison = []
    for tag, ckpt in CONFIG["checkpoints"].items():
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint 不存在: {ckpt}")
        summary, pixel_rows = evaluate_checkpoint(tag, ckpt)
        comparison.append(flatten_comparison(summary, pixel_rows))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(Path(CONFIG["output_dir"]) / "comparison.csv", comparison)
    print("\n全部完成，重点先看：")
    print(Path(CONFIG["output_dir"]) / "comparison.csv")


if __name__ == "__main__":
    main()