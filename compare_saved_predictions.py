# -*- coding: utf-8 -*-
"""
compare_saved_predictions_common_unified12.py

用途
----
对 export_model_predictions_common_unified12.py 导出的 A/B/C/D 四组冻结预测做统一评价。
本脚本完全不加载模型/checkpoint，只读取：

    saved_predictions/<profile>/common_unified12_val/predictions/*.png

四个模型必须来自同一套 D/unified12 公共验证集。脚本会读取每个模型导出的 metadata.json，
自动检查 image_dir / mask_dir / gt_scheme 是否一致，避免把不同验证集的结果误放到一起比较。

当前四组模型：
    A_simple7_correct : Simple-7 + correct token
    B_simple7_legacy  : Simple-7 + legacy token
    C_legacy_dual     : 旧 Strip-9 + Point-4 双模型，legacy token，最终融合为 unified12
    D_v2_legacy       : inclusion_v2 Gate + Experts，legacy token，unified12

主评价统一折叠为 canonical 7-class：
    0 Background / non-inclusion
    1 A
    2 B
    3 C
    4 D
    5 TINB/TINC
    6 TIND

unified12 GT / C / D 中的 7..11（HH/XW/XQL/HC/SZ）在主任务指标中折叠为 Background。
但是原始 GT 7..11 会继续用于统计抗干扰能力，例如 HH->inclusion、HC->D、SZ->D。

比较解释：
    A vs B：token 修正影响（最干净的单变量比较）
    B vs C：Simple-7 legacy 与旧双模型系统差异；同时变化了结构和标签体系，不能做单因素归因
    D vs C：共享 DINO 的 V2 与旧两套独立 DINO 双模型的系统差异；都属于 legacy token
    A vs D：当前 Simple-7 correct 与 V2 legacy 的最终系统比较
    A vs C：当前 Simple-7 correct 与旧双模型的最终系统比较

重点输出：
    comparison_output_common_unified12/
      key_metrics_wide.csv             # 最建议先看：关键指标横向表
      comparison_wide.csv              # 全指标横向表
      pair_effects_key.csv              # 关键指标的 pair 差值；beneficial_delta > 0 表示 target 更好
      pair_effects.csv                  # 全指标 pair 差值
      pair_image_win_summary.csv        # 逐图胜/平/负
      per_image_all_models.csv
      panels/                           # Original / GT / A / B / C / D 六联图
      <model_tag>/...                   # 每个模型的详细指标

说明：
- apparent FP 仍可能包含 GT 漏标，因此称“表观 FP”。
- object 指标基于 semantic connected components，不是实例标注。
- object recall 默认：GT 连通域内正确类别覆盖 >= 30% 视为检出。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # export_model_predictions_common_unified12.py 的输出根目录
    "prediction_root": r"./saved_predictions",

    # 四个 profile 都使用这个公共 eval-set 名称
    "eval_set_name": "common_unified12_val",

    # 用 D 的 metadata 作为公共 image_dir / mask_dir 的基准来源。
    # 不需要在本脚本重复填写验证集路径。
    "reference_model_for_gt": "D_v2_legacy",
    "strict_common_dataset_check": True,

    "models": {
        "A_simple7_correct": r"A_simple7_correct",
        "B_simple7_legacy": r"B_simple7_legacy",
        "C_legacy_dual": r"C_legacy_dual",
        "D_v2_legacy": r"D_v2_legacy",
    },

    # target - reference。beneficial_delta_positive_is_better > 0 表示 target 更好。
    "pairs": {
        "token_fix_A_vs_B": {
            "target": "A_simple7_correct",
            "reference": "B_simple7_legacy",
            "meaning": "Simple-7 中 correct token 相对 legacy token 的影响（最干净）",
        },
        "simple7_B_vs_legacy_dual_C": {
            "target": "B_simple7_legacy",
            "reference": "C_legacy_dual",
            "meaning": "同为 legacy token：Simple-7 单模型 vs 旧 Strip+Point 双模型；结构和标签体系同时变化，不能单因素归因",
        },
        "v2_D_vs_legacy_dual_C": {
            "target": "D_v2_legacy",
            "reference": "C_legacy_dual",
            "meaning": "同为 legacy token：V2 共享DINO双专家 vs 旧两套独立DINO双模型的系统差异",
        },
        "final_A_vs_D": {
            "target": "A_simple7_correct",
            "reference": "D_v2_legacy",
            "meaning": "当前 Simple-7 correct vs inclusion_v2 legacy 的最终系统级对比",
        },
        "final_A_vs_C": {
            "target": "A_simple7_correct",
            "reference": "C_legacy_dual",
            "meaning": "当前 Simple-7 correct vs 旧 Strip+Point 双模型的最终系统级对比",
        },
    },

    "output_dir": r"./comparison_output_common_unified12",
    "expected_hw": (1024, 1024),
    "um_per_pixel": 0.5448,
    "object_coverage_threshold": 0.30,
    "fragment_min_ratio": 0.05,
    "size_bins_um": (5.0, 10.0),
    "fp_min_area_px": 2,

    "make_panels": True,
    "panel_top_k_each_side": 20,
}

# 最适合先看的关键指标
KEY_METRICS = [
    "inclusion_precision", "inclusion_recall", "inclusion_f1",
    "macro_inclusion_f1", "mean_inclusion_iou",
    "A_f1", "B_f1", "C_f1", "D_f1", "TINBC_f1", "TIND_f1",
    "D_small_recall", "TIND_small_recall",
    "raw_BG_to_inclusion", "known_distractor_to_inclusion",
    "HH_to_any_inclusion", "XW_to_any_inclusion", "XQL_to_any_inclusion",
    "HC_to_D", "SZ_to_D",
    "pure_FP_ge64_count", "pure_FP_ge256_count",
    "pure_strip_FP_ge64_count", "pure_D_FP_ge64_count",
    "avg_inference_ms", "p95_inference_ms",
    "total_params_m", "checkpoint_total_size_mb", "peak_gpu_memory_mb",
]

CANONICAL_NAMES = ["bg", "A", "B", "C", "D", "TINB/TINC", "TIND"]
RAW12_NAMES = ["bg", "A", "B", "C", "D", "TINB/TINC", "TIND", "HH", "XW", "XQL", "HC", "SZ"]
KNOWN_DISTRACTORS = {7: "HH", 8: "XW", 9: "XQL", 10: "HC", 11: "SZ"}
INCLUSION_CLASSES = (1, 2, 3, 4, 5, 6)
STRIP_CLASSES = (1, 2, 3, 5, 6)
D_ID = 4
TIND_ID = 6
NUM_CANONICAL = 7
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 仅用于 panel 可视化；数值评价与颜色无关。
PALETTE_BGR = np.asarray([
    [0, 0, 0],       # bg
    [0, 0, 255],     # A
    [0, 165, 255],   # B
    [0, 255, 255],   # C
    [255, 0, 0],     # D
    [255, 0, 255],   # TINBC
    [0, 255, 0],     # TIND
], dtype=np.uint8)


# ============================================================
# 通用工具
# ============================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_div(a, b, zero=0.0):
    return float(a) / float(b) if b else zero


def f1_from_pr(p, r):
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        if not rows:
            path.write_text("", encoding="utf-8-sig")
            return
        keys = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"读取 mask 失败：{path}")
    return m


def validate_raw_labels(arr: np.ndarray, path: Path, max_id=11):
    u = np.unique(arr)
    bad = u[(u < 0) | (u > max_id)]
    if len(bad):
        raise ValueError(f"非法类别 {bad.tolist()}：{path}")


def canonicalize(arr: np.ndarray) -> np.ndarray:
    x = arr.copy()
    x[x >= 7] = 0
    return x.astype(np.uint8)


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # 兼容大写扩展名：扫描一次
    for p in image_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in IMAGE_EXTS:
            return p
    return None


def list_gt_masks(mask_dir: Path) -> List[Path]:
    masks = sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not masks:
        raise RuntimeError(f"没有找到 GT PNG：{mask_dir}")
    return masks


def load_metadata(model_dir: Path) -> dict:
    p = model_dir / "metadata.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def model_eval_dir(rel_dir: str) -> Path:
    return Path(CONFIG["prediction_root"]) / rel_dir / CONFIG["eval_set_name"]


def _norm_path_string(x) -> str:
    if x is None:
        return ""
    try:
        return str(Path(x).resolve()).replace("\\", "/").lower()
    except Exception:
        return str(x).replace("\\", "/").lower()


def resolve_common_dataset():
    """
    从导出 metadata 自动获得公共 image_dir/mask_dir，并检查 A/B/C/D 是否确实来自同一数据集。
    """
    metas = {}
    for tag, rel in CONFIG["models"].items():
        d = model_eval_dir(rel)
        m = load_metadata(d)
        if not m:
            raise FileNotFoundError(f"缺少 metadata.json：{d / 'metadata.json'}")
        metas[tag] = m

        if m.get("eval_set") != CONFIG["eval_set_name"]:
            raise ValueError(
                f"{tag} metadata eval_set={m.get('eval_set')}，"
                f"期望 {CONFIG['eval_set_name']}"
            )
        if m.get("gt_scheme") != "unified12":
            raise ValueError(f"{tag} gt_scheme={m.get('gt_scheme')}，主比较要求 unified12")

    ref_tag = CONFIG["reference_model_for_gt"]
    if ref_tag not in metas:
        raise KeyError(f"reference_model_for_gt={ref_tag} 不在 CONFIG['models'] 中")
    ref = metas[ref_tag]
    image_dir = Path(ref["image_dir"])
    mask_dir = Path(ref["mask_dir"])

    if CONFIG.get("strict_common_dataset_check", True):
        ref_img = _norm_path_string(ref.get("image_dir"))
        ref_mask = _norm_path_string(ref.get("mask_dir"))
        ref_n = int(ref.get("num_images", -1))
        for tag, m in metas.items():
            if _norm_path_string(m.get("image_dir")) != ref_img:
                raise RuntimeError(
                    f"公共验证集 image_dir 不一致：{tag}={m.get('image_dir')} vs {ref_tag}={ref.get('image_dir')}"
                )
            if _norm_path_string(m.get("mask_dir")) != ref_mask:
                raise RuntimeError(
                    f"公共验证集 mask_dir 不一致：{tag}={m.get('mask_dir')} vs {ref_tag}={ref.get('mask_dir')}"
                )
            if int(m.get("num_images", -1)) != ref_n:
                raise RuntimeError(
                    f"公共验证集图片数不一致：{tag}={m.get('num_images')} vs {ref_tag}={ref_n}"
                )

    if not image_dir.exists():
        raise FileNotFoundError(f"metadata 中的公共 image_dir 不存在：{image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"metadata 中的公共 mask_dir 不存在：{mask_dir}")

    return image_dir, mask_dir, metas


def connected_components(binary):
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
            "area": int(area), "cx": float(cx), "cy": float(cy),
        })
    return labels, comps


# ============================================================
# Pixel metrics
# ============================================================
def update_confusion(cm, gt7, pred7):
    idx = gt7.astype(np.int64) * NUM_CANONICAL + pred7.astype(np.int64)
    cm += np.bincount(idx.ravel(), minlength=NUM_CANONICAL ** 2).reshape(NUM_CANONICAL, NUM_CANONICAL)


def pixel_metrics_from_cm(cm):
    rows = []
    for c, name in enumerate(CANONICAL_NAMES):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f1 = f1_from_pr(p, r)
        iou = safe_div(tp, tp + fp + fn)
        rows.append({
            "class_id": c, "class_name": name,
            "tp_px": tp, "fp_px": fp, "fn_px": fn,
            "precision": p, "recall": r, "f1": f1, "iou": iou,
        })
    return rows


def binary_inclusion_counts(gt7, pred7):
    gt_inc = np.isin(gt7, INCLUSION_CLASSES)
    pred_inc = np.isin(pred7, INCLUSION_CLASSES)
    tp = int(np.logical_and(gt_inc, pred_inc).sum())
    fp = int(np.logical_and(~gt_inc, pred_inc).sum())
    fn = int(np.logical_and(gt_inc, ~pred_inc).sum())
    tn = int(np.logical_and(~gt_inc, ~pred_inc).sum())
    return tp, fp, fn, tn


def binary_metrics(tp, fp, fn, empty_is_one=False):
    if empty_is_one and tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return p, r, f1_from_pr(p, r)


# ============================================================
# Object metrics
# ============================================================
def equivalent_diameter_px(area_px):
    return 2.0 * math.sqrt(float(area_px) / math.pi)


def size_group(d_um):
    t1, t2 = CONFIG["size_bins_um"]
    if d_um <= t1:
        return f"small_<={t1:g}um"
    if d_um <= t2:
        return f"medium_{t1:g}-{t2:g}um"
    return f"large_>{t2:g}um"


def evaluate_objects(image_name, gt7, pred7):
    rows = []
    pred_inc_labels, pred_inc_comps = connected_components(np.isin(pred7, INCLUSION_CLASSES))
    pred_area_map = {c["component_id"]: c["area"] for c in pred_inc_comps}
    thr = float(CONFIG["object_coverage_threshold"])

    for cls_id in INCLUSION_CLASSES:
        gt_labels, comps = connected_components(gt7 == cls_id)
        for comp in comps:
            region = gt_labels == comp["component_id"]
            gt_area = comp["area"]
            pred_inside = pred7[region]
            pred_inc_inside = pred_inside[np.isin(pred_inside, INCLUSION_CLASSES)]

            inclusion_cov = safe_div(pred_inc_inside.size, gt_area)
            correct_cov = safe_div(int((pred_inside == cls_id).sum()), gt_area)

            if pred_inc_inside.size:
                counts = np.bincount(pred_inc_inside.astype(np.int64), minlength=NUM_CANONICAL)
                inc_counts = {c: int(counts[c]) for c in INCLUSION_CLASSES if counts[c] > 0}
                dominant_cls = max(inc_counts, key=inc_counts.get)
                dominant_ratio = safe_div(inc_counts[dominant_cls], pred_inc_inside.size)
                fragments = [
                    c for c, n in inc_counts.items()
                    if safe_div(n, pred_inc_inside.size) >= CONFIG["fragment_min_ratio"]
                ]
            else:
                dominant_cls = 0
                dominant_ratio = 0.0
                fragments = []

            pred_ids = pred_inc_labels[region]
            pred_ids = pred_ids[pred_ids > 0]
            purity = 0.0
            matched_area = 0
            matched_inter = 0
            if pred_ids.size:
                ids, nums = np.unique(pred_ids, return_counts=True)
                k = int(np.argmax(nums))
                matched_id = int(ids[k])
                matched_inter = int(nums[k])
                matched_area = int(pred_area_map.get(matched_id, 0))
                purity = safe_div(matched_inter, matched_area)

            dpx = equivalent_diameter_px(gt_area)
            dum = dpx * float(CONFIG["um_per_pixel"])
            rows.append({
                "image": image_name,
                "gt_class_id": cls_id,
                "gt_class_name": CANONICAL_NAMES[cls_id],
                "gt_component_id": comp["component_id"],
                "gt_area_px": gt_area,
                "eq_diameter_px": dpx,
                "eq_diameter_um": dum,
                "size_group": size_group(dum),
                "inclusion_coverage": inclusion_cov,
                "correct_class_coverage": correct_cov,
                "detected_correct_class": int(correct_cov >= thr),
                "dominant_pred_class_id": dominant_cls,
                "dominant_pred_class_name": CANONICAL_NAMES[dominant_cls],
                "dominant_ratio": dominant_ratio,
                "fragment_class_count": len(fragments),
                "fragment_classes": "|".join(CANONICAL_NAMES[c] for c in fragments),
                "matched_intersection_px": matched_inter,
                "matched_pred_area_px": matched_area,
                "purity": purity,
            })
    return rows


def summarize_objects(rows):
    out = {}
    for cls_id in INCLUSION_CLASSES:
        name = CANONICAL_NAMES[cls_id]
        rs = [r for r in rows if r["gt_class_id"] == cls_id]
        if not rs:
            out[name] = {
                "n_objects": 0, "object_recall": 0.0, "mean_coverage": 0.0,
                "mean_purity": 0.0, "mean_dominant_ratio": 0.0,
                "pct_multi_class_fragment": 0.0,
            }
            continue
        out[name] = {
            "n_objects": len(rs),
            "object_recall": float(np.mean([r["detected_correct_class"] for r in rs])),
            "mean_coverage": float(np.mean([r["correct_class_coverage"] for r in rs])),
            "mean_purity": float(np.mean([r["purity"] for r in rs])),
            "mean_dominant_ratio": float(np.mean([r["dominant_ratio"] for r in rs])),
            "pct_multi_class_fragment": float(np.mean([r["fragment_class_count"] >= 2 for r in rs])),
        }
    return out


def summarize_size(rows, cls_id):
    t1, t2 = CONFIG["size_bins_um"]
    groups = [f"small_<={t1:g}um", f"medium_{t1:g}-{t2:g}um", f"large_>{t2:g}um"]
    cls_rows = [r for r in rows if r["gt_class_id"] == cls_id]
    out = {}
    for g in groups:
        rs = [r for r in cls_rows if r["size_group"] == g]
        out[g] = {
            "n_objects": len(rs),
            "object_recall": float(np.mean([r["detected_correct_class"] for r in rs])) if rs else 0.0,
            "mean_correct_coverage": float(np.mean([r["correct_class_coverage"] for r in rs])) if rs else 0.0,
        }
    return out


# ============================================================
# 已知干扰物泄漏
# ============================================================
def distractor_pixel_stats(raw_gt, pred7):
    rows = []
    pred_inc = np.isin(pred7, INCLUSION_CLASSES)
    pred_strip = np.isin(pred7, STRIP_CLASSES)
    for raw_id, name in KNOWN_DISTRACTORS.items():
        region = raw_gt == raw_id
        total = int(region.sum())
        any_inc = int(np.logical_and(region, pred_inc).sum())
        strip_inc = int(np.logical_and(region, pred_strip).sum())
        d_px = int(np.logical_and(region, pred7 == D_ID).sum())
        rows.append({
            "gt_distractor_id": raw_id,
            "gt_distractor_name": name,
            "gt_pixels": total,
            "to_any_inclusion_px": any_inc,
            "to_any_inclusion_rate": safe_div(any_inc, total, zero=float("nan")),
            "to_strip_inclusion_px": strip_inc,
            "to_strip_inclusion_rate": safe_div(strip_inc, total, zero=float("nan")),
            "to_D_px": d_px,
            "to_D_rate": safe_div(d_px, total, zero=float("nan")),
        })
    return rows


def merge_distractor_rows(rows):
    agg = {i: defaultdict(int) for i in KNOWN_DISTRACTORS}
    for r in rows:
        i = int(r["gt_distractor_id"])
        for k in ("gt_pixels", "to_any_inclusion_px", "to_strip_inclusion_px", "to_D_px"):
            agg[i][k] += int(r[k])
    out = []
    for i, name in KNOWN_DISTRACTORS.items():
        d = agg[i]
        total = d["gt_pixels"]
        out.append({
            "gt_distractor_id": i,
            "gt_distractor_name": name,
            "gt_pixels": total,
            "to_any_inclusion_px": d["to_any_inclusion_px"],
            "to_any_inclusion_rate": safe_div(d["to_any_inclusion_px"], total, zero=float("nan")),
            "to_strip_inclusion_px": d["to_strip_inclusion_px"],
            "to_strip_inclusion_rate": safe_div(d["to_strip_inclusion_px"], total, zero=float("nan")),
            "to_D_px": d["to_D_px"],
            "to_D_rate": safe_div(d["to_D_px"], total, zero=float("nan")),
        })
    return out


# ============================================================
# Pure apparent FP：完整预测连通域与任何 GT inclusion 零重叠
# ============================================================
def component_distractor_overlap(raw_gt, region):
    total = int(region.sum())
    best_id, best_n = 0, 0
    for i in KNOWN_DISTRACTORS:
        n = int(np.logical_and(region, raw_gt == i).sum())
        if n > best_n:
            best_id, best_n = i, n
    if best_id == 0 or total == 0:
        return 0, "", 0.0
    return best_id, KNOWN_DISTRACTORS[best_id], safe_div(best_n, total)


def extract_fp_components(image_name, raw_gt, gt7, pred7):
    rows = []
    gt_inc = np.isin(gt7, INCLUSION_CLASSES)
    for cls_id in INCLUSION_CLASSES:
        labels, comps = connected_components(pred7 == cls_id)
        for comp in comps:
            if comp["area"] < int(CONFIG["fp_min_area_px"]):
                continue
            region = labels == comp["component_id"]
            if int(np.logical_and(region, gt_inc).sum()) > 0:
                continue

            area = comp["area"]
            raw_bg_px = int(np.logical_and(region, raw_gt == 0).sum())
            known_px = int(np.logical_and(region, raw_gt >= 7).sum())
            kd_id, kd_name, kd_frac = component_distractor_overlap(raw_gt, region)

            short_side = max(1, min(comp["w"], comp["h"]))
            long_side = max(comp["w"], comp["h"])
            ar = float(long_side / short_side)
            orientation = (
                "vertical" if comp["h"] >= 2 * comp["w"]
                else "horizontal" if comp["w"] >= 2 * comp["h"]
                else "compact"
            )
            origin = (
                "known_distractor" if known_px > raw_bg_px
                else "generic_bg" if raw_bg_px > known_px
                else "mixed"
            )
            rows.append({
                "image": image_name,
                "pred_class_id": cls_id,
                "pred_class_name": CANONICAL_NAMES[cls_id],
                "component_id": comp["component_id"],
                "area_px": area,
                "x": comp["x"], "y": comp["y"], "w": comp["w"], "h": comp["h"],
                "aspect_ratio": ar,
                "orientation": orientation,
                "raw_bg_px": raw_bg_px,
                "known_distractor_px": known_px,
                "origin_majority": origin,
                "majority_known_distractor_id": kd_id,
                "majority_known_distractor_name": kd_name,
                "majority_known_distractor_fraction": kd_frac,
            })
    return rows


def summarize_fp(rows):
    def one(rs):
        return {
            "count": len(rs),
            "area_px": int(sum(r["area_px"] for r in rs)),
            "count_ge64": int(sum(r["area_px"] >= 64 for r in rs)),
            "count_ge256": int(sum(r["area_px"] >= 256 for r in rs)),
            "area_ge64_px": int(sum(r["area_px"] for r in rs if r["area_px"] >= 64)),
            "area_ge256_px": int(sum(r["area_px"] for r in rs if r["area_px"] >= 256)),
            "max_area_px": int(max((r["area_px"] for r in rs), default=0)),
        }
    out = {
        "all_inclusion": one(rows),
        "strip_inclusion": one([r for r in rows if r["pred_class_id"] in STRIP_CLASSES]),
        "D": one([r for r in rows if r["pred_class_id"] == D_ID]),
        "generic_bg_origin": one([r for r in rows if r["origin_majority"] == "generic_bg"]),
        "known_distractor_origin": one([r for r in rows if r["origin_majority"] == "known_distractor"]),
    }
    for c in INCLUSION_CLASSES:
        out[CANONICAL_NAMES[c]] = one([r for r in rows if r["pred_class_id"] == c])
    return out


# ============================================================
# 单模型评价
# ============================================================
def save_confusion(cm, out_dir):
    out_dir = Path(out_dir)
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["GT\\Pred"] + CANONICAL_NAMES)
        for i, name in enumerate(CANONICAL_NAMES):
            w.writerow([name] + cm[i].tolist())

    denom = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm.astype(np.float64), denom, out=np.zeros_like(cm, dtype=np.float64), where=denom > 0)
    with open(out_dir / "confusion_matrix_normalized.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["GT\\Pred"] + CANONICAL_NAMES)
        for i, name in enumerate(CANONICAL_NAMES):
            w.writerow([name] + [f"{v:.6f}" for v in norm[i]])


def per_image_metrics(image_name, gt7, pred7, fp_rows_image):
    tp, fp, fn, tn = binary_inclusion_counts(gt7, pred7)
    p, r, f1 = binary_metrics(tp, fp, fn, empty_is_one=True)
    exact = float((gt7 == pred7).mean())
    return {
        "image": image_name,
        "inclusion_precision": p,
        "inclusion_recall": r,
        "inclusion_f1": f1,
        "tp_px": tp,
        "fp_px": fp,
        "fn_px": fn,
        "tn_px": tn,
        "canonical_pixel_accuracy": exact,
        "pure_fp_components": len(fp_rows_image),
        "pure_fp_ge64_count": int(sum(x["area_px"] >= 64 for x in fp_rows_image)),
        "pure_fp_ge256_count": int(sum(x["area_px"] >= 256 for x in fp_rows_image)),
        "pure_fp_area_px": int(sum(x["area_px"] for x in fp_rows_image)),
    }


def evaluate_saved_model(tag: str, rel_dir: str, gt_masks: List[Path]):
    model_dir = model_eval_dir(rel_dir)
    pred_dir = model_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction dir 不存在：{pred_dir}")

    metadata = load_metadata(model_dir)
    out_dir = Path(CONFIG["output_dir"]) / tag
    ensure_dir(out_dir)

    cm = np.zeros((NUM_CANONICAL, NUM_CANONICAL), dtype=np.int64)
    inc_tp = inc_fp = inc_fn = 0
    object_rows = []
    fp_rows = []
    dist_rows = []
    per_image_rows = []
    raw_bg_total = raw_bg_to_inc = 0
    known_total = known_to_inc = 0

    for i, gt_path in enumerate(gt_masks, 1):
        raw_gt = read_mask(gt_path)
        validate_raw_labels(raw_gt, gt_path, 11)
        if CONFIG["expected_hw"] is not None and tuple(raw_gt.shape) != tuple(CONFIG["expected_hw"]):
            raise ValueError(f"GT 尺寸 {raw_gt.shape} != expected_hw={CONFIG['expected_hw']}：{gt_path.name}")

        pred_path = pred_dir / f"{gt_path.stem}.png"
        if not pred_path.exists():
            raise FileNotFoundError(f"{tag} 缺少 prediction：{pred_path}")
        raw_pred = read_mask(pred_path)
        validate_raw_labels(raw_pred, pred_path, 11)
        if raw_pred.shape != raw_gt.shape:
            raise ValueError(f"尺寸不一致 {tag}/{gt_path.stem}: pred={raw_pred.shape}, gt={raw_gt.shape}")

        # metadata 可辅助检查输出类别是否符合预期
        pred_scheme = metadata.get("prediction_scheme", "")
        if pred_scheme == "simple7" and int(raw_pred.max()) > 6:
            raise ValueError(f"{tag} prediction_scheme=simple7，但预测出现类别 >6：{pred_path}")
        if pred_scheme == "unified12" and int(raw_pred.max()) > 11:
            raise ValueError(f"{tag} prediction_scheme=unified12，但预测出现类别 >11：{pred_path}")

        gt7 = canonicalize(raw_gt)
        pred7 = canonicalize(raw_pred)

        update_confusion(cm, gt7, pred7)
        tp, fp, fn, _ = binary_inclusion_counts(gt7, pred7)
        inc_tp += tp; inc_fp += fp; inc_fn += fn

        objs = evaluate_objects(gt_path.name, gt7, pred7)
        object_rows.extend(objs)

        fps = extract_fp_components(gt_path.name, raw_gt, gt7, pred7)
        fp_rows.extend(fps)
        per_image_rows.append(per_image_metrics(gt_path.name, gt7, pred7, fps))

        dist_rows.extend(distractor_pixel_stats(raw_gt, pred7))

        pred_inc = np.isin(pred7, INCLUSION_CLASSES)
        bg_region = raw_gt == 0
        kd_region = raw_gt >= 7
        raw_bg_total += int(bg_region.sum())
        raw_bg_to_inc += int(np.logical_and(bg_region, pred_inc).sum())
        known_total += int(kd_region.sum())
        known_to_inc += int(np.logical_and(kd_region, pred_inc).sum())

        if i % 20 == 0 or i == len(gt_masks):
            print(f"[{tag}] {i}/{len(gt_masks)}")

    pixel_rows = pixel_metrics_from_cm(cm)
    pc = {r["class_name"]: r for r in pixel_rows}
    inc_p, inc_r, inc_f1 = binary_metrics(inc_tp, inc_fp, inc_fn)
    macro_f1 = float(np.mean([pc[CANONICAL_NAMES[c]]["f1"] for c in INCLUSION_CLASSES]))
    mean_iou = float(np.mean([pc[CANONICAL_NAMES[c]]["iou"] for c in INCLUSION_CLASSES]))

    obj_summary = summarize_objects(object_rows)
    d_size = summarize_size(object_rows, D_ID)
    tind_size = summarize_size(object_rows, TIND_ID)
    dist_summary = merge_distractor_rows(dist_rows)
    fp_summary = summarize_fp(fp_rows)

    summary = {
        "model_tag": tag,
        "prediction_dir": str(pred_dir),
        "metadata": metadata,
        "num_images": len(gt_masks),
        "inclusion_pixel": {
            "precision": inc_p, "recall": inc_r, "f1": inc_f1,
            "tp_px": inc_tp, "fp_px": inc_fp, "fn_px": inc_fn,
        },
        "macro_inclusion_f1": macro_f1,
        "mean_inclusion_iou": mean_iou,
        "per_class": pc,
        "object_by_class": obj_summary,
        "D_size": d_size,
        "TIND_size": tind_size,
        "distractor_leakage": dist_summary,
        "background_leakage": {
            "raw_gt_bg_to_any_inclusion_rate": safe_div(raw_bg_to_inc, raw_bg_total, zero=float("nan")),
            "raw_gt_bg_to_any_inclusion_px": raw_bg_to_inc,
            "raw_gt_bg_pixels": raw_bg_total,
            "known_distractor_to_any_inclusion_rate": safe_div(known_to_inc, known_total, zero=float("nan")),
            "known_distractor_to_any_inclusion_px": known_to_inc,
            "known_distractor_pixels": known_total,
        },
        "pure_apparent_fp": fp_summary,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=True)

    write_csv(out_dir / "class_pixel_metrics.csv", pixel_rows)
    write_csv(out_dir / "object_metrics.csv", object_rows)
    write_csv(out_dir / "D_size_objects.csv", [r for r in object_rows if r["gt_class_id"] == D_ID])
    write_csv(out_dir / "TIND_size_objects.csv", [r for r in object_rows if r["gt_class_id"] == TIND_ID])
    write_csv(out_dir / "distractor_leakage.csv", dist_summary)
    write_csv(out_dir / "apparent_fp_components.csv", fp_rows)
    write_csv(out_dir / "per_image_metrics.csv", per_image_rows)
    save_confusion(cm, out_dir)

    print(
        f"{tag}: Inclusion P/R/F1={inc_p:.4f}/{inc_r:.4f}/{inc_f1:.4f}, "
        f"small-D={small_recall(d_size):.4f}, small-TIND={small_recall(tind_size):.4f}, "
        f"FP>=64={fp_summary['all_inclusion']['count_ge64']}"
    )
    return summary, per_image_rows


def small_recall(size_summary):
    t1 = CONFIG["size_bins_um"][0]
    return float(size_summary[f"small_<={t1:g}um"]["object_recall"])


# ============================================================
# 汇总表
# ============================================================
def get_dist(summary, name):
    for r in summary["distractor_leakage"]:
        if r["gt_distractor_name"] == name:
            return r
    return {}


def flatten_summary(summary):
    pc = summary["per_class"]
    obj = summary["object_by_class"]
    dsz = summary["D_size"]
    tsz = summary["TIND_size"]
    t1, t2 = CONFIG["size_bins_um"]
    sm = f"small_<={t1:g}um"; md = f"medium_{t1:g}-{t2:g}um"; lg = f"large_>{t2:g}um"
    fp = summary["pure_apparent_fp"]
    meta = summary.get("metadata", {})
    model_stats = meta.get("model_stats", {}) or {}
    checkpoints = model_stats.get("checkpoints", {}) or {}
    checkpoint_total_size_mb = 0.0
    for info in checkpoints.values():
        if isinstance(info, dict) and isinstance(info.get("size_mb"), (int, float)):
            checkpoint_total_size_mb += float(info["size_mb"])

    row = {
        "model": summary["model_tag"],
        "model_type": meta.get("model_type", ""),
        "prediction_scheme": meta.get("prediction_scheme", ""),
        "gt_scheme": meta.get("gt_scheme", ""),
        "token_mode": meta.get("token_mode", ""),
        "inclusion_precision": summary["inclusion_pixel"]["precision"],
        "inclusion_recall": summary["inclusion_pixel"]["recall"],
        "inclusion_f1": summary["inclusion_pixel"]["f1"],
        "macro_inclusion_f1": summary["macro_inclusion_f1"],
        "mean_inclusion_iou": summary["mean_inclusion_iou"],
        "A_f1": pc["A"]["f1"], "B_f1": pc["B"]["f1"], "C_f1": pc["C"]["f1"],
        "D_f1": pc["D"]["f1"], "TINBC_f1": pc["TINB/TINC"]["f1"], "TIND_f1": pc["TIND"]["f1"],
        "A_recall": pc["A"]["recall"], "B_recall": pc["B"]["recall"], "C_recall": pc["C"]["recall"],
        "D_recall": pc["D"]["recall"], "TINBC_recall": pc["TINB/TINC"]["recall"], "TIND_recall": pc["TIND"]["recall"],
        "A_object_recall": obj["A"]["object_recall"],
        "B_object_recall": obj["B"]["object_recall"],
        "C_object_recall": obj["C"]["object_recall"],
        "D_object_recall": obj["D"]["object_recall"],
        "TINBC_object_recall": obj["TINB/TINC"]["object_recall"],
        "TIND_object_recall": obj["TIND"]["object_recall"],
        "D_small_recall": dsz[sm]["object_recall"],
        "D_medium_recall": dsz[md]["object_recall"],
        "D_large_recall": dsz[lg]["object_recall"],
        "TIND_small_recall": tsz[sm]["object_recall"],
        "TIND_medium_recall": tsz[md]["object_recall"],
        "TIND_large_recall": tsz[lg]["object_recall"],
        "A_mean_coverage": obj["A"]["mean_coverage"],
        "B_mean_coverage": obj["B"]["mean_coverage"],
        "C_mean_coverage": obj["C"]["mean_coverage"],
        "TIND_mean_coverage": obj["TIND"]["mean_coverage"],
        "A_mean_purity": obj["A"]["mean_purity"],
        "B_mean_purity": obj["B"]["mean_purity"],
        "C_mean_purity": obj["C"]["mean_purity"],
        "TIND_mean_purity": obj["TIND"]["mean_purity"],
        "raw_BG_to_inclusion": summary["background_leakage"]["raw_gt_bg_to_any_inclusion_rate"],
        "known_distractor_to_inclusion": summary["background_leakage"]["known_distractor_to_any_inclusion_rate"],
        "pure_FP_components": fp["all_inclusion"]["count"],
        "pure_FP_area_px": fp["all_inclusion"]["area_px"],
        "pure_FP_ge64_count": fp["all_inclusion"]["count_ge64"],
        "pure_FP_ge256_count": fp["all_inclusion"]["count_ge256"],
        "pure_strip_FP_ge64_count": fp["strip_inclusion"]["count_ge64"],
        "pure_strip_FP_ge256_count": fp["strip_inclusion"]["count_ge256"],
        "pure_D_FP_ge64_count": fp["D"]["count_ge64"],
        "known_distractor_FP_ge64_count": fp["known_distractor_origin"]["count_ge64"],
        "avg_inference_ms": meta.get("avg_inference_ms"),
        "p95_inference_ms": meta.get("p95_inference_ms"),
        "total_params_m": model_stats.get("total_params_m"),
        "checkpoint_total_size_mb": checkpoint_total_size_mb,
        "peak_gpu_memory_mb": meta.get("peak_gpu_memory_mb"),
    }

    for name in ("HH", "XW", "XQL", "HC", "SZ"):
        d = get_dist(summary, name)
        row[f"{name}_to_any_inclusion"] = d.get("to_any_inclusion_rate", float("nan"))
        row[f"{name}_to_D"] = d.get("to_D_rate", float("nan"))
    return row


def write_wide(rows, path):
    if not rows:
        return
    meta_cols = {"model", "model_type", "prediction_scheme", "gt_scheme", "token_mode"}
    metrics = [k for k in rows[0].keys() if k not in meta_cols]
    out = []
    for metric in metrics:
        r = {"metric": metric}
        for model_row in rows:
            r[model_row["model"]] = model_row.get(metric)
        out.append(r)
    write_csv(path, out)


# ============================================================
# Pair effects：量化 A-B / B-C / D-C / A-D
# ============================================================
LOWER_IS_BETTER_PREFIXES = (
    "pure_FP", "pure_strip_FP", "pure_D_FP", "known_distractor_FP",
    "raw_BG_to_inclusion", "known_distractor_to_inclusion",
    "HH_to_", "XW_to_", "XQL_to_", "HC_to_", "SZ_to_",
    "avg_inference_ms", "p95_inference_ms", "peak_gpu_memory_mb",
    "checkpoint_total_size_mb", "total_params_m",
)

NON_NUMERIC = {"model", "model_type", "prediction_scheme", "gt_scheme", "token_mode"}


def is_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def lower_is_better(metric):
    return metric.startswith(LOWER_IS_BETTER_PREFIXES)


def make_pair_effects(summary_rows):
    by_model = {r["model"]: r for r in summary_rows}
    rows = []
    for pair_name, cfg in CONFIG["pairs"].items():
        target = cfg["target"]; ref = cfg["reference"]
        if target not in by_model or ref not in by_model:
            continue
        a = by_model[target]; b = by_model[ref]
        for metric, va in a.items():
            if metric in NON_NUMERIC or metric not in b:
                continue
            vb = b[metric]
            if not (is_number(va) and is_number(vb)):
                continue
            if np.isnan(float(va)) or np.isnan(float(vb)):
                delta = beneficial = float("nan")
            else:
                delta = float(va) - float(vb)
                beneficial = -delta if lower_is_better(metric) else delta
            rows.append({
                "pair": pair_name,
                "meaning": cfg["meaning"],
                "target": target,
                "reference": ref,
                "metric": metric,
                "target_value": va,
                "reference_value": vb,
                "raw_delta_target_minus_reference": delta,
                "beneficial_delta_positive_is_better": beneficial,
                "direction": "lower_is_better" if lower_is_better(metric) else "higher_is_better",
            })
    return rows


def combine_per_image(per_image_by_model):
    # 假设每个 model 都覆盖同一图集
    model_tags = list(per_image_by_model.keys())
    by_tag = {
        tag: {r["image"]: r for r in rows}
        for tag, rows in per_image_by_model.items()
    }
    common = set(by_tag[model_tags[0]])
    for tag in model_tags[1:]:
        common &= set(by_tag[tag])

    rows = []
    fields = [
        "inclusion_precision", "inclusion_recall", "inclusion_f1",
        "fp_px", "fn_px", "canonical_pixel_accuracy",
        "pure_fp_ge64_count", "pure_fp_ge256_count", "pure_fp_area_px",
    ]
    for image in sorted(common):
        row = {"image": image}
        for tag in model_tags:
            rr = by_tag[tag][image]
            for f in fields:
                row[f"{tag}__{f}"] = rr[f]
        rows.append(row)
    return rows


def pair_image_wins(combined_rows):
    detail_rows = []
    summary_rows = []
    eps = 1e-12
    for pair_name, cfg in CONFIG["pairs"].items():
        target, ref = cfg["target"], cfg["reference"]
        wins = ties = losses = 0
        fp_wins = fp_ties = fp_losses = 0
        for row in combined_rows:
            tf1 = float(row[f"{target}__inclusion_f1"])
            rf1 = float(row[f"{ref}__inclusion_f1"])
            d = tf1 - rf1
            if d > eps:
                result = "target_win"; wins += 1
            elif d < -eps:
                result = "target_loss"; losses += 1
            else:
                result = "tie"; ties += 1

            tfp = int(row[f"{target}__pure_fp_ge64_count"])
            rfp = int(row[f"{ref}__pure_fp_ge64_count"])
            if tfp < rfp:
                fp_result = "target_win"; fp_wins += 1
            elif tfp > rfp:
                fp_result = "target_loss"; fp_losses += 1
            else:
                fp_result = "tie"; fp_ties += 1

            detail_rows.append({
                "pair": pair_name,
                "meaning": cfg["meaning"],
                "image": row["image"],
                "target": target,
                "reference": ref,
                "target_inclusion_f1": tf1,
                "reference_inclusion_f1": rf1,
                "delta_f1": d,
                "f1_result": result,
                "target_fp_ge64": tfp,
                "reference_fp_ge64": rfp,
                "delta_fp_ge64": tfp - rfp,
                "fp_ge64_result": fp_result,
                "target_fp_px": row[f"{target}__fp_px"],
                "reference_fp_px": row[f"{ref}__fp_px"],
                "target_fn_px": row[f"{target}__fn_px"],
                "reference_fn_px": row[f"{ref}__fn_px"],
            })

        total = wins + ties + losses
        summary_rows.append({
            "pair": pair_name,
            "meaning": cfg["meaning"],
            "target": target,
            "reference": ref,
            "n_images": total,
            "f1_target_wins": wins,
            "f1_ties": ties,
            "f1_target_losses": losses,
            "f1_target_win_rate_excluding_ties": safe_div(wins, wins + losses, zero=float("nan")),
            "fp_ge64_target_wins": fp_wins,
            "fp_ge64_ties": fp_ties,
            "fp_ge64_target_losses": fp_losses,
            "fp_ge64_target_win_rate_excluding_ties": safe_div(fp_wins, fp_wins + fp_losses, zero=float("nan")),
        })
    return detail_rows, summary_rows


# ============================================================
# Panel：Original / GT / A / B / C / D
# ============================================================
def colorize_mask(mask7):
    return PALETTE_BGR[mask7]


def put_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(out, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def fit_tile(img, size=(512, 512)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def make_panel_for_image(image_name, model_dirs, out_path, image_dir: Path, mask_dir: Path):
    stem = Path(image_name).stem
    img_path = find_image_by_stem(image_dir, stem)
    if img_path is None:
        return False

    raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    gt_path = mask_dir / f"{stem}.png"
    if raw is None or not gt_path.exists():
        return False
    gt7 = canonicalize(read_mask(gt_path))

    tiles = [put_title(fit_tile(raw), "Original"), put_title(fit_tile(colorize_mask(gt7)), "GT canonical-7")]

    # 固定按 CONFIG models 顺序，通常 A/B/C/D
    for tag, rel in model_dirs.items():
        pred_path = model_eval_dir(rel) / "predictions" / f"{stem}.png"
        if not pred_path.exists():
            return False
        pred7 = canonicalize(read_mask(pred_path))
        tiles.append(put_title(fit_tile(colorize_mask(pred7)), tag))

    # 目标是 6 格；若不是 4 模型则动态补空/截断
    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]))
    tiles = tiles[:6]
    row1 = np.hstack(tiles[0:2])
    row2 = np.hstack(tiles[2:4])
    row3 = np.hstack(tiles[4:6])
    panel = np.vstack([row1, row2, row3])
    ensure_dir(Path(out_path).parent)
    return bool(cv2.imwrite(str(out_path), panel))


def generate_pair_panels(pair_detail_rows, image_dir: Path, mask_dir: Path):
    if not CONFIG.get("make_panels", True):
        return
    k = int(CONFIG.get("panel_top_k_each_side", 20))
    out_root = Path(CONFIG["output_dir"]) / "panels"
    models = CONFIG["models"]

    for pair_name in CONFIG["pairs"]:
        rows = [r for r in pair_detail_rows if r["pair"] == pair_name]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r["delta_f1"]))
        worst = rows[:k]
        best = list(reversed(rows[-k:]))

        for group_name, group in (("target_worse", worst), ("target_better", best)):
            for rank, r in enumerate(group, 1):
                stem = Path(r["image"]).stem
                delta = float(r["delta_f1"])
                out = out_root / pair_name / group_name / f"{rank:02d}_{delta:+.4f}_{stem}.jpg"
                make_panel_for_image(r["image"], models, out, image_dir=image_dir, mask_dir=mask_dir)


def write_key_metrics_wide(summary_rows, path):
    by_model = {r["model"]: r for r in summary_rows}
    rows = []
    for metric in KEY_METRICS:
        row = {"metric": metric, "direction": "lower_is_better" if lower_is_better(metric) else "higher_is_better"}
        for tag in CONFIG["models"]:
            row[tag] = by_model.get(tag, {}).get(metric, float("nan"))
        rows.append(row)
    write_csv(path, rows)


def filter_pair_effects_key(rows):
    keys = set(KEY_METRICS)
    return [r for r in rows if r.get("metric") in keys]


# ============================================================
# 主流程
# ============================================================
def main():
    out_root = Path(CONFIG["output_dir"])
    ensure_dir(out_root)

    image_dir, mask_dir, common_metas = resolve_common_dataset()
    gt_masks = list_gt_masks(mask_dir)
    print(f"Common image dir : {image_dir}")
    print(f"Common mask dir  : {mask_dir}")
    print(f"GT masks         : {len(gt_masks)}")

    summaries = {}
    per_image_by_model = {}
    summary_rows = []

    for tag, rel in CONFIG["models"].items():
        print("\n" + "=" * 90)
        print(f"Evaluating saved predictions: {tag}")
        print("=" * 90)
        summary, per_image = evaluate_saved_model(tag, rel, gt_masks)
        summaries[tag] = summary
        per_image_by_model[tag] = per_image
        summary_rows.append(flatten_summary(summary))

    # 主表：模型一行；宽表：指标一行
    write_csv(out_root / "comparison_summary.csv", summary_rows)
    write_wide(summary_rows, out_root / "comparison_wide.csv")
    write_key_metrics_wide(summary_rows, out_root / "key_metrics_wide.csv")

    # Pair 量化差异
    pair_effect_rows = make_pair_effects(summary_rows)
    write_csv(out_root / "pair_effects.csv", pair_effect_rows)
    write_csv(out_root / "pair_effects_key.csv", filter_pair_effects_key(pair_effect_rows))

    # 逐图对比
    combined = combine_per_image(per_image_by_model)
    write_csv(out_root / "per_image_all_models.csv", combined)
    pair_detail, pair_win_summary = pair_image_wins(combined)
    write_csv(out_root / "pair_image_details.csv", pair_detail)
    write_csv(out_root / "pair_image_win_summary.csv", pair_win_summary)

    # 差异最大样本可视化
    generate_pair_panels(pair_detail, image_dir=image_dir, mask_dir=mask_dir)

    # 额外保存一份实验结构说明，防止以后忘记 A/B/C/D 含义
    manifest = {
        "canonical_classes": CANONICAL_NAMES,
        "models": CONFIG["models"],
        "eval_set_name": CONFIG["eval_set_name"],
        "common_image_dir": str(image_dir.resolve()),
        "common_mask_dir": str(mask_dir.resolve()),
        "pairs": CONFIG["pairs"],
        "evaluation_rules": {
            "canonical_mapping": "raw class IDs >=7 -> background(0)",
            "inclusion_binary": "canonical classes 1..6 are inclusion",
            "object_detected_threshold": CONFIG["object_coverage_threshold"],
            "size_bins_um": CONFIG["size_bins_um"],
            "pure_apparent_fp": "full predicted-class CC with zero overlap to any GT inclusion",
        },
    }
    with open(out_root / "comparison_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("比较完成。建议优先查看：")
    print(f"1) {out_root / 'key_metrics_wide.csv'}")
    print(f"2) {out_root / 'pair_effects_key.csv'}")
    print(f"3) {out_root / 'pair_image_win_summary.csv'}")
    print(f"4) {out_root / 'panels'}")
    print(f"5) {out_root / 'comparison_wide.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()