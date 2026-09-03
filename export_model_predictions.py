# -*- coding: utf-8 -*-
"""
export_model_predictions.py

用途
----
把不同历史模型/checkpoint 的预测先“冻结”为 PNG mask，避免后续源码变化影响比较。

本版专门处理当前 4 组模型：

A = Simple-7-Correct
    - 单模型，7 类：0 BG, 1 A, 2 B, 3 C, 4 D, 5 TINBC, 6 TIND
    - token: feat[:, 1 + num_register_tokens:, :]
    - 主对比统一使用 D 的 unified12 全类别验证集

B = Simple-7-Legacy
    - 单模型，7 类
    - token: feat[:, 1:-4, :]
    - 主对比统一使用 D 的 unified12 全类别验证集

C = Legacy Dual (Strip + Point)
    - 条状模型 9 类：0 BG, 1 A, 2 B, 3 C, 4 HH, 5 XW, 6 XQL, 7 TINBC, 8 TIND
    - 点状模型 4 类：0 BG, 1 D, 2 HC, 3 SZ
    - 两个模型都使用 legacy token: feat[:, 1:-4, :]
    - 最终融合到 unified 12-ID 空间（0..11）：
        strip: 0->0, 1->1, 2->2, 3->3, 4->7, 5->8, 6->9, 7->5, 8->6
        point: 0->0, 1->4, 2->10, 3->11
      点状非背景区域优先覆盖条状结果。
    - 主对比统一使用 D 的 unified12 全类别验证集；对同一张图同时运行 strip + point，
      保存 strip_raw / point_raw / merged 三种预测，其中 merged 可直接和 unified12 GT 做系统级评价。

D = inclusion_v2
    - Gate + Strip/Point Experts
    - 使用自己的 full unified12 验证集（夹杂物 + 非夹杂物）
    - 历史 checkpoint 必须使用训练时对应的 inclusion_v2 encoder token 逻辑。

重要说明
--------
1) A/B/C 的 DINO encoder 在本脚本内实现，可以显式选择 correct / legacy token，
   不受你当前 models/dinov3_encoder.py 是否已经修改的影响。
2) D 仍调用仓库中的 inclusion_v2 模型，因此保留 source guard：若历史 checkpoint 声明 legacy，
   但 inclusion_v2/models/encoder.py 已改成 correct，默认直接报错，避免“旧权重 + 新 token 逻辑”。
3) A/B/C/D 的“主对比”强制统一使用 COMMON_EVAL，也就是 D 当前的 unified12 全类别验证集。
   这样四个模型面对完全相同的图片和 GT，后续才可以做 head-to-head 比较。
4) A/B 只输出 0..6；后续 evaluator 需要把 unified12 GT 中 7..11 折叠为 Background(0)。
   C/D 输出 0..11；四模型主指标比较时同样把 C/D 预测中的 7..11 折叠为 Background(0)。
5) 本脚本只导出原始预测，不在这里做类别折叠或计算分数。

输出结构
--------
    saved_predictions/
      A_simple7_correct/
        common_unified12_val/
          predictions/*.png
          metadata.json
          manifest.csv
          runtime_per_image.csv

      C_legacy_dual/
        common_unified12_val/
          predictions/*.png              # merged unified12 IDs
          strip_predictions_raw/*.png    # 0..8
          point_predictions_raw/*.png    # 0..3
          metadata.json
          manifest.csv

运行示例
--------
    python export_model_predictions.py --profile A_simple7_correct
    python export_model_predictions.py --profile B_simple7_legacy
    python export_model_predictions.py --profile C_legacy_dual
    python export_model_predictions.py --profile D_v2_legacy

四个 profile 都使用同一个 eval-set 名称：
    common_unified12_val
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


# ============================================================
# 公共主验证集：统一使用 D 当前的 unified12 全类别验证集
# A/B/C/D 都从这里读取，避免四处重复填写后路径不一致。
# ============================================================
COMMON_EVAL = {
    "image_dir": r"D:/lhb/datasets/JZW_v3/JZW_ALL/total/val/images",
    "mask_dir": r"D:/lhb/datasets/JZW_v3/JZW_ALL/total/val/masks",
    "gt_scheme": "unified12",
}


# ============================================================
# CONFIG：按你的本机路径填写
# ============================================================
CONFIG = {
    "output_root": r"./saved_predictions",
    "device": "cuda",
    "use_amp": True,
    "expected_hw": (1024, 1024),
    "overwrite": False,
    "hash_checkpoint_sha256": False,
    "latency_warmup_images": 5,

    # 与现有验证/推理 Normalize 保持一致
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),

    "profiles": {
        # ----------------------------------------------------
        # A：Simple-7 + correct token
        # ----------------------------------------------------
        "A_simple7_correct": {
            "model_type": "simple",
            "checkpoint": r"./checkpoints/JZW_2/best_iou.pth",
            "backbone_name": r"dinov3_model",
            "num_classes": 7,
            "token_mode": "correct",
            "weight_source": "auto",
            "prediction_scheme": "simple7",
            "notes": "Simple-7 + correct token extraction",
            "eval_sets": {
                "common_unified12_val": dict(COMMON_EVAL),
            },
        },

        # ----------------------------------------------------
        # B：Simple-7 + legacy token
        # ----------------------------------------------------
        "B_simple7_legacy": {
            "model_type": "simple",
            "checkpoint": r"./checkpoints/JZW/best_iou.pth",
            "backbone_name": r"dinov3_model",
            "num_classes": 7,
            "token_mode": "legacy",
            "weight_source": "auto",
            "prediction_scheme": "simple7",
            "notes": "Simple-7 + feat[:,1:-4,:]",
            "eval_sets": {
                "common_unified12_val": dict(COMMON_EVAL),
            },
        },

        # ----------------------------------------------------
        # C：旧版条状 + 点状双模型；两个权重，legacy token
        # ----------------------------------------------------
        "C_legacy_dual": {
            "model_type": "legacy_dual",
            "backbone_name": r"dinov3_model",
            "token_mode": "legacy",
            "weight_source": "auto",  # teacher / student / auto

            # 条状模型：9 类
            "strip_checkpoint": r"./checkpoints/ABCTIN_1024_v2/ABCTIN_slim.pth",
            "strip_num_classes": 9,

            # 点状模型：4 类
            "point_checkpoint": r"./checkpoints/D_1024_v2/DDS_slim.pth",
            "point_num_classes": 4,

            # 与 infer_end2end.py 保持一致。1024 val patch 时等价于单窗口推理。
            "confidence_threshold": 0.1,
            "prediction_scheme": "unified12",
            "notes": "Historical strip9 + point4 dual-model pipeline; point nonzero overrides strip",

            # 主对比：在 D 的 unified12 全类别验证集上，对每张同一图片同时跑 strip + point，
            # 再按 MAPPING_STRIP / MAPPING_POINT 重映射并融合。
            "eval_sets": {
                "common_unified12_val": dict(COMMON_EVAL),
            },
        },

        # ----------------------------------------------------
        # D：inclusion_v2，使用自己的 unified12 full validation
        # ----------------------------------------------------
        "D_v2_legacy": {
            "model_type": "v2",
            "checkpoint": r"./checkpoints/inclusion_v2_mvp1/best_inclusion_f1.pth",
            "backbone_name": r"dinov3_model",
            "token_mode": "legacy",
            "prediction_scheme": "unified12",
            "weight_source": "auto",
            "fusion_alpha": 0.5,
            "freeze_backbone": True,
            "encoder_layers": (4, 8, 12),
            "feat_dim": 768,
            "fusion_dim": 512,
            "decoder_dim": 32,
            "source_guard_file": r"inclusion_v2/models/encoder.py",
            "strict_source_guard": True,
            "notes": "inclusion_v2 historical checkpoint + legacy token extraction",
            "eval_sets": {
                "common_unified12_val": dict(COMMON_EVAL),
            },
        },
    },
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# C：与旧 infer_end2end.py / merge_masks.py 一致的映射
MAPPING_STRIP = {
    0: 0,   # BG
    1: 1,   # A
    2: 2,   # B
    3: 3,   # C
    4: 7,   # HH
    5: 8,   # XW
    6: 9,   # XQL
    7: 5,   # TIN-B/TIN-C
    8: 6,   # TIN-D
}

MAPPING_POINT = {
    0: 0,   # BG
    1: 4,   # D
    2: 10,  # HC
    3: 11,  # SZ
}

GT_MAX_ID = {
    "simple7": 6,
    "strip9": 8,
    "point4": 3,
    "unified12": 11,
}

PRED_MAX_ID = {
    "simple7": 6,
    "unified12": 11,
}


# ============================================================
# 通用工具
# ============================================================
def ensure_dir(path: Path | str):
    Path(path).mkdir(parents=True, exist_ok=True)


def device_from_config() -> torch.device:
    req = str(CONFIG["device"])
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动改用 CPU")
        return torch.device("cpu")
    return torch.device(req)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def git_commit(repo_root: Path = Path(".")) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        return out or None
    except Exception:
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_module_prefix(state: Dict[str, torch.Tensor]):
    if state and all(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def select_state_dict(ckpt, weight_source: str = "auto"):
    """
    兼容：
      - EMA/双权重 checkpoint: teacher / student / model
      - 普通 checkpoint: model / state_dict / model_state_dict
      - 纯 state_dict

    返回: state_dict, meta, loaded_source
    """
    if not isinstance(ckpt, dict):
        return ckpt, {}, "raw"

    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, {}, "raw_state_dict"

    if weight_source == "teacher":
        priority = ["teacher", "student", "model", "state_dict", "model_state_dict"]
    elif weight_source == "student":
        priority = ["student", "teacher", "model", "state_dict", "model_state_dict"]
    else:
        priority = ["teacher", "student", "model", "state_dict", "model_state_dict"]

    for key in priority:
        if key in ckpt and isinstance(ckpt[key], dict):
            meta = {
                k: v for k, v in ckpt.items()
                if k != key and not torch.is_tensor(v) and not isinstance(v, dict)
            }
            return ckpt[key], meta, key

    raise ValueError(
        "无法识别 checkpoint 格式；未找到 teacher/student/model/state_dict/model_state_dict。"
        f" keys={list(ckpt.keys())[:30]}"
    )


def classify_encoder_source(path: Path) -> str:
    """粗略识别 inclusion_v2 encoder 是 legacy 还是 correct token slice。"""
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    compact = re.sub(r"\s+", "", text)

    if "[:,1:-4,:]" in compact or "[:,1:-4]" in compact:
        return "legacy"

    if (
        "num_register_tokens" in text
        and ("1+num_register" in compact or "1+self." in compact)
    ):
        return "correct"

    if "register" in text.lower() and re.search(r"1\s*\+", text):
        return "correct"

    return "unknown"


def check_v2_source_guard(profile: dict):
    expected = profile.get("token_mode", "repo")
    if expected not in ("legacy", "correct"):
        return

    guard = Path(profile.get("source_guard_file", "inclusion_v2/models/encoder.py"))
    detected = classify_encoder_source(guard)
    strict = bool(profile.get("strict_source_guard", True))

    print(f"[V2 source guard] {guard}: detected={detected}, expected={expected}")
    if detected != expected:
        msg = (
            f"V2 encoder 源码与 checkpoint 声明不一致：expected={expected}, detected={detected}.\n"
            f"文件：{guard}\n"
            "历史 V2 checkpoint 必须用训练时同一 token 逻辑推理。"
        )
        if strict:
            raise RuntimeError(msg)
        print("[WARN] " + msg)


def preprocess_image(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    mean = np.asarray(CONFIG["mean"], dtype=np.float32)
    std = np.asarray(CONFIG["std"], dtype=np.float32)
    x = image_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).unsqueeze(0)
    return x.to(device, non_blocking=True)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir 不存在：{image_dir}")
    imgs = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        raise RuntimeError(f"没有找到验证图片：{image_dir}")
    stems = [p.stem for p in imgs]
    if len(stems) != len(set(stems)):
        raise RuntimeError(f"{image_dir} 中存在不同文件但 stem 相同，导出 PNG 会覆盖")
    return imgs


def find_mask_by_stem(mask_dir: Path, stem: str) -> Path | None:
    direct = mask_dir / f"{stem}.png"
    if direct.exists():
        return direct
    for p in mask_dir.glob(f"{stem}.*"):
        if p.suffix.lower() in (".png", ".bmp", ".tif", ".tiff"):
            return p
    return None


def validate_eval_set(eval_name: str, spec: dict, images: Iterable[Path]) -> list[dict]:
    """检查 image/mask 对齐，并生成 manifest 基础行。"""
    gt_scheme = spec.get("gt_scheme")
    if gt_scheme not in GT_MAX_ID:
        raise ValueError(f"eval_set={eval_name} 的 gt_scheme 无效：{gt_scheme}")

    mask_dir_raw = spec.get("mask_dir")
    mask_dir = Path(mask_dir_raw) if mask_dir_raw else None
    if mask_dir is not None and not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir 不存在：{mask_dir}")

    rows = []
    missing = []
    for p in images:
        mask_path = find_mask_by_stem(mask_dir, p.stem) if mask_dir is not None else None
        if mask_dir is not None and mask_path is None:
            missing.append(p.name)
        rows.append({
            "image": p.name,
            "stem": p.stem,
            "image_path": str(p.resolve()),
            "gt_mask_path": str(mask_path.resolve()) if mask_path is not None else "",
            "gt_scheme": gt_scheme,
        })

    if missing:
        raise RuntimeError(
            f"eval_set={eval_name} 有 {len(missing)} 张图片找不到对应 mask，例如：{missing[:5]}"
        )
    return rows


def validate_prediction_ids(pred: np.ndarray, scheme: str, context: str):
    max_allowed = PRED_MAX_ID[scheme]
    u = np.unique(pred)
    if int(u.max()) > max_allowed or int(u.min()) < 0:
        raise RuntimeError(
            f"{context} 预测类别超范围：unique={u.tolist()[:30]}, scheme={scheme}, max={max_allowed}"
        )


def save_mask(path: Path, mask: np.ndarray):
    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), mask.astype(np.uint8)):
        raise RuntimeError(f"保存 mask 失败：{path}")


# ============================================================
# A/B/C 共用的兼容 Simple encoder（模型配置保持在本脚本中）
# ============================================================
class DINOv3EncoderCompat(nn.Module):
    """参数命名保持 encoder.backbone.*，但 token slicing 可显式选择。"""

    def __init__(self, model_name: str, token_mode: str, trainable: bool = False):
        super().__init__()
        if token_mode not in ("legacy", "correct"):
            raise ValueError("token_mode 必须是 legacy 或 correct")

        self.backbone = AutoModel.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.token_mode = token_mode
        self.trainable = trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def set_trainable(self, trainable: bool):
        self.trainable = trainable
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def _take_patch_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        if self.token_mode == "legacy":
            # 仅用于复现历史 checkpoint。
            return feat[:, 1:-4, :]

        nreg = int(getattr(self.backbone.config, "num_register_tokens", 4))
        return feat[:, 1 + nreg:, :]

    def _reshape(self, feat: torch.Tensor, input_hw: Tuple[int, int]) -> torch.Tensor:
        B, _, C = feat.shape
        feat = self._take_patch_tokens(feat)
        n = int(feat.shape[1])

        h = w = int(round(n ** 0.5))
        if h * w != n:
            raise RuntimeError(f"patch token 数 {n} 不能组成正方形 feature map")

        patch = getattr(self.backbone.config, "patch_size", 16)
        if isinstance(patch, (tuple, list)):
            ph, pw = int(patch[0]), int(patch[1])
        else:
            ph = pw = int(patch)

        ih, iw = input_hw
        expected_h, expected_w = ih // ph, iw // pw
        if expected_h * expected_w == n:
            h, w = expected_h, expected_w

        feat = feat.reshape(B, h, w, C).permute(0, 3, 1, 2)
        return feat.contiguous()

    def forward(self, x: torch.Tensor):
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        hs = outputs.hidden_states
        f_low, f_mid, f_high = hs[-3], hs[-2], hs[-1]
        hw = (int(x.shape[-2]), int(x.shape[-1]))
        return {
            "f4": self._reshape(f_low, hw),
            "f8": self._reshape(f_mid, hw),
            "f16": self._reshape(f_high, hw),
        }


class SimpleDINOv3SegCompat(nn.Module):
    """与仓库 DINOv3Seg 的 state_dict key 保持兼容。"""

    def __init__(self, backbone_name: str, num_classes: int, token_mode: str):
        super().__init__()
        from models.decoder_v3 import DecoderV3

        self.encoder = DINOv3EncoderCompat(
            backbone_name, token_mode=token_mode, trainable=False
        )
        self.decoder = DecoderV3(num_classes=num_classes, feat_dim=768)

    def forward(self, x):
        feats = self.encoder(x)
        seg, boundary = self.decoder(feats, output_size=x.shape[2:])
        return seg, boundary


# ============================================================
# 模型加载
# ============================================================
def load_simple_checkpoint(
    checkpoint_path: str | Path,
    backbone_name: str,
    num_classes: int,
    token_mode: str,
    weight_source: str,
    device: torch.device,
):
    model = SimpleDINOv3SegCompat(
        backbone_name=backbone_name,
        num_classes=int(num_classes),
        token_mode=token_mode,
    ).to(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state, meta, loaded_source = select_state_dict(ckpt, weight_source=weight_source)
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, meta, loaded_source


def build_v2_cfg(profile: dict):
    return SimpleNamespace(
        backbone_name=profile["backbone_name"],
        freeze_backbone=profile.get("freeze_backbone", True),
        encoder_layers=tuple(profile.get("encoder_layers", (4, 8, 12))),
        feat_dim=profile.get("feat_dim", 768),
        fusion_dim=profile.get("fusion_dim", 512),
        decoder_dim=profile.get("decoder_dim", 32),
    )


def load_profile_models(profile: dict, device: torch.device):
    typ = profile["model_type"].lower()

    if typ == "simple":
        model, meta, source = load_simple_checkpoint(
            checkpoint_path=profile["checkpoint"],
            backbone_name=profile["backbone_name"],
            num_classes=profile["num_classes"],
            token_mode=profile["token_mode"],
            weight_source=profile.get("weight_source", "auto"),
            device=device,
        )
        return {
            "type": typ,
            "model": model,
            "checkpoint_meta": meta,
            "loaded_source": source,
        }

    if typ == "legacy_dual":
        strip_model, strip_meta, strip_source = load_simple_checkpoint(
            checkpoint_path=profile["strip_checkpoint"],
            backbone_name=profile["backbone_name"],
            num_classes=profile["strip_num_classes"],
            token_mode=profile["token_mode"],
            weight_source=profile.get("weight_source", "auto"),
            device=device,
        )
        point_model, point_meta, point_source = load_simple_checkpoint(
            checkpoint_path=profile["point_checkpoint"],
            backbone_name=profile["backbone_name"],
            num_classes=profile["point_num_classes"],
            token_mode=profile["token_mode"],
            weight_source=profile.get("weight_source", "auto"),
            device=device,
        )
        return {
            "type": typ,
            "strip_model": strip_model,
            "point_model": point_model,
            "strip_checkpoint_meta": strip_meta,
            "point_checkpoint_meta": point_meta,
            "strip_loaded_source": strip_source,
            "point_loaded_source": point_source,
        }

    if typ == "v2":
        check_v2_source_guard(profile)
        from inclusion_v2.models.model import InclusionDualExpertNet

        model = InclusionDualExpertNet(build_v2_cfg(profile)).to(device)
        ckpt_path = Path(profile["checkpoint"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在：{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state, meta, source = select_state_dict(
            ckpt, weight_source=profile.get("weight_source", "auto")
        )
        state = strip_module_prefix(state)
        model.load_state_dict(state, strict=True)
        model.eval()
        return {
            "type": typ,
            "model": model,
            "checkpoint_meta": meta,
            "loaded_source": source,
        }

    raise ValueError(f"未知 model_type: {typ}")


# ============================================================
# 推理与 C 的 mask 重映射/融合
# ============================================================
def remap_mask(mask: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for old_class, new_class in mapping.items():
        new_mask[mask == old_class] = new_class
    return new_mask


def merge_legacy_dual_masks(mask_strip: np.ndarray, mask_point: np.ndarray) -> np.ndarray:
    """与旧 infer_end2end.py / merge_masks.py 一致：point 非0优先覆盖 strip。"""
    strip_new = remap_mask(mask_strip, MAPPING_STRIP)
    point_new = remap_mask(mask_point, MAPPING_POINT)
    merged = strip_new.copy()
    merged[point_new != 0] = point_new[point_new != 0]
    return merged.astype(np.uint8)


@torch.no_grad()
def infer_simple_logits(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    amp = bool(CONFIG["use_amp"] and device.type == "cuda")
    if amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            seg, _ = model(x)
    else:
        seg, _ = model(x)
    return seg


@torch.no_grad()
def infer_one(models: dict, profile: dict, image_rgb: np.ndarray, device: torch.device):
    """
    返回:
      result: dict[str, np.ndarray]
        simple/v2: {"merged": pred}
        legacy_dual: {"merged": ..., "strip_raw": ..., "point_raw": ...}
      elapsed_ms: 整个 profile 的实际系统推理时间
    """
    x = preprocess_image(image_rgb, device)
    typ = profile["model_type"].lower()

    sync(device)
    t0 = time.perf_counter()

    if typ == "simple":
        seg = infer_simple_logits(models["model"], x, device)
        pred = torch.argmax(seg, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        result = {"merged": pred}

    elif typ == "legacy_dual":
        # 与用户现有双模型推理一致：各自 softmax -> argmax -> confidence threshold -> remap -> point override
        strip_seg = infer_simple_logits(models["strip_model"], x, device)
        point_seg = infer_simple_logits(models["point_model"], x, device)

        strip_prob = torch.softmax(strip_seg, dim=1)
        point_prob = torch.softmax(point_seg, dim=1)

        strip_conf, strip_pred_t = torch.max(strip_prob, dim=1)
        point_conf, point_pred_t = torch.max(point_prob, dim=1)

        threshold = float(profile.get("confidence_threshold", 0.1))
        strip_pred_t = strip_pred_t.clone()
        point_pred_t = point_pred_t.clone()
        strip_pred_t[strip_conf < threshold] = 0
        point_pred_t[point_conf < threshold] = 0

        strip_raw = strip_pred_t[0].detach().cpu().numpy().astype(np.uint8)
        point_raw = point_pred_t[0].detach().cpu().numpy().astype(np.uint8)
        merged = merge_legacy_dual_masks(strip_raw, point_raw)
        result = {
            "merged": merged,
            "strip_raw": strip_raw,
            "point_raw": point_raw,
        }

    elif typ == "v2":
        from inclusion_v2.utils.output_fusion import fuse_outputs

        amp = bool(CONFIG["use_amp"] and device.type == "cuda")
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = models["model"](x)
                fused = fuse_outputs(
                    outputs["gate"], outputs["strip"], outputs["point"],
                    alpha=float(profile.get("fusion_alpha", 0.5)),
                )
        else:
            outputs = models["model"](x)
            fused = fuse_outputs(
                outputs["gate"], outputs["strip"], outputs["point"],
                alpha=float(profile.get("fusion_alpha", 0.5)),
            )
        pred = torch.argmax(fused, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        result = {"merged": pred}

    else:
        raise ValueError(f"未知 model_type: {typ}")

    sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


# ============================================================
# Metadata / 统计
# ============================================================
def sanitize_meta(meta: dict) -> dict:
    return {
        k: v for k, v in meta.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }


def checkpoint_info(path: str | Path) -> dict:
    p = Path(path)
    info = {
        "path": str(p.resolve()),
        "size_mb": float(p.stat().st_size / (1024 ** 2)),
        "sha256": None,
    }
    if CONFIG.get("hash_checkpoint_sha256", False):
        print(f"计算 SHA256: {p}")
        info["sha256"] = file_sha256(p)
    return info


def profile_model_stats(models: dict, profile: dict) -> dict:
    typ = profile["model_type"].lower()
    if typ == "legacy_dual":
        strip_params = int(sum(p.numel() for p in models["strip_model"].parameters()))
        point_params = int(sum(p.numel() for p in models["point_model"].parameters()))
        return {
            "strip_params": strip_params,
            "point_params": point_params,
            "total_params": strip_params + point_params,
            "total_params_m": (strip_params + point_params) / 1e6,
            "checkpoints": {
                "strip": checkpoint_info(profile["strip_checkpoint"]),
                "point": checkpoint_info(profile["point_checkpoint"]),
            },
            "strip_loaded_source": models.get("strip_loaded_source"),
            "point_loaded_source": models.get("point_loaded_source"),
            "strip_checkpoint_metadata": sanitize_meta(models.get("strip_checkpoint_meta", {})),
            "point_checkpoint_metadata": sanitize_meta(models.get("point_checkpoint_meta", {})),
        }

    model = models["model"]
    total_params = int(sum(p.numel() for p in model.parameters()))
    return {
        "total_params": total_params,
        "total_params_m": total_params / 1e6,
        "checkpoints": {"main": checkpoint_info(profile["checkpoint"])},
        "loaded_source": models.get("loaded_source"),
        "checkpoint_metadata": sanitize_meta(models.get("checkpoint_meta", {})),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================
# 单 eval_set 导出
# ============================================================
def run_eval_set(
    tag: str,
    profile: dict,
    eval_name: str,
    eval_spec: dict,
    models: dict,
    model_stats: dict,
    device: torch.device,
    args,
):
    image_dir = Path(eval_spec["image_dir"])
    images = list_images(image_dir)
    manifest_rows = validate_eval_set(eval_name, eval_spec, images)

    output_root = Path(args.output_root or CONFIG["output_root"])
    out_dir = output_root / tag / eval_name
    pred_dir = out_dir / "predictions"
    ensure_dir(pred_dir)

    typ = profile["model_type"].lower()
    strip_dir = out_dir / "strip_predictions_raw"
    point_dir = out_dir / "point_predictions_raw"
    if typ == "legacy_dual":
        ensure_dir(strip_dir)
        ensure_dir(point_dir)

    print("=" * 100)
    print(f"Profile      : {tag}")
    print(f"Eval set     : {eval_name}")
    print(f"Model type   : {typ}")
    print(f"Token mode   : {profile.get('token_mode')}")
    print(f"GT scheme    : {eval_spec.get('gt_scheme')}")
    print(f"Image dir    : {image_dir}")
    print(f"Mask dir     : {eval_spec.get('mask_dir')}")
    print(f"Output       : {out_dir}")
    print(f"Images       : {len(images)}")
    print(f"Device       : {device}")
    print("=" * 100)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    runtime_rows = []
    latency_after_warmup = []
    overwrite = bool(args.overwrite or CONFIG["overwrite"])
    expected_hw_raw = profile.get("expected_hw", CONFIG["expected_hw"])
    expected_hw = tuple(expected_hw_raw) if expected_hw_raw is not None else None

    for i, src in enumerate(images, 1):
        dst = pred_dir / f"{src.stem}.png"
        strip_dst = strip_dir / f"{src.stem}.png"
        point_dst = point_dir / f"{src.stem}.png"

        all_exist = dst.exists()
        if typ == "legacy_dual":
            all_exist = all_exist and strip_dst.exists() and point_dst.exists()
        if all_exist and not overwrite:
            print(f"[{i}/{len(images)}] skip existing: {src.name}")
            continue

        bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"读取图片失败：{src}")

        hw = tuple(int(v) for v in bgr.shape[:2])
        if expected_hw is not None and hw != expected_hw:
            raise ValueError(f"图片尺寸 {hw} != expected_hw={expected_hw}：{src.name}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result, ms = infer_one(models, profile, rgb, device)

        merged = result["merged"]
        if merged.shape != bgr.shape[:2]:
            raise RuntimeError(f"预测尺寸 {merged.shape} != 输入尺寸 {bgr.shape[:2]}：{src.name}")
        validate_prediction_ids(merged, profile["prediction_scheme"], f"{tag}/{eval_name}/{src.name}")
        save_mask(dst, merged)

        if typ == "legacy_dual":
            strip_raw = result["strip_raw"]
            point_raw = result["point_raw"]
            if int(strip_raw.max()) > 8:
                raise RuntimeError(f"strip raw 超出 0..8：{src.name}")
            if int(point_raw.max()) > 3:
                raise RuntimeError(f"point raw 超出 0..3：{src.name}")
            save_mask(strip_dst, strip_raw)
            save_mask(point_dst, point_raw)

        runtime_rows.append({
            "image": src.name,
            "inference_ms": ms,
        })
        if i > int(CONFIG["latency_warmup_images"]):
            latency_after_warmup.append(ms)

        if i % 10 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {src.name}  {ms:.2f} ms")

    # 最终文件完整性校验
    missing_pred = [p.name for p in images if not (pred_dir / f"{p.stem}.png").exists()]
    if missing_pred:
        raise RuntimeError(f"导出后缺少 {len(missing_pred)} 个 merged prediction，例如：{missing_pred[:5]}")

    if typ == "legacy_dual":
        missing_strip = [p.name for p in images if not (strip_dir / f"{p.stem}.png").exists()]
        missing_point = [p.name for p in images if not (point_dir / f"{p.stem}.png").exists()]
        if missing_strip or missing_point:
            raise RuntimeError(
                f"C raw prediction 不完整：strip missing={len(missing_strip)}, point missing={len(missing_point)}"
            )

    # manifest 加预测路径
    for row in manifest_rows:
        stem = row["stem"]
        row["merged_prediction_path"] = str((pred_dir / f"{stem}.png").resolve())
        row["strip_prediction_path"] = (
            str((strip_dir / f"{stem}.png").resolve()) if typ == "legacy_dual" else ""
        )
        row["point_prediction_path"] = (
            str((point_dir / f"{stem}.png").resolve()) if typ == "legacy_dual" else ""
        )

    write_csv(
        out_dir / "manifest.csv",
        manifest_rows,
        fieldnames=[
            "image", "stem", "image_path", "gt_mask_path", "gt_scheme",
            "merged_prediction_path", "strip_prediction_path", "point_prediction_path",
        ],
    )

    if runtime_rows:
        write_csv(
            out_dir / "runtime_per_image.csv",
            runtime_rows,
            fieldnames=["image", "inference_ms"],
        )

    avg_ms = float(np.mean(latency_after_warmup)) if latency_after_warmup else None
    p95_ms = float(np.percentile(latency_after_warmup, 95)) if latency_after_warmup else None
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda" else 0.0
    )

    metadata = {
        "profile": tag,
        "eval_set": eval_name,
        "model_type": profile["model_type"],
        "token_mode": profile.get("token_mode"),
        "prediction_scheme": profile.get("prediction_scheme"),
        "gt_scheme": eval_spec.get("gt_scheme"),
        "is_common_benchmark": eval_name == "common_unified12_val",
        "image_dir": str(image_dir.resolve()),
        "mask_dir": str(Path(eval_spec["mask_dir"]).resolve()) if eval_spec.get("mask_dir") else None,
        "num_images": len(images),
        "expected_hw": expected_hw,
        "mean": CONFIG["mean"],
        "std": CONFIG["std"],
        "use_amp": CONFIG["use_amp"],
        "git_commit_at_export": git_commit(Path(".")),
        "export_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "avg_inference_ms": avg_ms,
        "p95_inference_ms": p95_ms,
        "peak_gpu_memory_mb": peak_mb,
        "notes": profile.get("notes", ""),
        "model_stats": model_stats,
        "prediction_format": "uint8 PNG class-index mask",
    }

    if typ == "legacy_dual":
        metadata.update({
            "strip_num_classes": int(profile["strip_num_classes"]),
            "point_num_classes": int(profile["point_num_classes"]),
            "confidence_threshold": float(profile.get("confidence_threshold", 0.1)),
            "mapping_strip": {str(k): int(v) for k, v in MAPPING_STRIP.items()},
            "mapping_point": {str(k): int(v) for k, v in MAPPING_POINT.items()},
            "merge_rule": "point remapped nonzero pixels override strip remapped pixels",
            "saved_outputs": {
                "merged": "predictions/*.png (unified12 IDs 0..11)",
                "strip_raw": "strip_predictions_raw/*.png (local IDs 0..8)",
                "point_raw": "point_predictions_raw/*.png (local IDs 0..3)",
            },
            "evaluation_note": (
                "This is the common unified12 benchmark. merged prediction can be scored directly in unified12; "
                "for the four-model canonical comparison, fold IDs 7..11 to Background in the evaluator."
            ),
        })

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n导出完成")
    print(f"merged prediction : {pred_dir}")
    if typ == "legacy_dual":
        print(f"strip raw         : {strip_dir}")
        print(f"point raw         : {point_dir}")
    print(f"manifest          : {out_dir / 'manifest.csv'}")
    print(f"metadata          : {out_dir / 'metadata.json'}")
    if avg_ms is not None:
        print(f"latency avg/p95   : {avg_ms:.2f}/{p95_ms:.2f} ms")


# ============================================================
# Profile 入口
# ============================================================
def run_profile(tag: str, profile: dict, args):
    device = device_from_config()

    # CLI 临时覆盖 checkpoint：只允许单模型 profile，避免 C 两权重歧义。
    if args.checkpoint:
        if profile["model_type"].lower() == "legacy_dual":
            raise ValueError("C_legacy_dual 有两个 checkpoint，请直接在 CONFIG 中填写，不支持 --checkpoint 单值覆盖")
        profile = dict(profile)
        profile["checkpoint"] = args.checkpoint

    eval_sets = profile.get("eval_sets", {})
    if not eval_sets:
        raise ValueError(f"profile={tag} 未配置 eval_sets")

    selected_names = list(eval_sets.keys())
    if args.eval_set:
        if args.eval_set not in eval_sets:
            raise ValueError(
                f"profile={tag} 没有 eval_set={args.eval_set}; 可选={list(eval_sets.keys())}"
            )
        selected_names = [args.eval_set]

    print("加载模型...")
    models = load_profile_models(profile, device)
    model_stats = profile_model_stats(models, profile)

    for eval_name in selected_names:
        run_eval_set(
            tag=tag,
            profile=profile,
            eval_name=eval_name,
            eval_spec=eval_sets[eval_name],
            models=models,
            model_stats=model_stats,
            device=device,
            args=args,
        )

    # profile 级索引，方便后续 evaluator 自动发现所有 eval_set
    output_root = Path(args.output_root or CONFIG["output_root"])
    profile_dir = output_root / tag
    ensure_dir(profile_dir)
    profile_index = {
        "profile": tag,
        "model_type": profile["model_type"],
        "token_mode": profile.get("token_mode"),
        "prediction_scheme": profile.get("prediction_scheme"),
        "eval_sets": selected_names,
        "all_configured_eval_sets": list(eval_sets.keys()),
        "notes": profile.get("notes", ""),
    }
    with open(profile_dir / "profile_index.json", "w", encoding="utf-8") as f:
        json.dump(profile_index, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Export frozen predictions for A/B/C/D historical comparison")
    p.add_argument("--profile", required=True, choices=sorted(CONFIG["profiles"].keys()))
    p.add_argument("--eval-set", default=None, help="只运行该 profile 的某一个 eval_set")
    p.add_argument("--output-root", default=None)
    p.add_argument("--checkpoint", default=None, help="仅 A/B/D：临时覆盖 CONFIG 中单个 checkpoint")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    profile = dict(CONFIG["profiles"][args.profile])
    run_profile(args.profile, profile, args)


if __name__ == "__main__":
    main()