"""
DINOv3 红墨水识别推理模块。

功能：
- 加载 DINOv3Seg 分割模型；
- 对输入图像执行滑窗推理；
- 使用前景并集规则融合重叠区域；
- 读取 JSON 中的 model_detail 配置；
- 输出单通道类别索引 mask。
"""

import os
import math
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from pred.pred_base import PreditionBase


# ==========================================================
# 推理常量
# ==========================================================

_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)
_GRAY = (128, 128, 128)

# ==========================================================
# 滑窗位置
# ==========================================================

def _sliding_positions(length, window_size, stride):
    """固定步长，并保证最后一个窗口贴住图像边缘。"""
    if length <= window_size:
        return [0]

    positions = list(range(0, length - window_size + 1, stride))
    last = length - window_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def _adaptive_positions(length, window_size):
    """自适应均匀分布窗口：首尾贴边，中间均匀重叠。"""
    if length <= window_size:
        return [0]

    span = length - window_size
    n = math.ceil(span / window_size) + 1
    n = max(n, 2)

    step = span / (n - 1)
    positions = [round(i * step) for i in range(n - 1)]
    positions.append(span)

    unique = [positions[0]]
    for pos in positions[1:]:
        if pos - unique[-1] >= 2:
            unique.append(pos)
        else:
            unique[-1] = max(unique[-1], pos)

    return unique


# ==========================================================
# DINO PTH 推理引擎
# ==========================================================

class DinoEngine:
    """
    红墨水 DINOv3 PTH 推理引擎。

    使用 DINOv3Seg 进行 FP16 推理，并采用 ImageNet mean/std 归一化。
    """

    def __init__(self, model_path, params):
        self.params = params
        use_cuda = torch.cuda.is_available() and bool(params.get("cuda", 1))
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        import sys
        import numpy as np

        # NumPy checkpoint 兼容
        sys.modules.setdefault("numpy._core", np)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

        # 当前文件所在目录
        self_dir = os.path.dirname(os.path.abspath(__file__))

        # DINOv3 模型代码目录
        project_path = self.params.get("project_path", os.path.join(self_dir, "pred_dinov3"))

        if project_path not in sys.path:
            sys.path.insert(0, project_path)

        try:
            from models.dinov3_segmentation import DINOv3Seg
        except ImportError as e:
            raise ImportError(f"找不到 DINOv3 项目代码，当前 project_path={project_path}") from e

        cfg = SimpleNamespace(
            backbone_name=self.params.get("backbone_name", "dinov3_model"),
            freeze_backbone=True,
            num_classes=int(self.params.get("num_classes", 5)),
        )

        model = DINOv3Seg(cfg).half().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        model.device = self.device
        print(f"[DinoEngine] loaded={model_path} device={self.device}")
        return model

    def _preprocess(self, image_rgb):
        """
        image_rgb: [H,W,3] uint8
        return: [1,3,H,W] FP16
        """
        image = image_rgb.astype(np.float32) / 255.0
        batch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).half().to(self.device)
        mean = _MEAN.to(self.device)
        std = _STD.to(self.device)

        return (batch - mean) / std

    def infer(self, image_rgb):
        """
        单窗口推理。

        Returns:
            probs: [C,H,W] float32 numpy
        """
        batch = self._preprocess(image_rgb)

        with torch.no_grad():
            seg, _ = self.model(batch)
            probs = torch.softmax(seg, dim=1)

        return probs[0].float().cpu().numpy()


# ==========================================================
# 滑窗推理
# ==========================================================

def sliding_detect(engine, image_rgb, params):
    """
    滑窗推理 + 前景并集融合。

    融合规则：
    1. stride > 0：固定步长；
       stride == 0：自适应窗口；
    2. 当前窗口背景不覆盖已有前景；
    3. 当前可靠前景可覆盖背景；
    4. 前景重叠时保留前景概率更高者；
    5. 前景概率必须达到 foreground_threshold。
    """
    height, width = image_rgb.shape[:2]

    window_size = int(params.get("input_shape", [1024, 1024])[0])
    stride = int(params.get("stride", 768))
    num_classes = int(params.get("num_classes", 5))
    foreground_threshold = float(
        params.get("foreground_threshold", 0.60)
    )

    if stride > 0:
        xs = _sliding_positions(width, window_size, stride)
        ys = _sliding_positions(height, window_size, stride)
    else:
        xs = _adaptive_positions(width, window_size)
        ys = _adaptive_positions(height, window_size)

    # 初始化整图为背景
    best_probs = np.zeros((num_classes, height, width), dtype=np.float32)
    best_probs[0] = 1.0
    best_class = np.zeros((height, width), dtype=np.uint8)
    best_fg_prob = np.zeros((height, width), dtype=np.float32)

    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + window_size, width)
            y2 = min(y1 + window_size, height)

            patch = image_rgb[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]

            # 不足窗口大小时使用 128 灰色进行 padding
            if ph != window_size or pw != window_size:
                padded = np.full((window_size, window_size, 3), _GRAY, dtype=image_rgb.dtype)
                padded[:ph, :pw] = patch
                patch = padded

            probs = engine.infer(patch)[:, :ph, :pw]

            if probs.shape[0] != num_classes:
                raise ValueError(f"模型输出通道数={probs.shape[0]}，配置 num_classes={num_classes}")

            window_class = np.argmax(probs, axis=0).astype(np.uint8)
            fg_probs = probs[1:]
            window_fg_prob = np.max(fg_probs, axis=0)
            window_fg_class = (np.argmax(fg_probs, axis=0) + 1).astype(np.uint8)
            region_class = best_class[y1:y2, x1:x2]
            region_fg_prob = best_fg_prob[y1:y2, x1:x2]
            region_probs = best_probs[:, y1:y2, x1:x2]

            # 当前像素必须真正被 argmax 判定成前景，
            # 同时达到前景阈值
            window_is_valid_fg = (window_class != 0) & (window_fg_prob >= foreground_threshold)

            # 前景并集融合
            update = window_is_valid_fg & ((region_class == 0) | (window_fg_prob > region_fg_prob))

            if np.any(update):
                region_class[update] = window_fg_class[update]
                region_fg_prob[update] = window_fg_prob[update]

                for class_id in range(num_classes):
                    region_probs[class_id][update] = probs[class_id][update]

    return best_probs


# ==========================================================
# PreditionDINO 类
# ==========================================================

class PreditionDINO(PreditionBase):
    """
    红墨水 DINOv3 推理类。

    对外接口：
        load_model(params)
        pred(img)

    img:
        OpenCV BGR uint8 图像

    默认 pred 返回：
        uint8 [H,W] 类别索引 mask
        0 background
        1 hd_w
        2 hd_y
        3 hd_t
        4 red
    """

    def __init__(self, catagory, item):
        super().__init__(catagory, item)

        self.engine = None
        self.params = None

    # ------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------

    def load_model(self, params):
        """
        params 推荐字段：

        {
            "model_name": "best_iou.pth",
            "num_classes": 5,

            "backbone_name": "dinov3_model",
            "freeze_backbone": True,
            "crop_size": 1024,
            "stride": 768,
            "confidence_threshold": 0.0,
            "foreground_threshold": 0.60,

            "cuda": 1
        }
        """
        self.params = dict(params)

        # 推理默认参数
        self.params.setdefault("num_classes", 5)
        self.params.setdefault("backbone_name", "dinov3_model")

        self.params.setdefault("input_shape", [1024, 1024])
        self.params.setdefault("stride", 768)
        self.params.setdefault("confidence_threshold", 0.0)
        self.params.setdefault("foreground_threshold", 0.60)

        self.params.setdefault("cuda", 1)

        input_shape = self.params["input_shape"]
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("input_shape 必须为 [height, width]")

        if int(self.params["num_classes"]) != 5:
            raise ValueError("当前红墨水模型为 1 个背景类 + 4 个前景类，num_classes 应为 5")

        # 支持两种形式：
        # 1. model_path 直接传完整路径
        # 2. model_name 按 pred/data/models/ 查找
        model_path = self.params.get("model_path")

        if not model_path:
            model_name = self.params["model_name"]

            if os.path.isabs(model_name):
                model_path = model_name
            else:
                self_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(self_dir, "data", "models", model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重不存在：{model_path}")

        self.params["model_path"] = model_path

        self.engine = DinoEngine(model_path, self.params)

    # ------------------------------------------------------
    # 推理
    # ------------------------------------------------------

    def pred(self, img):
        """
        输入:
            img: OpenCV BGR uint8 [H,W,3]

        返回:
            默认 uint8 [H,W] 类别索引 mask
        """
        try:
            if self.engine is None:
                raise RuntimeError("模型尚未加载，请先调用 load_model(params)")

            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            probs = sliding_detect(self.engine, image_rgb, self.params)

            confidence = np.max(probs, axis=0).astype(np.float32)

            index_mask = np.argmax(probs, axis=0).astype(np.uint8)

            # 根据置信度阈值过滤低置信度像素
            confidence_threshold = float(self.params.get("confidence_threshold", 0.0))

            index_mask[confidence < confidence_threshold] = 0

            # 默认直接返回类别索引 mask
            if self.params.get("return_confidence", 0) == 1:
                return index_mask, confidence

            return index_mask

        except Exception:
            traceback.print_exc()
            raise