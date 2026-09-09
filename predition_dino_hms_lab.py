"""
DINOv3 + Lab a* 红墨水识别推理模块。

功能：
- 加载 HMSLabRedSeg 分割模型；
- 输入仍为普通 OpenCV BGR 三通道图像；
- 内部自动生成 RGB 3 通道 + Lab a* 1 通道；
- 对输入图像执行滑窗推理；
- 使用前景并集规则融合重叠区域；
- 输出单通道类别索引 mask。

类别：
    0: background
    1: hd_w
    2: hd_y
    3: hd_t
    4: red
"""

import math
import os
import sys
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from pred.pred_base import PreditionBase


_RGB_MEAN = (0.485, 0.456, 0.406)
_RGB_STD = (0.229, 0.224, 0.225)
_GRAY = (128, 128, 128)


def _sliding_positions(length, window_size, stride):
    if length <= window_size:
        return [0]

    positions = list(range(0, length - window_size + 1, stride))
    last = length - window_size

    if positions[-1] != last:
        positions.append(last)

    return positions


def _adaptive_positions(length, window_size):
    if length <= window_size:
        return [0]

    span = length - window_size
    n = max(math.ceil(span / window_size) + 1, 2)
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


class DinoEngine:
    """
    HMS Lab-Red DINOv3 推理引擎。

    外部输入：
        RGB [H, W, 3]

    内部模型输入：
        RGB    [1, 3, H, W]
        Lab a* [1, 1, H, W]
    """

    def __init__(self, model_path, params):
        self.params = params

        use_cuda = torch.cuda.is_available() and bool(params.get("cuda", 1))
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.use_fp16 = self.device.type == "cuda" and bool(params.get("fp16", 1))

        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.mean = torch.tensor(_RGB_MEAN, dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_RGB_STD, dtype=dtype, device=self.device).view(1, 3, 1, 1)

        self.model = self._load_model(model_path)

    def _resolve_project_path(self):
        self_dir = os.path.dirname(os.path.abspath(__file__))

        project_path = self.params.get("project_path")
        if not project_path:
            project_path = os.path.join(self_dir, "pred_dinov3")

        project_path = os.path.abspath(project_path)

        if not os.path.isdir(project_path):
            raise FileNotFoundError(f"DINOv3/HMS 项目目录不存在：{project_path}")

        if project_path not in sys.path:
            sys.path.insert(0, project_path)

        return project_path

    def _resolve_backbone_name(self, project_path):
        backbone_name = str(self.params.get("backbone_name", "dinov3_model"))

        if os.path.isabs(backbone_name):
            return backbone_name

        local_backbone = os.path.join(project_path, backbone_name)
        if os.path.exists(local_backbone):
            return local_backbone

        return backbone_name

    def _build_model_config(self, project_path):
        return SimpleNamespace(
            backbone_name=self._resolve_backbone_name(project_path),
            freeze_backbone=bool(self.params.get("freeze_backbone", True)),
            num_classes=int(self.params.get("num_classes", 5)),
            red_class=int(self.params.get("red_class", 4)),
            lab_channels=int(self.params.get("lab_channels", 32)),
            lab_fusion_init=float(self.params.get("lab_fusion_init", 0.5)),
        )

    @staticmethod
    def _extract_state_dict(ckpt):
        if not isinstance(ckpt, dict):
            return ckpt, "raw"

        for key in ("model", "state_dict", "student", "teacher"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value, key

        tensor_values = [value for value in ckpt.values() if torch.is_tensor(value)]
        if tensor_values:
            return ckpt, "raw_dict"

        raise KeyError(f"无法从 checkpoint 中找到模型参数，keys={list(ckpt.keys())}")

    @staticmethod
    def _strip_module_prefix(state_dict):
        if not state_dict:
            return state_dict

        if all(key.startswith("module.") for key in state_dict.keys()):
            return {key[7:]: value for key, value in state_dict.items()}

        return state_dict

    def _load_model(self, model_path):
        sys.modules.setdefault("numpy._core", np)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

        project_path = self._resolve_project_path()

        try:
            from HMS.model import HMSLabRedSeg
        except ImportError as e:
            raise ImportError(
                f"找不到 HMSLabRedSeg，请检查 project_path={project_path} 下是否存在 HMS/model.py"
            ) from e

        cfg = self._build_model_config(project_path)
        model = HMSLabRedSeg(cfg).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict, source = self._extract_state_dict(ckpt)
        state_dict = self._strip_module_prefix(state_dict)

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        if self.use_fp16:
            model.half()
        else:
            model.float()

        model.device = self.device

        print(
            f"[DinoEngine] loaded={model_path} "
            f"source={source} "
            f"device={self.device} "
            f"dtype={'fp16' if self.use_fp16 else 'fp32'}"
        )

        return model

    def _preprocess_rgb(self, image_rgb):
        image = image_rgb.astype(np.float32) / 255.0
        batch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.use_fp16:
            batch = batch.half()
        else:
            batch = batch.float()

        return (batch - self.mean) / self.std

    def _preprocess_lab_a(self, image_rgb):
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        lab_a = lab[:, :, 1].astype(np.float32)

        lab_a = (lab_a - 128.0) / 127.0
        lab_a = np.clip(lab_a, -1.0, 1.0)

        batch = torch.from_numpy(lab_a).unsqueeze(0).unsqueeze(0).to(self.device)

        if self.use_fp16:
            batch = batch.half()
        else:
            batch = batch.float()

        return batch

    def _preprocess(self, image_rgb):
        rgb_batch = self._preprocess_rgb(image_rgb)
        lab_a_batch = self._preprocess_lab_a(image_rgb)
        return rgb_batch, lab_a_batch

    def infer(self, image_rgb):
        rgb_batch, lab_a_batch = self._preprocess(image_rgb)

        with torch.inference_mode():
            outputs = self.model(rgb_batch, lab_a_batch)
            seg = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            probs = torch.softmax(seg, dim=1)

        return probs[0].float().cpu().numpy()


def sliding_detect(engine, image_rgb, params):
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb 必须为 [H, W, 3] RGB 图像")

    height, width = image_rgb.shape[:2]

    input_shape = params.get("input_shape", [1024, 1024])
    window_height = int(input_shape[0])
    window_width = int(input_shape[1])

    if window_height != window_width:
        raise ValueError(f"当前滑窗实现要求 input_shape 为正方形，当前={input_shape}")

    window_size = window_height
    stride = int(params.get("stride", 768))
    num_classes = int(params.get("num_classes", 5))
    foreground_threshold = float(params.get("foreground_threshold", 0.60))

    if stride < 0:
        raise ValueError("stride 必须 >= 0")

    if not 0.0 <= foreground_threshold <= 1.0:
        raise ValueError("foreground_threshold 必须在 0~1 之间")

    if stride > 0:
        xs = _sliding_positions(width, window_size, stride)
        ys = _sliding_positions(height, window_size, stride)
    else:
        xs = _adaptive_positions(width, window_size)
        ys = _adaptive_positions(height, window_size)

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

            if ph != window_size or pw != window_size:
                padded = np.full((window_size, window_size, 3), _GRAY, dtype=image_rgb.dtype)
                padded[:ph, :pw] = patch
                patch = padded

            probs = engine.infer(patch)[:, :ph, :pw]

            if probs.shape[0] != num_classes:
                raise ValueError(
                    f"模型输出通道数={probs.shape[0]}，配置 num_classes={num_classes}"
                )

            window_class = np.argmax(probs, axis=0).astype(np.uint8)

            fg_probs = probs[1:]
            window_fg_prob = np.max(fg_probs, axis=0)
            window_fg_class = (np.argmax(fg_probs, axis=0) + 1).astype(np.uint8)

            region_class = best_class[y1:y2, x1:x2]
            region_fg_prob = best_fg_prob[y1:y2, x1:x2]
            region_probs = best_probs[:, y1:y2, x1:x2]

            window_is_valid_fg = (
                (window_class != 0)
                & (window_fg_prob >= foreground_threshold)
            )

            update = (
                window_is_valid_fg
                & (
                    (region_class == 0)
                    | (window_fg_prob > region_fg_prob)
                )
            )

            if np.any(update):
                region_class[update] = window_fg_class[update]
                region_fg_prob[update] = window_fg_prob[update]

                for class_id in range(num_classes):
                    region_probs[class_id][update] = probs[class_id][update]

    return best_probs


class PreditionDINO(PreditionBase):
    """
    HMS Lab-Red DINOv3 推理类。

    对外接口：
        load_model(params)
        pred(img)

    img:
        OpenCV BGR uint8 [H, W, 3]

    默认返回：
        uint8 [H, W] 类别索引 mask

    return_confidence=1 时返回：
        index_mask, confidence
    """

    def __init__(self, catagory, item):
        super().__init__(catagory, item)
        self.engine = None
        self.params = None

    def load_model(self, params):
        self.params = dict(params)

        self.params.setdefault("num_classes", 5)
        self.params.setdefault("red_class", 4)

        self.params.setdefault("backbone_name", "dinov3_model")
        self.params.setdefault("freeze_backbone", True)

        self.params.setdefault("lab_channels", 32)
        self.params.setdefault("lab_fusion_init", 0.5)

        self.params.setdefault("input_shape", [1024, 1024])
        self.params.setdefault("stride", 768)

        self.params.setdefault("confidence_threshold", 0.0)
        self.params.setdefault("foreground_threshold", 0.60)

        self.params.setdefault("cuda", 1)
        self.params.setdefault("fp16", 1)
        self.params.setdefault("return_confidence", 0)

        input_shape = self.params["input_shape"]

        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("input_shape 必须为 [height, width]")

        input_height = int(input_shape[0])
        input_width = int(input_shape[1])

        if input_height <= 0 or input_width <= 0:
            raise ValueError("input_shape 中的 height 和 width 必须 > 0")

        if input_height != input_width:
            raise ValueError(f"当前滑窗推理要求 input_shape 为正方形，当前={input_shape}")

        num_classes = int(self.params["num_classes"])
        red_class = int(self.params["red_class"])

        if num_classes != 5:
            raise ValueError("当前 HMS 模型为 1 个背景类 + 4 个前景类，num_classes 应为 5")

        if red_class != 4:
            raise ValueError("当前 HMS 红墨水类别 red_class 应为 4")

        confidence_threshold = float(self.params["confidence_threshold"])
        foreground_threshold = float(self.params["foreground_threshold"])

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 必须在 0~1 之间")

        if not 0.0 <= foreground_threshold <= 1.0:
            raise ValueError("foreground_threshold 必须在 0~1 之间")

        stride = int(self.params["stride"])

        if stride < 0:
            raise ValueError("stride 必须 >= 0")

        model_path = self.params.get("model_path")

        if not model_path:
            model_name = self.params.get("model_name")

            if not model_name:
                raise ValueError("必须提供 model_path 或 model_name")

            if os.path.isabs(model_name):
                model_path = model_name
            else:
                self_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(self_dir, "data", "models", model_name)

        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型权重不存在：{model_path}")

        self.params["model_path"] = model_path
        self.engine = DinoEngine(model_path, self.params)

    def pred(self, img):
        try:
            if self.engine is None:
                raise RuntimeError("模型尚未加载，请先调用 load_model(params)")

            if img is None:
                raise ValueError("输入图像 img 不能为 None")

            if not isinstance(img, np.ndarray):
                raise TypeError(f"img 必须为 numpy.ndarray，当前类型={type(img)}")

            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"img 必须为 BGR 三通道图像 [H,W,3]，当前 shape={img.shape}"
                )

            if img.dtype != np.uint8:
                raise ValueError(f"img dtype 必须为 uint8，当前 dtype={img.dtype}")

            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            probs = sliding_detect(self.engine, image_rgb, self.params)

            confidence = np.max(probs, axis=0).astype(np.float32)
            index_mask = np.argmax(probs, axis=0).astype(np.uint8)

            confidence_threshold = float(
                self.params.get("confidence_threshold", 0.0)
            )

            if confidence_threshold > 0.0:
                index_mask[confidence < confidence_threshold] = 0

            if int(self.params.get("return_confidence", 0)) == 1:
                return index_mask, confidence

            return index_mask

        except Exception:
            traceback.print_exc()
            raise
