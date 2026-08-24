# -*- coding: utf-8 -*-
"""
DINOv3 + DecoderV3 单张图片语义分割推理与叠加显示。

功能：
1. 输入单张图片路径；
2. 使用滑动窗口完成语义分割推理；
3. 根据 DISPLAY_CLASSES 只显示指定类别；
4. 仅保存一张“左侧原图、右侧叠加图”的结果图；
5. 模型可能预测多个类别，但可以只给其中一个或几个类别配置颜色。

注意：
- 模型输出通道数 MODEL["num_classes"] 包含背景类别 0。
- DISPLAY_CLASSES 只控制哪些类别显示颜色。
- 未配置颜色的预测类别不会着色，也不会影响左右对比图输出。
- OpenCV 使用 BGR 颜色顺序，不是 RGB。
"""

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import cv2
import numpy as np
import torch

from models.dinov3_segmentation import DINOv3Seg


# ============================================================
# 一、用户配置
# ============================================================

CONFIG = {
    # ---------- 单张图片 ----------
    "image_path": r"F:/liuhaibo/datasets/LG_JZW/20260709152249-200X-B1/06-17.jpg",

    # ---------- 输出目录 ----------
    "output_dir": r"./output/single_image_infer",

    # ---------- 模型结构：必须与训练时保持一致 ----------
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,

    # 包含背景类别 0。
    # 例如：背景 + 3 个前景类别，应填写 4。
    "num_classes": 4,

    # ---------- 权重 ----------
    "checkpoint_dir": r"./checkpoints/D_newdata",
    # 可以填写 best_iou、best_dice、last，也可以填写完整 .pth 路径。
    "checkpoint_name": "best_iou",
    # teacher / student / auto
    "weight_source": "auto",

    # ---------- 滑动窗口 ----------
    "crop_size": 784,
    # 0：自适应均匀分布并允许重叠；大于 0：固定步长。
    "stride": 0,
    # 形态学闭运算核大小，0 表示关闭。
    "infer_close_kernel": 0,

    # ---------- 设备 ----------
    "device": "cuda",

    # ---------- 叠加显示 ----------
    "alpha": 0.50,                  # 掩码颜色透明度：0.0为完全显示原图，1.0为完全显示类别颜色
    "draw_contours": False,          # 是否绘制目标轮廓及类别标签：True启用，False关闭
    "contour_thickness": 0,         # 轮廓线宽，单位为像素；设置为0时不画轮廓线，但仍可显示类别标签
    "min_component_area": 10,       # 最小连通区域面积，小于该面积的区域不绘制轮廓和类别标签
    "label_font_scale": 0.65,       # 类别标签文字大小
    "label_font_thickness": 2,      # 类别标签文字线宽
    "label_padding": 4,             # 标签文字与标签背景边缘之间的留白，单位为像素

    # 输出 JPEG 时的质量。
    "jpeg_quality": 95,
}


# ============================================================
# 二、需要显示的类别、名称与颜色
# ============================================================

# 这里只配置“需要在叠加图中显示”的类别。
# 模型即使预测出其他类别，也不会报错；其他类别不会被着色或标注。
#
# 每项格式：
#     类别编号: (类别名称, BGR颜色)
#
# 示例：模型 Mask 中可能有类别 0、1、2、3、4，
# 但只想显示类别 4，则只保留：
#     4: ("D", (155, 0, 128))
DISPLAY_CLASSES: Dict[int, Tuple[str, Tuple[int, int, int]]] = {
    # 1: ("A", (255, 0, 0)),
    # 2: ("B", (0, 255, 0)),
    # 3: ("C", (0, 0, 255)),
    1: ("D", (0, 255, 0)),
    # 5: ("HH", (255, 0, 128)),
    # 6: ("XW", (155, 255, 128)),

    # 仅显示一个类别时，可注释上面的配置，只保留类似下面一项：
    # 4: ("D", (155, 0, 128)),
}

BACKGROUND_CLASS_ID = 0
GRAY = (128, 128, 128)


# ============================================================
# 三、模型配置与归一化参数
# ============================================================

MODEL_CFG = SimpleNamespace(
    backbone_name=CONFIG["backbone_name"],
    freeze_backbone=CONFIG["freeze_backbone"],
    num_classes=CONFIG["num_classes"],
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# 四、通用文件读写
# ============================================================

def read_image_unicode(image_path: Path) -> np.ndarray:
    """读取图片，兼容 Windows 中文路径。返回 BGR 图像。"""
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError(f"无法读取图片文件：{image_path}") from exc

    if data.size == 0:
        raise RuntimeError(f"图片文件为空：{image_path}")

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法解码图片，文件可能损坏：{image_path}")
    return image


def save_image_unicode(
    output_path: Path,
    image: np.ndarray,
    jpeg_quality: int = 95,
) -> None:
    """保存图片，兼容 Windows 中文路径。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    extension_map = {
        ".jpg": ".jpg",
        ".jpeg": ".jpg",
        ".png": ".png",
        ".bmp": ".bmp",
        ".tif": ".tif",
        ".tiff": ".tiff",
    }
    encode_extension = extension_map.get(suffix, ".png")

    encode_params: List[int] = []
    if encode_extension == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    success, encoded = cv2.imencode(
        encode_extension,
        image,
        encode_params,
    )
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")

    encoded.tofile(str(output_path))


# ============================================================
# 五、配置检查
# ============================================================

def validate_config() -> None:
    num_classes = int(CONFIG["num_classes"])
    crop_size = int(CONFIG["crop_size"])
    stride = int(CONFIG["stride"])
    alpha = float(CONFIG["alpha"])

    if num_classes <= 1:
        raise ValueError("num_classes 至少应为 2：背景类别 + 至少一个前景类别。")
    if crop_size <= 0:
        raise ValueError("crop_size 必须大于 0。")
    if stride < 0:
        raise ValueError("stride 不能小于 0。")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须位于 0.0 到 1.0 之间。")
    if int(CONFIG["infer_close_kernel"]) < 0:
        raise ValueError("infer_close_kernel 不能小于 0。")

    for class_id, class_config in DISPLAY_CLASSES.items():
        if not isinstance(class_id, int):
            raise TypeError(f"显示类别编号必须是整数：{class_id!r}")
        if class_id == BACKGROUND_CLASS_ID:
            raise ValueError("DISPLAY_CLASSES 不需要配置背景类别 0。")
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"显示类别 {class_id} 超出模型输出范围；"
                f"当前有效类别编号为 0 到 {num_classes - 1}。"
            )

        if not isinstance(class_config, tuple) or len(class_config) != 2:
            raise ValueError(
                f"类别 {class_id} 的配置必须是 (名称, BGR颜色)：{class_config!r}"
            )

        class_name, bgr_color = class_config
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(f"类别 {class_id} 的名称不能为空。")
        if len(bgr_color) != 3:
            raise ValueError(f"类别 {class_id} 的颜色必须包含 3 个通道。")
        if any(int(channel) < 0 or int(channel) > 255 for channel in bgr_color):
            raise ValueError(f"类别 {class_id} 的颜色值必须位于 0 到 255。")


# ============================================================
# 六、权重路径与模型加载
# ============================================================

def resolve_checkpoint_path() -> Path:
    checkpoint_name = str(CONFIG["checkpoint_name"])
    direct_path = Path(checkpoint_name)

    # 用户直接填写了存在的完整路径。
    if direct_path.is_file():
        return direct_path

    checkpoint_dir = Path(CONFIG["checkpoint_dir"])

    # 没有扩展名时，自动补 .pth。
    candidate_name = checkpoint_name
    if Path(candidate_name).suffix.lower() != ".pth":
        candidate_name += ".pth"

    candidate = checkpoint_dir / candidate_name
    if candidate.is_file():
        return candidate

    # 与原代码一致，找不到指定权重时按顺序回退。
    for fallback_name in ("best_dice.pth", "best_iou.pth", "last.pth"):
        fallback = checkpoint_dir / fallback_name
        if fallback.is_file():
            print(f"[提示] 未找到指定权重，自动使用：{fallback}")
            return fallback

    raise FileNotFoundError(
        "未找到模型权重。\n"
        f"checkpoint_name：{CONFIG['checkpoint_name']}\n"
        f"checkpoint_dir：{checkpoint_dir.resolve()}"
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """加载 teacher、student 或 model 权重。"""
    model = DINOv3Seg(MODEL_CFG).to(device)

    # CUDA 使用 FP16；CPU 使用 FP32，避免部分 CPU 算子不支持 float16。
    if device.type == "cuda":
        model = model.half()

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    weight_source = str(CONFIG["weight_source"]).lower()
    if weight_source not in {"teacher", "student", "auto"}:
        raise ValueError("weight_source 只能是 teacher、student 或 auto。")

    if weight_source in {"teacher", "auto"}:
        priority = ["teacher", "student", "model"]
    else:
        priority = ["student", "teacher", "model"]

    loaded_source = None
    if isinstance(checkpoint, dict):
        for key in priority:
            if key in checkpoint:
                model.load_state_dict(checkpoint[key], strict=True)
                loaded_source = key
                break

    # 兼容权重文件本身就是 state_dict 的情况。
    if loaded_source is None:
        try:
            model.load_state_dict(checkpoint, strict=True)
            loaded_source = "state_dict"
        except Exception as exc:
            keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
            raise KeyError(
                f"无法从权重中找到可加载参数。checkpoint keys/type={keys}"
            ) from exc

    model.eval()

    epoch = checkpoint.get("epoch", "?") if isinstance(checkpoint, dict) else "?"
    iou = checkpoint.get("iou", 0.0) if isinstance(checkpoint, dict) else 0.0
    dice = checkpoint.get("dice", 0.0) if isinstance(checkpoint, dict) else 0.0

    print(f"权重路径：{checkpoint_path}")
    print(
        f"权重来源：{loaded_source}，epoch={epoch}，"
        f"IoU={float(iou):.4f}，Dice={float(dice):.4f}"
    )
    return model


def print_model_info(model: torch.nn.Module, checkpoint_path: Path) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    checkpoint_bytes = checkpoint_path.stat().st_size

    print("=" * 80)
    print(f"参数量：{total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"模型参数大小：{model_bytes / (1024 ** 2):.2f} MB")
    print(f"权重文件大小：{checkpoint_bytes / (1024 ** 2):.2f} MB")
    print("=" * 80)


# ============================================================
# 七、滑动窗口推理
# ============================================================

def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def preprocess_batch(
    images_rgb: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """images_rgb: [N,H,W,3]，取值范围 0~255。"""
    images = images_rgb.astype(np.float32) / 255.0
    images = (images - IMAGENET_MEAN) / IMAGENET_STD

    batch = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    if device.type == "cuda":
        batch = batch.half()
    else:
        batch = batch.float()
    return batch


def predict_probabilities(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> np.ndarray:
    """水平翻转 TTA，返回 [N,C,H,W] 概率。"""
    with torch.inference_mode():
        segmentation, _ = model(batch)
        probabilities = torch.softmax(segmentation, dim=1)

        flipped_batch = torch.flip(batch, dims=[3])
        flipped_segmentation, _ = model(flipped_batch)
        flipped_probabilities = torch.softmax(flipped_segmentation, dim=1)
        flipped_probabilities = torch.flip(flipped_probabilities, dims=[3])

        probabilities = (probabilities + flipped_probabilities) / 2.0

    return probabilities.float().cpu().numpy()


def fixed_positions(
    length: int,
    window_size: int,
    stride: int,
) -> List[int]:
    """固定步长窗口位置；末端窗口不足时在推理阶段补灰。"""
    if length <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError(
            f"固定窗口参数不合法：length={length}, "
            f"window_size={window_size}, stride={stride}"
        )
    return list(range(0, length, stride))


def adaptive_positions(length: int, window_size: int) -> List[int]:
    """首尾贴边、中间均匀分布的自适应窗口位置。"""
    if length <= 0 or window_size <= 0:
        raise ValueError(
            f"自适应窗口参数不合法：length={length}, window_size={window_size}"
        )
    if length <= window_size:
        return [0]

    span = length - window_size
    window_count = max(math.ceil(span / window_size) + 1, 2)
    step = span / (window_count - 1)

    positions = [round(index * step) for index in range(window_count - 1)]
    positions.append(span)

    unique_positions = [positions[0]]
    for position in positions[1:]:
        if position - unique_positions[-1] >= 2:
            unique_positions.append(position)
        else:
            unique_positions[-1] = max(unique_positions[-1], position)
    return unique_positions


def sliding_window_infer(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """返回 [C,H,W] 的整图概率。重叠位置使用概率平均。"""
    height, width = image_rgb.shape[:2]
    window_size = int(CONFIG["crop_size"])
    stride = int(CONFIG["stride"])
    num_classes = int(CONFIG["num_classes"])

    if stride > 0:
        x_positions = fixed_positions(width, window_size, stride)
        y_positions = fixed_positions(height, window_size, stride)
        mode_name = f"固定步长 stride={stride}"
    else:
        x_positions = adaptive_positions(width, window_size)
        y_positions = adaptive_positions(height, window_size)
        mode_name = "自适应均匀分布"

    print(
        f"滑窗模式：{mode_name}；网格："
        f"{len(y_positions)} 行 x {len(x_positions)} 列；"
        f"原图尺寸：{width}x{height}"
    )

    probability_sum = np.zeros(
        (num_classes, height, width),
        dtype=np.float32,
    )
    probability_count = np.zeros((height, width), dtype=np.float32)

    for y1 in y_positions:
        for x1 in x_positions:
            x2 = min(x1 + window_size, width)
            y2 = min(y1 + window_size, height)

            patch = image_rgb[y1:y2, x1:x2]
            patch_height, patch_width = patch.shape[:2]

            if patch_height != window_size or patch_width != window_size:
                padded_patch = np.full(
                    (window_size, window_size, 3),
                    GRAY,
                    dtype=image_rgb.dtype,
                )
                padded_patch[:patch_height, :patch_width] = patch
                patch = padded_patch

            batch = preprocess_batch(
                np.expand_dims(patch, axis=0),
                device,
            )
            patch_probabilities = predict_probabilities(model, batch)[0]

            if patch_probabilities.shape[0] != num_classes:
                raise RuntimeError(
                    f"模型实际输出通道数为 {patch_probabilities.shape[0]}，"
                    f"但 CONFIG['num_classes']={num_classes}。"
                )

            probability_sum[:, y1:y2, x1:x2] += (
                patch_probabilities[:, :patch_height, :patch_width]
            )
            probability_count[y1:y2, x1:x2] += 1.0

    if np.any(probability_count == 0):
        raise RuntimeError("滑动窗口未覆盖完整图片，请检查 crop_size 和 stride。")

    return probability_sum / probability_count[None, :, :]


def enhance_connectivity(
    index_mask: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """按类别执行闭运算，不覆盖原本已有的其他前景类别。"""
    if kernel_size <= 0:
        return index_mask

    if kernel_size % 2 == 0:
        print(f"[提示] 闭运算核 {kernel_size} 为偶数，自动调整为 {kernel_size + 1}。")
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    result = index_mask.copy()
    occupied_foreground = index_mask > BACKGROUND_CLASS_ID

    for class_id in range(1, int(CONFIG["num_classes"])):
        binary_mask = (index_mask == class_id).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        new_pixels = (
            (closed > 0)
            & (~occupied_foreground)
            & (result == BACKGROUND_CLASS_ID)
        )
        result[new_pixels] = class_id

    return result


def infer_single_image(
    model: torch.nn.Module,
    original_bgr: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回完整类别索引 Mask 和置信度图。"""
    image_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    probabilities = sliding_window_infer(model, image_rgb, device)
    index_mask = np.argmax(probabilities, axis=0).astype(np.uint8)
    index_mask = enhance_connectivity(
        index_mask,
        int(CONFIG["infer_close_kernel"]),
    )
    confidence = np.max(probabilities, axis=0).astype(np.float32)
    return index_mask, confidence


# ============================================================
# 八、叠加图与标签
# ============================================================

def choose_text_color(
    bgr_color: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    blue, green, red = bgr_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness >= 150 else (255, 255, 255)


def draw_class_contours_and_labels(
    image_bgr: np.ndarray,
    index_mask: np.ndarray,
) -> np.ndarray:
    result = image_bgr.copy()
    image_height, image_width = result.shape[:2]

    for class_id, (class_name, bgr_color) in DISPLAY_CLASSES.items():
        binary_mask = (index_mask == class_id).astype(np.uint8) * 255
        if not np.any(binary_mask):
            continue

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < float(CONFIG["min_component_area"]):
                continue

            contour_thickness = int(CONFIG["contour_thickness"])
            if contour_thickness > 0:
                cv2.drawContours(
                    result,
                    [contour],
                    contourIdx=-1,
                    color=bgr_color,
                    thickness=contour_thickness,
                    lineType=cv2.LINE_AA,
                )

            x, y, width, height = cv2.boundingRect(contour)
            font_scale = float(CONFIG["label_font_scale"])
            font_thickness = int(CONFIG["label_font_thickness"])
            padding = int(CONFIG["label_padding"])

            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )

            label_width = text_width + padding * 2
            label_height = text_height + baseline + padding * 2

            label_x1 = max(0, min(x, max(0, image_width - label_width)))
            if y >= label_height:
                label_y1 = y - label_height
            else:
                label_y1 = min(
                    y + height,
                    max(0, image_height - label_height),
                )

            label_x2 = min(image_width - 1, label_x1 + label_width)
            label_y2 = min(image_height - 1, label_y1 + label_height)

            cv2.rectangle(
                result,
                (label_x1, label_y1),
                (label_x2, label_y2),
                bgr_color,
                thickness=-1,
            )

            cv2.putText(
                result,
                class_name,
                (label_x1 + padding, label_y1 + padding + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                choose_text_color(bgr_color),
                font_thickness,
                cv2.LINE_AA,
            )

    return result


def create_overlay(
    original_bgr: np.ndarray,
    index_mask: np.ndarray,
) -> np.ndarray:
    """只对 DISPLAY_CLASSES 中配置的类别进行着色。"""
    overlay = original_bgr.copy()
    alpha = float(CONFIG["alpha"])

    for class_id, (_, bgr_color) in DISPLAY_CLASSES.items():
        class_region = index_mask == class_id
        if not np.any(class_region):
            continue

        original_pixels = overlay[class_region].astype(np.float32)
        color = np.asarray(bgr_color, dtype=np.float32)
        blended = original_pixels * (1.0 - alpha) + color * alpha
        overlay[class_region] = np.clip(blended, 0, 255).astype(np.uint8)

    if bool(CONFIG["draw_contours"]):
        overlay = draw_class_contours_and_labels(overlay, index_mask)

    return overlay




# ============================================================
# 九、输出保存
# ============================================================

def save_output(
    image_path: Path,
    original_bgr: np.ndarray,
    index_mask: np.ndarray,
) -> Path:
    """仅保存左侧原图、右侧叠加图。"""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_comparison.jpg"

    overlay = create_overlay(original_bgr, index_mask)
    comparison = np.hstack((original_bgr, overlay))

    save_image_unicode(
        output_path,
        comparison,
        jpeg_quality=int(CONFIG["jpeg_quality"]),
    )
    return output_path



# ============================================================
# 十、主程序
# ============================================================

def main() -> None:
    validate_config()

    image_path = Path(CONFIG["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在：{image_path}")

    requested_device = str(CONFIG["device"]).lower()
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
    else:
        if requested_device.startswith("cuda"):
            print("[提示] CUDA 不可用，自动改用 CPU。")
        device = torch.device("cpu")

    checkpoint_path = resolve_checkpoint_path()
    model = load_model(checkpoint_path, device)
    print_model_info(model, checkpoint_path)

    original_bgr = read_image_unicode(image_path)
    image_height, image_width = original_bgr.shape[:2]

    print(f"输入图片：{image_path}")
    print(f"图片尺寸：{image_width}x{image_height}")
    print(f"推理设备：{device}")
    print(f"模型类别编号：0 到 {int(CONFIG['num_classes']) - 1}")
    print(f"叠加显示类别：{sorted(DISPLAY_CLASSES.keys())}")
    print("=" * 80)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_device(device)
    start_time = time.perf_counter()

    index_mask, _ = infer_single_image(
        model,
        original_bgr,
        device,
    )

    synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    class_values, class_counts = np.unique(index_mask, return_counts=True)
    class_statistics = dict(
        zip(class_values.tolist(), class_counts.tolist())
    )

    output_path = save_output(
        image_path,
        original_bgr,
        index_mask,
    )

    print("=" * 80)
    print(f"推理耗时：{elapsed_ms:.2f} ms")
    print(f"Mask 类别统计：{class_statistics}")

    predicted_classes = set(int(value) for value in class_values.tolist())
    displayed_classes = predicted_classes.intersection(DISPLAY_CLASSES.keys())
    hidden_classes = sorted(
        predicted_classes
        - {BACKGROUND_CLASS_ID}
        - set(DISPLAY_CLASSES.keys())
    )

    print(f"实际着色类别：{sorted(displayed_classes)}")
    if hidden_classes:
        print(
            f"未着色但保留在完整 Mask 中的类别：{hidden_classes}"
        )

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU 峰值显存：{peak_memory_mb:.2f} MB")

    print(f"左右对比图：{output_path.resolve()}")
    print("=" * 80)
    print("完成。")


if __name__ == "__main__":
    main()
