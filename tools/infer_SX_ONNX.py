"""
infer_SX_onnx.py —— DINOv3 ONNX 整图缩放推理

流程：
    1. 加载一个分割模型
    2. 整图 resize 至 crop_size × crop_size 后推理，恢复至原图尺寸
    3. 输出单通道类别索引 mask
    4. 将 mask 半透明叠加到原图并保存

类别：
    0: background
    1: A

注意：
    当前仅识别 A 1个前景类别，加上背景类别0，
    因此模型输出通道数 num_classes = 2。

依赖：numpy、opencv-python、onnxruntime（CPU）或 onnxruntime-gpu（GPU）。
外部权重：如果模型带有 .onnx.data 文件，须与 .onnx 一起部署。

可选参数：
    python infer_SX_onnx.py --limit N
"""

import os

import cv2
import numpy as np
import sys
import time
import onnxruntime as ort


# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有推理配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    # "test_image_dir": "D:/lhb/datasets/testsets/JZW/JZW_test/JZW_test",
    # "test_image_dir": "D:/lhb/datasets/testsets/JZW/HQL_JZW_0810/2",
    "test_image_dir": "D:/lhb/datasets/testsets/DZ/SX1009/1",

    # ========== ONNX 模型 ==========
    "onnx_path": "./checkpoints/SX1009/SX.onnx",
    "seg_output_name": "seg",       # 导出脚本中的分割 logits 输出名
    "num_classes": 2,               # 0=background, 1=A

    # ========== 整图缩放 ==========
    "crop_size": 1024,               # 整图缩放至 crop_size × crop_size
    "confidence_threshold": 0.1,

    # ========== 输出 ==========
    # "output_dir": "D:/lhb/datasets/testsets/JZW/HQL_JZW_0810/result/2/3_JZW",
    # "output_dir": "D:/lhb/datasets/testsets/JZW/JZW_test/result_3/JZW_4_1",
    "output_dir": "D:/lhb/datasets/testsets/DZ/SX1009/result/Dino_2_ONNX",
    "save_mask": True,               # 保存类别索引 mask
    "output_mode": "overlay",        # overlay / side_by_side

    # ========== 叠加显示 ==========
    "display_class_ids": {1},
    "alpha": 0.40,
    "draw_contours": True,
    "draw_labels": False,             # True=绘制类别标签；False=隐藏标签
    "contour_thickness": 0,          # >0 时绘制不透明轮廓
    "min_component_area": 2,
    "label_font_scale": 0.65,
    "label_font_thickness": 2,
    "label_padding": 4,

    # ========== 设备 / 性能统计 ==========
    "device": "cuda",              # cuda / cpu；CUDA 不可用时打印提示并回退 CPU
    "gpu_device_id": 0,
    "warmup_images": 3,
}


# BGR 颜色。可按项目需要自行调整。
CLASS_COLORS = {
    1: (255, 0,   0),    # A  蓝
    # 2: (0,   255, 0),    # B  绿
    # 3: (0,   0,   255),  # C  红
    # 4: (255, 0,   255),  # D  紫红

    # 5: (255, 255, 0),    # TIN-B/TIN-C  青
    # 6: (0,   165, 255),  # TIN-D        橙
}

CLASS_NAMES = {
    1: "A",
    # 2: "B",
    # 3: "C",
    # 4: "D",

    # 5: "TIN-B/TIN-C",
    # 6: "TIN-D",
}


num_classes          = int(INFER["num_classes"])
display_class_ids     = set(INFER["display_class_ids"])
alpha                 = float(INFER["alpha"])
draw_contours         = bool(INFER["draw_contours"])
draw_labels           = bool(INFER["draw_labels"])
contour_thickness     = int(INFER["contour_thickness"])
min_component_area    = int(INFER["min_component_area"])
label_font_scale      = float(INFER["label_font_scale"])
label_font_thickness  = int(INFER["label_font_thickness"])
label_padding         = int(INFER["label_padding"])
output_mode           = INFER["output_mode"]
crop_size             = int(INFER["crop_size"])
confidence_threshold  = float(INFER["confidence_threshold"])

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        工具函数                               ║
# ╚══════════════════════════════════════════════════════════════╝
def load_model():
    """加载 ONNX；外部权重文件须保持在模型引用的相对位置。"""
    model_path = INFER["onnx_path"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX 模型不存在：{model_path}")

    providers = ["CPUExecutionProvider"]
    if INFER["device"] == "cuda":
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, (
                "CUDAExecutionProvider", {"device_id": int(INFER["gpu_device_id"])}
            ))
        else:
            print("CUDAExecutionProvider 不可用，使用 CPU 推理。")

    session = ort.InferenceSession(model_path, providers=providers)
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("期望 ONNX 模型只有一个 NCHW 图像输入")
    inp = inputs[0]
    if inp.type not in {"tensor(float)", "tensor(float16)"}:
        raise ValueError(f"不支持的输入类型：{inp.type}")
    expected = (1, 3, crop_size, crop_size)
    if len(inp.shape) != 4:
        raise ValueError(f"期望 NCHW 输入，实际为 {inp.shape}")
    for actual, configured in zip(inp.shape, expected):
        if isinstance(actual, int) and actual != configured:
            raise ValueError(
                f"ONNX 输入尺寸 {inp.shape} 与配置 {expected} 不匹配，请检查 crop_size"
            )
    outputs = {out.name: out for out in session.get_outputs()}
    output_name = INFER["seg_output_name"]
    if output_name not in outputs:
        raise ValueError(f"找不到分割输出 {output_name}，可用输出：{list(outputs)}")
    print(f"Loaded: {model_path}")
    print(f"Providers: {session.get_providers()}")
    if INFER["device"] == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        print("CUDA 初始化未成功，当前使用 CPU 推理。")
    print(f"Input: {inp.name}, shape={inp.shape}, dtype={inp.type}")
    print(f"Seg output: {output_name}")
    return session


def _predict_probabilities(model, batch):
    """分割 logits -> 数值稳定的 softmax 概率 [N,C,H,W]。"""
    input_name = model.get_inputs()[0].name
    seg = model.run([INFER["seg_output_name"]], {input_name: batch})[0]
    if seg.ndim != 4 or seg.shape[:2] != (1, num_classes):
        raise ValueError(f"分割输出应为 [1,{num_classes},H,W]，实际为 {seg.shape}")
    seg = seg.astype(np.float32)
    seg -= np.max(seg, axis=1, keepdims=True)
    probs = np.exp(seg)
    return probs / np.sum(probs, axis=1, keepdims=True)


def _preprocess_batch(images, input_dtype):
    """RGB / 255 -> NCHW -> ImageNet 归一化 -> ONNX 输入类型。"""
    batch = (images.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
    batch = (batch - MEAN) / STD
    return np.ascontiguousarray(batch, dtype=input_dtype)


def infer_single(model, image_bgr):
    """整图缩放推理 -> 原图尺寸的 index mask + confidence。"""
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        image_rgb, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR
    )
    input_dtype = np.float16 if model.get_inputs()[0].type == "tensor(float16)" else np.float32
    batch = _preprocess_batch(np.expand_dims(resized, axis=0), input_dtype)
    probs = _predict_probabilities(model, batch)[0]

    assert probs.shape[0] == num_classes, (
        f"模型输出通道数={probs.shape[0]}，配置 num_classes={num_classes}"
    )

    # 先将各类别概率恢复至原图尺寸，再确定类别，避免插值类别索引。
    probs = np.stack([
        cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
        for prob in probs
    ])
    confidence = np.max(probs, axis=0).astype(np.float32)
    index_mask = np.argmax(probs, axis=0).astype(np.uint8)
    index_mask[confidence < confidence_threshold] = 0
    return index_mask, confidence


# ╔══════════════════════════════════════════════════════════════╗
# ║                     半透明叠加展示                             ║
# ╚══════════════════════════════════════════════════════════════╝
def choose_text_color(bgr_color):
    blue, green, red = bgr_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness >= 150 else (255, 255, 255)


def draw_class_contours_and_labels(image, mask):
    result = image.copy()
    image_height, image_width = result.shape[:2]

    for class_id in sorted(display_class_ids):
        bgr_color = CLASS_COLORS[class_id]
        class_name = CLASS_NAMES[class_id]

        binary_mask = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_component_area:
                continue

            if draw_contours and contour_thickness > 0:
                cv2.drawContours(
                    result,
                    [contour],
                    contourIdx=-1,
                    color=bgr_color,
                    thickness=contour_thickness,
                    lineType=cv2.LINE_AA,
                )

            if not draw_labels:
                continue

            x, y, width, height = cv2.boundingRect(contour)

            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                label_font_thickness,
            )

            label_width = text_width + label_padding * 2
            label_height = text_height + baseline + label_padding * 2

            label_x1 = max(0, min(x, image_width - label_width))
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

            text_color = choose_text_color(bgr_color)
            text_x = label_x1 + label_padding
            text_y = label_y1 + label_padding + text_height

            cv2.putText(
                result,
                class_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                text_color,
                label_font_thickness,
                cv2.LINE_AA,
            )

    return result


def create_overlay(image, mask):
    """将显示类别的 mask 半透明叠加到原图。"""
    overlay = image.copy()

    for class_id in sorted(display_class_ids):
        class_region = mask == class_id
        if not np.any(class_region):
            continue

        color = np.asarray(CLASS_COLORS[class_id], dtype=np.float32)
        original = overlay[class_region].astype(np.float32)

        blended = original * (1.0 - alpha) + color * alpha
        overlay[class_region] = np.clip(blended, 0, 255).astype(np.uint8)

    if draw_contours or draw_labels:
        overlay = draw_class_contours_and_labels(overlay, mask)

    return overlay


def create_output_image(original_image, overlay_image):
    if output_mode == "overlay":
        return overlay_image

    if output_mode == "side_by_side":
        return np.hstack((original_image, overlay_image))

    raise ValueError(f"不支持的 output_mode：{output_mode}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        文件保存                               ║
# ╚══════════════════════════════════════════════════════════════╝
def save_image(output_path, image):
    """保存彩色图片，兼容中文路径。"""
    suffix = os.path.splitext(output_path)[1].lower()
    extension_map = {
        ".jpg": ".jpg",
        ".jpeg": ".jpg",
        ".png": ".png",
        ".bmp": ".bmp",
        ".tif": ".tif",
        ".tiff": ".tiff",
    }
    encode_extension = extension_map.get(suffix, ".png")

    encode_params = []
    if encode_extension == ".jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    success, encoded_image = cv2.imencode(
        encode_extension, image, encode_params
    )
    if not success:
        raise RuntimeError(f"图片编码失败：{output_path}")

    encoded_image.tofile(output_path)


def save_mask_png(output_path, mask):
    """
    保存 uint8 单通道类别索引 mask：
        0 background
        1 A
    """
    success, encoded_image = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError(f"Mask 编码失败：{output_path}")

    encoded_image.tofile(output_path)


# ╔══════════════════════════════════════════════════════════════╗
# ║                        配置校验                               ║
# ╚══════════════════════════════════════════════════════════════╝
def validate_config():
    expected_foreground_ids = set(CLASS_NAMES.keys())
    expected_num_classes = len(expected_foreground_ids) + 1

    if num_classes != expected_num_classes:
        raise ValueError(
            f"num_classes 配置错误：当前定义了 {len(expected_foreground_ids)} 个前景类别，"
            f"加上背景后应为 {expected_num_classes} 类，实际配置为 {num_classes}。"
        )

    if set(CLASS_COLORS.keys()) != expected_foreground_ids:
        raise ValueError(
            "CLASS_COLORS 与 CLASS_NAMES 的类别编号不一致"
        )

    invalid_display = display_class_ids - expected_foreground_ids
    if invalid_display:
        raise ValueError(
            f"display_class_ids 中存在无效类别：{sorted(invalid_display)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在 0~1 之间")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold 必须在 0~1 之间")

    if crop_size <= 0:
        raise ValueError("crop_size 必须 > 0")

    if INFER["device"] not in {"cuda", "cpu"}:
        raise ValueError("device 只能是 cuda 或 cpu")

    if not 2 <= num_classes <= 256:
        raise ValueError("uint8 mask 要求 num_classes 在 2~256 之间")

    if output_mode not in {"overlay", "side_by_side"}:
        raise ValueError(
            'output_mode 只能是 "overlay" 或 "side_by_side"'
        )
# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    validate_config()

    model = load_model()

    print(f"num_classes={num_classes}（背景1类 + 前景{num_classes - 1}类）")
    print(
        f"resize={crop_size}x{crop_size}  "
        f"confidence_threshold={confidence_threshold}"
    )
    print("=" * 70)

    test_dir = INFER["test_image_dir"]
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        sys.exit(1)

    image_names = sorted(
        name
        for name in os.listdir(test_dir)
        if name.lower().endswith(VALID_EXTS)
    )

    if not image_names:
        print(f"No images found in: {test_dir}")
        sys.exit(1)

    # 可选：--limit N
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        if len(sys.argv) < 3:
            raise ValueError("--limit 后必须提供图片数量")

        limit = int(sys.argv[2])
        if limit <= 0:
            raise ValueError("--limit 必须 > 0")
        image_names = image_names[:limit]
        print(f"[限制] 仅处理前 {limit} 张图片")

    output_dir = INFER["output_dir"]
    mask_dir = os.path.join(output_dir, "mask")
    overlay_dir = os.path.join(output_dir, "overlay")

    os.makedirs(overlay_dir, exist_ok=True)
    if INFER["save_mask"]:
        os.makedirs(mask_dir, exist_ok=True)

    print(f"Total images: {len(image_names)}")
    print(f"Mask目录：{mask_dir}")
    print(f"Overlay目录：{overlay_dir}")
    print("=" * 70)

    measured_times = []

    for idx, name in enumerate(image_names):
        image_path = os.path.join(test_dir, name)
        stem = os.path.splitext(name)[0]

        image_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[跳过] 无法读取图片：{image_path}")
            continue

        t0 = time.perf_counter()

        # ① 单模型推理 -> mask
        mask, _ = infer_single(model, image_bgr)

        # ② mask 直接叠加原图，不做任何 mask 合并/remap
        overlay = create_overlay(image_bgr, mask)
        output_image = create_output_image(image_bgr, overlay)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if idx >= INFER["warmup_images"]:
            measured_times.append(elapsed_ms)

        # ③ 保存类别索引 mask
        if INFER["save_mask"]:
            save_mask_png(
                os.path.join(mask_dir, stem + ".png"),
                mask,
            )

        # ④ 保存原图叠加结果
        save_image(
            os.path.join(overlay_dir, name),
            output_image,
        )

        present_ids = [
            cid for cid in range(1, num_classes)
            if np.any(mask == cid)
        ]
        present_names = [
            CLASS_NAMES[cid] for cid in present_ids
        ]

        warmup = (
            " [预热]"
            if idx < INFER["warmup_images"]
            else ""
        )

        print(
            f"[{idx + 1}/{len(image_names)}] "
            f"Infer: {name}  "
            f"classes={present_names}  "
            f"Time: {elapsed_ms:.2f} ms{warmup}"
        )

    print("=" * 70)

    if measured_times:
        avg_ms = float(np.mean(measured_times))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        print(f"有效统计图片数：{len(measured_times)}")
        print(f"平均单图推理时间：{avg_ms:.2f} ms")
        print(f"平均推理速度：{fps:.2f} FPS")
    else:
        print("有效统计图片数不足，请减少 warmup_images。")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
