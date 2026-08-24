"""
batch_infer_onnx.py

两个 DINOv3 ONNX 模型批量联合推理

逻辑：
1. 批量读取文件夹图片
2. 784x784 滑窗
3. Batch ONNX 推理
4. softmax + threshold + argmax
5. model_2 先上色
6. model_1 非背景区域覆盖 model_2
7. 保存最终彩色结果
8. 统计每张图片和整体耗时

ONNX 输出：
    seg
    boundary

这里只使用 seg，忽略 boundary。
"""

import os
import math
import time
import cv2
import numpy as np
import torch
import onnxruntime as ort


# ============================================================
# 预加载 PyTorch 自带 CUDA / cuDNN DLL
# ============================================================

ort.preload_dlls()


# ============================================================
# 配置
# ============================================================

CONFIG = {
    "input_dir": r"F:/liuhaibo/datasets/test/LG_JZW_0807/20260703154505",
    "output_dir": r"output/infer/20260703154505_test_results_onnx",

    "model_1_path": r"checkpoints\ABCTIN784\ABCTIN_784_slim.onnx",
    "model_2_path": r"checkpoints\D784\D784_slim.onnx",

    "input_shape": 784,
    "stride": 0,
    "infer_batch_size": 12,
    "cuda": 1,

    # 先验证 CUDAExecutionProvider，成功后再改 True 测 TensorRT
    "use_tensorrt": False,

    "infer_threshold": 0.5,
    "infer_close_kernel": 3,
    "infer_min_area": 10,

    "model_1": {
        "num_classes": 9,
        "colors": {
            0: [0, 0, 0],
            1: [0, 0, 255],
            2: [0, 255, 0],
            3: [255, 0, 0],
            7: [0, 255, 0],
            8: [128, 0, 255],
        },
    },

    "model_2": {
        "num_classes": 4,
        "colors": {
            0: [0, 0, 0],
            1: [128, 0, 255],
        },
    },
}


# ============================================================
# 常量
# ============================================================

PAD_VALUE = 128

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# ONNX 模型
# ============================================================

class ONNXSegModel:
    def __init__(self, model_path, num_classes, input_shape, use_cuda=True, use_tensorrt=False):
        self.model_path = model_path
        self.num_classes = num_classes
        self.input_shape = input_shape

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        available = ort.get_available_providers()
        providers = []

        print(f"ORT version         : {ort.__version__}")
        print(f"Torch version       : {torch.__version__}")
        print(f"Torch CUDA          : {torch.version.cuda}")
        print(f"Available providers : {available}")

        if use_cuda and use_tensorrt:
            if "TensorrtExecutionProvider" not in available:
                raise RuntimeError(f"当前 ORT 不支持 TensorRT，providers={available}")

            cache_dir = os.path.join(os.path.dirname(model_path), "trt_cache")
            os.makedirs(cache_dir, exist_ok=True)

            providers.append((
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_max_workspace_size": 4 * 1024 ** 3,
                },
            ))

        if use_cuda:
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(f"当前 ORT 没有 CUDAExecutionProvider，providers={available}")

            providers.append(("CUDAExecutionProvider", {"device_id": 0}))

        providers.append("CPUExecutionProvider")

        print(f"Requested providers : {providers}")

        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

        active_providers = self.session.get_providers()

        if use_cuda and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                f"CUDAExecutionProvider 加载失败，当前实际 providers={active_providers}，程序停止，不允许回退 CPU。"
            )

        self.input_name = self.session.get_inputs()[0].name

        output_names = [x.name for x in self.session.get_outputs()]
        self.seg_output_name = "seg" if "seg" in output_names else output_names[0]

        print("=" * 70)
        print(f"Loaded ONNX : {model_path}")
        print(f"Providers   : {active_providers}")
        print(f"Input       : {self.input_name}")
        print(f"Outputs     : {output_names}")
        print(f"Seg output  : {self.seg_output_name}")
        print("=" * 70)

    def predict(self, batch):
        seg = self.session.run([self.seg_output_name], {self.input_name: batch})[0]

        if seg.ndim != 4:
            raise RuntimeError(f"ONNX seg 输出维度错误: {seg.shape}")

        if seg.shape[1] != self.num_classes:
            raise RuntimeError(f"ONNX 输出类别数不一致: 模型输出={seg.shape[1]}, 配置={self.num_classes}")

        return seg


# ============================================================
# 工具
# ============================================================

def create_color_lut(colors, num_classes):
    lut = np.zeros((256, 3), dtype=np.uint8)

    for class_id, rgb in colors.items():
        class_id = int(class_id)

        if class_id >= num_classes:
            raise ValueError(f"class_id={class_id} 超过 num_classes={num_classes}")

        lut[class_id] = np.asarray(rgb[::-1], dtype=np.uint8)

    return lut


def colorize(mask, lut):
    return lut[mask]


def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def deduplicate_positions(positions):
    result = []

    for value in positions:
        value = int(value)

        if not result or value != result[-1]:
            result.append(value)

    return result


def adaptive_positions(length, window):
    if length <= window:
        return [0]

    span = length - window
    count = max(math.ceil(span / window) + 1, 2)
    step = span / (count - 1)

    positions = [round(i * step) for i in range(count - 1)] + [span]
    return deduplicate_positions(positions)


def fixed_positions(length, window, stride):
    if length <= window:
        return [0]

    last = length - window
    positions = list(range(0, last + 1, stride))

    if not positions or positions[-1] != last:
        positions.append(last)

    return deduplicate_positions(positions)


def build_positions(length, window, stride):
    if stride == 0:
        return adaptive_positions(length, window)

    return fixed_positions(length, window, stride)


def preprocess_batch(patches):
    array = np.stack(patches, axis=0).astype(np.float32) / 255.0
    batch = np.transpose(array, (0, 3, 1, 2))
    batch = (batch - MEAN) / STD

    return np.ascontiguousarray(batch, dtype=np.float32)


# ============================================================
# 滑窗推理
# ============================================================

def sliding_window_infer(model, image_rgb, num_classes, input_shape, stride, batch_size):
    height, width = image_rgb.shape[:2]
    window = input_shape

    xs = build_positions(width, window, stride)
    ys = build_positions(height, window, stride)

    probability_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    patches = []
    patch_infos = []

    for y1 in ys:
        for x1 in xs:
            x2 = min(x1 + window, width)
            y2 = min(y1 + window, height)

            patch = image_rgb[y1:y2, x1:x2]
            patch_h, patch_w = patch.shape[:2]

            if patch_h != window or patch_w != window:
                padded = np.full((window, window, 3), PAD_VALUE, dtype=np.uint8)
                padded[:patch_h, :patch_w] = patch
                patch = padded

            patches.append(patch)
            patch_infos.append((y1, y2, x1, x2, patch_h, patch_w))

    print(f"    patches: {len(patches)}")

    for start in range(0, len(patches), batch_size):
        batch_patches = patches[start:start + batch_size]
        batch_infos = patch_infos[start:start + batch_size]

        batch = preprocess_batch(batch_patches)

        logits = model.predict(batch)
        probabilities = softmax(logits, axis=1)

        for probability, info in zip(probabilities, batch_infos):
            y1, y2, x1, x2, patch_h, patch_w = info

            valid_probability = probability[:, :patch_h, :patch_w]

            probability_sum[:, y1:y2, x1:x2] += valid_probability
            count_map[y1:y2, x1:x2] += 1.0

    if np.any(count_map == 0):
        raise RuntimeError("滑窗配置错误：存在没有被覆盖的像素")

    return probability_sum / count_map[None, :, :]


# ============================================================
# 单模型推理
# ============================================================

def infer_one(model, image_rgb, model_cfg):
    start = time.perf_counter()

    probabilities = sliding_window_infer(
        model=model,
        image_rgb=image_rgb,
        num_classes=model_cfg["num_classes"],
        input_shape=CONFIG["input_shape"],
        stride=CONFIG["stride"],
        batch_size=CONFIG["infer_batch_size"],
    )

    mask = np.argmax(probabilities, axis=0).astype(np.uint8)
    max_probability = np.max(probabilities, axis=0)

    mask[max_probability < CONFIG["infer_threshold"]] = 0

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return mask, elapsed_ms


# ============================================================
# 双模型联合推理
# ============================================================

def infer_image(image_bgr, model_1, model_2, lut_1, lut_2):
    total_start = time.perf_counter()

    prepare_start = time.perf_counter()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prepare_ms = (time.perf_counter() - prepare_start) * 1000.0

    print("  Model 2 inference...")
    mask_2, model_2_ms = infer_one(model_2, image_rgb, CONFIG["model_2"])

    color_start = time.perf_counter()
    result = colorize(mask_2, lut_2)
    color_2_ms = (time.perf_counter() - color_start) * 1000.0

    print("  Model 1 inference...")
    mask_1, model_1_ms = infer_one(model_1, image_rgb, CONFIG["model_1"])

    merge_start = time.perf_counter()

    write_area = mask_1 != 0

    if np.any(write_area):
        color_1 = colorize(mask_1, lut_1)
        result[write_area] = color_1[write_area]

    merge_ms = (time.perf_counter() - merge_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    times = {
        "prepare": prepare_ms,
        "model_1": model_1_ms,
        "model_2": model_2_ms,
        "color_2": color_2_ms,
        "merge": merge_ms,
        "total": total_ms,
    }

    return result, times


# ============================================================
# 图片列表
# ============================================================

def get_image_files(folder):
    files = []

    for name in os.listdir(folder):
        path = os.path.join(folder, name)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(name)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            files.append(path)

    return sorted(files)


# ============================================================
# Main
# ============================================================

def main():
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")

    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)
    print(f"Torch          : {torch.__version__}")
    print(f"Torch CUDA     : {torch.version.cuda}")
    print(f"cuDNN          : {torch.backends.cudnn.version()}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")

    print(f"ORT            : {ort.__version__}")
    print(f"ORT providers  : {ort.get_available_providers()}")
    print("=" * 80)

    print("\nLoading models...")

    model_1 = ONNXSegModel(
        CONFIG["model_1_path"],
        CONFIG["model_1"]["num_classes"],
        CONFIG["input_shape"],
        use_cuda=bool(CONFIG["cuda"]),
        use_tensorrt=CONFIG["use_tensorrt"],
    )

    model_2 = ONNXSegModel(
        CONFIG["model_2_path"],
        CONFIG["model_2"]["num_classes"],
        CONFIG["input_shape"],
        use_cuda=bool(CONFIG["cuda"]),
        use_tensorrt=CONFIG["use_tensorrt"],
    )

    lut_1 = create_color_lut(CONFIG["model_1"]["colors"], CONFIG["model_1"]["num_classes"])
    lut_2 = create_color_lut(CONFIG["model_2"]["colors"], CONFIG["model_2"]["num_classes"])

    image_files = get_image_files(input_dir)

    if not image_files:
        raise RuntimeError(f"输入目录没有找到图片: {input_dir}")

    print()
    print("=" * 80)
    print(f"Images     : {len(image_files)}")
    print(f"Input dir  : {input_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Input size : {CONFIG['input_shape']}")
    print(f"Batch size : {CONFIG['infer_batch_size']}")
    print(f"Stride     : {CONFIG['stride']}")
    print(f"Threshold  : {CONFIG['infer_threshold']}")
    print(f"TensorRT   : {CONFIG['use_tensorrt']}")
    print("=" * 80)

    all_times = []
    batch_start = time.perf_counter()

    for index, image_path in enumerate(image_files, 1):
        print()
        print("=" * 80)
        print(f"[{index}/{len(image_files)}] {os.path.basename(image_path)}")
        print("=" * 80)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            print(f"[Warning] 无法读取，跳过: {image_path}")
            continue

        print(f"  image shape: {image.shape[1]} x {image.shape[0]}")

        result, times = infer_image(image, model_1, model_2, lut_1, lut_2)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, filename + "_pred.png")

        if not cv2.imwrite(save_path, result):
            raise RuntimeError(f"保存失败: {save_path}")

        all_times.append(times)

        print()
        print(f"  Prepare : {times['prepare']:.2f} ms")
        print(f"  Model 1 : {times['model_1']:.2f} ms")
        print(f"  Model 2 : {times['model_2']:.2f} ms")
        print(f"  Color 2 : {times['color_2']:.2f} ms")
        print(f"  Merge   : {times['merge']:.2f} ms")
        print(f"  Total   : {times['total']:.2f} ms")
        print(f"  Saved   : {save_path}")

    batch_total_ms = (time.perf_counter() - batch_start) * 1000.0

    if not all_times:
        print("没有成功处理任何图片。")
        return

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for key, title in [
        ("prepare", "Prepare"),
        ("model_1", "Model 1"),
        ("model_2", "Model 2"),
        ("color_2", "Color 2"),
        ("merge", "Merge"),
        ("total", "Total"),
    ]:
        values = [x[key] for x in all_times]

        print(
            f"{title:<10}: "
            f"avg={np.mean(values):8.2f} ms   "
            f"min={np.min(values):8.2f} ms   "
            f"max={np.max(values):8.2f} ms"
        )

    print("-" * 80)
    print(f"Processed      : {len(all_times)} images")
    print(f"Batch total    : {batch_total_ms / 1000.0:.2f} s")
    print(f"Average/image  : {batch_total_ms / len(all_times):.2f} ms")
    print(f"Throughput     : {len(all_times) / (batch_total_ms / 1000.0):.2f} images/s")
    print(f"Results        : {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()