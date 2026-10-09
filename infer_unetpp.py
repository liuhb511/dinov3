"""
infer_unetpp.py -- DINOv3 + UNet++ standalone sliding-window inference.

Output class IDs:
0 BG, 1 A, 2 B, 3 C, 4 D, 5 TIN-B/TIN-C, 6 TIN-D
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from models.unetpp.model import DINOv3UNetPP


# ╔══════════════════════════════════════════════════════════════╗
# ║                         推理配置                              ║
# ╚══════════════════════════════════════════════════════════════╝
INFER = {
    # ========== 测试数据 ==========
    "test_image_dir": r"D:/lhb/datasets/testsets/JZW/JZW_test/JZW_test",

    # ========== checkpoint：只读取 ckpt['model'] ==========
    "checkpoint_dir": "./checkpoints/A_unetpp_v2",
    "checkpoint_name": "best_iou",

    # ========== 模型结构（必须与训练时一致） ==========
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "num_classes": 7,
    "feature_layers": (-4, -3, -2, -1),
    "pyramid_channels": (32, 64, 128, 256),
    "dino_dim": 768,

    # ========== 滑动窗口 ==========
    "crop_size": 1024,
    "stride": 0,                  # 0=自适应重叠；>0=固定步长
    "infer_batch_size": 4,        # UNet++显存高于基础Decoder，先用4更稳妥
    "confidence_threshold": 0.1,

    # ========== 输出 ==========
    "output_dir": r"D:/lhb/datasets/testsets/JZW/JZW_test/result/dino_unetpp_v2",
    "save_mask": True,
    "output_mode": "overlay",   # overlay / side_by_side

    # ========== 叠加显示：与旧版一致 ==========
    "display_class_ids": {1, 2, 3, 4, 5, 6},
    "alpha": 0.40,
    "draw_contours": True,
    "contour_thickness": 0,
    "min_component_area": 2,      # 仅影响显示，不修改预测mask
    "label_font_scale": 0.65,
    "label_font_thickness": 2,
    "label_padding": 4,

    # ========== 设备 / 性能统计 ==========
    "device": "cuda",
    "warmup_images": 3,
}


CLASS_NAMES = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "TIN-B/TIN-C",
    6: "TIN-D",
}

CLASS_COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 0, 255),
    5: (255, 255, 0),
    6: (0, 165, 255),
}

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
PAD_VALUE = 128

num_classes = int(INFER["num_classes"])
display_class_ids = set(INFER["display_class_ids"])
alpha = float(INFER["alpha"])
draw_contours = bool(INFER["draw_contours"])
contour_thickness = int(INFER["contour_thickness"])
min_component_area = int(INFER["min_component_area"])
label_font_scale = float(INFER["label_font_scale"])
label_font_thickness = int(INFER["label_font_thickness"])
label_padding = int(INFER["label_padding"])
output_mode = INFER["output_mode"]
crop_size = int(INFER["crop_size"])
stride_len = int(INFER["stride"])
infer_batch_size = int(INFER["infer_batch_size"])
confidence_threshold = float(INFER["confidence_threshold"])

MEAN_FP32 = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(1, 3, 1, 1)
STD_FP32 = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(1, 3, 1, 1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_checkpoint(checkpoint_dir, checkpoint_name):
    path = checkpoint_name
    if not os.path.isabs(path) and not os.path.exists(path):
        name = (
            checkpoint_name
            if checkpoint_name.lower().endswith(".pth")
            else checkpoint_name + ".pth"
        )
        path = os.path.join(checkpoint_dir, name)
    return path


def strip_module_prefix(state_dict):
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def print_model_info(model, checkpoint_path):
    total_params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    checkpoint_bytes = os.path.getsize(checkpoint_path)

    print("=" * 70)
    print(f"参数量：{total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"模型参数大小：{model_bytes / (1024 ** 2):.2f} MB")
    print(f"权重文件大小：{checkpoint_bytes / (1024 ** 2):.2f} MB")
    print("=" * 70)


def load_model(path, device):
    cfg = SimpleNamespace(
        backbone_name=INFER["backbone_name"],
        freeze_backbone=INFER["freeze_backbone"],
        num_classes=INFER["num_classes"],
        feature_layers=INFER["feature_layers"],
        pyramid_channels=INFER["pyramid_channels"],
        dino_dim=INFER["dino_dim"],
    )

    model = DINOv3UNetPP(cfg).to(device)
    if device.type == "cuda":
        model = model.half()

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint必须是dict，实际={type(ckpt).__name__}")
    if "model" not in ckpt:
        raise KeyError(
            f"checkpoint keys={list(ckpt.keys())}，未找到唯一允许的权重字段'model'"
        )

    state = ckpt["model"]
    if not isinstance(state, dict):
        raise TypeError("ckpt['model']必须是state_dict")
    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"Loaded: {path}")
    print(
        f"  source=model, epoch={ckpt.get('epoch', '?')}, "
        f"IoU={ckpt.get('iou', 0):.4f}, Dice={ckpt.get('dice', 0):.4f}"
    )
    return model


def adaptive_positions(length, window):
    if length <= window:
        return [0]

    span = length - window
    count = max(math.ceil(span / window) + 1, 2)
    step = span / (count - 1)
    positions = [round(i * step) for i in range(count - 1)] + [span]

    unique = [int(positions[0])]
    for pos in positions[1:]:
        pos = int(pos)
        if pos - unique[-1] >= 2:
            unique.append(pos)
        else:
            unique[-1] = max(unique[-1], pos)
    return unique


def fixed_positions(length, window, stride):
    if length <= window:
        return [0]
    last = length - window
    positions = list(range(0, last + 1, stride))
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def preprocess_batch(patches, device):
    arr = np.stack(patches).astype(np.float32) / 255.0
    batch = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    batch = batch.to(dtype=dtype)
    mean = MEAN_FP32.to(device=device, dtype=dtype)
    std = STD_FP32.to(device=device, dtype=dtype)
    return (batch - mean) / std


@torch.inference_mode()
def predict_probabilities(model, batch):
    seg, _ = model(batch)
    probs = torch.softmax(seg, dim=1)
    return probs.float().cpu().numpy()


def sliding_window_infer(model, image_rgb, device):
    h, w = image_rgb.shape[:2]
    window = crop_size
    stride = stride_len

    xs = (
        fixed_positions(w, window, stride)
        if stride > 0
        else adaptive_positions(w, window)
    )
    ys = (
        fixed_positions(h, window, stride)
        if stride > 0
        else adaptive_positions(h, window)
    )

    prob_sum = np.zeros((num_classes, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    patches = []
    infos = []

    for y1 in ys:
        for x1 in xs:
            y2 = min(y1 + window, h)
            x2 = min(x1 + window, w)
            patch = image_rgb[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]

            if ph != window or pw != window:
                padded = np.full(
                    (window, window, 3),
                    PAD_VALUE,
                    dtype=np.uint8,
                )
                padded[:ph, :pw] = patch
                patch = padded

            patches.append(patch)
            infos.append((y1, y2, x1, x2, ph, pw))

    for start in range(0, len(patches), infer_batch_size):
        bp = patches[start : start + infer_batch_size]
        bi = infos[start : start + infer_batch_size]
        batch = preprocess_batch(bp, device)
        probs = predict_probabilities(model, batch)

        if probs.ndim != 4 or probs.shape[1] != num_classes:
            raise RuntimeError(
                f"模型输出应为[N,{num_classes},H,W]，实际={probs.shape}"
            )

        for p, info in zip(probs, bi):
            y1, y2, x1, x2, ph, pw = info
            prob_sum[:, y1:y2, x1:x2] += p[:, :ph, :pw]
            count[y1:y2, x1:x2] += 1.0

    if np.any(count == 0):
        raise RuntimeError("存在滑窗未覆盖像素")

    return prob_sum / count[None, ...]


def infer_single(model, image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    probs = sliding_window_infer(model, image_rgb, device)

    confidence = np.max(probs, axis=0).astype(np.float32)
    mask = np.argmax(probs, axis=0).astype(np.uint8)
    mask[confidence < confidence_threshold] = 0
    return mask, confidence


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
            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                label_font_thickness,
            )

            label_width = text_width + label_padding * 2
            label_height = text_height + baseline + label_padding * 2
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

            text_color = choose_text_color(bgr_color)
            cv2.putText(
                result,
                class_name,
                (label_x1 + label_padding, label_y1 + label_padding + text_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_font_scale,
                text_color,
                label_font_thickness,
                cv2.LINE_AA,
            )

    return result


def create_overlay(image, mask):
    overlay = image.copy()
    for class_id in sorted(display_class_ids):
        region = mask == class_id
        if not np.any(region):
            continue
        color = np.asarray(CLASS_COLORS[class_id], dtype=np.float32)
        original = overlay[region].astype(np.float32)
        overlay[region] = np.clip(
            original * (1.0 - alpha) + color * alpha,
            0,
            255,
        ).astype(np.uint8)

    if draw_contours:
        overlay = draw_class_contours_and_labels(overlay, mask)
    return overlay


def create_output_image(original_image, overlay_image):
    if output_mode == "overlay":
        return overlay_image
    if output_mode == "side_by_side":
        return np.hstack((original_image, overlay_image))
    raise ValueError(f"不支持的output_mode: {output_mode}")


def save_image(output_path, image):
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
    params = []
    if encode_extension == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]

    ok, encoded = cv2.imencode(encode_extension, image, params)
    if not ok:
        raise RuntimeError(f"图片编码失败: {output_path}")
    encoded.tofile(output_path)


def save_mask_png(output_path, mask):
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Mask编码失败: {output_path}")
    encoded.tofile(output_path)


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def validate_config():
    expected_ids = set(CLASS_NAMES.keys())
    expected_num_classes = len(expected_ids) + 1

    if num_classes != expected_num_classes:
        raise ValueError(
            f"num_classes应为{expected_num_classes}，实际={num_classes}"
        )
    if set(CLASS_COLORS.keys()) != expected_ids:
        raise ValueError("CLASS_COLORS与CLASS_NAMES类别编号不一致")
    invalid_display = display_class_ids - expected_ids
    if invalid_display:
        raise ValueError(f"display_class_ids存在无效类别: {sorted(invalid_display)}")

    if crop_size <= 0 or crop_size % 32 != 0:
        raise ValueError("crop_size必须>0且能被32整除")
    if stride_len < 0:
        raise ValueError("stride必须>=0")
    if infer_batch_size <= 0:
        raise ValueError("infer_batch_size必须>0")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold必须在0~1之间")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha必须在0~1之间")
    if output_mode not in {"overlay", "side_by_side"}:
        raise ValueError('output_mode只能是"overlay"或"side_by_side"')
    if len(INFER["feature_layers"]) != 4:
        raise ValueError("feature_layers必须包含4个层索引")
    if len(INFER["pyramid_channels"]) != 4:
        raise ValueError("pyramid_channels必须包含4个通道值")
    if any(int(v) <= 0 for v in INFER["pyramid_channels"]):
        raise ValueError("pyramid_channels必须全部>0")
    if int(INFER["warmup_images"]) < 0:
        raise ValueError("warmup_images必须>=0")
    if not os.path.isdir(INFER["test_image_dir"]):
        raise FileNotFoundError(INFER["test_image_dir"])


def main():
    validate_config()

    device = torch.device(
        INFER["device"] if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = resolve_checkpoint(
        INFER["checkpoint_dir"],
        INFER["checkpoint_name"],
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    print("=" * 70)
    print("加载 DINOv3 + UNet++ 模型...")
    model = load_model(checkpoint_path, device)
    print_model_info(model, checkpoint_path)

    mode_label = "adaptive" if stride_len == 0 else f"fixed stride={stride_len}"
    print(
        f"feature_layers={INFER['feature_layers']} | "
        f"pyramid_channels={INFER['pyramid_channels']}"
    )
    print(
        f"crop_size={crop_size} | mode={mode_label} | "
        f"infer_batch_size={infer_batch_size} | "
        f"confidence_threshold={confidence_threshold}"
    )
    print("=" * 70)

    names = sorted(
        name
        for name in os.listdir(INFER["test_image_dir"])
        if name.lower().endswith(VALID_EXTS)
    )
    if not names:
        raise RuntimeError(f"No images found in: {INFER['test_image_dir']}")

    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        if len(sys.argv) < 3:
            raise ValueError("--limit后必须提供图片数量")
        limit = int(sys.argv[2])
        if limit <= 0:
            raise ValueError("--limit N中的N必须>0")
        names = names[:limit]
        print(f"[限制] 仅处理前{limit}张图片")

    mask_dir = os.path.join(INFER["output_dir"], "mask")
    overlay_dir = os.path.join(INFER["output_dir"], "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    if INFER["save_mask"]:
        os.makedirs(mask_dir, exist_ok=True)

    print(f"Total images: {len(names)}")
    print(f"Overlay目录: {overlay_dir}")
    if INFER["save_mask"]:
        print(f"Mask目录: {mask_dir}")
    print("=" * 70)

    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for i, name in enumerate(names):
        image_path = os.path.join(INFER["test_image_dir"], name)
        image_bgr = read_image(image_path)
        if image_bgr is None:
            print(f"[skip] 无法读取图片: {image_path}")
            continue

        sync(device)
        t0 = time.perf_counter()

        mask, _ = infer_single(model, image_bgr, device)
        overlay = create_overlay(image_bgr, mask)
        output_image = create_output_image(image_bgr, overlay)

        sync(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if i >= int(INFER["warmup_images"]):
            times.append(elapsed_ms)

        stem = os.path.splitext(name)[0]
        if INFER["save_mask"]:
            save_mask_png(os.path.join(mask_dir, stem + ".png"), mask)
        save_image(os.path.join(overlay_dir, name), output_image)

        present_ids = [
            cid for cid in range(1, num_classes) if np.any(mask == cid)
        ]
        present_names = [CLASS_NAMES[cid] for cid in present_ids]
        warmup = " [预热]" if i < int(INFER["warmup_images"]) else ""

        print(
            f"[{i + 1}/{len(names)}] Infer: {name}  "
            f"classes={present_names}  Time={elapsed_ms:.2f} ms{warmup}"
        )

    print("=" * 70)
    if times:
        avg_ms = float(np.mean(times))
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        print(f"有效统计图片数: {len(times)}")
        print(f"平均单图推理时间: {avg_ms:.2f} ms")
        print(f"平均推理速度: {fps:.2f} FPS")
    else:
        print("有效统计图片数不足，请减少warmup_images。")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"GPU峰值显存: {peak_mb:.2f} MB")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
