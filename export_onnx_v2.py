import os
import sys
import numpy as np
import torch
from types import SimpleNamespace

from models.dinov3_segmentation import DINOv3Seg


# NumPy 兼容
sys.modules.setdefault("numpy._core", np)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


# ============================================================
# 配置：必须与训练时一致
# ============================================================

BACKBONE_NAME = "dinov3_model"
NUM_CLASSES = 4
IMAGE_SIZE = 784
FREEZE_BACKBONE = True

# pth 模型
WEIGHT_PATH = r"checkpoints\D784\D784_slim.pth"

# ONNX 输出位置
ONNX_PATH = r"checkpoints\D784\D784_slim.onnx"


MODEL_CFG = SimpleNamespace(
    backbone_name=BACKBONE_NAME,
    freeze_backbone=FREEZE_BACKBONE,
    num_classes=NUM_CLASSES,
)


def export_onnx(weight_path, onnx_path, device):
    print("=" * 70)
    print("DINOv3 + DecoderV3 ONNX Export")
    print("=" * 70)
    print(f"Device      : {device}")
    print(f"Backbone    : {BACKBONE_NAME}")
    print(f"Num classes : {NUM_CLASSES}")
    print(f"Image size  : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Checkpoint  : {weight_path}")
    print(f"ONNX output : {onnx_path}")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    print("\n[1/5] Building model...")
    model = DINOv3Seg(MODEL_CFG).to(device)

    print("[2/5] Loading checkpoint...")
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)

    print("Checkpoint keys:", list(ckpt.keys()))

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    print(
        f"Loaded successfully "
        f"(epoch={ckpt.get('epoch', '?')}, "
        f"IoU={ckpt.get('iou', '?')}, "
        f"Dice={ckpt.get('dice', '?')})"
    )

    print("\n[3/5] Running PyTorch sanity check...")

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32, device=device)

    with torch.inference_mode():
        seg, boundary = model(dummy_input)

    print("Input shape    :", tuple(dummy_input.shape))
    print("Seg shape      :", tuple(seg.shape))
    print("Boundary shape :", tuple(boundary.shape))

    assert seg.shape == (1, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
    assert boundary.shape[0] == 1
    assert boundary.shape[2:] == (IMAGE_SIZE, IMAGE_SIZE)

    print("PyTorch forward check: OK")

    output_dir = os.path.dirname(onnx_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n[4/5] Exporting ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["seg", "boundary"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "seg": {0: "batch_size"},
            "boundary": {0: "batch_size"},
        },
    )

    print("\n[5/5] Export completed.")

    size_mb = os.path.getsize(onnx_path) / 1024 ** 2

    print("=" * 70)
    print(f"ONNX saved: {onnx_path}")
    print(f"ONNX size : {size_mb:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    export_onnx(WEIGHT_PATH, ONNX_PATH, device)
