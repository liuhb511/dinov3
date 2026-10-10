import math
from types import SimpleNamespace

import numpy as np
import torch

from .common import PROJECT, input_path


def load_model(cfg, checkpoint=None):
    import sys
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from models.dinov3_segmentation import DINOv3Seg
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose cpu explicitly")
    model = DINOv3Seg(SimpleNamespace(backbone_name=str(input_path(cfg["backbone"])),
                                     freeze_backbone=cfg.get("freeze_backbone", True), num_classes=2))
    ckpt = torch.load(input_path(checkpoint or cfg["checkpoint"]), map_location="cpu", weights_only=False)
    order = ["student", "teacher", "model", "state_dict"] if cfg.get("weight_source") == "student" else ["teacher", "student", "model", "state_dict"]
    key = next((k for k in order if k in ckpt and isinstance(ckpt[k], dict)), None)
    state = ckpt[key] if key else ckpt
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device), device, key or "raw"


def positions(length, size):
    if length <= size:
        return [0]
    span = length-size
    count = max(2, math.ceil(span/size)+1)
    return sorted(set(round(i*span/(count-1)) for i in range(count)))


@torch.no_grad()
def predict(model, bgr, device, cfg):
    size = int(cfg.get("window_size", 1024))
    if size <= 0:
        raise ValueError("window_size must be positive")
    mode = cfg.get("overlap_mode", "overwrite")
    if mode not in {"overwrite", "average"}:
        raise ValueError("Unknown overlap mode")
    h, w = bgr.shape[:2]
    result = np.zeros((h, w), np.float32)
    counts = np.zeros((h, w), np.float32)
    use_half = cfg.get("fp16", True) and device.type == "cuda"
    model.eval()
    model.half() if use_half else model.float()
    dtype = torch.float16 if use_half else torch.float32
    mean = torch.tensor([.485, .456, .406], device=device, dtype=dtype)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device, dtype=dtype)[None, :, None, None]
    for y in positions(h, size):
        for x in positions(w, size):
            ph, pw = min(size, h-y), min(size, w-x)
            patch = np.full((size, size, 3), 128, np.uint8)
            patch[:ph, :pw] = bgr[y:y+ph, x:x+pw, ::-1]
            tensor = torch.from_numpy(patch.copy()).permute(2, 0, 1)[None].to(device, dtype)/255
            logits, _ = model((tensor-mean)/std)
            prob = logits.softmax(1)[0, 1, :ph, :pw].float().cpu().numpy()
            if mode == "overwrite":
                result[y:y+ph, x:x+pw] = prob
            else:
                result[y:y+ph, x:x+pw] += prob
            counts[y:y+ph, x:x+pw] += 1
    return result / counts if mode == "average" else result
