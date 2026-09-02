import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from HMS.config import HMSConfig
from HMS.model import HMSLabRedSeg
from HMS.transforms import letterbox_resize, rgb_to_lab_a, rgb_to_tensor, a_to_tensor

def build_model(cfg, checkpoint, device):
    base_cfg = SimpleNamespace(
        backbone_name=cfg.backbone_name,
        freeze_backbone=False,
        num_classes=cfg.num_classes,
    )
    model = HMSLabRedSeg(
        base_cfg,
        red_class=cfg.red_class,
        lab_channels=cfg.lab_channels,
        fusion_init=cfg.lab_fusion_init,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="prediction.png")
    args = ap.parse_args()

    cfg = HMSConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, args.checkpoint, device)

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise RuntimeError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb, _ = letterbox_resize(rgb, None, cfg.image_size)

    a = rgb_to_lab_a(rgb)
    rgb_t = rgb_to_tensor(rgb).unsqueeze(0).to(device)
    a_t = a_to_tensor(a).unsqueeze(0).to(device)

    seg, _, _ = model(rgb_t, a_t)
    pred = seg.argmax(dim=1)[0].byte().cpu().numpy()
    cv2.imwrite(args.output, pred)
    print(f"saved: {args.output}")

if __name__ == "__main__":
    main()
