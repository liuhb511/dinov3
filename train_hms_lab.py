import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from HMS.config import HMSConfig
from HMS.dataset import HMSDataset
from HMS.transforms import get_train_augmentation, get_val_augmentation
from HMS.model import HMSLabRedSeg
from HMS.losses import HMSRedLoss
from HMS.metrics import SegmentationMeter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_model_cfg(cfg):
    return SimpleNamespace(
        backbone_name=cfg.backbone_name,
        freeze_backbone=cfg.freeze_backbone,
        num_classes=cfg.num_classes,
    )

def set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)

def build_loaders(cfg):
    root = Path(cfg.dataset_root)
    train_ds = HMSDataset(
        root / "train" / "images",
        root / "train" / "masks",
        augmentation=get_train_augmentation(),
        image_size=cfg.image_size,
    )
    val_ds = HMSDataset(
        root / "val" / "images",
        root / "val" / "masks",
        augmentation=get_val_augmentation(),
        image_size=cfg.image_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader

def build_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)

    for rgb, lab_a, mask, _ in pbar:
        rgb = rgb.to(device, non_blocking=True)
        lab_a = lab_a.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            seg, boundary, red_residual = model(rgb, lab_a)
            loss, parts = criterion(seg, mask)

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            red_tv=f"{parts['red_tversky']:.4f}",
            lab_scale=f"{model.fusion_scale.item():.3f}",
        )

    return running / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    meter = SegmentationMeter(cfg.num_classes, cfg.red_class, device=device)
    running = 0.0

    for rgb, lab_a, mask, _ in tqdm(loader, desc="Val", leave=False):
        rgb = rgb.to(device, non_blocking=True)
        lab_a = lab_a.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            seg, _, _ = model(rgb, lab_a)
            loss, _ = criterion(seg, mask)

        running += loss.item()
        pred = seg.argmax(dim=1)
        meter.update(pred, mask)

    return running / max(1, len(loader)), meter.compute()

def main():
    cfg = HMSConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_loader, val_loader = build_loaders(cfg)

    model = HMSLabRedSeg(
        make_model_cfg(cfg),
        red_class=cfg.red_class,
        lab_channels=cfg.lab_channels,
        fusion_init=cfg.lab_fusion_init,
    ).to(device)

    criterion = HMSRedLoss(
        num_classes=cfg.num_classes,
        red_class=cfg.red_class,
        class_weights=cfg.class_weights,
        ce_weight=cfg.ce_weight,
        dice_weight=cfg.dice_weight,
        red_tversky_weight=cfg.red_tversky_weight,
        alpha=cfg.red_tversky_alpha,
        beta=cfg.red_tversky_beta,
        gamma=cfg.red_tversky_gamma,
    ).to(device)

    # Stage 1: freeze DINOv3 encoder, train decoder + LAB branch.
    set_encoder_trainable(model, False)
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_score = -1.0

    for epoch in range(cfg.epochs):
        if epoch == cfg.freeze_epochs:
            print("Unfreezing DINOv3 encoder...")
            set_encoder_trainable(model, True)
            optimizer = build_optimizer(model, cfg.unfreeze_lr, cfg.weight_decay)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg
        )
        val_loss, m = validate(model, val_loader, criterion, device, cfg)

        # RED completeness is important, so selection includes RED recall.
        score = 0.50 * m["mIoU_fg"] + 0.30 * m["red_iou"] + 0.20 * m["red_recall"]

        print(
            f"Epoch {epoch+1:03d}/{cfg.epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"mIoU={m['mIoU_fg']:.4f} "
            f"RED IoU={m['red_iou']:.4f} "
            f"RED Dice={m['red_dice']:.4f} "
            f"RED Recall={m['red_recall']:.4f} "
            f"RED Precision={m['red_precision']:.4f} "
            f"LAB scale={model.fusion_scale.item():.3f}"
        )

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(cfg),
            "metrics": m,
        }
        torch.save(state, Path(cfg.save_dir) / "last.pt")

        if score > best_score:
            best_score = score
            torch.save(state, Path(cfg.save_dir) / "best.pt")
            print(f"  saved best.pt, score={score:.4f}")

if __name__ == "__main__":

    main()
