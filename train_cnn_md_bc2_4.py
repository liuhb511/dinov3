"""
train_cnn_md1024.py -- A-CNN-MD-v1 combined structural model.

Main structural changes vs A0:
    1) DINO hidden states: L4 / L8 / L12
    2) Lightweight local CNN: true spatial F4/F8 features
    3) Gated residual fusion inside Decoder

Intentionally unchanged:
    - original training dataset
    - original get_train_transform/get_val_transform
    - original TotalLoss (CE + Dice + optional boundary)
    - original two-stage freeze/unfreeze schedule
    - no Small-v1 auxiliary head
    - no size-aware loss
    - no tiny-object oversampling
    - no hard-negative data in this first experiment
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

TRAIN = {
    "dataset_root": r"D:/lhb/datasets/JZW_v3/JZW",

    # ===== Model =====
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "num_classes": 7,
    "image_size": 1024,
    "feature_layers": (4, 8, 12),
    "local_stem_channels": 16,
    "local_f4_channels": 32,
    "local_f8_channels": 64,
    "fusion_init_alpha": 0.1,

    # ===== Training =====
    "freeze_batch_size": 4,
    "unfreeze_batch_size": 2,
    "val_batch_size": 4,
    "num_workers": 4,
    "epochs": 100,
    "learning_rate": 1e-4,
    "unfreeze_lr": 1e-5,
    "freeze_epochs": 60,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "amp": True,
    "seed": 42,

    # ===== Loss: unchanged =====
    "ce_weight": 0.7,
    "dice_weight": 0.3,
    "boundary_weight": 0.0,

    # ===== Save =====
    "save_dir": "./checkpoints/A_cnn_md_v2",
    "log_dir": "./logs/A_cnn_md_v2",
    "resume": "",
    "save_best_iou": True,
    "save_best_dice": True,

    "device": "cuda",
}

_dataset_root = TRAIN["dataset_root"]
train_image_dir = os.path.join(_dataset_root, "train", "images")
train_mask_dir = os.path.join(_dataset_root, "train", "masks")
val_image_dir = os.path.join(_dataset_root, "val", "images")
val_mask_dir = os.path.join(_dataset_root, "val", "masks")

_model_cfg = SimpleNamespace(
    backbone_name=TRAIN["backbone_name"],
    freeze_backbone=TRAIN["freeze_backbone"],
    num_classes=TRAIN["num_classes"],
    feature_layers=TRAIN["feature_layers"],
    local_stem_channels=TRAIN["local_stem_channels"],
    local_f4_channels=TRAIN["local_f4_channels"],
    local_f8_channels=TRAIN["local_f8_channels"],
    fusion_init_alpha=TRAIN["fusion_init_alpha"],
    ce_weight=TRAIN["ce_weight"],
    bce_weight=TRAIN["ce_weight"],
    dice_weight=TRAIN["dice_weight"],
    boundary_weight=TRAIN["boundary_weight"],
)

from data.grain_dataset import GrainDataset
from data.transforms import get_train_transform, get_val_transform
from models.cnn_md.model import DINOv3SegCNNMD
from losses.loss_JZW import TotalLoss
from utils.metric_JZW import update_confusion_matrix, calculate_metrics
from utils.seed import set_seed
from utils.scheduler import WarmupCosineScheduler
from utils.logger import TrainLogger


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def create_train_loader(dataset, batch_size, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=TRAIN["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def create_val_loader(dataset, device):
    return DataLoader(
        dataset,
        batch_size=TRAIN["val_batch_size"],
        shuffle=False,
        num_workers=TRAIN["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, logger, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Train")
    amp_enabled = TRAIN["amp"] and device.type == "cuda"

    for batch_idx, (images, masks, _) in enumerate(loop):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            seg, boundary = model(images)
            loss = criterion(seg, boundary, masks)

        scaler.scale(loss).backward()
        if TRAIN["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        avg_loss = total_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]["lr"]
        global_step = epoch * len(loader) + batch_idx + 1
        logger.log_batch(
            epoch, batch_idx, global_step,
            batch_loss, avg_loss, current_lr,
        )
        loop.set_postfix(
            loss=f"{batch_loss:.4f}",
            avg_loss=f"{avg_loss:.4f}",
            lr=f"{current_lr:.2e}",
        )

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_cls = TRAIN["num_classes"]
    amp_enabled = TRAIN["amp"] and device.type == "cuda"
    confusion_matrix = torch.zeros(
        (num_cls, num_cls), dtype=torch.long, device=device
    )

    loop = tqdm(loader, desc="Val")
    for images, masks, _ in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            seg, boundary = model(images)
            loss = criterion(seg, boundary, masks)
        preds = torch.argmax(seg, dim=1)
        total_loss += float(loss.item())
        update_confusion_matrix(
            confusion_matrix, preds, masks, num_classes=num_cls
        )
        loop.set_postfix(loss=f"{loss.item():.4f}")

    mean_iou, mean_dice, _, _ = calculate_metrics(
        confusion_matrix, include_background=False
    )
    return total_loss / max(len(loader), 1), mean_iou, mean_dice


def main():
    set_seed(TRAIN["seed"])
    device = torch.device(
        TRAIN["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print("Experiment: A-CNN-MD-v1 (DINO L4/L8/L12 + CNN F8/F4)")
    print(f"Dataset: {TRAIN['dataset_root']}")
    print(f"Classes: {TRAIN['num_classes']}")
    print(f"DINO feature layers: {TRAIN['feature_layers']}")
    print(
        "Local CNN: stem={} / F4={} / F8={}".format(
            TRAIN["local_stem_channels"],
            TRAIN["local_f4_channels"],
            TRAIN["local_f8_channels"],
        )
    )
    print("Loss/Data/Augmentation: same as A0; no HN; no Small-v1 loss")

    logger = TrainLogger(TRAIN["log_dir"])
    logger.save_config(TRAIN)

    train_dataset = GrainDataset(
        train_image_dir, train_mask_dir,
        transform=get_train_transform(), is_train=True,
        image_size=TRAIN["image_size"],
    )
    val_dataset = GrainDataset(
        val_image_dir, val_mask_dir,
        transform=get_val_transform(), is_train=False,
        image_size=TRAIN["image_size"],
    )

    val_loader = create_val_loader(val_dataset, device)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    model = DINOv3SegCNNMD(_model_cfg).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Params: {trainable:,} trainable / {total:,} total "
        f"({100.0 * trainable / total:.1f}%)"
    )

    criterion = TotalLoss(_model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN["learning_rate"],
        weight_decay=TRAIN["weight_decay"],
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=TRAIN["warmup_epochs"],
        total_epochs=TRAIN["epochs"],
        min_lr=TRAIN["min_lr"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=TRAIN["amp"] and device.type == "cuda"
    )

    best_iou = 0.0
    best_dice = 0.0
    start_epoch = 0
    stage2_activated = False

    resume_path = TRAIN["resume"]
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)

        start_epoch = int(ckpt["epoch"]) + 1
        best_iou = float(ckpt.get("best_iou", 0.0))
        best_dice = float(ckpt.get("best_dice", 0.0))
        stage2_activated = start_epoch >= TRAIN["freeze_epochs"]

        if stage2_activated:
            model.encoder.set_trainable(True)
            for param_group in optimizer.param_groups:
                param_group["lr"] = TRAIN["unfreeze_lr"]
                param_group["initial_lr"] = TRAIN["unfreeze_lr"]
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=TRAIN["epochs"] - TRAIN["freeze_epochs"],
                min_lr=TRAIN["min_lr"],
            )

        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        print(f"Resumed from epoch {start_epoch + 1}")

    current_batch_size = (
        TRAIN["unfreeze_batch_size"]
        if stage2_activated
        else TRAIN["freeze_batch_size"]
    )
    train_loader = create_train_loader(
        train_dataset, current_batch_size, device
    )

    os.makedirs(TRAIN["save_dir"], exist_ok=True)

    total_epochs = TRAIN["epochs"]
    for epoch in range(start_epoch, total_epochs):
        if epoch == TRAIN["freeze_epochs"] and not stage2_activated:
            stage2_activated = True
            print("=" * 60)
            print(f"Stage 2: unfreeze DINO backbone, lr={TRAIN['unfreeze_lr']}")
            print(
                f"Train batch size: {TRAIN['freeze_batch_size']} "
                f"-> {TRAIN['unfreeze_batch_size']}"
            )
            print("=" * 60)

            model.encoder.set_trainable(True)

            train_loader = create_train_loader(
                train_dataset, TRAIN["unfreeze_batch_size"], device
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = TRAIN["unfreeze_lr"]
                param_group["initial_lr"] = TRAIN["unfreeze_lr"]

            remaining = total_epochs - epoch
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=remaining,
                min_lr=TRAIN["min_lr"],
            )

        current_lr = optimizer.param_groups[0]["lr"]
        stage_name = "Stage2 unfrozen" if stage2_activated else "Stage1 frozen"
        print("=" * 60)
        print(f"Epoch [{epoch+1}/{total_epochs}] LR={current_lr:.2e} [{stage_name}]")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, logger, epoch,
        )
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device
        )
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val IoU   : {val_iou:.4f}")
        print(f"Val Dice  : {val_dice:.4f}")

        logger.log_epoch(
            epoch, train_loss, val_loss,
            val_iou, val_dice, current_lr,
        )
        if (epoch + 1) % 5 == 0 or epoch + 1 == total_epochs:
            logger.plot()

        iou_improved = val_iou > best_iou
        dice_improved = val_dice > best_dice

        if iou_improved:
            best_iou = val_iou
        if dice_improved:
            best_dice = val_dice

        scheduler.step()

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_iou": best_iou,
            "best_dice": best_dice,
        }

        if iou_improved and TRAIN["save_best_iou"]:
            save_checkpoint(
                ckpt_state,
                os.path.join(TRAIN["save_dir"], "best_iou.pth"),
            )
            print(f"[Save] best_iou.pth -> {best_iou:.4f}")

        if dice_improved and TRAIN["save_best_dice"]:
            save_checkpoint(
                ckpt_state,
                os.path.join(TRAIN["save_dir"], "best_dice.pth"),
            )
            print(f"[Save] best_dice.pth -> {best_dice:.4f}")

        save_checkpoint(
            ckpt_state,
            os.path.join(TRAIN["save_dir"], "last.pth"),
        )

    print("Training done.")


if __name__ == "__main__":
    main()