"""
train_unetpp.py -- DINOv3 + UNet++ two-stage training.

Training policy intentionally follows the project's train_new.py:
Stage 1: freeze DINO backbone, train reassembly + UNet++ decoder + heads.
Stage 2: unfreeze DINO backbone and fine-tune the whole model at lower LR.

Loss / dataset / transforms / scheduler / metrics all reuse the existing project
implementations so the experiment is directly comparable with the baseline.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.grain_dataset import GrainDataset
from data.transforms import get_train_transform, get_val_transform
from losses.loss_JZW import TotalLoss
from models.unetpp.model import DINOv3UNetPP
from utils.logger import TrainLogger
from utils.metric_JZW import calculate_metrics, update_confusion_matrix
from utils.scheduler import WarmupCosineScheduler
from utils.seed import set_seed


# ╔══════════════════════════════════════════════════════════════╗
# ║                         训练配置                              ║
# ╚══════════════════════════════════════════════════════════════╝
TRAIN = {
    # ========== 数据集 ==========
    # 目录结构：
    # dataset_root/
    #   train/images, train/masks
    #   val/images,   val/masks
    "dataset_root": r"D:/lhb/datasets/JZW_v3/JZW",

    # ========== DINOv3 + UNet++ ==========
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "num_classes": 7,
    "image_size": 1024,

    # 四个晚期 Transformer hidden states。
    # 这是第一版的保守选择，尽量只验证 UNet++ nested decoder，
    # 避免把很浅层 DINO 特征的纹理敏感性同时引入实验。
    "feature_layers": (-4, -3, -2, -1),

    # DINO token feature 经过 reassembly 后形成：
    # p4=32@256, p8=64@128, p16=128@64, p32=256@32（1024输入）
    "pyramid_channels": (32, 64, 128, 256),
    "dino_dim": 768,

    # ========== 训练超参：保持 train_new.py 策略 ==========
    "batch_size": 2,
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

    # ========== 损失：按基础训练入口保持不变 ==========
    "ce_weight": 0.7,
    "dice_weight": 0.3,
    "boundary_weight": 0.2,

    # ========== 保存 / 日志 ==========
    "save_dir": "./checkpoints/A_unetpp_v2",
    "log_dir": "./logs/A_unetpp_v2",
    "resume": "",
    "save_best_iou": True,
    "save_best_dice": True,

    # ========== 设备 ==========
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
    pyramid_channels=TRAIN["pyramid_channels"],
    dino_dim=TRAIN["dino_dim"],
    ce_weight=TRAIN["ce_weight"],
    bce_weight=TRAIN["ce_weight"],
    dice_weight=TRAIN["dice_weight"],
    boundary_weight=TRAIN["boundary_weight"],
)


def validate_config() -> None:
    if TRAIN["num_classes"] < 2:
        raise ValueError("num_classes must include background and be >= 2")
    if TRAIN["image_size"] <= 0 or TRAIN["image_size"] % 32 != 0:
        raise ValueError("image_size must be >0 and divisible by 32")
    if len(TRAIN["feature_layers"]) != 4:
        raise ValueError("feature_layers must contain exactly 4 indices")
    if len(TRAIN["pyramid_channels"]) != 4:
        raise ValueError("pyramid_channels must contain exactly 4 values")
    if any(int(v) <= 0 for v in TRAIN["pyramid_channels"]):
        raise ValueError("all pyramid_channels must be > 0")
    if TRAIN["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if TRAIN["epochs"] <= 0:
        raise ValueError("epochs must be > 0")
    if not 0 <= TRAIN["freeze_epochs"] <= TRAIN["epochs"]:
        raise ValueError("freeze_epochs must be in [0, epochs]")
    if TRAIN["learning_rate"] <= 0 or TRAIN["unfreeze_lr"] <= 0:
        raise ValueError("learning rates must be > 0")

    for path_name, path in (
        ("train_image_dir", train_image_dir),
        ("train_mask_dir", train_mask_dir),
        ("val_image_dir", val_image_dir),
        ("val_mask_dir", val_mask_dir),
    ):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path_name} not found: {path}")


def save_checkpoint(state, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _amp_enabled(device: torch.device) -> bool:
    return bool(TRAIN["amp"] and device.type == "cuda")


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    logger,
    epoch,
):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Train")
    amp_enabled = _amp_enabled(device)
    grad_clip_val = float(TRAIN["grad_clip"])

    for batch_idx, (images, masks, _) in enumerate(loop):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            seg, boundary = model(images)
            loss = criterion(seg, boundary, masks)

        scaler.scale(loss).backward()

        if grad_clip_val > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

        scaler.step(optimizer)
        scaler.update()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        avg_loss = total_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]["lr"]
        global_step = epoch * len(loader) + batch_idx + 1

        logger.log_batch(
            epoch,
            batch_idx,
            global_step,
            batch_loss,
            avg_loss,
            current_lr,
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
    num_cls = int(TRAIN["num_classes"])
    amp_enabled = _amp_enabled(device)
    confusion_matrix = torch.zeros(
        (num_cls, num_cls),
        dtype=torch.long,
        device=device,
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
            confusion_matrix,
            preds,
            masks,
            num_classes=num_cls,
        )
        loop.set_postfix(loss=f"{loss.item():.4f}")

    mean_iou, mean_dice, _, _ = calculate_metrics(
        confusion_matrix,
        include_background=False,
    )
    return total_loss / max(len(loader), 1), mean_iou, mean_dice


def _build_optimizer(model, lr: float):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=TRAIN["weight_decay"],
    )


def main():
    validate_config()
    set_seed(TRAIN["seed"])

    device = torch.device(
        TRAIN["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(
        f"Model: DINOv3UNetPP | backbone={TRAIN['backbone_name']} | "
        f"num_classes={TRAIN['num_classes']}"
    )
    print(
        f"feature_layers={TRAIN['feature_layers']} | "
        f"pyramid_channels={TRAIN['pyramid_channels']}"
    )
    print(f"Dataset: {TRAIN['dataset_root']}")
    print(
        f"Config: image_size={TRAIN['image_size']}, batch={TRAIN['batch_size']}, "
        f"epochs={TRAIN['epochs']}, lr={TRAIN['learning_rate']}"
    )
    print(
        f"Loss: ce={TRAIN['ce_weight']}, dice={TRAIN['dice_weight']}, "
        f"boundary={TRAIN['boundary_weight']}"
    )
    print(
        f"Stage1: freeze {TRAIN['freeze_epochs']} epochs -> "
        f"Stage2: unfreeze lr={TRAIN['unfreeze_lr']}"
    )

    logger = TrainLogger(TRAIN["log_dir"])
    logger.save_config(TRAIN)

    train_dataset = GrainDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_train_transform(),
        is_train=True,
        image_size=TRAIN["image_size"],
    )
    val_dataset = GrainDataset(
        val_image_dir,
        val_mask_dir,
        transform=get_val_transform(),
        is_train=False,
        image_size=TRAIN["image_size"],
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN["batch_size"],
        shuffle=True,
        num_workers=TRAIN["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN["batch_size"],
        shuffle=False,
        num_workers=TRAIN["num_workers"],
        pin_memory=True,
    )

    model = DINOv3UNetPP(_model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Params: {trainable_params:,} trainable / {total_params:,} total "
        f"({100.0 * trainable_params / total_params:.1f}%)"
    )

    criterion = TotalLoss(_model_cfg)
    optimizer = _build_optimizer(model, TRAIN["learning_rate"])
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=TRAIN["warmup_epochs"],
        total_epochs=TRAIN["epochs"],
        min_lr=TRAIN["min_lr"],
    )

    amp_enabled = _amp_enabled(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_iou = 0.0
    best_dice = 0.0
    start_epoch = 0
    stage2_activated = False

    resume_path = TRAIN["resume"]
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_iou = float(ckpt.get("iou", 0.0))
        best_dice = float(ckpt.get("dice", 0.0))

        # Reconstruct the correct training stage before optimizer restoration.
        if start_epoch >= TRAIN["freeze_epochs"]:
            model.encoder.set_trainable(True)
            stage2_activated = True
            optimizer = _build_optimizer(model, TRAIN["unfreeze_lr"])
            remaining = max(TRAIN["epochs"] - start_epoch, 1)
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=remaining,
                min_lr=TRAIN["min_lr"],
            )
        else:
            optimizer = _build_optimizer(model, TRAIN["learning_rate"])
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=TRAIN["warmup_epochs"],
                total_epochs=TRAIN["epochs"],
                min_lr=TRAIN["min_lr"],
            )

        # Optimizer state is loaded only when compatible with reconstructed stage.
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print("[resume] optimizer state incompatible with reconstructed stage; using fresh optimizer")

        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(TRAIN["save_dir"], exist_ok=True)

    for epoch in range(start_epoch, TRAIN["epochs"]):
        if epoch == TRAIN["freeze_epochs"] and not stage2_activated:
            stage2_activated = True
            print("\n" + "=" * 60)
            print(f"Stage 2: Unfreezing DINO backbone, lr={TRAIN['unfreeze_lr']}")
            print("=" * 60)

            model.encoder.set_trainable(True)
            optimizer = _build_optimizer(model, TRAIN["unfreeze_lr"])
            remaining = TRAIN["epochs"] - epoch
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=remaining,
                min_lr=TRAIN["min_lr"],
            )

        current_lr = optimizer.param_groups[0]["lr"]
        print("\n" + "=" * 60)
        print(
            f"Epoch [{epoch + 1}/{TRAIN['epochs']}] LR: {current_lr:.2e} "
            f"{'[Stage2 finetune]' if stage2_activated else '[Stage1 frozen]'}"
        )
        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            logger,
            epoch,
        )
        val_loss, val_iou, val_dice = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val IoU   : {val_iou:.4f}")
        print(f"Val Dice  : {val_dice:.4f}")

        logger.log_epoch(
            epoch,
            train_loss,
            val_loss,
            val_iou,
            val_dice,
            current_lr,
        )
        if (epoch + 1) % 5 == 0 or (epoch + 1) == TRAIN["epochs"]:
            logger.plot()

        scheduler.step()

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iou": val_iou,
            "dice": val_dice,
        }

        if val_iou > best_iou and TRAIN["save_best_iou"]:
            best_iou = val_iou
            save_checkpoint(
                ckpt_state,
                os.path.join(TRAIN["save_dir"], "best_iou.pth"),
            )
            print(f" -> Saved best_iou.pth (IoU={best_iou:.4f})")

        if val_dice > best_dice and TRAIN["save_best_dice"]:
            best_dice = val_dice
            save_checkpoint(
                ckpt_state,
                os.path.join(TRAIN["save_dir"], "best_dice.pth"),
            )
            print(f" -> Saved best_dice.pth (Dice={best_dice:.4f})")

        save_checkpoint(
            ckpt_state,
            os.path.join(TRAIN["save_dir"], "last.pth"),
        )

    print(
        f"\nTraining done! Best IoU: {best_iou:.4f}, "
        f"Best Dice: {best_dice:.4f}"
    )


if __name__ == "__main__":
    main()
