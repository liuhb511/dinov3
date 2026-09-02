"""
train_newJZW.py  ——  DINOv3 + DecoderV3  两阶段训练
══════════════════════════════════════════════════════
所有配置在本文件顶部 TRAIN 字典中，零外部 config 依赖。
数据集路径通过 dataset_root 自动拼接子目录。

Stage1: 冻结 backbone → 训练 decoder
Stage2: 解冻 backbone → 全模型微调（更低 LR）
损失: CE + Dice   |   评估: 混淆矩阵 + mIoU/mDice
调度: Warmup + Cosine   |   日志: 实时 batch/epoch 级
══════════════════════════════════════════════════════
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有训练配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
TRAIN = {
    # ========== 数据集 ==========
    "dataset_root":    "D:/lhb/datasets/JZW_v3/JZW",     # 根目录，自动拼接子路径
    # "dataset_root":    "F:/liuhaibo/DINOv3-CARNet-Doo-main/dataset/BG_HQL_JZW_ABC_V2",     # 根目录，自动拼接子路径

    # ========== 模型结构（必须与预训练/checkpoint一致）==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,          # 模型构造参数

    # ========== num_classes 类别数，包含背景 ==========
    # 4(D类) / 5(ABCD) / 7(ABC) / 5(D+TIN-D) / 9(ABC+TIN-B+TIN-C) / 9（ABC+TIN）
    "num_classes":     7,

    "image_size":      1024,

    # ========== 训练超参 ==========
    "batch_size":      4,
    "num_workers":     4,
    "epochs":          100,
    "learning_rate":   1e-4,          # Stage1 学习率
    "unfreeze_lr":     1e-5,          # Stage2 更低的微调学习率
    "freeze_epochs":   60,           # 前 N 轮冻结 backbone
    "warmup_epochs":   5,
    "min_lr":          1e-6,
    "weight_decay":    1e-4,
    "grad_clip":       1.0,
    "amp":             True,          # 混合精度训练
    "seed":            42,

    # ========== 损失权重 ==========
    "ce_weight":       0.7,
    "dice_weight":     0.3,
    "boundary_weight": 0.2,             # Boundary Head 默认不启用

    # ========== 保存 / 日志 ==========
    "save_dir":        "./checkpoints/JZW_2",
    "log_dir":         "./logs/JZW_2",
    "resume":          "",            # 断点续训 checkpoint 路径，空=不续训
    "save_best_iou":   True,
    "save_best_dice":  True,

    # ========== 设备 ==========
    "device": "cuda",
}

# ---- 自动拼接数据集路径 ----
_dataset_root = TRAIN["dataset_root"]
train_image_dir = os.path.join(_dataset_root, "train", "images")
train_mask_dir  = os.path.join(_dataset_root, "train", "masks")
val_image_dir   = os.path.join(_dataset_root, "val",   "images")
val_mask_dir    = os.path.join(_dataset_root, "val",   "masks")

# ---- 模型构造函数所需的最小 cfg 对象 ----
_model_cfg = SimpleNamespace(
    backbone_name   = TRAIN["backbone_name"],
    freeze_backbone = TRAIN["freeze_backbone"],
    num_classes     = TRAIN["num_classes"],
    ce_weight       = TRAIN["ce_weight"],
    bce_weight      = TRAIN["ce_weight"],   # 兼容旧 TotalLoss 的 bce_weight 字段
    dice_weight     = TRAIN["dice_weight"],
    boundary_weight = TRAIN["boundary_weight"],
)

# ╔══════════════════════════════════════════════════════════════╗
# ║                        模块导入                               ║
# ╚══════════════════════════════════════════════════════════════╝
from data.grain_dataset import GrainDataset
from data.transforms import get_train_transform, get_val_transform
from models.dinov3_segmentation import DINOv3Seg
from losses.loss_JZW import TotalLoss
from utils.metric_JZW import update_confusion_matrix, calculate_metrics
from utils.seed import set_seed
from utils.scheduler import WarmupCosineScheduler
from utils.logger import TrainLogger


# ╔══════════════════════════════════════════════════════════════╗
# ║                        工具函数                               ║
# ╚══════════════════════════════════════════════════════════════╝
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, logger, epoch):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Train")
    amp_enabled = TRAIN["amp"]
    grad_clip_val = TRAIN["grad_clip"]

    for batch_idx, (images, masks, _) in enumerate(loop):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()
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

        batch_loss = loss.item()
        total_loss += batch_loss
        avg_loss = total_loss / (batch_idx + 1)
        current_lr = optimizer.param_groups[0]["lr"]
        global_step = epoch * len(loader) + batch_idx + 1
        logger.log_batch(epoch, batch_idx, global_step, batch_loss, avg_loss, current_lr)
        loop.set_postfix(loss=f"{batch_loss:.4f}", avg_loss=f"{avg_loss:.4f}", lr=f"{current_lr:.2e}")

    optimizer.zero_grad(set_to_none=True)
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_cls = TRAIN["num_classes"]
    amp_enabled = TRAIN["amp"]
    confusion_matrix = torch.zeros((num_cls, num_cls), dtype=torch.long, device=device)
    loop = tqdm(loader, desc="Val")

    for images, masks, _ in loop:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            seg, boundary = model(images)
            loss = criterion(seg, boundary, masks)

        preds = torch.argmax(seg, dim=1)
        total_loss += loss.item()
        update_confusion_matrix(confusion_matrix, preds, masks, num_classes=num_cls)
        loop.set_postfix(loss=f"{loss.item():.4f}")

    mean_iou, mean_dice, class_iou, class_dice = calculate_metrics(
        confusion_matrix, include_background=False
    )
    return total_loss / len(loader), mean_iou, mean_dice


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    set_seed(TRAIN["seed"])
    device = torch.device(TRAIN["device"] if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Backbone: {TRAIN['backbone_name']}  |  num_classes: {TRAIN['num_classes']}")
    print(f"Dataset: {TRAIN['dataset_root']}")
    print(f"  Train: {train_image_dir}")
    print(f"  Val:   {val_image_dir}")
    print(f"Config: image_size={TRAIN['image_size']}, batch={TRAIN['batch_size']}, "
          f"epochs={TRAIN['epochs']}, lr={TRAIN['learning_rate']}")
    print(f"Loss: ce={TRAIN['ce_weight']}, dice={TRAIN['dice_weight']}, "
          f"boundary={TRAIN['boundary_weight']}")
    print(f"Stage1: freeze {TRAIN['freeze_epochs']} epochs  "
          f"-> Stage2: unfreeze lr={TRAIN['unfreeze_lr']}")

    # --- Logger ---
    logger = TrainLogger(TRAIN["log_dir"])
    logger.save_config(TRAIN)

    # --- Dataset ---
    train_dataset = GrainDataset(
        train_image_dir, train_mask_dir,
        transform=get_train_transform(), is_train=True,
        image_size=TRAIN["image_size"])
    val_dataset = GrainDataset(
        val_image_dir, val_mask_dir,
        transform=get_val_transform(), is_train=False,
        image_size=TRAIN["image_size"])
    print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN["batch_size"], shuffle=True,
        num_workers=TRAIN["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=TRAIN["batch_size"], shuffle=False,
        num_workers=TRAIN["num_workers"], pin_memory=True)

    # --- Model ---
    model = DINOv3Seg(_model_cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total "
          f"({100*trainable/total:.1f}%)")

    # --- Loss / Optimizer / Scheduler / Scaler ---
    criterion = TotalLoss(_model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAIN["learning_rate"],
        weight_decay=TRAIN["weight_decay"])
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=TRAIN["warmup_epochs"],
        total_epochs=TRAIN["epochs"], min_lr=TRAIN["min_lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=TRAIN["amp"])

    best_iou, best_dice = 0.0, 0.0
    start_epoch = 0
    stage2_activated = False

    # --- Resume ---
    resume_path = TRAIN["resume"]
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou  = ckpt.get("iou", 0.0)
        best_dice = ckpt.get("dice", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(TRAIN["save_dir"], exist_ok=True)

    # --- Training Loop ---
    total_epochs    = TRAIN["epochs"]
    freeze_epochs   = TRAIN["freeze_epochs"]
    unfreeze_lr_val = TRAIN["unfreeze_lr"]
    weight_decay    = TRAIN["weight_decay"]
    min_lr_val      = TRAIN["min_lr"]
    save_dir        = TRAIN["save_dir"]
    save_best_iou   = TRAIN["save_best_iou"]
    save_best_dice  = TRAIN["save_best_dice"]

    for epoch in range(start_epoch, total_epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch == freeze_epochs and not stage2_activated:
            stage2_activated = True
            print(f"\n{'='*50}")
            print(f"Stage 2: Unfreezing backbone, lr={unfreeze_lr_val}")
            print(f"{'='*50}")
            model.encoder.set_trainable(True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=unfreeze_lr_val, weight_decay=weight_decay)
            remaining = total_epochs - epoch
            scheduler = WarmupCosineScheduler(
                optimizer, warmup_epochs=0, total_epochs=remaining, min_lr=min_lr_val)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Params: {trainable:,} trainable / {total:,} total "
                  f"({100*trainable/total:.1f}%)")

        print(f"\n{'='*50}")
        print(f"Epoch [{epoch+1}/{total_epochs}]  LR: {current_lr:.2e}  "
              f"{'[Stage2 finetune]' if stage2_activated else '[Stage1 frozen]'}")
        print(f"{'='*50}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      scaler, device, logger, epoch)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   IoU : {val_iou:.4f}")
        print(f"Val   Dice: {val_dice:.4f}")

        logger.log_epoch(epoch, train_loss, val_loss, val_iou, val_dice, current_lr)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs:
            logger.plot()
        scheduler.step()

        ckpt_state = {"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iou": val_iou, "dice": val_dice}

        if val_iou > best_iou and save_best_iou:
            best_iou = val_iou
            save_checkpoint(ckpt_state, os.path.join(save_dir, "best_iou.pth"))
            print(f"  -> Saved best_iou.pth (IoU={best_iou:.4f})")

        if val_dice > best_dice and save_best_dice:
            best_dice = val_dice
            save_checkpoint(ckpt_state, os.path.join(save_dir, "best_dice.pth"))
            print(f"  -> Saved best_dice.pth (Dice={best_dice:.4f})")

        save_checkpoint(ckpt_state, os.path.join(save_dir, "last.pth"))

    print(f"\nTraining done! Best IoU: {best_iou:.4f}, Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
