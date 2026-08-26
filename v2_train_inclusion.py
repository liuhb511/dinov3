"""
v2_train_inclusion.py —— 非金属夹杂物识别 MVP-1 训练入口（共享 DINOv3 + 双专家 Head）
══════════════════════════════════════════════════════
- 数据：用户提供的全类别（12 类 + bg）unified mask，train/val 划分
- 模型：inclusion_v2.InclusionDualExpertNet（官方 DINOv3 随机初始化，不 warm-start）
- 两阶段：Stage1 冻结 backbone → Stage2 解冻低 LR 微调
- 损失：Gate + Strip(0.6·OHEM + 0.4·Dice + λ·Rejection) + Point(同构) + Boundary
- 指标：Inclusion Precision/Recall/F1 + 定向 FP rate
- Checkpoint：best_inclusion_precision / best_inclusion_f1 / last
- 训练步数控制：max_total_steps 可限制总训练步数（对齐旧训练预算）

所有配置在本文件顶部 TRAIN 字典中修改。
══════════════════════════════════════════════════════
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

# ╔══════════════════════════════════════════════════════════════╗
# ║                    所有训练配置（在此修改）                    ║
# ╚══════════════════════════════════════════════════════════════╝
TRAIN = {
    # ========== 数据集（用户提供全类别 mask，train/val 已划分）==========
    "dataset_root":    "F:/liuhaibo/datasets/JZW_v3/inclusion_unified",  # 改为你的数据根目录

    # ========== 模型结构 ==========
    "backbone_name":   "dinov3_model",
    "freeze_backbone": True,          # 构造参数；两阶段训练时 Stage2 会解冻
    "encoder_layers":  (4, 8, 12),    # L4 / L8 / L12
    "feat_dim":        768,
    "fusion_dim":      512,
    "decoder_dim":     32,
    "image_size":      1024,          # 训练 patch 尺寸（滑窗推理用 1024）

    # ========== 融合 ==========
    "fusion_alpha":    0.5,           # P(cls)=P_expert×P_gate^α，验证用

    # ========== 训练超参 ==========
    "batch_size":      4,
    "num_workers":     4,
    "epochs":          100,
    "learning_rate":   1e-4,
    "unfreeze_lr":     1e-5,
    "freeze_epochs":   60,
    "warmup_epochs":   5,
    "min_lr":          1e-6,
    "weight_decay":    1e-4,
    "grad_clip":       1.0,
    "amp":             True,
    "seed":            42,
    "max_total_steps": 0,             # >0 时达到该总步数即停（对齐旧训练预算）

    # ========== 损失权重 ==========
    "gate_weight":             1.0,
    "strip_weight":            1.0,
    "point_weight":            1.0,
    "boundary_weight":         0.1,
    "strip_ce_weight":         0.6,
    "strip_dice_weight":       0.4,
    "strip_rejection_weight":  0.1,
    "point_ce_weight":         0.6,
    "point_dice_weight":       0.4,
    "point_rejection_weight":  0.1,
    "use_focal":               False,   # True 用 Focal-CE，False 用 OHEM
    "ohem_min_kept":           100000,

    # ========== 保存 / 日志 ==========
    "save_dir":        "./checkpoints/inclusion_v2_mvp1",
    "log_dir":         "./logs/inclusion_v2_mvp1",
    "resume":          "",

    # ========== 设备 ==========
    "device": "cuda",
}

# ---- 数据集路径 ----
_dataset_root = TRAIN["dataset_root"]
train_image_dir = os.path.join(_dataset_root, "train", "images")
train_mask_dir  = os.path.join(_dataset_root, "train", "masks")
val_image_dir   = os.path.join(_dataset_root, "val",   "images")
val_mask_dir    = os.path.join(_dataset_root, "val",   "masks")

# ---- 模型/损失所需的最小 cfg ----
_model_cfg = SimpleNamespace(
    backbone_name   = TRAIN["backbone_name"],
    freeze_backbone = TRAIN["freeze_backbone"],
    encoder_layers  = TRAIN["encoder_layers"],
    feat_dim        = TRAIN["feat_dim"],
    fusion_dim      = TRAIN["fusion_dim"],
    decoder_dim     = TRAIN["decoder_dim"],
)

# ---- 损失权重 cfg ----
_LOSS_KEYS = [
    "gate_weight", "strip_weight", "point_weight", "boundary_weight",
    "strip_ce_weight", "strip_dice_weight", "strip_rejection_weight",
    "point_ce_weight", "point_dice_weight", "point_rejection_weight",
    "use_focal", "ohem_min_kept",
]
_loss_cfg = SimpleNamespace(**{k: TRAIN[k] for k in _LOSS_KEYS})


# ╔══════════════════════════════════════════════════════════════╗
# ║                        模块导入                               ║
# ╚══════════════════════════════════════════════════════════════╝
from utils.seed import set_seed
from utils.scheduler import WarmupCosineScheduler
from utils.logger import TrainLogger

from inclusion_v2.models import InclusionDualExpertNet
from inclusion_v2.losses import InclusionTotalLoss
from inclusion_v2.data import InclusionDataset, quick_validate_labels
from inclusion_v2.data.transforms import get_train_transform, get_val_transform
from inclusion_v2.utils.output_fusion import fuse_outputs
from inclusion_v2.metrics import InclusionMetricsAccumulator


# ╔══════════════════════════════════════════════════════════════╗
# ║                        工具函数                               ║
# ╚══════════════════════════════════════════════════════════════╝
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def mean_iou_from_cm(cm, include_background=False, eps=1e-6):
    """从 12 类混淆矩阵计算 mIoU（不含背景）。"""
    cm = cm.float()
    tp = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - tp
    iou = (tp + eps) / (union + eps)
    valid = union > 0
    if not include_background:
        valid[0] = False
    if valid.any():
        return float(iou[valid].mean().item())
    return 0.0


def mean_dice_from_cm(cm, include_background=False, eps=1e-6):
    cm = cm.float()
    tp = torch.diag(cm)
    denom = cm.sum(dim=1) + cm.sum(dim=0)
    dice = (2.0 * tp + eps) / (denom + eps)
    valid = denom > 0
    if not include_background:
        valid[0] = False
    if valid.any():
        return float(dice[valid].mean().item())
    return 0.0


def write_inclusion_csv(path, epoch, metrics):
    """把业务指标追加写入独立 CSV。"""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({"epoch": epoch + 1, **{k: round(v, 6) for k, v in metrics.items()}})


# ╔══════════════════════════════════════════════════════════════╗
# ║                        训练循环                               ║
# ╚══════════════════════════════════════════════════════════════╝
def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    logger, epoch, global_step, amp_enabled, grad_clip_val):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc=f"Train E{epoch + 1}")

    for batch_idx, (images, targets, _) in enumerate(loop):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)

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
        global_step += 1

        logger.log_batch(epoch, batch_idx, global_step, batch_loss, avg_loss, current_lr)
        loop.set_postfix(loss=f"{batch_loss:.4f}",
                         gate=f"{loss_dict['gate']:.3f}",
                         strip=f"{loss_dict['strip_ce']:.3f}",
                         point=f"{loss_dict['point_ce']:.3f}",
                         lr=f"{current_lr:.2e}")

        if TRAIN["max_total_steps"] > 0 and global_step >= TRAIN["max_total_steps"]:
            print(f"[Train] 达到 max_total_steps={TRAIN['max_total_steps']}，提前停止训练。")
            break

    optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, (batch_idx + 1)), global_step


@torch.no_grad()
def validate(model, loader, criterion, device, alpha):
    model.eval()
    total_loss = 0.0
    amp_enabled = TRAIN["amp"]
    acc = InclusionMetricsAccumulator(device=device)

    loop = tqdm(loader, desc="Val")
    for images, targets, _ in loop:
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            loss, _ = criterion(outputs, targets)

        probs = fuse_outputs(outputs["gate"], outputs["strip"], outputs["point"], alpha=alpha)
        pred = torch.argmax(probs, dim=1)
        total_loss += loss.item()
        acc.update(pred, targets["mask"])
        loop.set_postfix(loss=f"{loss.item():.4f}")

    metrics = acc.compute()
    cm = acc.cm
    m_iou = mean_iou_from_cm(cm)
    m_dice = mean_dice_from_cm(cm)
    return total_loss / max(1, len(loader)), m_iou, m_dice, metrics


# ╔══════════════════════════════════════════════════════════════╗
# ║                          主入口                               ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    set_seed(TRAIN["seed"])
    device = torch.device(TRAIN["device"] if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Backbone: {TRAIN['backbone_name']}  |  layers={TRAIN['encoder_layers']}")
    print(f"Dataset: {TRAIN['dataset_root']}")
    print(f"  Train: {train_image_dir}")
    print(f"  Val:   {val_image_dir}")

    # --- 数据校验（mask 编码必须是 unified 12 类）---
    print("\n[Data] 校验 mask 类别值（抽查前 64 张）...")
    train_stats = quick_validate_labels(train_mask_dir, num_files=64)
    val_stats = quick_validate_labels(val_mask_dir, num_files=64)
    print(f"[Data] train 出现的类别: {sorted(train_stats.keys())}")
    print(f"[Data] val   出现的类别: {sorted(val_stats.keys())}")

    # --- Logger ---
    logger = TrainLogger(TRAIN["log_dir"])
    logger.save_config(TRAIN)
    incl_csv = os.path.join(TRAIN["log_dir"], "inclusion_metrics.csv")

    # --- Dataset / DataLoader ---
    train_dataset = InclusionDataset(train_image_dir, train_mask_dir,
                                     transform=get_train_transform())
    val_dataset = InclusionDataset(val_image_dir, val_mask_dir,
                                   transform=get_val_transform())
    print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=TRAIN["batch_size"], shuffle=True,
                              num_workers=TRAIN["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN["batch_size"], shuffle=False,
                            num_workers=TRAIN["num_workers"], pin_memory=True)

    # --- Model ---
    model = InclusionDualExpertNet(_model_cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.1f}%)")

    # --- Loss / Optimizer / Scheduler / Scaler ---
    criterion = InclusionTotalLoss(_loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["learning_rate"],
                                  weight_decay=TRAIN["weight_decay"])
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=TRAIN["warmup_epochs"],
                                      total_epochs=TRAIN["epochs"], min_lr=TRAIN["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN["amp"])

    best_precision, best_f1 = 0.0, 0.0
    start_epoch = 0
    global_step = 0
    stage2_activated = False

    # --- Resume ---
    resume_path = TRAIN["resume"]
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_precision = ckpt.get("precision", 0.0)
        best_f1 = ckpt.get("f1", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(TRAIN["save_dir"], exist_ok=True)

    total_epochs = TRAIN["epochs"]
    freeze_epochs = TRAIN["freeze_epochs"]
    unfreeze_lr_val = TRAIN["unfreeze_lr"]
    weight_decay = TRAIN["weight_decay"]
    min_lr_val = TRAIN["min_lr"]
    save_dir = TRAIN["save_dir"]
    alpha = TRAIN["fusion_alpha"]

    for epoch in range(start_epoch, total_epochs):
        if epoch == freeze_epochs and not stage2_activated:
            stage2_activated = True
            print(f"\n{'=' * 50}")
            print(f"Stage 2: Unfreezing backbone, lr={unfreeze_lr_val}")
            print(f"{'=' * 50}")
            model.encoder.set_trainable(True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=unfreeze_lr_val,
                                          weight_decay=weight_decay)
            remaining = total_epochs - epoch
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=0,
                                              total_epochs=remaining, min_lr=min_lr_val)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Params: {trainable:,} trainable / {total:,} total "
                  f"({100 * trainable / total:.1f}%)")

        print(f"\n{'=' * 50}")
        print(f"Epoch [{epoch + 1}/{total_epochs}]  LR: {optimizer.param_groups[0]['lr']:.2e}  "
              f"{'[Stage2 finetune]' if stage2_activated else '[Stage1 frozen]'}")
        print(f"{'=' * 50}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            logger, epoch, global_step, TRAIN["amp"], TRAIN["grad_clip"])

        val_loss, val_iou, val_dice, val_metrics = validate(
            model, val_loader, criterion, device, alpha)

        print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_iou:.4f}  |  Val mDice: {val_dice:.4f}")
        print(f"Inclusion Precision: {val_metrics['inclusion_precision']:.4f}  "
              f"Recall: {val_metrics['inclusion_recall']:.4f}  "
              f"F1: {val_metrics['inclusion_f1']:.4f}")
        print(f"  HH→A/C: {val_metrics['hh_to_ac_fp_rate']:.4f}  "
              f"HC→D: {val_metrics['hc_to_d_fp_rate']:.4f}  "
              f"SZ→D: {val_metrics['sz_to_d_fp_rate']:.4f}  "
              f"Bg→D: {val_metrics['bg_to_d_fp_rate']:.4f}")

        logger.log_epoch(epoch, train_loss, val_loss, val_iou, val_dice,
                         optimizer.param_groups[0]["lr"])
        write_inclusion_csv(incl_csv, epoch, val_metrics)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == total_epochs:
            logger.plot()
        scheduler.step()

        ckpt_state = {
            "epoch": epoch, "global_step": global_step,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "precision": best_precision, "f1": best_f1,
            "val_metrics": val_metrics,
        }

        if val_metrics["inclusion_precision"] > best_precision:
            best_precision = val_metrics["inclusion_precision"]
            save_checkpoint(ckpt_state, os.path.join(save_dir, "best_inclusion_precision.pth"))
            print(f"  -> Saved best_inclusion_precision.pth (P={best_precision:.4f})")

        if val_metrics["inclusion_f1"] > best_f1:
            best_f1 = val_metrics["inclusion_f1"]
            save_checkpoint(ckpt_state, os.path.join(save_dir, "best_inclusion_f1.pth"))
            print(f"  -> Saved best_inclusion_f1.pth (F1={best_f1:.4f})")

        save_checkpoint(ckpt_state, os.path.join(save_dir, "last.pth"))

        if TRAIN["max_total_steps"] > 0 and global_step >= TRAIN["max_total_steps"]:
            print(f"\n达到 max_total_steps={TRAIN['max_total_steps']}，训练结束。")
            break

    print(f"\nTraining done! Best Inclusion Precision: {best_precision:.4f}, "
          f"Best Inclusion F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()




