# -*- coding: utf-8 -*-
"""Train A-Small-v1 on the untouched baseline 7-class dataset.

A-Small-v1 =
  baseline DINOv3 + DecoderV3 main path
  + training-only tiny-D / small-TIND auxiliary head
  + object-size-aware CE weighting
  + tiny-target patch oversampling

No hard-negative data is used in this script. That belongs to A-HN/A-Combined.
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.grain_dataset import GrainDataset
from data.transforms import get_val_transform
from models.small import (
    DINOv3SegSmall,
    SmallTargetLoss,
    SmallTargetTargetBuilder,
    build_small_target_sampler,
    get_small_train_transform,
)
from utils.metric_JZW import update_confusion_matrix, calculate_metrics
from utils.seed import set_seed
from utils.scheduler import WarmupCosineScheduler
from utils.logger import TrainLogger


TRAIN = {
    # ---------- baseline dataset: DO NOT replace with HN-modified data ----------
    "dataset_root": r"D:/lhb/datasets/JZW_v3/JZW",

    # ---------- model ----------
    "backbone_name": "dinov3_model",
    "freeze_backbone": True,
    "num_classes": 7,     # 0 BG,1 A,2 B,3 C,4 D,5 TINBC,6 TIND
    "image_size": 1024,
    "small_aux_hidden": 32,

    # Optional: initialize shared weights from current A baseline checkpoint.
    # For the cleanest from-scratch ablation leave empty and keep the same seed/
    # pretrained DINO initialization as A0.
    "init_from_baseline": "",

    # ---------- physical scale / class IDs ----------
    "um_per_px": 0.5488,
    "d_class_id": 4,
    "tind_class_id": 6,

    # ---------- size-aware main CE ----------
    "d_weight_le3": 4.0,
    "d_weight_3_4": 3.0,
    "d_weight_4_5": 1.5,
    "tind_small_weight": 1.5,

    # ---------- auxiliary target ----------
    "d_aux_max_um": 5.0,
    "tind_aux_max_um": 5.0,
    "d_aux_dilate_px": 2,
    "tind_aux_dilate_px": 1,
    "d_aux_ring_px": 5,
    "tind_aux_ring_px": 3,
    "aux_bg_weight": 0.03,
    "aux_ring_weight": 0.25,
    "aux_pos_weight": 1.0,

    # ---------- auxiliary focal loss ----------
    "aux_d_loss_weight": 0.30,
    "aux_tind_loss_weight": 0.15,
    "aux_focal_alpha": 0.75,
    "aux_focal_gamma": 2.0,

    # ---------- patch oversampling ----------
    "use_small_sampler": True,
    "sample_weight_d_le3": 4.0,
    "sample_weight_d_3_4": 3.0,
    "sample_weight_d_4_5": 1.5,
    "sample_weight_tind_small": 1.5,
    "sampler_epoch_factor": 1.0,  # keep one epoch roughly same length as baseline

    # ---------- training: match A baseline unless you intentionally ablate ----------
    "batch_size": 4,
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

    # baseline main loss
    "ce_weight": 0.7,
    "dice_weight": 0.3,
    "boundary_weight": 0.0,

    # ---------- outputs ----------
    "save_dir": "./checkpoints/A_small_v1",
    "log_dir": "./logs/A_small_v1",
    "resume": "",
    "save_best_iou": True,
    "save_best_dice": True,
    "device": "cuda",
}


ROOT = TRAIN["dataset_root"]
train_image_dir = os.path.join(ROOT, "train", "images")
train_mask_dir = os.path.join(ROOT, "train", "masks")
val_image_dir = os.path.join(ROOT, "val", "images")
val_mask_dir = os.path.join(ROOT, "val", "masks")

CFG = SimpleNamespace(**TRAIN)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_shared_baseline_weights(model, path, device):
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    aux_missing = [k for k in missing if "small_aux_head" in k]
    non_aux_missing = [k for k in missing if "small_aux_head" not in k]
    print(f"Init from baseline: {path}")
    print(f"  expected new aux params: {len(aux_missing)} missing")
    if non_aux_missing:
        print("  WARNING non-aux missing keys:", non_aux_missing[:20])
    if unexpected:
        print("  WARNING unexpected keys:", unexpected[:20])


def train_one_epoch(model, loader, optimizer, criterion, target_builder, scaler, device, logger, epoch):
    model.train()
    total_loss = 0.0
    amp_enabled = TRAIN["amp"]
    grad_clip_val = TRAIN["grad_clip"]
    loop = tqdm(loader, desc="Train Small-v1")

    epoch_obj = {"d_le3": 0, "d_3_4": 0, "d_4_5": 0, "d_gt5": 0, "tind_small": 0, "tind_other": 0}

    for batch_idx, (images, masks, _) in enumerate(loop):
        # Build object-aware labels on CPU before moving GT to GPU.
        pixel_w, aux_target, aux_w, obj_stats = target_builder.build(masks)
        for k in epoch_obj:
            epoch_obj[k] += int(obj_stats.get(k, 0))

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        pixel_w = pixel_w.to(device, non_blocking=True)
        aux_target = aux_target.to(device, non_blocking=True)
        aux_w = aux_w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            seg, boundary, aux = model(images, return_aux=True)
            loss, parts = criterion(
                seg, boundary, aux, masks,
                pixel_weight=pixel_w,
                aux_target=aux_target,
                aux_weight=aux_w,
            )

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
        logger.log_batch(epoch, batch_idx, global_step, batch_loss, avg_loss, current_lr)

        loop.set_postfix(
            loss=f"{batch_loss:.4f}",
            main=f"{float(parts['main']):.3f}",
            daux=f"{float(parts['aux_d']):.3f}",
            taux=f"{float(parts['aux_tind']):.3f}",
        )

    print("Train small-object exposure:", epoch_obj)
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_cls = TRAIN["num_classes"]
    amp_enabled = TRAIN["amp"]
    cm = torch.zeros((num_cls, num_cls), dtype=torch.long, device=device)
    loop = tqdm(loader, desc="Val")

    for images, masks, _ in loop:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            # Auxiliary head is skipped during validation/inference.
            seg, boundary = model(images, return_aux=False)
            loss, _ = criterion.main_loss(seg, boundary, masks, pixel_weight=None)
        preds = torch.argmax(seg, dim=1)
        total_loss += float(loss.item())
        update_confusion_matrix(cm, preds, masks, num_classes=num_cls)
        loop.set_postfix(loss=f"{float(loss.item()):.4f}")

    mean_iou, mean_dice, class_iou, class_dice = calculate_metrics(cm, include_background=False)

    # Useful guardrails during training; final size-object evaluation still uses
    # the common unified12 benchmark/diagnostics script.
    denom_d = cm[4].sum().clamp_min(1)
    denom_tind = cm[6].sum().clamp_min(1)
    d_pixel_recall = (cm[4, 4].float() / denom_d.float()).item()
    tind_pixel_recall = (cm[6, 6].float() / denom_tind.float()).item()

    return total_loss / max(len(loader), 1), mean_iou, mean_dice, d_pixel_recall, tind_pixel_recall


def main():
    set_seed(TRAIN["seed"])
    device = torch.device(TRAIN["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(TRAIN["save_dir"], exist_ok=True)

    print("=" * 70)
    print("A-Small-v1 | baseline data only")
    print(f"Device: {device}")
    print(f"Dataset: {ROOT}")
    print(f"Scale: {TRAIN['um_per_px']} um/px | D={TRAIN['d_class_id']} | TIND={TRAIN['tind_class_id']}")
    print("Tiny-D weights:", TRAIN["d_weight_le3"], TRAIN["d_weight_3_4"], TRAIN["d_weight_4_5"])
    print("Aux loss weights: D=", TRAIN["aux_d_loss_weight"], " TIND=", TRAIN["aux_tind_loss_weight"])
    print("=" * 70)

    logger = TrainLogger(TRAIN["log_dir"])
    logger.save_config(TRAIN)

    train_dataset = GrainDataset(
        train_image_dir,
        train_mask_dir,
        transform=get_small_train_transform(),
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

    sampler = None
    if TRAIN["use_small_sampler"]:
        sampler, sample_summary = build_small_target_sampler(
            train_dataset,
            train_mask_dir,
            CFG,
            summary_json=os.path.join(TRAIN["save_dir"], "small_sampler_summary.json"),
        )
        print("Small sampler summary:", sample_summary)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
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

    model = DINOv3SegSmall(CFG).to(device)
    load_shared_baseline_weights(model, TRAIN["init_from_baseline"], device)

    criterion = SmallTargetLoss(CFG)
    target_builder = SmallTargetTargetBuilder(CFG)

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
    scaler = torch.cuda.amp.GradScaler(enabled=TRAIN["amp"])

    start_epoch = 0
    best_iou = 0.0
    best_dice = 0.0
    stage2_activated = False

    resume_path = TRAIN["resume"]
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_iou = float(ckpt.get("iou", 0.0))
        best_dice = float(ckpt.get("dice", 0.0))
        stage2_activated = bool(ckpt.get("stage2_activated", start_epoch > TRAIN["freeze_epochs"]))
        print(f"Resumed from epoch {start_epoch}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {trainable:,} trainable / {total_params:,} total")

    for epoch in range(start_epoch, TRAIN["epochs"]):
        if epoch == TRAIN["freeze_epochs"] and not stage2_activated:
            stage2_activated = True
            model.encoder.set_trainable(True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=TRAIN["unfreeze_lr"],
                weight_decay=TRAIN["weight_decay"],
            )
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=0,
                total_epochs=TRAIN["epochs"] - epoch,
                min_lr=TRAIN["min_lr"],
            )
            print(f"Stage2 activated at epoch {epoch + 1}, lr={TRAIN['unfreeze_lr']}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch+1}/{TRAIN['epochs']}] lr={current_lr:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, target_builder,
            scaler, device, logger, epoch,
        )
        val_loss, val_iou, val_dice, d_recall, tind_recall = validate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val IoU   : {val_iou:.4f}")
        print(f"Val Dice  : {val_dice:.4f}")
        print(f"Val D pixel recall    : {d_recall:.4f}")
        print(f"Val TIND pixel recall : {tind_recall:.4f}")

        logger.log_epoch(epoch, train_loss, val_loss, val_iou, val_dice, current_lr)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == TRAIN["epochs"]:
            logger.plot()
        scheduler.step()

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iou": val_iou,
            "dice": val_dice,
            "d_pixel_recall": d_recall,
            "tind_pixel_recall": tind_recall,
            "stage2_activated": stage2_activated,
            "train_config": dict(TRAIN),
        }
        if val_iou > best_iou and TRAIN["save_best_iou"]:
            best_iou = val_iou
            save_checkpoint(state, os.path.join(TRAIN["save_dir"], "best_iou.pth"))
            print(f" -> Saved best_iou.pth (IoU={best_iou:.4f})")
        if val_dice > best_dice and TRAIN["save_best_dice"]:
            best_dice = val_dice
            save_checkpoint(state, os.path.join(TRAIN["save_dir"], "best_dice.pth"))
            print(f" -> Saved best_dice.pth (Dice={best_dice:.4f})")
        save_checkpoint(state, os.path.join(TRAIN["save_dir"], "last.pth"))

    print(f"Training done. Best IoU={best_iou:.4f}, Best Dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
