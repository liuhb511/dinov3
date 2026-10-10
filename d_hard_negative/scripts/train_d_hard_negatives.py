"""Fine-tune independent D; audit images are never used for training."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from lib.common import config, output_path, read_csv, save_json, write_csv
from lib.dataset import DDataset
from lib.inference import load_model
from lib.loss import DLoss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, help="Smoke-test limit per epoch")
    args = p.parse_args()
    cfg = config(args.config)
    root = output_path(f"runs/{cfg['training_run']}")
    if root.exists():
        raise FileExistsError("Use a fresh training_run")
    all_rows = read_csv(output_path(f"data/masks/{cfg['supervision_id']}/manifest.csv"))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    audit_groups = {r["group_id"] for r in all_rows if r["split"] == "audit"}
    if not train_rows or any(r["group_id"] in audit_groups for r in train_rows):
        raise ValueError("Empty training data or audit leakage")
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = DDataset(train_rows, int(cfg.get("image_size", 1024)))
    factor = float(cfg.get("hard_negative_sampling_weight", 1))
    if factor < 1:
        raise ValueError("Sampling weight must be >=1")
    sampler = None
    if factor > 1:
        weights = [factor if int(r["has_negative"]) else 1 for r in train_rows]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True,
                                        generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(dataset, batch_size=int(cfg.get("batch_size", 2)),
                        shuffle=sampler is None, sampler=sampler,
                        num_workers=int(cfg.get("num_workers", 0)), drop_last=False)
    model, device, source = load_model(cfg)
    model.float()
    criterion = DLoss(cfg)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=float(cfg.get("learning_rate", 1e-5)),
                                  weight_decay=float(cfg.get("weight_decay", 1e-4)))
    use_amp = cfg.get("fp16", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    save_json(root / "config" / "config.json", dict(cfg, loaded_source=source, max_steps=args.max_steps))
    write_csv(root / "config" / "training_manifest.csv", train_rows)
    history = []
    epochs = int(cfg.get("epochs", 5))
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    for epoch in range(epochs):
        model.train()
        if cfg.get("freeze_backbone", True):
            model.encoder.eval()
        totals = {}
        steps = 0
        for images, gt, negative, _ in loader:
            images, gt, negative = images.to(device), gt.to(device), negative.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits, boundary = model(images)
            loss, parts = criterion(logits, boundary, gt, negative)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            for key, value in dict(total=loss.detach(), **parts).items():
                totals[key] = totals.get(key, 0)+float(value)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        row = dict(epoch=epoch+1, learning_rate=optimizer.param_groups[0]["lr"],
                   sampling_weight=factor, hard_negative_weight=criterion.hn_weight,
                   **{k: v/steps for k, v in totals.items()})
        history.append(row)
        write_csv(root / "logs" / "epochs.csv", history)
        ckpt = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), config=cfg, supervision_id=cfg["supervision_id"])
        path = output_path(root / "checkpoints" / f"epoch_{epoch+1:03d}.pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        print(row, flush=True)


if __name__ == "__main__":
    main()
