"""Compare checkpoints on confirmed local audit targets, not full-image recall."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from lib.common import (components, config, exact_region, output_path, read_csv,
                        read_image, save_image, save_json, write_csv)
from lib.inference import load_model, predict


def target_hit(pred, region, fraction):
    return float((pred & region).sum())/max(1, int(region.sum())) >= fraction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="New checkpoint; baseline comes from config")
    args = p.parse_args()
    cfg = config(args.config)
    root = output_path(f"runs/{cfg['audit_run']}/evaluation")
    if root.exists():
        raise FileExistsError("Use a fresh audit_run")
    masks_root = output_path(f"data/masks/{cfg['supervision_id']}")
    images = [r for r in read_csv(masks_root / "manifest.csv") if r["split"] == "audit"]
    targets = read_csv(masks_root / "audit_targets.csv")
    fraction = float(cfg.get("audit_overlap_fraction", .1))
    if not images or not 0 < fraction <= 1:
        raise ValueError("Need audit images and valid overlap fraction")
    details = []
    baseline_predictions = {}
    for name, checkpoint in (("baseline", cfg["checkpoint"]), ("new", args.checkpoint)):
        model, device, source = load_model(cfg, checkpoint)
        for row in images:
            image = read_image(row["image_path"])
            prob = predict(model, image, device, cfg)
            pred = prob > float(cfg["d_threshold"])
            gt = read_image(row["gt_path"], True)
            image_targets = []
            confirmed_regions = []
            for target in targets:
                if target["image_id"] == row["image_id"]:
                    region = exact_region(target, gt.shape)
                    image_targets.append((target["candidate_id"], "negative" if target["review_label"] == "0" else "positive_review", region, int(region.sum()), target.get("positive_difficulty", "")))
                    confirmed_regions.append(region)
            labels, cc = components(gt == 1)
            # These GT components still require confidence in the original label.
            for cid, (_, _, _, _, area) in cc:
                region = labels == cid
                if not any((region & reviewed).any() for reviewed in confirmed_regions):
                    image_targets.append((f"gt_{cid}", "positive_gt_unverified", region, area, ""))
            for tid, kind, region, area, difficulty in image_targets:
                details.append(dict(model=name, image_id=row["image_id"], target_id=tid, kind=kind,
                                    area_px=area, positive_difficulty=difficulty, hit=int(target_hit(pred, region, fraction)),
                                    mean_d_probability=float(prob[region].mean())))
            overlay = image.copy()
            overlay[pred] = (.6*overlay[pred]+.4*np.array([255, 0, 255])).astype(np.uint8)
            save_image(root / name / (row["image_id"]+".png"), overlay)
            save_image(root / name / (row["image_id"]+"_mask.png"), pred.astype(np.uint8))
            if name == "baseline":
                baseline_predictions[row["image_id"]] = overlay
            else:
                save_image(root / "comparison" / (row["image_id"]+".png"), np.hstack([baseline_predictions[row["image_id"]], overlay]))
        del model
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    write_csv(root / "targets.csv", details)
    summary = {}
    for name in ("baseline", "new"):
        summary[name] = {}
        for kind in ("negative", "positive_review", "positive_gt_unverified"):
            subset = [r for r in details if r["model"] == name and r["kind"] == kind]
            summary[name][kind] = dict(targets=len(subset), hits=sum(r["hit"] for r in subset),
                                       misses=sum(1-r["hit"] for r in subset))
        small = [r for r in details if r["model"] == name and r["kind"] == "positive_review" and r["area_px"] <= 32]
        summary[name]["small_positive"] = dict(targets=len(small), hits=sum(r["hit"] for r in small))
        weak = [r for r in details if r["model"] == name and r["kind"] == "positive_review" and r["positive_difficulty"] == "weak"]
        summary[name]["weak_positive"] = dict(targets=len(weak), hits=sum(r["hit"] for r in weak))
    save_json(root / "summary.json", summary)
    save_json(root / "config.json", dict(cfg, new_checkpoint=args.checkpoint,
                                         limitation="Local overlap audit; positive_gt not independently verified; not full-image recall"))
    print(summary)


if __name__ == "__main__":
    main()
