"""Build exact task-specific supervision; original GT is never overwritten."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from lib.common import (CANDIDATE_FIELDS, config, exact_region, input_path, map_gt, output_path,
                        read_csv, read_image, save_image, save_json, write_csv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = config(args.config)
    root = output_path(f"data/masks/{cfg['supervision_id']}")
    if root.exists():
        raise FileExistsError(root)
    reviews = read_csv(output_path(cfg["review_csv"]))
    images = read_csv(output_path(cfg["split_csv"]))
    if len({r["image_id"] for r in images}) != len(images):
        raise ValueError("Duplicate image IDs in split")
    if len({str(input_path(r["image_path"])) for r in images}) != len(images):
        raise ValueError("Duplicate source image paths in split")
    by_image = {}
    for row in reviews:
        if row["review_label"] not in {"", "0", "1", "3"}:
            raise ValueError("Unexpected review label")
        by_image.setdefault(row["image_id"], []).append(row)
    if set(by_image) - {r["image_id"] for r in images}:
        raise ValueError("Reviewed images absent from split")
    groups = {}
    manifest, conflicts, audit_rows = [], [], []
    total_negative_pixels = 0
    for image_row in images:
        split = image_row["split"]
        if split not in {"train", "audit"}:
            raise ValueError("split must be train/audit")
        group = image_row["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError("Specimen/image group leaks across splits")
        groups[group] = split
        gt = map_gt(read_image(image_row["gt_mask_path"], True), cfg["gt_mapping"])
        negative = np.zeros(gt.shape, bool)
        ignored = gt == 255
        positive = np.zeros(gt.shape, bool)
        regions = []
        for row in by_image.get(image_row["image_id"], []):
            region = exact_region(row, gt.shape)
            regions.append((row, region))
            label = row["review_label"]
            if split == "audit" and label in {"0", "1"}:
                audit_rows.append(dict(row, split=split, group_id=group))
            if label == "0":
                negative |= region
            elif label == "3":
                ignored |= region
            elif label == "1":
                if row.get("positive_mask_path"):
                    correction = read_image(input_path(row["positive_mask_path"]), True) > 0
                    if correction.shape != gt.shape:
                        raise ValueError("Positive correction must be full-image size")
                    positive |= correction
                else:
                    # Knowing an object is D does not make its predicted outline a true label.
                    ignored |= region & (gt != 1)
        gt[positive] = 1
        collision = negative & ((gt == 1) | ignored)
        for row, region in regions:
            if row["review_label"] == "0" and (region & collision).any():
                conflicts.append(dict(candidate_id=row["candidate_id"], image_id=row["image_id"],
                                      conflict_pixels=int((region & collision).sum())))
                # Exclude entire conflicting candidate, not just intersecting pixels.
                negative[region] = False
                ignored[region] = True
        negative &= ~ignored
        total_negative_pixels += int(negative.sum())
        gt[negative] = 0
        gt[ignored] = 255
        paths = {}
        for name, array in (("gt", gt), ("negative", negative.astype(np.uint8)), ("ignore", ignored.astype(np.uint8))):
            path = root / split / name / (image_row["image_id"]+".png")
            save_image(path, array)
            paths[name+"_path"] = str(path)
        manifest.append(dict(image_row, **paths, has_negative=int(negative.any())))
    write_csv(root / "manifest.csv", manifest)
    write_csv(root / "review_snapshot.csv", reviews)
    write_csv(root / "split_snapshot.csv", images)
    write_csv(root / "conflicts.csv", conflicts, ["candidate_id", "image_id", "conflict_pixels"])
    write_csv(root / "audit_targets.csv", audit_rows, (list(reviews[0]) if reviews else CANDIDATE_FIELDS)+["split", "group_id"])
    save_json(root / "config.json", cfg)
    conflict_ids = {r["candidate_id"] for r in conflicts}
    save_json(root / "summary.json", dict(images=len(manifest), candidates=len(reviews),
                                          reviewed=sum(bool(r["review_label"]) for r in reviews),
                                          used_negative_candidates=sum(r["review_label"] == "0" and r["candidate_id"] not in conflict_ids for r in reviews),
                                          conflicts=len(conflicts),
                                          train_images=sum(r["split"] == "train" for r in manifest),
                                          negative_pixels=total_negative_pixels,
                                          audit_targets=len(audit_rows)))
    print(f"Built {root}; conflicts={len(conflicts)} (excluded from supervision)")


if __name__ == "__main__":
    main()
