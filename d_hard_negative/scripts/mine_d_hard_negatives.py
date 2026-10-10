"""Mine D candidates; GT mismatch is never automatic negative confirmation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from lib.common import (CANDIDATE_FIELDS, components, config, identity, input_path, map_gt,
                        output_path, read_image, save_image, save_json, write_csv)
from lib.inference import load_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = config(args.config)
    root = output_path(f"data/mining/{cfg['run_id']}")
    if root.exists():
        raise FileExistsError(f"Use a fresh run_id: {root}")
    images = sorted(p for p in input_path(cfg["image_dir"]).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError("No input images")
    model, device, source = load_model(cfg)
    save_json(root / "config.json", dict(cfg, loaded_source=source))
    rows, image_rows = [], []
    for image_path in images:
        image = read_image(image_path)
        gt_path = input_path(cfg["mask_dir"]) / (image_path.stem + ".png")
        raw_gt = read_image(gt_path, True)
        if raw_gt.shape != image.shape[:2]:
            raise ValueError(f"GT dimensions differ: {image_path}")
        gt = map_gt(raw_gt, cfg["gt_mapping"])
        image_id = identity(image_path)
        image_rows.append(dict(image_id=image_id, image_path=str(image_path.resolve()),
                               gt_mask_path=str(gt_path.resolve()), group_id=""))
        probability = predict(model, image, device, cfg)
        threshold = float(cfg["d_threshold"])
        if not 0 < threshold < 1:
            raise ValueError("d_threshold must be between 0 and 1")
        labels, cc = components(probability > threshold)
        margin = int(cfg.get("d_margin", 3))
        if margin < 0:
            raise ValueError("d_margin must be nonnegative")
        protected = cv2.dilate((gt == 1).astype(np.uint8), np.ones((2*margin+1, 2*margin+1), np.uint8)) > 0
        groups = {}
        for cid, (x, y, w, h, area) in cc:
            region = labels == cid
            if area < int(cfg.get("min_area", 1)) or (region & protected).any() or (region & (gt == 255)).any():
                continue
            mean = float(probability[region].mean())
            size_bin = 0 if area <= 8 else 1 if area <= 32 else 2 if area <= 128 else 3
            conf_bin = 0 if mean < .7 else 1 if mean < .9 else 2
            groups.setdefault((size_bin, conf_bin), []).append((cid, x, y, w, h, area, mean))
        # Round-robin bins prevents large/high-confidence candidates consuming all slots.
        for group in groups.values():
            group.sort(key=lambda item: (-item[-1], item[0]))
        selected = []
        limit = int(cfg.get("per_image_limit", 60))
        if limit <= 0:
            raise ValueError("per_image_limit must be positive")
        while len(selected) < limit and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(selected) < limit:
                    selected.append(groups[key].pop(0))
        for cid, x, y, w, h, area, mean in selected:
            candidate_id = f"{image_id}_{cid}"
            base = root / "candidates" / candidate_id
            exact = (labels[y:y+h, x:x+w] == cid).astype(np.uint8)*255
            paths = {}
            for name, pad in (("raw", cfg.get("crop_padding", 40)), ("context", cfg.get("context_padding", 120))):
                pad = int(pad)
                x0, y0 = max(0, x-pad), max(0, y-pad)
                crop = image[y0:min(image.shape[0], y+h+pad), x0:min(image.shape[1], x+w+pad)]
                path = base.with_name(base.name+f"_{name}.png")
                save_image(path, crop)
                paths[name+"_path"] = str(path)
                if name == "raw":
                    save_image(base.with_name(base.name+"_zoom.png"), cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
                    overlay = crop.copy()
                    cv2.rectangle(overlay, (x-x0, y-y0), (x-x0+w-1, y-y0+h-1), (255, 0, 255), 1)
                    save_image(base.with_name(base.name+"_overlay.png"), overlay)
            component_path = base.with_name(base.name+"_component.png")
            save_image(component_path, exact)
            rows.append(dict(candidate_id=candidate_id, image_id=image_id,
                             image_path=str(image_path.resolve()), gt_mask_path=str(gt_path.resolve()),
                             x=x, y=y, w=w, h=h, area_px=area,
                             mean_confidence=mean, max_confidence=float(probability[labels == cid].max()),
                             gt_values=";".join(map(str, np.unique(raw_gt[labels == cid]).tolist())),
                             component_mask_path=str(component_path), **paths,
                             review_label="", negative_type="", positive_difficulty="", review_note="", positive_mask_path=""))
        print(f"{image_path.name}: {len(selected)} candidates", flush=True)
    write_csv(root / "candidates.csv", rows, CANDIDATE_FIELDS)
    write_csv(root / "images.csv", image_rows)
    print(root)


if __name__ == "__main__":
    main()
