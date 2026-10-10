"""Append original-GT D components for independent review, without auto-confirmation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from lib.common import (CANDIDATE_FIELDS, components, config, input_path, map_gt, output_path,
                        read_csv, read_image, save_image, save_json, write_csv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True, help="New task-relative bundle directory")
    args = p.parse_args()
    cfg = config(args.config)
    root = output_path(args.output)
    if root.exists():
        raise FileExistsError(root)
    rows = read_csv(input_path(args.candidates))
    images = read_csv(output_path(cfg["split_csv"]))
    for row in images:
        if row["split"] != "audit":
            continue
        image = read_image(row["image_path"])
        gt = map_gt(read_image(row["gt_mask_path"], True), cfg["gt_mapping"])
        labels, cc = components(gt == 1)
        for cid, (x, y, w, h, area) in cc:
            candidate_id = f"{row['image_id']}_gt_{cid}"
            base = root / "targets" / candidate_id
            component_path = base.with_name(base.name+"_component.png")
            save_image(component_path, (labels[y:y+h, x:x+w] == cid).astype(np.uint8)*255)
            paths = {}
            for name, pad in (("raw", 40), ("context", 120)):
                crop = image[max(0, y-pad):min(image.shape[0], y+h+pad),
                             max(0, x-pad):min(image.shape[1], x+w+pad)]
                path = base.with_name(base.name+f"_{name}.png")
                save_image(path, crop)
                paths[name+"_path"] = str(path)
                if name == "raw":
                    save_image(base.with_name(base.name+"_zoom.png"), cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
            entry = {key: "" for key in CANDIDATE_FIELDS}
            entry.update(candidate_id=candidate_id, image_id=row["image_id"], image_path=row["image_path"],
                         gt_mask_path=row["gt_mask_path"], x=x, y=y, w=w, h=h, area_px=area,
                         mean_confidence="GT待复核", max_confidence="", gt_values="D",
                         component_mask_path=str(component_path), **paths)
            rows.append(entry)
    write_csv(root / "candidates.csv", rows, CANDIDATE_FIELDS)
    save_json(root / "config.json", cfg)
    print(f"Review {root / 'candidates.csv'}; label 1 confirms D, label 3 marks uncertainty")


if __name__ == "__main__":
    main()
