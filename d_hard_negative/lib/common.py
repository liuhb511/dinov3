import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CANDIDATE_FIELDS = ["candidate_id", "image_id", "image_path", "gt_mask_path", "x", "y", "w", "h",
                    "area_px", "mean_confidence", "max_confidence", "gt_values", "component_mask_path",
                    "raw_path", "context_path", "review_label", "negative_type", "positive_difficulty",
                    "review_note", "positive_mask_path"]


def output_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output must stay within {ROOT}: {path}")
    return path


def input_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["candidate_id"])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def save_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_image(path, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def save_image(path, image):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path}")
    encoded.tofile(str(path))


def identity(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:16]


def map_gt(raw, mapping):
    mapping = {int(k): int(v) for k, v in mapping.items()}
    unknown = set(np.unique(raw).tolist()) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped GT values: {unknown}")
    if set(mapping.values()) - {0, 1, 255}:
        raise ValueError("D targets must be 0, 1 or 255")
    result = np.full(raw.shape, 255, np.uint8)
    for source, target in mapping.items():
        result[raw == source] = target
    return result


def components(binary):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    return labels, [(i, tuple(map(int, stats[i]))) for i in range(1, n)]


def exact_region(row, shape):
    x, y, w, h = [int(row[k]) for k in ("x", "y", "w", "h")]
    if x < 0 or y < 0 or x+w > shape[1] or y+h > shape[0]:
        raise ValueError("Candidate bounds outside image")
    component = read_image(row["component_mask_path"], True) > 0
    if component.shape != (h, w):
        raise ValueError("Component dimensions mismatch")
    region = np.zeros(shape, bool)
    region[y:y+h, x:x+w] = component
    return region
