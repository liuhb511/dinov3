"""Create grouped train/audit split before mask generation."""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.common import input_path, output_path, read_csv, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="Mining images.csv; fill group_id with specimen ID when available")
    p.add_argument("--output", default="data/manifests/split.csv")
    p.add_argument("--audit-fraction", type=float, default=.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-by-image", action="store_true", help="Explicitly declare input images independent originals, not adjacent crops")
    args = p.parse_args()
    destination = output_path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    rows = read_csv(input_path(args.images))
    if not args.group_by_image and any(not r.get("group_id") for r in rows):
        raise ValueError("Fill group_id with source-image/specimen IDs; use --group-by-image only for independent originals")
    if not 0 < args.audit_fraction < 1:
        raise ValueError("audit-fraction must be between 0 and 1")
    groups = sorted({r.get("group_id") or r["image_id"] for r in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two independent image/specimen groups")
    random.Random(args.seed).shuffle(groups)
    audit = set(groups[:max(1, min(len(groups)-1, round(len(groups)*args.audit_fraction)))])
    for row in rows:
        row["group_id"] = row.get("group_id") or row["image_id"]
        row["split"] = "audit" if row["group_id"] in audit else "train"
    write_csv(destination, rows)


if __name__ == "__main__":
    main()
