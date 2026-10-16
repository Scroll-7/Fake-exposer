"""Build a balanced image split (`real/` + `ai/`) from a labelled CSV.

Generic helper for the honest image benchmarks: reads a CSV with an image-path
column and a label column, and copies (or moves) up to `--limit` images per
class into `<out>/real` and `<out>/ai`. The `--real-label` value marks the real
class; every other label is treated as AI.

Example — 2025 Women-in-AI "AI vs Human-Generated Images" (label 0 = human,
1 = AI-generated):

    python -m eval.make_split --csv "archive (1)/train.csv" \
        --root "archive (1)" --out eval/data/aihuman --limit 1000
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def label_to_dir(label, real_label=0):
    """Return 'real' or 'ai' for a label (everything not real is AI)."""
    return "real" if str(label) == str(real_label) else "ai"


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="labelled CSV")
    ap.add_argument("--root", default=".", help="root the CSV paths are relative to")
    ap.add_argument("--out", required=True, help="output dir (gets real/ and ai/)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--path-col", default="file_name")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--real-label", default="0")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    args = ap.parse_args(argv)

    counts = {"real": 0, "ai": 0}
    transfer = shutil.move if args.move else shutil.copyfile

    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = (row.get(args.path_col) or "").strip()
            label = (row.get(args.label_col) or "").strip()
            if not rel or label == "" or not is_image(rel):
                continue
            d = label_to_dir(label, args.real_label)
            if counts[d] >= args.limit:
                if counts["real"] >= args.limit and counts["ai"] >= args.limit:
                    break
                continue
            path = os.path.join(args.root, rel)
            if not os.path.exists(path):
                continue
            dest_dir = os.path.join(args.out, d)
            os.makedirs(dest_dir, exist_ok=True)
            ext = os.path.splitext(rel)[1].lower()
            transfer(path, os.path.join(dest_dir, "%05d%s" % (counts[d], ext)))
            counts[d] += 1

    print("real: %d, ai: %d -> %s" % (counts["real"], counts["ai"], args.out))
    return 0 if (counts["real"] or counts["ai"]) else 1


if __name__ == "__main__":
    sys.exit(main())
