"""Fetch a balanced CIFAKE subset from a HuggingFace mirror (free, no auth).

CIFAKE: 32x32 CIFAR-10 reals + Stable-Diffusion 1.4 fakes. Writes a small
balanced subset to <out>/real and <out>/ai for eval.image_eval.

Prefer this over Kaggle (no account needed).
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_REPO = "dragonintelligence/CIFAKE-image-dataset"
DEFAULT_FILE = "data/test-00000-of-00001.parquet"
LABEL_REAL, LABEL_AI = 0, 1


def target_dir(label: int) -> str:
    """Map CIFAKE label to output subdir. 0=real, 1=AI/fake."""
    return "ai" if int(label) == LABEL_AI else "real"


def write_images(rows, out_dir: str) -> int:
    """rows: iterable of raw JPEG bytes. Returns count written."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, raw in enumerate(rows):
        with open(os.path.join(out_dir, "%05d.jpg" % i), "wb") as f:
            f.write(raw)
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("eval", "data", "images"))
    ap.add_argument("--limit", type=int, default=1000, help="images per class")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--file", default=DEFAULT_FILE)
    args = ap.parse_args(argv)

    try:
        import pandas as pd  # type: ignore
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:
        print("missing dependency: %s (pip install pandas pyarrow huggingface_hub)" % e)
        return 1

    try:
        path = hf_hub_download(args.repo, args.file, repo_type="dataset")
    except Exception as e:
        print("download failed: %s" % e)
        return 1

    df = pd.read_parquet(path)
    if "image" not in df.columns or "label" not in df.columns:
        print("unexpected parquet schema: %s" % list(df.columns))
        return 1

    total = 0
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == int(label)].head(args.limit)
        d = os.path.join(args.out, target_dir(int(label)))
        n = write_images((r["bytes"] for r in sub["image"]), d)
        print("%-4s -> %s (%d)" % (target_dir(int(label)), d, n))
        total += n
    print("wrote %d images to %s" % (total, args.out))
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())