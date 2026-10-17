"""Build a labelled image set for the "sports north-star" evaluation.

Why this exists
---------------
The only image benchmark that predicts production behaviour is one drawn from
the real deployment distribution (sports/news photos), not CIFAR-32 or a
single diffusion model. Every published image model here is domain-sensitive,
so the honest way to improve one is to *measure it on your own labelled set*.

This helper turns two folders of raw images into the three artefacts the rest
of the tooling already consumes:

  * ``<out>/real/`` and ``<out>/ai/``       — normalised folder layout
  * ``<out>/sports.csv``                    — ``file_name,label`` for
                                               ``bin/probe-nonescape.py features``
  * ``<out>/sports-manifest.jsonl``         — pipeline manifest for
                                               ``python -m eval.pipeline_eval``

Drop files into ``--real`` and ``--ai`` (or straight into ``<out>/real`` and
``<out>/ai``) and run it. Images are validated (decodable, minimum size) and
exact duplicates are dropped, because a duplicated image is a leaked example
and quietly inflates every metric.

Usage
-----
    python bin/build-sports-set.py --real path/to/real --ai path/to/ai \
        --out eval/data/sports --provenance "own photos + Flux free tier"

    # already dropped into the default folders:
    python bin/build-sports-set.py --out eval/data/sports --provenance "mixed"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MIN_PER_CLASS_HINT = 50


def iter_images(folder: str) -> list[str]:
    """Return sorted image paths under ``folder`` (recursive)."""
    out: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(root, fn))
    return sorted(out)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: str) -> tuple[int, int] | None:
    """Return ``(width, height)`` if the file is a decodable image, else None."""
    try:
        from PIL import Image
    except ImportError:
        return (-1, -1)

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def unique_dest(dest_dir: str, name: str) -> str:
    stem, ext = os.path.splitext(name)
    candidate = os.path.join(dest_dir, name)
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    return candidate


def collect_class(
    src_folder: str,
    dest_folder: str,
    label: int,
    limit: int,
    min_size: int,
    drop_duplicates: bool,
    seen_hashes: dict[str, int],
    in_place: bool,
) -> dict:
    """Copy/validate one class; return per-class report and accepted records."""
    os.makedirs(dest_folder, exist_ok=True)
    accepted: list[dict] = []
    report = {
        "label": label,
        "found": 0,
        "accepted": 0,
        "invalid": 0,
        "duplicate": 0,
        "conflict": 0,
        "too_small": 0,
    }

    for path in iter_images(src_folder):
        if limit and report["accepted"] >= limit:
            break
        report["found"] += 1

        size = image_size(path)
        if size is None:
            report["invalid"] += 1
            continue
        w, h = size
        if min(w, h) < min_size:
            report["too_small"] += 1
            continue

        digest = sha256_of(path)
        if drop_duplicates and digest in seen_hashes:
            if seen_hashes[digest] != label:
                report["conflict"] += 1
                print(
                    f"  [!] conflict: {os.path.basename(path)} has the same bytes as "
                    f"a label-{seen_hashes[digest]} image; skipping",
                    file=sys.stderr,
                )
            else:
                report["duplicate"] += 1
            continue
        seen_hashes[digest] = label

        if in_place:
            dest = path
        else:
            dest = unique_dest(dest_folder, os.path.basename(path))
            shutil.copy2(path, dest)

        accepted.append({"path": dest, "label": label, "sha256": digest})
        report["accepted"] += 1

    return {"report": report, "accepted": accepted}


def write_csv(path: str, records: list[dict], root: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file_name", "label"])
        for rec in records:
            rel = os.path.relpath(os.path.abspath(rec["path"]), os.path.abspath(root))
            writer.writerow([rel.replace("\\", "/"), rec["label"]])


def write_manifest(
    path: str, records: list[dict], data_dir: str, provenance: str
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            rel = os.path.relpath(os.path.abspath(rec["path"]), os.path.abspath(data_dir))
            item = {
                "id": f"s{i}",
                "type": "image",
                "input": rel.replace("\\", "/"),
                "label": rec["label"],
                "source": "sports-set",
                "provenance": provenance,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", default=None, help="Folder of real photos (label 0).")
    parser.add_argument("--ai", default=None, help="Folder of AI-generated images (label 1).")
    parser.add_argument("--out", default="eval/data/sports", help="Destination set folder.")
    parser.add_argument(
        "--provenance",
        default="unspecified",
        help="Free-text note recorded on every manifest item.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap per class (0 = all).")
    parser.add_argument("--min-size", type=int, default=32, help="Min shorter edge in px.")
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not drop byte-identical images.",
    )
    parser.add_argument(
        "--data-dir",
        default="eval/data",
        help="Base dir that manifest image paths are relative to.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Base dir that the probe CSV paths are relative to.",
    )
    args = parser.parse_args(argv)

    out = os.path.abspath(args.out)
    dest_real = os.path.join(out, "real")
    dest_ai = os.path.join(out, "ai")
    real_src = os.path.abspath(args.real or dest_real)
    ai_src = os.path.abspath(args.ai or dest_ai)

    if not os.path.isdir(real_src):
        print(f"[!] real source not found: {real_src}", file=sys.stderr)
        return 1
    if not os.path.isdir(ai_src):
        print(f"[!] ai source not found: {ai_src}", file=sys.stderr)
        return 1

    in_place = os.path.normcase(real_src) == os.path.normcase(dest_real) and os.path.normcase(
        ai_src
    ) == os.path.normcase(dest_ai)

    print(f"[+] real: {real_src}")
    print(f"[+] ai:   {ai_src}")
    print(f"[+] out:  {out}{'  (in place)' if in_place else ''}")

    seen: dict[str, int] = {}
    drop_dupes = not args.keep_duplicates

    real = collect_class(
        real_src, dest_real, 0, args.limit, args.min_size, drop_dupes, seen, in_place
    )
    ai = collect_class(
        ai_src, dest_ai, 1, args.limit, args.min_size, drop_dupes, seen, in_place
    )

    records = real["accepted"] + ai["accepted"]
    if not records:
        print("[!] no usable images found in either folder.", file=sys.stderr)
        return 1

    csv_path = os.path.join(out, "sports.csv")
    manifest_path = os.path.join(out, "sports-manifest.jsonl")
    write_csv(csv_path, records, args.root)
    write_manifest(manifest_path, records, args.data_dir, args.provenance)

    for name, cls in (("real", real), ("ai", ai)):
        r = cls["report"]
        print(
            f"  {name}: accepted={r['accepted']} invalid={r['invalid']} "
            f"too_small={r['too_small']} duplicate={r['duplicate']} "
            f"conflict={r['conflict']}"
        )
    print(f"[+] wrote {csv_path}")
    print(f"[+] wrote {manifest_path}")

    for name, cls in (("real", real), ("ai", ai)):
        n = cls["report"]["accepted"]
        if n < MIN_PER_CLASS_HINT:
            print(
                f"  [i] {name} has {n} images; ~{MIN_PER_CLASS_HINT}/class is the "
                "suggested floor before the numbers mean much."
            )
    if not in_place:
        print(
            "\nNext: python bin/probe-nonescape.py features "
            f"--csv {os.path.relpath(csv_path, args.root)} --root {args.root} "
            "--exclude-dir eval/data/aihuman --cache eval/data/.sports-probe.npz"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
