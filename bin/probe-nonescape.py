"""Linear-probe fine-tune of the Nonescape mini *head* on a labelled dataset.

Why this exists
---------------
The upstream checkpoint is fine on realistic images (ROC-AUC ~0.89) but
domain-sensitive. Full backbone fine-tuning is not feasible on CPU (hours per
epoch), but the backbone is frozen and only the final ``Linear(1024, 2)`` head
moves — so we can (1) run every image through the backbone **once**, cache the
1024-d features, then (2) fit a logistic-regression head on the cache in
seconds. This is a linear probe.

It is deliberately conservative: we only produce an artifact worth shipping if
the probe beats the original head on a held-out split *and* does not regress on
an independent benchmark. Otherwise we keep the upstream model and record a
negative result.

Subcommands
-----------
    features  extract+cache backbone features for a balanced sample
    train     fit/train, compare to the original head on the val split, save head
    export    write a new checkpoint (backbone + trained head) and ONNX

Example
-------
    python bin/probe-nonescape.py features \
        --csv "archive (1)/train.csv" --root "archive (1)" \
        --per-class 2500 --val-per-class 500 \
        --exclude-dir eval/data/aihuman --cache eval/data/.probe.npz
    python bin/probe-nonescape.py train --cache eval/data/.probe.npz \
        --head-out eval/data/.probe-head.npz
    python bin/probe-nonescape.py export --head eval/data/.probe-head.npz \
        --safetensors nonescape-mini-v0.safetensors --onnx nonescape-mini-v1.onnx
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SD_FILE = os.path.join(_REPO_ROOT, "nonescape-mini-v0.safetensors")


def read_labels(csv_path, root, path_col="file_name", label_col="label", real_label="0"):
    """Return [(absolute_path, y)] with y=1 for AI, 0 for real."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel = r[path_col].replace("\\", "/")
            full = rel if os.path.isabs(rel) else os.path.join(root, rel)
            y = 0 if str(r[label_col]) == str(real_label) else 1
            rows.append((full, y))
    return rows


def _names_in(dirpath):
    try:
        return set(os.listdir(dirpath))
    except OSError:
        return set()


def sample_balanced(rows, per_class, val_per_class, exclude_dirs, seed=42):
    """Balanced sample; returns (train_items, val_items). Excludes eval filenames."""
    excluded = set()
    for d in exclude_dirs or []:
        excluded |= _names_in(d)
    rng = random.Random(seed)
    pools = {0: [], 1: []}
    for path, y in rows:
        if os.path.basename(path) in excluded:
            continue
        if os.path.isfile(path):
            pools[y].append((path, y))
    for y in pools:
        pools[y].sort()
        rng.shuffle(pools[y])
    train, val = [], []
    for y in (0, 1):
        val += pools[y][:val_per_class]
        train += pools[y][val_per_class:val_per_class + per_class]
    return train, val


def build_backbone():
    import torch
    import torchvision.models as models

    model = models.efficientnet_v2_s(weights=None, num_classes=1024, dropout=0.2)
    from safetensors.torch import load_file

    sd = load_file(_SD_FILE)
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    model.load_state_dict(backbone_sd)
    model.eval()
    return model, sd


def extract_features(model, items, batch=32):
    import torch
    from nonescape_preprocess import preprocess_nonescape

    X = np.zeros((len(items), 1024), dtype=np.float32)
    t0 = time.time()
    step = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        feats = []
        for path, _ in chunk:
            with open(path, "rb") as fh:
                feats.append(preprocess_nonescape(fh.read())[0])
        xb = torch.from_numpy(np.stack(feats)).float()
        with torch.no_grad():
            X[i:i + len(chunk)] = model(xb).numpy()
        step += 1
        if step % 10 == 0 or i + len(chunk) >= len(items):
            done = min(i + len(chunk), len(items))
            el = time.time() - t0
            print(f"  {done}/{len(items)}  {el:.0f}s  {done / el:.2f} img/s", flush=True)
    return X


def softmax_ai(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def ece_score(y, prob, bins=15):
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(prob)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi)
        if not m.any():
            continue
        conf = prob[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / total) * abs(acc - conf)
    return float(ece)


def cmd_features(args):
    model, _ = build_backbone()
    rows = read_labels(args.csv, args.root, args.path_col, args.label_col, args.real_label)
    print(f"[+] {len(rows)} labelled rows")
    train, val = sample_balanced(rows, args.per_class, args.val_per_class, args.exclude_dir)
    print(f"[+] train={len(train)} val={len(val)}")
    print("[+] extracting train features ...")
    Xtr = extract_features(model, train, args.batch)
    print("[+] extracting val features ...")
    Xva = extract_features(model, val, args.batch)
    ytr = np.array([y for _, y in train], dtype=np.int64)
    yva = np.array([y for _, y in val], dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    np.savez_compressed(args.cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
    print(f"[+] wrote {args.cache}")


def _metrics(y, prob):
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    return {
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, (np.asarray(prob) >= 0.5).astype(int))),
        "f1": float(f1_score(y, (np.asarray(prob) >= 0.5).astype(int))),
        "ece": ece_score(y, prob),
        "mean_real": float(np.mean(np.asarray(prob)[np.asarray(y) == 0])),
        "mean_ai": float(np.mean(np.asarray(prob)[np.asarray(y) == 1])),
    }


def cmd_train(args):
    from sklearn.linear_model import LogisticRegression

    from safetensors.torch import load_file

    d = np.load(args.cache)
    Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]

    sd = load_file(_SD_FILE)
    W = sd["head.weight"].numpy()
    b = sd["head.bias"].numpy()
    orig = softmax_ai(Xva @ W.T + b)
    print("[orig]", {k: round(v, 4) for k, v in _metrics(yva, orig).items()})

    best = None
    for C in (args.C if args.C else [0.01, 0.1, 1.0, 10.0]):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xva)[:, 1]
        m = _metrics(yva, prob)
        print(f"[lr C={C}]", {k: round(v, 4) for k, v in m.items()})
        if best is None or m["auc"] > best[1]["auc"]:
            best = (clf, m)

    clf, m = best
    np.savez_compressed(
        args.head_out,
        weight=clf.coef_.astype(np.float32),
        bias=clf.intercept_.astype(np.float32),
        auc=m["auc"],
    )
    print(f"[+] saved head -> {args.head_out} (val AUC {m['auc']:.4f})")


def cmd_export(args):
    import torch
    from safetensors.torch import save_file
    from safetensors.torch import load_file

    h = np.load(args.head)
    coef = h["weight"]  # (1, 1024) for the AI (positive) class
    bias = h["bias"]  # (1,)
    weight = np.zeros((2, coef.shape[1]), dtype=np.float32)
    weight[1] = coef[0]  # AI logit = coef . x + bias; class 0 logit fixed at 0
    new_bias = np.zeros(2, dtype=np.float32)
    new_bias[1] = bias[0]

    sd = dict(load_file(args.safetensors))
    sd["head.weight"] = torch.tensor(weight)
    sd["head.bias"] = torch.tensor(new_bias)
    save_file(sd, args.safetensors_out, metadata={"format": "pt"})
    print(f"[+] wrote checkpoint -> {args.safetensors_out}")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "convert_nonescape", os.path.join(_REPO_ROOT, "bin", "convert-nonescape.py")
    )
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    convert.convert(args.safetensors_out, args.onnx)
    convert.verify(args.onnx)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features")
    f.add_argument("--csv", required=True)
    f.add_argument("--root", required=True)
    f.add_argument("--path-col", default="file_name")
    f.add_argument("--label-col", default="label")
    f.add_argument("--real-label", default="0")
    f.add_argument("--per-class", type=int, default=2500)
    f.add_argument("--val-per-class", type=int, default=500)
    f.add_argument("--exclude-dir", action="append", default=[])
    f.add_argument("--cache", default="eval/data/.probe.npz")
    f.add_argument("--batch", type=int, default=32)
    f.set_defaults(func=cmd_features)

    t = sub.add_parser("train")
    t.add_argument("--cache", default="eval/data/.probe.npz")
    t.add_argument("--head-out", default="eval/data/.probe-head.npz")
    t.add_argument("--C", type=float, action="append")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("export")
    e.add_argument("--head", default="eval/data/.probe-head.npz")
    e.add_argument("--safetensors", default="nonescape-mini-v0.safetensors")
    e.add_argument("--safetensors-out", default="nonescape-mini-v1.safetensors")
    e.add_argument("--onnx", default="nonescape-mini-v1.onnx")
    e.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
