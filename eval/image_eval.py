"""Evaluate the local Nonescape ONNX AI-image detector directly.

Zero cost, no server, no API calls: this loads `nonescape-mini-v0.onnx` and
scores two folders of images — real vs AI-generated — then reports a full
metric panel plus the best decision threshold.

Why this harness exists
-----------------------
The end-to-end image endpoint calls paid LLMs, but the *local* AI-image model
is the only detector we can measure for free and at scale. Measuring it on a
labelled set tells us how much signal it actually contributes, instead of
assuming.

Free public datasets to populate the folders (no budget required):
  * CIFAKE            — https://huggingface.co/datasets/diffusers/cifake
                        or https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
  * AI vs Human       — https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset
Expected layout:
    eval/data/images/real/*.jpg
    eval/data/images/ai/*.jpg

Usage:
    python -m eval.image_eval --real eval/data/images/real --ai eval/data/images/ai
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def collect_images(folder: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(EXTS):
                out.append(os.path.join(root, fn))
    return sorted(out)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Nonescape preprocessing: shared with the runtime API (port 8001)."""
    from nonescape_preprocess import preprocess_nonescape

    return preprocess_nonescape(image_bytes)


def score_images(session, paths: list[str]) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    probs = []
    for p in paths:
        with open(p, "rb") as f:
            data = f.read()
        logits = session.run(None, {input_name: preprocess(data)})[0][0]
        exp = np.exp(logits - np.max(logits))
        probs.append(float((exp / exp.sum())[1]))  # index 1 == AI
    return np.asarray(probs, dtype=float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local AI-image detector evaluation.")
    parser.add_argument("--real", required=True, help="Folder of real photos.")
    parser.add_argument("--ai", required=True, help="Folder of AI-generated images.")
    from nonescape_model import resolve_model_path

    parser.add_argument("--model", default=resolve_model_path())
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Cap per class (0 = all).")
    args = parser.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"\n[!] Model not found: {args.model}")
        print("    Download it with:  npm run download-models\n")
        return 1

    if not os.path.isdir(args.real) or not os.path.isdir(args.ai):
        print("\n[!] Need two folders of images to evaluate.")
        print("    Layout:  eval/data/images/real/   and   eval/data/images/ai/")
        print("    Free datasets: CIFAKE or 'AI vs Human' (see eval/README.md)\n")
        return 1

    real = collect_images(args.real)
    ai = collect_images(args.ai)
    if args.limit:
        real, ai = real[: args.limit], ai[: args.limit]
    if not real or not ai:
        print("[!] One of the folders is empty. Nothing to evaluate.")
        return 1

    try:
        import onnxruntime as ort
    except ImportError:
        print("[!] onnxruntime is not installed. Run: pip install onnxruntime")
        return 1

    from eval.metrics import (
        best_threshold,
        classification_report_dict,
        format_report,
    )

    print(f"\n  model:  {args.model}")
    print(f"  real:   {len(real)} images")
    print(f"  ai:     {len(ai)} images")

    session = ort.InferenceSession(args.model)
    print("  scoring images ...")
    prob_real = score_images(session, real)
    prob_ai = score_images(session, ai)

    y_true = np.concatenate([np.zeros(len(real)), np.ones(len(ai))]).astype(int)
    y_prob = np.concatenate([prob_real, prob_ai])

    pred = (y_prob >= args.threshold).astype(int)
    report = classification_report_dict(y_true, pred, y_prob)
    print()
    print(format_report(f"nonescape @ threshold={args.threshold}", report))

    bt, bf = best_threshold(y_true, y_prob)
    print(f"\n    best threshold={bt:.2f}  macro-F1={bf:.4f}")
    print("    (a large distance between the two means signals a separable "
          "signal; a near-0.5 best threshold with low F1 means it isn't)")
    print(f"    mean P(AI) on real={prob_real.mean():.3f}  "
          f"on ai={prob_ai.mean():.3f}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
