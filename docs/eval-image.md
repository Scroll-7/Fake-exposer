# Local AI-image model evaluation — Nonescape mini on CIFAKE

Run: `python -m eval.image_eval --real eval/data/images/real --ai eval/data/images/ai --limit 1000`
Date: 2026-10-15 · model `nonescape-mini-v0.onnx` · data CIFAKE test
(1000 real + 1000 AI, 32×32 upscaled to 224)

## Result

| Metric | Value |
|---|---|
| Accuracy | 0.5095 |
| macro-F1 | 0.3582 |
| Precision (macro) | 0.6662 |
| Recall (macro) | 0.5095 |
| ROC-AUC | **0.6305** |
| PR-AUC | 0.6173 |
| Brier | 0.2889 |
| **ECE** | **0.2295** |

Confusion at threshold 0.5: **TN=24, FP=976, FN=5, TP=995** (n=2000).
Mean P(AI): real **0.704**, AI **0.755**. Best threshold 0.70 → macro-F1 0.599.

## What this actually says

1. **On CIFAKE the model is barely above chance** (ROC-AUC 0.63) and horribly
   miscalibrated (ECE 0.23). At the default 0.5 threshold it calls **97.6% of
   real images AI** (976/1000 false positives). It is not a usable standalone
   detector on this distribution.
2. **The bias flips by domain.** On the repo's own high-resolution real photos
   (`known_faces/`, 1024px+) it scores *low* P(AI) (≈0.00–0.05), but on 32×32
   upscaled CIFAKE it scores *high* P(AI) for **both** classes (0.70 / 0.755).
   A detector whose meaning depends this strongly on input domain cannot be
   trusted to threshold across domains.

## Why (likely)

CIFAKE real images are CIFAR-10 (32×32, upscaled and blurry); CIFAKE fakes are
Stable Diffusion 1.4 at the same resolution. This is **far** from the model's
training distribution — Nonescape was trained on ARIA (1024px+ real photos and
modern generative output). So this number is an *out-of-distribution floor*,
not a verdict on the model in its intended regime. It does still demonstrate
that the shipped model is not a safe default for arbitrary user uploads.

## Caveats

- CIFAKE images are 32×32 upscaled — a **proxy**, not the deployment
  distribution. Read it as a floor for "does this fail visibly?", not as an
  expected production accuracy.
- Data source: `dragonintelligence/CIFAKE-image-dataset` (HuggingFace mirror of
  the Kaggle CIFAKE set). Columns: `image` (JPEG bytes), `label` (0=real,
  1=AI-generated).
- n=2000; confidence intervals are wide, but the FP rate is unambiguous.

## Reproduce

```bash
# fetch a balanced subset (default 1000 per class) from HF mirror (no auth)
python -m eval.fetch_cifake --limit 1000

# run evaluation
python -m eval.image_eval --real eval/data/images/real --ai eval/data/images/ai --limit 1000
```

The fetcher downloads the small test parquet (8.4 MB) via `huggingface_hub` and
writes JPEGs to `eval/data/images/real` and `eval/data/images/ai`. You can also
run the inline one-liner shown in commit history; the script above is the
reproducible, committed version.

## Recommendation

Do **not** surface the Nonescape probability as a confident "AI-generated"
verdict. Either (a) keep it as one weak signal behind the existing gate, with
the UI framing it as low-confidence, or (b) replace it. Any replacement should
be re-measured on the sports-realistic set described in `eval/README.md`, not
on CIFAKE alone.
