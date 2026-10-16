# Local AI-image model evaluation — Nonescape mini

Model: `nonescape-mini-v0.onnx` · metric: P(AI-generated), index 1 = AI.
Same shared preprocessing as the runtime API (`nonescape_preprocess.py`).

Two benchmarks, and the contrast between them is the whole story:

| Benchmark | What it is | Closeness to production |
|---|---|---|
| **Realistic 512px** | 2025 Women-in-AI *AI vs Human-Generated Images* (Shutterstock photos paired with modern generative output) | high — photo-like, modern models |
| **CIFAKE 32px** | CIFAR-10 reals vs Stable-Diffusion 1.4, upscaled from 32×32 | low — far outside the model's training distribution |

## Benchmark A — realistic 512px (policy-defining)

Run: `python -m eval.image_eval --real eval/data/aihuman/real --ai eval/data/aihuman/ai --limit 1000`
Date: 2026-10-15 · 1000 real + 1000 AI

| Metric | Value |
|---|---|
| Accuracy | 0.8045 |
| macro-F1 | 0.8038 |
| Precision (macro) | 0.8090 |
| Recall (macro) | 0.8045 |
| ROC-AUC | **0.8898** |
| PR-AUC | 0.8970 |
| Brier | 0.1441 |
| **ECE** | 0.0876 |

Confusion at 0.5: **TN=865, FP=135, FN=256, TP=744**. Mean P(AI): real **0.171**,
AI **0.712**. Best threshold 0.30 → macro-F1 0.815.

## Benchmark B — CIFAKE 32px (out-of-distribution)

Run: `python -m eval.image_eval --real eval/data/images/real --ai eval/data/images/ai --limit 1000`

| Metric | Value |
|---|---|
| Accuracy | 0.5095 |
| macro-F1 | 0.3582 |
| ROC-AUC | **0.6305** |
| PR-AUC | 0.6173 |
| Brier | 0.2889 |
| **ECE** | 0.2295 |

Confusion at 0.5: **TN=24, FP=976, FN=5, TP=995**. Mean P(AI): real **0.704**,
AI **0.755**. Best threshold 0.70 → macro-F1 0.599.

## Threshold sweep — why the cap is at 0.9

Share of images above each threshold (1000 per set):

```
                 mean   >0.50  >0.65  >0.70  >0.80  >0.90  >0.95
realistic real   0.171   13.5%   9.6%   8.4%   6.0%   3.1%   1.1%
realistic AI     0.712   74.4%  67.2%  63.8%  58.1%  46.6%  37.7%
CIFAKE real      0.702   97.1%  67.4%  47.1%  19.4%   3.5%   1.1%
```

Above **0.9** the false-positive rate on real images is ~3% in **both** domains
(3.1% realistic, 3.5% CIFAKE), while catching ~47% of realistic AI images —
i.e. ~94% precision. Below 0.9 the model is too domain-sensitive to act on
hard. So 0.9 is where the pipeline draws the line.

## What this actually says

1. **On realistic images the model is genuinely useful** (ROC-AUC 0.89,
   ~80% accuracy / macro-F1). It is *not* a broken model.
2. **On 32px CIFAKE it is near chance** (ROC-AUC 0.63) and flags ~97% of real
   images as AI at 0.5. This is a **domain-shift artifact**, not a
   property of the model's target use. The earlier "the model is useless"
   reading came from this benchmark alone.
3. **The real defect is domain sensitivity**, not memorisation. The honest fix
   is a high-confidence threshold (0.9) plus advisory-only behaviour below it —
   which the pipeline now implements (see `nonescapeAdvisory()` in
   `services/groq.js`).

## Can we just retrain it? (linear probe result)

Full backbone fine-tuning is infeasible on CPU, but the backbone can be frozen
and the final `Linear(1024, 2)` head re-fit on cached features — a *linear
probe* (`bin/probe-nonescape.py`). We extracted features for 2500/class train +
500/class val from the realistic dataset (val AUC 0.974) and then scored v1 on
the **independent** realistic eval set, CIFAKE, and 32 real known-face photos:

| model | realistic ROC-AUC | realistic acc | realistic ECE | CIFAKE ROC-AUC | real faces >0.5 |
|---|---|---|---|---|---|
| v0 (upstream) | 0.890 | 0.804 | 0.088 | **0.631** | 1/32 |
| **v1 (linear probe)** | **0.971** | **0.917** | **0.018** | 0.498 | 1/32 |
| ensemble mean(v0,v1) | 0.959 | 0.887 | — | 0.551 | 1/32 |

Read it honestly:

- **v1 nearly doubles the same-domain signal and is far better calibrated**
  (ECE 0.088 → 0.018), and real-photo false positives are unchanged.
- **v1 loses the cross-domain signal** on CIFAKE (0.631 → 0.498). Both scores
  are in "unusable" territory, but v0 really was carrying *some* out-of-distribution
  cue that the probe no longer uses — a reminder that the probe is fit to one
  image generator.
- **No ensemble dominates**: the mean blend is worse than v1 same-domain and
  worse than v0 cross-domain, so there is no free lunch here.

**Decision.** v0 stays the default — it is the only checkpoint a fresh clone
has, and it is the more general of the two. The probe is available as an opt-in
for realistic-image workloads:

```bash
python bin/probe-nonescape.py features --csv "archive (1)/train.csv" --root "archive (1)" \
    --per-class 2500 --val-per-class 500 --exclude-dir eval/data/aihuman \
    --cache eval/data/.probe.npz
python bin/probe-nonescape.py train  --cache eval/data/.probe.npz --head-out eval/data/.probe-head.npz
python bin/probe-nonescape.py export --head eval/data/.probe-head.npz \
    --safetensors nonescape-mini-v0.safetensors --onnx nonescape-mini-v1.onnx
# then point the runtime and eval at it:
set NONESCAPE_MODEL=nonescape-mini-v1.onnx        # PowerShell:  $env:NONESCAPE_MODEL="..."
```

## Caveats

- CIFAKE is 32×32 upscaled — a proxy, not the deployment distribution; read it
  as an out-of-distribution floor.
- The realistic benchmark pairs each Shutterstock photo with a generative
  counterpart, so its AI half is a *specific* generator set (not all real-world
  AI imagery). Treat the 0.89 as indicative, not final.
- Neither benchmark is a sports-realistic set. The north-star dataset in
  `eval/README.md` is still worth building.
- n=1000 per class per benchmark; the FP/recall rates are stable enough to set
  a threshold, not to claim fine-grained calibration.

## Reproduce

```bash
# Benchmark B — fetch CIFAKE from the HF mirror (no auth)
python -m eval.fetch_cifake --limit 1000
python -m eval.image_eval --real eval/data/images/real --ai eval/data/images/ai --limit 1000

# Benchmark A — build the split from the downloaded dataset (label 0 = human)
python -m eval.make_split --csv "archive (1)/train.csv" --root "archive (1)" \
    --out eval/data/aihuman --limit 1000
python -m eval.image_eval --real eval/data/aihuman/real --ai eval/data/aihuman/ai --limit 1000
```
