# Evaluation

Honest measurement for the Fake Exposer models. Everything here runs **locally
and free** — no paid APIs, no cloud GPU. Where the live pipeline is needed we
use the free Groq/Gemini tiers, and we say so.

## Why

The original trainer reported one accuracy number on a random split taken
*before* deduplication. WELFake is a concatenation of several corpora full of
near-duplicates, so that number measured memorisation. This directory replaces
it with leak-free, multi-metric measurement.

## What's here

| Module | What it measures | Cost |
|---|---|---|
| `report.py` | Text classifier on a leak-free WELFake split | free |
| `image_eval.py` | The local Nonescape AI-image ONNX model | **free, unlimited** |
| `pipeline_eval.py` | The full `/api/analyze/*` pipeline end-to-end | free-tier API calls |
| `dedupe.py` / `split.py` / `metrics.py` | The reusable harness | free |
| `test_eval.py` | Self-tests for the harness (no data needed) | free |

## Quick runs

```bash
# Harness self-tests
python -m unittest eval.test_eval -v

# Text classifier, leak-free (needs WELFake_Dataset.csv in the repo root)
python -m eval.report --data WELFake_Dataset.csv --compare-legacy

# Local AI-image model (needs two folders of images, see below)
python -m eval.image_eval --real eval/data/images/real --ai eval/data/images/ai

# Full pipeline (start the server first)
python -m eval.pipeline_eval --manifest eval/data/sample_manifest.jsonl
```

## Getting data for free

### Text — WELFake (required for `report.py`)
Free Kaggle download, no payment:
<https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification>
Drop `WELFake_Dataset.csv` in the project root.

### Images — for `image_eval.py`
Both are free downloads; pick one and split into `real/` and `ai/`:

- **CIFAKE** (60k real CIFAR-10 + 60k Stable-Diffusion 1.4) —
  <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images>
- **AI vs Human generated** (HuggingFace, 500k images) —
  <https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset>

Layout:
```
eval/data/images/real/*.jpg
eval/data/images/ai/*.jpg
```

> Caveat: CIFAKE images are 32×32 upscaled, so they are a *proxy* for the real
> deployment distribution (modern 1024px+ generative photos). Treat the number
> as a floor for model behaviour, not a guarantee.

#### Model file & preprocessing (fixed)

The shipped `nonescape-mini-v0.onnx` was **broken**: `bin/download-models.ps1`
fell back to the HuggingFace *safetensors* checkpoint but saved it under the
`.onnx` name, which onnxruntime cannot parse (`INVALID_PROTOBUF`), so the local
image detector silently did nothing. The script now downloads the safetensors
under its real name and converts it with `bin/convert-nonescape.py`, which
reconstructs the upstream `NonescapeClassifierMini` (`efficientnet_v2_s`
backbone → `Linear(1024, 2)`, returns raw logits).

`image_api.py` (port 8001) and `image_eval.py` share a single preprocessing
module (`nonescape_preprocess.py`) that mirrors the upstream SDK: aspect
preserving resize (shorter edge → 256) → center-crop 224 → JPEG q100 → ImageNet
normalise. The earlier runtime code forced a 256×256 square, which distorted the
aspect ratio and produced nonsensical probabilities (a real photo scored
P(AI)=0.99); with the fix real photos score low (≈0.00–0.05).


### A real sports set (the north star — Phase 3 in the plan)
The only dataset that predicts production behaviour is labelled sports
misinformation. Build it incrementally:
- Real: your own licensed/self-taken photos, or public-domain sports photos.
- AI: images you generate yourself through any free tool you already have
  access to (e.g. a free Colab diffusion notebook, Krea, etc.).
Label each item with a `provenance` field. Start with ~50 per class; this
beats 10,000 irrelevant political-text rows.

## Manifest format (`pipeline_eval.py`)

```json
{"id": "s1", "type": "text",  "input": "article text...", "label": 1, "source": "sports-set"}
{"id": "s2", "type": "url",   "input": "https://...",     "label": 0}
{"id": "s3", "type": "image", "input": "images/ai1.jpg",  "label": 1}
```

`label`: **1 = fake/AI/manipulated**, **0 = real/authentic**.
Image `input` paths are relative to `--data-dir` (default `eval/data`).

## Rate limits

`/api/analyze/text` is capped at 60 requests/hour/IP and `/api/analyze/image`
at 15/hour/IP. Large pipeline runs will hit these; the runner reports errored
items so you can resume. For unlimited image scoring, use `image_eval.py`.
