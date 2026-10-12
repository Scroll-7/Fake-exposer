# Text classifier evaluation — WELFake (leak-free)

Run: `python -m eval.report --data WELFake_Dataset.csv --compare-legacy`
Date: 2026-10-12 · seed 42 · WELFake_Dataset.csv (72,134 rows)

## Result

| Metric | Corrected (dedup + group split + calibrated) | Legacy (random split, no dedup) |
|---|---|---|
| Accuracy | **0.9552** | 0.9483 |
| macro-F1 | **0.9551** | 0.9483 |
| Precision (macro) | 0.9551 | 0.9484 |
| Recall (macro) | 0.9552 | 0.9482 |
| ROC-AUC | 0.9911 | 0.9882 |
| PR-AUC | 0.9914 | 0.9886 |
| Brier | 0.0339 | 0.0434 |
| **ECE** | **0.0077** | **0.0527** |
| Train-val gap | +0.0120 (ok) | n/a |

Confusion (corrected): TN=6723, FP=309, FN=338, TP=7057 (n=14,427).

Deduplication found **9,708 duplicate rows (13.46%)**, the largest cluster
holding 48 rows — real leakage that a random split would spread across both
sides.

## What this actually says

1. **The old ~95% is real, not a duplication artifact.** On a dedup'd, group
   split the corrected pipeline scores *higher* (0.9552 vs 0.9483), so the
   headline accuracy was never mainly leakage-inflated. The nested
   `--compare-legacy` delta (`macro-F1 = -0.0069`) is *negative*.
2. **Calibration was the real defect.** Legacy ECE = 5.27% means its
   probabilities were materially over/under-confident; the calibrated pipeline
   drops ECE to **0.77%**. For a system that thresholds "is_fake", that is the
   difference that matters.
3. **No overfitting**: train-val gap +0.012.

## Caveats (why not to over-read this)

- The legacy comparison changes **two** things at once (split *and*
  features/calibration), so it does **not** isolate leakage's contribution. To
  isolate it you would run the *same* feature pipeline on both a random and a
  dedup'd split. The negative delta here mostly reflects that the corrected
  feature set is simply better.
- WELFake is **2016–2018 US political news**. This is an *in-domain* benchmark.
  The deployed product sees modern, largely sports, text and images. 95% here
  does **not** predict production accuracy — treat it as a sanity ceiling for
  the model family, not a deployment estimate.
- After clustering, the 14,427 test rows correspond to far fewer independent
  articles, so effective confidence intervals are wider than `n` suggests.
- `eval/report.py` near-duplicate dedup over 72k long articles is **slow**
  (~13 min CPU). Acceptable as a periodic baseline, not as an every-push gate.

## Reproduce

```bash
python -m eval.report --data WELFake_Dataset.csv --compare-legacy
```
