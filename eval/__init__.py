"""Honest evaluation harness for the Fake Exposer ML models.

Why this exists
---------------
The original training code reported a single accuracy number on a random 80/20
split of WELFake taken *before* deduplication. WELFake is a concatenation of
several source corpora and contains many exact/near-duplicate articles, so that
number was inflated by memorisation rather than generalisation.

This package builds a leak-free split (deduplicate first, then group-split),
reports a full metric panel (macro-F1, PR-AUC, calibration, train/val gap),
and can score both freshly-trained models and the shipped `.pkl` artifacts.

Entry point:
    python -m eval.report --data WELFake_Dataset.csv
"""
