"""CLI evaluation report for the fake-news text classifier.

Usage
-----
    python -m eval.report --data WELFake_Dataset.csv
    python -m eval.report --data WELFake_Dataset.csv --compare-legacy
    python -m eval.report --data WELFake_Dataset.csv \
        --model fake_news_model.pkl --vectorizer fake_news_vectorizer.pkl

What it does
------------
1. Loads the CSV and builds the same ``title + text`` content the trainer uses.
2. Deduplicates (exact + near-duplicate) and group-splits so no duplicate
   cluster leaks across train/test.
3. Trains the *corrected* pipeline (word 1-2 grams, sublinear tf, min/max df,
   calibrated probabilities) and reports the full metric panel.
4. Optionally re-runs the *legacy* pipeline on a random, un-deduplicated split
   to show how much the old number was inflated.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from eval.dedupe import cluster_stats, cluster_texts
from eval.metrics import classification_report_dict, format_report, train_val_gap
from eval.split import assert_no_cluster_leak, group_stratified_split

SEP = "-" * 66


def banner(text: str) -> None:
    print(f"\n{SEP}\n  {text}\n{SEP}")


def build_content(df, text_col: str | None) -> "list[str]":
    if text_col and text_col in df.columns:
        return df[text_col].fillna("").astype(str).tolist()
    title = df["title"].fillna("").astype(str) if "title" in df.columns else ""
    text = df["text"].fillna("").astype(str) if "text" in df.columns else ""
    if isinstance(title, str) and isinstance(text, str):
        raise SystemExit(
            "CSV has neither a usable --text-col nor title/text columns."
        )
    return (title + " " + text).tolist()


def build_improved_pipeline():
    """Corrected pipeline: richer features + calibrated probabilities."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9,
        stop_words="english",
        max_features=60_000,
    )
    classifier = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, C=1.0),
        method="sigmoid",
        cv=3,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def build_legacy_pipeline():
    """The original (pre-fix) configuration, for comparison only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    vectorizer = TfidfVectorizer(max_features=25_000, stop_words="english")
    classifier = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def predict_proba_fake(pipeline, texts) -> np.ndarray:
    """Probability of the positive class (label 1 == fake in WELFake)."""
    proba = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def evaluate_pipeline(
    pipeline, X_train, y_train, X_test, y_test, train_cluster_ids, seed=42
):
    """Fit and return (test_report, train_val_gap) using a grouped inner split."""
    # Carve a validation set out of the training clusters so we can measure the
    # train/val gap without leaking duplicate clusters between inner train/val.
    tr_idx, va_idx = group_stratified_split(
        list(train_cluster_ids), list(y_train), test_size=0.15, seed=seed
    )
    X_tr = [X_train[i] for i in tr_idx]
    y_tr = [y_train[i] for i in tr_idx]
    X_va = [X_train[i] for i in va_idx]
    y_va = [y_train[i] for i in va_idx]

    pipeline.fit(X_tr, y_tr)
    train_pred = pipeline.predict(X_tr)
    val_pred = pipeline.predict(X_va)

    from sklearn.metrics import accuracy_score

    gap = train_val_gap(
        accuracy_score(y_tr, train_pred), accuracy_score(y_va, val_pred)
    )

    # Refit on the full training set for the final test evaluation.
    pipeline.fit(X_train, y_train)
    test_pred = pipeline.predict(X_test)
    test_prob = predict_proba_fake(pipeline, X_test)
    report = classification_report_dict(y_test, test_pred, test_prob)
    return report, gap


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Leak-free evaluation report.")
    parser.add_argument("--data", default="WELFake_Dataset.csv")
    parser.add_argument("--text-col", default=None,
                        help="Column to use as content (default: title + text).")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--model", default=None,
                        help="Existing model .pkl to evaluate instead of training.")
    parser.add_argument("--vectorizer", default=None,
                        help="Vectorizer .pkl to pair with --model.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=float, default=0.8)
    parser.add_argument("--compare-legacy", action="store_true",
                        help="Also score the old leaky pipeline for comparison.")
    args = parser.parse_args(argv)

    import os

    if not args.model and not os.path.exists(args.data):
        print(f"\n[!] Dataset not found: {args.data}\n")
        print("    Download WELFake_Dataset.csv from Kaggle and place it in the")
        print("    project root:")
        print("    https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n")
        return 1

    import pandas as pd

    banner("Fake Exposer - text classifier evaluation (leak-free)")

    if args.model:
        import pickle

        print(f"  Loading model:      {args.model}")
        print(f"  Loading vectorizer: {args.vectorizer}")
        df = pd.read_csv(args.data)
        texts = build_content(df, args.text_col)
        labels = df[args.label_col].astype(int).tolist()

        cluster_ids, near_ok = cluster_texts(
            texts, threshold=args.near_dup_threshold, seed=args.seed
        )
        stats = cluster_stats(cluster_ids)
        _print_cluster_stats(stats, near_ok)
        train_idx, test_idx = group_stratified_split(
            cluster_ids, labels, args.test_size, args.seed
        )
        assert_no_cluster_leak(cluster_ids, train_idx, test_idx)

        from eval.text_pipeline import PipelineModel

        with open(args.model, "rb") as f:
            model = pickle.load(f)
        with open(args.vectorizer, "rb") as f:
            vectorizer = pickle.load(f)
        wrapper = PipelineModel(model, vectorizer)

        X_test = [texts[i] for i in test_idx]
        y_test = [labels[i] for i in test_idx]
        pred = wrapper.predict(X_test)
        prob = predict_proba_fake(wrapper, X_test)
        report = classification_report_dict(y_test, pred, prob)
        print()
        print(format_report("shipped model (leak-free test)", report))
        print()
        return 0

    # --- Train + evaluate the corrected pipeline ---------------------------
    df = pd.read_csv(args.data)
    texts = build_content(df, args.text_col)
    labels = df[args.label_col].astype(int).tolist()
    print(f"  Rows loaded:        {len(texts):,}")

    cluster_ids, near_ok = cluster_texts(texts, threshold=args.near_dup_threshold,
                                         seed=args.seed)
    stats = cluster_stats(cluster_ids)
    _print_cluster_stats(stats, near_ok)

    train_idx, test_idx = group_stratified_split(
        cluster_ids, labels, args.test_size, args.seed
    )
    assert_no_cluster_leak(cluster_ids, train_idx, test_idx)
    print(f"  Train rows:         {len(train_idx):,}")
    print(f"  Test rows:          {len(test_idx):,}")

    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    train_clusters = [cluster_ids[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    banner("Corrected pipeline (dedup + group split + calibrated)")
    report, gap = evaluate_pipeline(
        build_improved_pipeline(), X_train, y_train, X_test, y_test,
        train_clusters, args.seed,
    )
    print()
    print(format_report("corrected (leak-free test)", report))
    print(f"    train-val gap={gap:+.4f}  "
          f"({'overfitting' if gap > 0.10 else 'ok'})")

    if args.compare_legacy:
        banner("Legacy pipeline (random split, no dedup) — inflated baseline")
        legacy_report = _evaluate_legacy(
            build_legacy_pipeline(), texts, labels, args.test_size, args.seed
        )
        print()
        print(format_report("legacy (leaky test)", legacy_report))
        infl = legacy_report["f1_macro"] - report["f1_macro"]
        print(f"\n  macro-F1 inflation from leakage + weak features: {infl:+.4f}")

    print()
    return 0


def _print_cluster_stats(stats: dict, near_ok: bool) -> None:
    print(f"  Duplicate clusters: {stats['clusters']:,}")
    print(f"  Duplicate rows:     {stats['duplicate_rows_removed']:,} "
          f"({stats['duplicate_pct']}%)")
    print(f"  Largest cluster:    {stats['largest_cluster']}")
    if not near_ok:
        print("  [warn] datasketch not installed — exact-duplicate dedup only.")
        print("         pip install datasketch  (for near-duplicate detection)")


def _evaluate_legacy(pipeline, texts, labels, test_size, seed):
    from sklearn.metrics import accuracy_score

    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed
    )
    pipeline.fit(X_tr, y_tr)
    pred = pipeline.predict(X_te)
    prob = predict_proba_fake(pipeline, X_te)
    report = classification_report_dict(y_te, pred, prob)
    report["train_accuracy"] = float(accuracy_score(y_tr, pipeline.predict(X_tr)))
    return report


if __name__ == "__main__":
    sys.exit(main())
