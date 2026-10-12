"""
auto_train.py

Training orchestrator for the Fake News Detector's text classifier.

Trains the one local model that actually contributes signal:
  - Text -> fake_news_model.pkl + fake_news_vectorizer.pkl + .card.json

The old "image -> animal_model.onnx" (AFHQ + humans) domain was removed. A
cat/dog/wild/human classifier is orthogonal to fake-news detection, its class
order could silently desync from the hardcoded API list, and it only produced
cosmetic flags. Local AI-image detection is handled by the pretrained Nonescape
ONNX model (see image_api.py), and human presence by InsightFace (face_api.py).

Training here is intentionally honest:
  * duplicate / near-duplicate articles are removed BEFORE splitting,
  * the split keeps whole duplicate clusters together (no leakage),
  * probabilities are calibrated,
  * a full metric panel + model card is written.

Evaluate an already-trained artifact with:
    python -m eval.report --data WELFake_Dataset.csv \
        --model fake_news_model.pkl --vectorizer fake_news_vectorizer.pkl

Usage:
  python auto_train.py           # train only if the model is missing
  python auto_train.py --force   # retrain unconditionally
"""

import argparse
import json
import os
import sys
import time

# ─── ANSI helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def banner(text):
    line = "-" * 60
    print(f"\n{CYAN}{BOLD}{line}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{BOLD}{line}{RESET}\n")


def ok(text):   print(f"{GREEN}[OK] {text}{RESET}")
def warn(text): print(f"{YELLOW}[WARN] {text}{RESET}")
def err(text):  print(f"{RED}[ERR] {text}{RESET}")
def info(text): print(f"  [..] {text}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 1 -- Text fake-news classifier (WELFake)
# ═══════════════════════════════════════════════════════════════════════════════

def train_text_model():
    """Train a calibrated TF-IDF + Logistic Regression on WELFake_Dataset.csv.

    Uses the leak-free pipeline from `eval/`: deduplicate -> group split ->
    train -> calibrate -> measure. Writes a model card next to the artifacts.
    """
    import pickle

    import pandas as pd

    from eval.dedupe import cluster_stats, cluster_texts
    from eval.metrics import (
        classification_report_dict,
        format_report,
        train_val_gap,
    )
    from eval.split import assert_no_cluster_leak, group_stratified_split

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    CSV_PATH = "WELFake_Dataset.csv"
    MODEL_OUT = "fake_news_model.pkl"
    VEC_OUT = "fake_news_vectorizer.pkl"
    CARD_OUT = "fake_news_model.card.json"
    SEED = 42
    TEST_SIZE = 0.2

    info("Loading WELFake dataset ...")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    info(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]

    label_values = sorted(df["label"].unique().tolist())
    if label_values != [0, 1]:
        raise SystemExit(
            f"Unexpected label values {label_values}; expected [0, 1]. "
            "Verify the CSV before training."
        )
    # WELFake convention: 1 = fake, 0 = real. Persisted in the model card so the
    # API never has to guess (this was a silent-failure risk before).
    FAKE_LABEL = 1

    info("Label distribution:")
    for label, count in df["label"].value_counts().items():
        info(f"  label={label} -> {count:,} rows")

    texts = df["content"].tolist()
    labels = df["label"].astype(int).tolist()

    info("Deduplicating (exact + near-duplicate) ...")
    cluster_ids, near_dup_ran = cluster_texts(texts, threshold=0.8, seed=SEED)
    stats = cluster_stats(cluster_ids)
    info(f"  clusters={stats['clusters']:,}  duplicate rows removed="
         f"{stats['duplicate_rows_removed']:,} ({stats['duplicate_pct']}%)")
    if not near_dup_ran:
        warn("datasketch not installed — exact-duplicate dedup only "
             "(pip install datasketch for near-duplicate detection).")

    info("Splitting (grouped by duplicate cluster, leak-free) ...")
    train_idx, test_idx = group_stratified_split(
        cluster_ids, labels, test_size=TEST_SIZE, seed=SEED
    )
    assert_no_cluster_leak(cluster_ids, train_idx, test_idx)
    info(f"  train={len(train_idx):,}  test={len(test_idx):,}")

    X_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    train_clusters = [cluster_ids[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    info("Vectorising (word 1-2 grams, sublinear tf, min_df=3, max_df=0.9) ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9,
        stop_words="english",
        max_features=60_000,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    info("Training calibrated Logistic Regression ...")
    base = LogisticRegression(max_iter=1000, C=1.0)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train_tfidf, y_train)

    # ── Metrics ────────────────────────────────────────────────────────────
    classes = list(model.classes_)
    if FAKE_LABEL not in classes:
        raise SystemExit(f"Fake label {FAKE_LABEL} missing from classes {classes}")
    fake_col = classes.index(FAKE_LABEL)

    test_pred = model.predict(X_test_tfidf)
    test_prob = model.predict_proba(X_test_tfidf)[:, fake_col]
    report = classification_report_dict(y_test, test_pred, test_prob)

    # Train-val gap on an inner grouped split (no cluster leaks into val).
    from sklearn.metrics import accuracy_score

    tr_idx, va_idx = group_stratified_split(
        train_clusters, y_train, test_size=0.15, seed=SEED
    )
    inner = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, C=1.0), method="sigmoid", cv=3
    )
    inner.fit(
        vectorizer.transform([X_train[i] for i in tr_idx]),
        [y_train[i] for i in tr_idx],
    )
    gap = train_val_gap(
        accuracy_score(
            [y_train[i] for i in tr_idx],
            inner.predict(vectorizer.transform([X_train[i] for i in tr_idx])),
        ),
        accuracy_score(
            [y_train[i] for i in va_idx],
            inner.predict(vectorizer.transform([X_train[i] for i in va_idx])),
        ),
    )

    print()
    ok("Leak-free evaluation:")
    print(format_report("fake_news_model (held-out test)", report))
    print(f"    train-val gap={gap:+.4f}  "
          f"({'OVERFITTING' if gap > 0.10 else 'ok'})")

    # ── Save artifacts + model card ────────────────────────────────────────
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_OUT, "wb") as f:
        pickle.dump(vectorizer, f)

    card = {
        "name": "fake_news_text",
        "dataset": os.path.basename(CSV_PATH),
        "dataset_rows": int(len(df)),
        "dedup": stats,
        "near_dup_dedup": bool(near_dup_ran),
        "split": {"seed": SEED, "test_size": TEST_SIZE,
                  "n_train": int(len(train_idx)), "n_test": int(len(test_idx))},
        "label_map": {"real": 0, "fake": FAKE_LABEL},
        "features": {"ngram_range": [1, 2], "min_df": 3, "max_df": 0.9,
                     "max_features": 60000},
        "calibration": "sigmoid (CalibratedClassifierCV)",
        "metrics": report,
        "train_val_gap": round(float(gap), 4),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "warning": (
            "WELFake is 2016-2018 US political news. This model does NOT "
            "generalise to modern sports/multimodal content; treat its output "
            "as a weak, calibrated signal only."
        ),
    }
    with open(CARD_OUT, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)

    ok(f"Saved -> {MODEL_OUT}, {VEC_OUT}, {CARD_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain registry
# ═══════════════════════════════════════════════════════════════════════════════

DOMAINS = [
    {
        "name": "Text Fake-News Classifier  (WELFake -> calibrated TF-IDF + LogReg)",
        "data_files": ["WELFake_Dataset.csv"],
        "model_files": ["fake_news_model.pkl", "fake_news_vectorizer.pkl"],
        "train_fn": train_text_model,
        "data_hint": (
            "Download WELFake_Dataset.csv from Kaggle:\n"
            "  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            "  and place it in the project root."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto-train the Fake News Detector's text model."
    )
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if model files already exist.")
    parser.add_argument("--domain", type=str, default=None,
                        help="Train only a specific domain by index (0=text).")
    args = parser.parse_args()

    banner("Fake News Detector -- Auto Training Pipeline")

    results = []

    for idx, domain in enumerate(DOMAINS):
        if args.domain is not None and str(idx) != args.domain:
            continue

        banner(f"Domain {idx}: {domain['name']}")

        available_data = [f for f in domain["data_files"] if os.path.exists(f)]
        if not available_data:
            warn("No data files found for this domain.")
            warn(f"Expected (any of): {domain['data_files']}")
            print(f"\n  {YELLOW}How to get the data:{RESET}")
            for line in domain["data_hint"].splitlines():
                print(f"    {line}")
            results.append((domain["name"], "SKIPPED -- data missing", ""))
            continue

        info(f"Data found: {available_data}")

        trained_files = [f for f in domain["model_files"] if os.path.exists(f)]
        if trained_files and not args.force:
            ok(f"Already trained: {trained_files}")
            info("Skipping (use --force to retrain).")
            results.append((domain["name"], "SKIPPED -- already trained", ""))
            continue

        if args.force and trained_files:
            warn("--force flag set -- retraining even though model exists.")

        t_start = time.time()
        try:
            domain["train_fn"]()
            elapsed = time.time() - t_start
            ok(f"Finished in {elapsed:.1f}s")
            results.append((domain["name"], "TRAINED", f"{elapsed:.1f}s"))
        except Exception as exc:
            import traceback
            err(f"Training failed: {exc}")
            traceback.print_exc()
            results.append((domain["name"], "FAILED", str(exc)))

    banner("Training Summary")
    col_w = max(len(d["name"]) for d in DOMAINS) + 2
    for name, status, detail in results:
        detail_str = f"  ({detail})" if detail else ""
        print(f"  {name:<{col_w}}  {status}{detail_str}")

    if not results:
        warn("No domains were processed. Check your --domain argument.")

    print()
    if all("SKIPPED -- already trained" in s for _, s, _ in results) and results:
        ok("Everything is up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
