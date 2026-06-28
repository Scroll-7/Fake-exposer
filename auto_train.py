"""
auto_train.py

Master training orchestrator for the Fake News Detector project.

Run once and it trains EVERY domain automatically:
  - Text  -> fake_news_model.pkl + fake_news_vectorizer.pkl   (WELFake dataset)
  - Image -> animal_model.onnx                                (AFHQ + humans)

Usage:
  python auto_train.py           # train only what is missing
  python auto_train.py --force   # retrain every domain unconditionally

The script detects which data files are present and skips domains whose data
is not available, printing clear instructions for what to download.
"""

import argparse
import os
import sys
import time

# ─── ANSI helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def banner(text):
    line = "-" * 60
    print(f"\n{CYAN}{BOLD}{line}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{BOLD}{line}{RESET}\n")

def ok(text):    print(f"{GREEN}[OK] {text}{RESET}")
def warn(text):  print(f"{YELLOW}[WARN] {text}{RESET}")
def err(text):   print(f"{RED}[ERR] {text}{RESET}")
def info(text):  print(f"  [..] {text}")

# --- Domain definitions -------------------------------------------------------
# Each domain is a dict describing:
#   name        – human-readable label
#   data_files  – list of files that must exist to train this domain
#                 (any one of them is enough to attempt training)
#   model_files – list of output model files that signal the domain is trained
#   train_fn    – callable that performs the actual training
#   data_hint   – instructions printed when data files are missing

DOMAINS = []  # populated below after the training functions are defined


# ═══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 1 -- Text fake-news classifier (WELFake)
# ═══════════════════════════════════════════════════════════════════════════════

def train_text_model():
    """Train TF-IDF + Logistic Regression on WELFake_Dataset.csv."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import pickle

    CSV_PATH = "WELFake_Dataset.csv"
    MODEL_OUT = "fake_news_model.pkl"
    VEC_OUT   = "fake_news_vectorizer.pkl"

    info("Loading WELFake dataset ...")
    t0 = time.time()
    df = pd.read_csv(CSV_PATH)
    info(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]

    info("Label distribution:")
    for label, count in df["label"].value_counts().items():
        info(f"  label={label} -> {count:,} rows")

    X = df["content"]
    y = df["label"]

    info("Splitting data (80 / 20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    info("Vectorising text (TF-IDF, max 25 000 features) ...")
    vectorizer = TfidfVectorizer(max_features=25_000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    info("Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    accuracy = model.score(X_test_tfidf, y_test)
    ok(f"Accuracy on test set: {accuracy * 100:.2f}%")

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    with open(VEC_OUT, "wb") as f:
        pickle.dump(vectorizer, f)

    ok(f"Saved -> {MODEL_OUT}  &  {VEC_OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 2 -- Image classifier (cat / dog / wild / humans)
# ═══════════════════════════════════════════════════════════════════════════════

def train_image_model():
    """
    Fine-tune MobileNetV2 on AFHQ (cat/dog/wild) + humans dataset.

    Data sources expected in the current directory:
      archive.zip  -- AFHQ dataset  (cat / dog / wild)
      humans.zip   -- Humans images (flat folder structure)

    If only one zip is present, trains with whatever classes are available.
    """
    import random
    import shutil
    import zipfile

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader

    AFHQ_ZIP    = "archive.zip"
    HUMANS_ZIP  = "humans.zip"
    DATASET_DIR = "dataset_auto_train_tmp"
    ONNX_OUTPUT = "animal_model.onnx"
    BATCH_SIZE  = 32
    NUM_EPOCHS  = 10
    VAL_SPLIT   = 0.15
    SEED        = 42
    IMG_SIZE    = 224
    MAX_HUMANS  = 5000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    info(f"Device: {device}")

    random.seed(SEED)
    torch.manual_seed(SEED)

    has_afhq   = os.path.exists(AFHQ_ZIP)
    has_humans = os.path.exists(HUMANS_ZIP)

    # ── Prepare dataset directory ──────────────────────────────────────────
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    classes_to_train = []
    if has_afhq:
        classes_to_train += ["cat", "dog", "wild"]
    if has_humans:
        classes_to_train += ["humans"]

    for split in ["train", "val"]:
        for cls in classes_to_train:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

    # ── Extract AFHQ ───────────────────────────────────────────────────────
    if has_afhq:
        info("Extracting AFHQ (cat / dog / wild) from archive.zip ...")
        t0 = time.time()
        with zipfile.ZipFile(AFHQ_ZIP, "r") as z:
            for entry in z.namelist():
                parts = entry.replace("\\", "/").split("/")
                if len(parts) < 4:
                    continue
                split_name = parts[-3]
                cls_name   = parts[-2]
                filename   = parts[-1]
                if split_name not in ("train", "val"):
                    continue
                if cls_name not in ("cat", "dog", "wild"):
                    continue
                if not filename or filename.endswith("/"):
                    continue
                dest = os.path.join(DATASET_DIR, split_name, cls_name, filename)
                with z.open(entry) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
        info(f"AFHQ extracted in {time.time()-t0:.1f}s")
    else:
        warn("archive.zip not found -- skipping cat/dog/wild classes")

    # ── Extract Humans ─────────────────────────────────────────────────────
    if has_humans:
        info(f"Extracting humans from humans.zip (max {MAX_HUMANS} images) ...")
        t0 = time.time()
        with zipfile.ZipFile(HUMANS_ZIP, "r") as z:
            entries = [
                e for e in z.namelist()
                if not e.endswith("/")
                and e.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        random.shuffle(entries)
        entries = entries[:MAX_HUMANS]
        val_n   = int(len(entries) * VAL_SPLIT)
        val_e   = entries[:val_n]
        train_e = entries[val_n:]

        def extract_humans(lst, split_name):
            with zipfile.ZipFile(HUMANS_ZIP, "r") as z:
                for entry in lst:
                    fn = os.path.basename(entry)
                    if not fn:
                        continue
                    dest = os.path.join(DATASET_DIR, split_name, "humans", fn)
                    with z.open(entry) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

        extract_humans(train_e, "train")
        extract_humans(val_e,   "val")
        info(f"Humans: {len(train_e)} train | {len(val_e)} val -- done in {time.time()-t0:.1f}s")
    else:
        warn("humans.zip not found -- skipping humans class")

    # ── Class counts ───────────────────────────────────────────────────────
    info("Dataset distribution:")
    for split in ["train", "val"]:
        for cls in sorted(os.listdir(os.path.join(DATASET_DIR, split))):
            cls_path = os.path.join(DATASET_DIR, split, cls)
            n = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            info(f"  [{split}] {cls}: {n}")

    # ── Transforms ────────────────────────────────────────────────────────
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=(x == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for x in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    num_classes  = len(class_names)
    info(f"Classes detected ({num_classes}): {class_names}")

    # ── Build model ────────────────────────────────────────────────────────
    info(f"Loading pre-trained MobileNetV2 -> fine-tuning for {num_classes} classes ...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for block in model.features[15:]:
        for param in block.parameters():
            param.requires_grad = True
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.features[15:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(),    "lr": 5e-4},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(NUM_EPOCHS):
        info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc  = running_corrects.double() / len(image_datasets[phase])
            info(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    ok(f"Best val accuracy: {best_val_acc:.4f}")
    if best_weights:
        model.load_state_dict(best_weights)

    # ── Export ONNX ────────────────────────────────────────────────────────
    info(f"Exporting to {ONNX_OUTPUT} ...")
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, ONNX_OUTPUT,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,
        )
    size_mb = os.path.getsize(ONNX_OUTPUT) / (1024**2)
    ok(f"Saved -> {ONNX_OUTPUT}  ({size_mb:.1f} MB)   classes: {class_names}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    shutil.rmtree(DATASET_DIR)
    info("Temporary dataset directory cleaned up.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain registry
# ═══════════════════════════════════════════════════════════════════════════════

DOMAINS = [
    {
        "name":       "Text Fake-News Classifier  (WELFake -> TF-IDF + LogReg)",
        "data_files": ["WELFake_Dataset.csv"],
        "model_files": ["fake_news_model.pkl", "fake_news_vectorizer.pkl"],
        "train_fn":   train_text_model,
        "data_hint":  (
            "Download WELFake_Dataset.csv from Kaggle:\n"
            "  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n"
            "  and place it in the project root."
        ),
    },
    {
        "name":       "Image Classifier  (AFHQ + Humans -> MobileNetV2 ONNX)",
        "data_files": ["archive.zip", "humans.zip"],    # any one is enough
        "model_files": ["animal_model.onnx"],
        "train_fn":   train_image_model,
        "data_hint":  (
            "Need at least one of:\n"
            "  archive.zip  -- AFHQ dataset (cat/dog/wild)\n"
            "                 https://www.kaggle.com/datasets/andrewmvd/animal-faces\n"
            "  humans.zip   -- Human face images dataset\n"
            "Place the zip file(s) in the project root, then re-run auto_train.py."
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Auto-train all ML models for the Fake News Detector."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain every domain even if model files already exist."
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Train only a specific domain by index (0=text, 1=image)."
    )
    args = parser.parse_args()

    banner("Fake News Detector -- Auto Training Pipeline")

    results = []  # (domain_name, status, detail)

    for idx, domain in enumerate(DOMAINS):
        # ── Filter by --domain flag ────────────────────────────────────────
        if args.domain is not None and str(idx) != args.domain:
            continue

        banner(f"Domain {idx}: {domain['name']}")

        # ── Check if data is available ─────────────────────────────────────
        available_data = [f for f in domain["data_files"] if os.path.exists(f)]
        if not available_data:
            warn(f"No data files found for this domain.")
            warn(f"Expected (any of): {domain['data_files']}")
            print(f"\n  {YELLOW}How to get the data:{RESET}")
            for line in domain["data_hint"].splitlines():
                print(f"    {line}")
            results.append((domain["name"], "SKIPPED -- data missing", ""))
            continue

        info(f"Data found: {available_data}")

        # ── Check if model already trained ────────────────────────────────
        trained_files = [f for f in domain["model_files"] if os.path.exists(f)]
        if trained_files and not args.force:
            ok(f"Already trained: {trained_files}")
            info("Skipping (use --force to retrain).")
            results.append((domain["name"], "SKIPPED -- already trained", ""))
            continue

        if args.force and trained_files:
            warn("--force flag set -- retraining even though model exists.")

        # ── Run training ──────────────────────────────────────────────────
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

    # ── Summary ────────────────────────────────────────────────────────────────
    banner("Training Summary")
    col_w = max(len(d["name"]) for d in DOMAINS) + 2
    for name, status, detail in results:
        detail_str = f"  ({detail})" if detail else ""
        print(f"  {name:<{col_w}}  {status}{detail_str}")

    if not results:
        warn("No domains were processed. Check your --domain argument.")

    trained_any = any("TRAINED" in s for _, s, _ in results)
    had_data = any("SKIPPED -- data missing" not in s for _, s, _ in results)

    print()

    if not had_data:
        err("No training data found for any domain. See instructions above.")
        return 1
    if not trained_any and results:
        warn("All domains skipped (already trained or no data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
