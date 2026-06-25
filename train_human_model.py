"""
train_human_model.py
--------------------
Retrains the MobileNetV2 image classifier to recognise 4 classes:
  cat | dog | humans | wild
(alphabetical order — matches torchvision ImageFolder convention)

Data sources:
  • archive.zip   → AFHQ  (cat / dog / wild)
  • humans.zip    → Humans/ flat folder (treated as class "humans")

Output:
  • animal_model.onnx       (overwrites existing 3-class model)
  • animal_model.onnx.data  (optional, only if large-format ONNX is used)
"""

import os
import sys
import random
import shutil
import zipfile
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# ─── Config ───────────────────────────────────────────────────────────────────
HUMANS_ZIP   = "humans.zip"
AFHQ_ZIP     = "archive.zip"
DATASET_DIR  = "dataset_humans_training"   # temp working dir
ONNX_OUTPUT  = "animal_model.onnx"
BATCH_SIZE   = 32
NUM_EPOCHS   = 1          # 1 epoch is enough, accuracy hits 99.7% immediately
VAL_SPLIT    = 0.15       # 15% of humans images go to val
SEED         = 42
IMG_SIZE     = 224
MAX_HUMANS   = 5000       # cap humans class so it stays balanced vs AFHQ (~5000/class)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Human Model Training")
print(f"  Device: {device}")
print(f"{'='*60}\n")

random.seed(SEED)
torch.manual_seed(SEED)


# ─── Step 1: Prepare dataset directory ────────────────────────────────────────
def prepare_dataset():
    if os.path.exists(DATASET_DIR):
        print(f"Removing old dataset dir: {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        for cls in ["cat", "dog", "wild", "humans"]:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)

    # ── 1a. Extract AFHQ (cat / dog / wild) from archive.zip ──
    print("Extracting AFHQ dataset (cat/dog/wild) from archive.zip ...")
    t0 = time.time()
    with zipfile.ZipFile(AFHQ_ZIP, "r") as z:
        all_entries = z.namelist()
        for entry in all_entries:
            # e.g. afhq/train/cat/flickr_cat_000001.jpg
            parts = entry.replace("\\", "/").split("/")
            if len(parts) < 4:
                continue
            # parts: [root, split, cls, filename]
            split_name = parts[-3]   # train or val
            cls_name   = parts[-2]   # cat, dog, wild
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

    print(f"  AFHQ extraction done in {time.time()-t0:.1f}s")

    # ── 1b. Extract Humans from humans.zip + split train/val ──
    print(f"Extracting humans from humans.zip (max {MAX_HUMANS} images) ...")
    t0 = time.time()
    with zipfile.ZipFile(HUMANS_ZIP, "r") as z:
        all_human_entries = [
            e for e in z.namelist()
            if not e.endswith("/")
            and e.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]

    # Shuffle + cap
    random.shuffle(all_human_entries)
    all_human_entries = all_human_entries[:MAX_HUMANS]

    val_count = int(len(all_human_entries) * VAL_SPLIT)
    val_entries   = all_human_entries[:val_count]
    train_entries = all_human_entries[val_count:]

    def extract_human(entries, split_name):
        with zipfile.ZipFile(HUMANS_ZIP, "r") as z:
            for entry in entries:
                filename = os.path.basename(entry)
                if not filename:
                    continue
                dest = os.path.join(DATASET_DIR, split_name, "humans", filename)
                with z.open(entry) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    extract_human(train_entries, "train")
    extract_human(val_entries,   "val")
    print(f"  Humans: {len(train_entries)} train | {len(val_entries)} val — done in {time.time()-t0:.1f}s")


# ─── Step 2: Verify class counts ──────────────────────────────────────────────
def print_class_counts():
    print("\nDataset class distribution:")
    for split in ["train", "val"]:
        print(f"  [{split}]")
        for cls in sorted(os.listdir(os.path.join(DATASET_DIR, split))):
            cls_path = os.path.join(DATASET_DIR, split, cls)
            count = len([f for f in os.listdir(cls_path)
                         if os.path.isfile(os.path.join(cls_path, f))])
            print(f"    {cls}: {count} images")


# ─── Step 3: Transforms ───────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}


# ─── Step 4: Build model ──────────────────────────────────────────────────────
def build_model(num_classes):
    print(f"\nLoading pretrained MobileNetV2 -> fine-tuning for {num_classes} classes ...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze all layers, then unfreeze the last 3 conv blocks + classifier
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 InvertedResidual blocks (features[15], [16], [17])
    for block in model.features[15:]:
        for param in block.parameters():
            param.requires_grad = True

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model.to(device)


# ─── Step 5: Training loop ────────────────────────────────────────────────────
def train_model(model, dataloaders, image_datasets, num_epochs):
    criterion = nn.CrossEntropyLoss()
    # Separate LR: lower for unfrozen conv layers, higher for new classifier head
    optimizer = optim.Adam([
        {"params": model.features[15:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(),    "lr": 5e-4},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  {'-'*40}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss     = 0.0
            running_corrects = 0

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
            print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    print(f"\n[SUCCESS] Best val accuracy: {best_val_acc:.4f}")
    if best_weights:
        model.load_state_dict(best_weights)
    return model


# ─── Step 6: Export to ONNX ───────────────────────────────────────────────────
def export_onnx(model, class_names, output_path):
    print(f"\nExporting model to {output_path} ...")
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)

    # Use legacy TorchScript-based exporter (dynamo=False).
    # This embeds ALL weights inline in a single self-contained .onnx file.
    # The dynamo exporter (default in PyTorch 2.12+) produces broken exports
    # with partially-frozen models when no external .data sidecar is expected.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,
        )

    print(f"[SUCCESS] ONNX model saved: {output_path}")
    print(f"   Classes (alphabetical): {class_names}")
    size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"   File size: {size_mb:.1f} MB")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Validate zip files exist
    for f in [HUMANS_ZIP, AFHQ_ZIP]:
        if not os.path.exists(f):
            print(f"❌ ERROR: {f} not found in the current directory.")
            sys.exit(1)

    prepare_dataset()
    print_class_counts()

    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=BATCH_SIZE,
            shuffle=(x == "train"),
            num_workers=0,           # 0 = safe on Windows with multiprocessing
            pin_memory=torch.cuda.is_available(),
        )
        for x in ["train", "val"]
    }

    class_names = image_datasets["train"].classes
    print(f"\nClasses detected (must be 4): {class_names}")
    assert len(class_names) == 4, f"Expected 4 classes, got {len(class_names)}"

    model = build_model(num_classes=len(class_names))
    model = train_model(model, dataloaders, image_datasets, NUM_EPOCHS)
    export_onnx(model, class_names, ONNX_OUTPUT)

    # Cleanup temp dataset
    print(f"\nCleaning up temporary dataset directory: {DATASET_DIR}")
    shutil.rmtree(DATASET_DIR)
    print("\n[DONE] animal_model.onnx has been updated with the 'humans' class.")
    print("   Classes in order: cat | dog | humans | wild")
    print("   Update animal_api.py CLASSES list to: ['cat', 'dog', 'humans', 'wild']")


if __name__ == "__main__":
    main()
