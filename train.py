"""Train a waste classifier (MobileNetV2 transfer learning).

Dataset layout (TrashNet or any similar dataset):
    dataset/
        cardboard/*.jpg
        glass/*.jpg
        metal/*.jpg
        paper/*.jpg
        plastic/*.jpg
        trash/*.jpg
Run:  python train.py
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from utils import IMG_SIZE, MEAN, STD

DATA_DIR = Path("dataset")
OUT_DIR = Path("model")
OUT_DIR.mkdir(exist_ok=True)
BATCH, HEAD_EPOCHS, FINE_EPOCHS, SEED = 32, 5, 5, 42

random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])

# Two views of the same folder: one with augmentation, one without.
full_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
full_val = datasets.ImageFolder(DATA_DIR, transform=val_tf)
classes = full_train.classes
print("Classes:", classes, "| images:", len(full_train))
if len(classes) < 2:
    raise SystemExit("Put class sub-folders inside dataset/ (see README).")

idx = list(range(len(full_train)))
random.shuffle(idx)
cut = int(0.8 * len(idx))
train_dl = DataLoader(Subset(full_train, idx[:cut]), BATCH, shuffle=True, num_workers=0)
val_dl = DataLoader(Subset(full_val, idx[cut:]), BATCH, shuffle=False, num_workers=0)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.to(device)
loss_fn = nn.CrossEntropyLoss()


def run_epoch(dl, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


best_acc, history = 0.0, {"train_acc": [], "val_acc": []}


def fit(epochs, opt, tag):
    global best_acc
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(train_dl, opt)
        vl, va = run_epoch(val_dl)
        history["train_acc"].append(ta); history["val_acc"].append(va)
        print(f"[{tag}] epoch {ep}/{epochs}  train_acc={ta:.3f}  val_acc={va:.3f}  val_loss={vl:.3f}")
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), OUT_DIR / "waste_model.pth")


# Phase 1: train only the new classification head
for p in model.features.parameters():
    p.requires_grad = False
fit(HEAD_EPOCHS, torch.optim.Adam(model.classifier.parameters(), lr=1e-3), "head")

# Phase 2: fine-tune the whole network with a small learning rate
for p in model.parameters():
    p.requires_grad = True
fit(FINE_EPOCHS, torch.optim.Adam(model.parameters(), lr=1e-4), "fine-tune")

# Final report using the best saved weights
model.load_state_dict(torch.load(OUT_DIR / "waste_model.pth", map_location=device))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in val_dl:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()

(OUT_DIR / "classes.json").write_text(json.dumps(classes))
print("\nBest validation accuracy: %.3f" % best_acc)
print(classification_report(y_true, y_pred, target_names=classes))

ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(xticks_rotation=45)
plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150); plt.close()
plt.plot(history["train_acc"], label="train"); plt.plot(history["val_acc"], label="validation")
plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.savefig(OUT_DIR / "accuracy_curve.png", dpi=150)
print("Saved model + charts in ./model")
