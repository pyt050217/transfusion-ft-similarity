import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Locate the model implementation bundled with this repository.
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = next(
    (
        parent for parent in _THIS_FILE.parents
        if (parent / "vendor" / "tab-transformer-pytorch").is_dir()
    ),
    None,
)
if REPO_ROOT is None:
    raise FileNotFoundError(
        "Cannot locate vendor/tab-transformer-pytorch. "
        "Run this script from a complete repository checkout."
    )
sys.path.insert(
    0, str(REPO_ROOT / "vendor" / "tab-transformer-pytorch")
)

from tab_transformer_pytorch import FTTransformer


# ============================================================
# Config
# ============================================================

SEED = 42

DATA_DIR = Path(
    os.environ.get(
        "TRANSFUSION_DATA_DIR",
        str(REPO_ROOT / "data" / "processed_ft_dedup_train_seed42"),
    )
).expanduser()
OUTPUT_DIR = Path(
    os.environ.get(
        "TRANSFUSION_OUTPUT_DIR",
        str(REPO_ROOT / "experiments" / "ft_rbc_dedup_seed42" / "output"),
    )
).expanduser()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 512
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load metadata
# ============================================================

with open(
    DATA_DIR / "preprocessing_metadata.json",
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)

category_sizes = tuple(metadata["category_sizes"])


# ============================================================
# Load arrays
# ============================================================

def load_split(name):
    data = np.load(DATA_DIR / f"{name}.npz")

    x_categ = torch.from_numpy(
        data["x_categ"]
    ).long()

    x_cont = torch.from_numpy(
        data["x_cont"]
    ).float()

    y = torch.from_numpy(
        data["y_binary"]
    ).float()

    return x_categ, x_cont, y


print("Loading datasets...")

xcat_train, xcont_train, y_train = load_split("train")
xcat_val, xcont_val, y_val = load_split("val")
xcat_test, xcont_test, y_test = load_split("test")


print("Train:", xcat_train.shape, xcont_train.shape, y_train.shape)
print("Val:", xcat_val.shape, xcont_val.shape, y_val.shape)
print("Test:", xcat_test.shape, xcont_test.shape, y_test.shape)


# ============================================================
# Sanity checks
# ============================================================

assert xcat_train.shape[1] == len(category_sizes)
assert xcont_train.shape[1] == 26

for i, size in enumerate(category_sizes):
    assert xcat_train[:, i].min() >= 0
    assert xcat_train[:, i].max() < size


# ============================================================
# Dataset / DataLoader
# ============================================================

train_ds = TensorDataset(
    xcat_train,
    xcont_train,
    y_train,
)

val_ds = TensorDataset(
    xcat_val,
    xcont_val,
    y_val,
)

test_ds = TensorDataset(
    xcat_test,
    xcont_test,
    y_test,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


# ============================================================
# Model
# ============================================================

model = FTTransformer(
    categories=category_sizes,
    num_continuous=xcont_train.shape[1],
    dim=32,
    dim_out=1,
    depth=4,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
).to(DEVICE)

print("\nDevice:", DEVICE)

n_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", n_params)


# ============================================================
# Class imbalance
# ============================================================

n_pos = float(y_train.sum().item())
n_neg = float(len(y_train) - n_pos)

pos_weight = n_neg / max(n_pos, 1.0)

print("Positive samples:", int(n_pos))
print("Negative samples:", int(n_neg))
print("Positive weight:", pos_weight)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [pos_weight],
        dtype=torch.float32,
        device=DEVICE,
    )
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(loader):
    model.eval()

    all_y = []
    all_prob = []

    for x_categ, x_cont, y in loader:
        x_categ = x_categ.to(
            DEVICE,
            non_blocking=True,
        )

        x_cont = x_cont.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(
            x_categ,
            x_cont,
        ).squeeze(1)

        prob = torch.sigmoid(logits)

        all_y.append(
            y.numpy()
        )

        all_prob.append(
            prob.cpu().numpy()
        )

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_prob)

    y_pred = (
        y_prob >= 0.5
    ).astype(np.int64)

    metrics = {
        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics


# ============================================================
# Train
# ============================================================

best_pr_auc = -1.0
best_epoch = -1

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    for step, (
        x_categ,
        x_cont,
        y
    ) in enumerate(train_loader, 1):

        x_categ = x_categ.to(
            DEVICE,
            non_blocking=True,
        )

        x_cont = x_cont.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        logits = model(
            x_categ,
            x_cont,
        ).squeeze(1)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if step % 200 == 0:
            print(
                f"Epoch {epoch:02d} "
                f"Step {step:04d}/{len(train_loader)} "
                f"Loss {loss.item():.4f}"
            )

    avg_loss = (
        total_loss
        / len(train_loader)
    )

    val_metrics = evaluate(
        val_loader
    )

    print(
        f"\nEpoch {epoch:02d} | "
        f"loss={avg_loss:.4f} | "
        f"ROC-AUC={val_metrics['roc_auc']:.4f} | "
        f"PR-AUC={val_metrics['pr_auc']:.4f} | "
        f"Precision={val_metrics['precision']:.4f} | "
        f"Recall={val_metrics['recall']:.4f} | "
        f"F1={val_metrics['f1']:.4f}"
    )

    if val_metrics["pr_auc"] > best_pr_auc:

        best_pr_auc = val_metrics["pr_auc"]
        best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "category_sizes": category_sizes,
            "num_continuous": xcont_train.shape[1],
            "val_metrics": val_metrics,
            "metadata_path": str(
                DATA_DIR
                / "preprocessing_metadata.json"
            ),
        }

        torch.save(
            checkpoint,
            OUTPUT_DIR
            / "best_fttransformer.pt"
        )

        print(
            "Saved new best model."
        )


# ============================================================
# Final test
# ============================================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    OUTPUT_DIR / "best_fttransformer.pt",
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_metrics = evaluate(
    test_loader
)

print("\n===================================")
print("FINAL RESULTS")
print("===================================")

print("Best epoch:", best_epoch)
print("Best validation PR-AUC:", best_pr_auc)

for key, value in test_metrics.items():
    print(
        f"Test {key}: {value:.4f}"
    )

with open(
    OUTPUT_DIR / "test_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        test_metrics,
        f,
        indent=2,
    )

print("\nTraining finished.")
