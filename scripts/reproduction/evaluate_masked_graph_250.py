import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    balanced_accuracy_score,
    classification_report
)

from torch_geometric.data import Data
from datasets import GraphDataset
from models import Encoder

SEED = 42
MAX_EPOCHS = 250
EVAL_EVERY = 5
PATIENCE = 12
DEVICE = "cuda"

OUT = Path(f"results/masked_transductive_seed{SEED}_ep250")
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with open("data/TonsilBE_processed_0.5.pkl", "rb") as f:
    X, y, X_target, edges, edges_target, names = pickle.load(f)

dataset = GraphDataset(
    X, y, X_target, edges, edges_target
)
full = dataset.labeled_data

y_np = full.y.cpu().numpy().astype(int)
classes = np.unique(y_np)
assert np.array_equal(classes, np.arange(len(classes)))
n_classes = len(classes)

# Keep the original 20% test split
trainval_idx, test_idx = train_test_split(
    np.arange(len(y_np)),
    test_size=0.20,
    stratify=y_np,
    random_state=SEED
)

# 70% train, 10% validation, 20% test
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.125,
    stratify=y_np[trainval_idx],
    random_state=SEED
)

train_t = torch.tensor(train_idx, device=DEVICE)
y_t = full.y.to(DEVICE)

print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print("Original graph edges:", full.num_edges)

results = []

for mode in ["spatial", "no_edges"]:
    print(f"\n=== {mode} ===", flush=True)

    if mode == "spatial":
        edge_index = full.edge_index.to(DEVICE)
    else:
        edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=DEVICE
        )

    # ALL nodes remain in the graph
    graph = Data(
        x=full.x.to(DEVICE),
        edge_index=edge_index
    )

    # Identical initialization
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = Encoder(
        x_dim=full.num_node_features,
        num_cls=n_classes
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=5e-2
    )

    best_f1 = -1
    best_epoch = 0
    best_state = None
    bad_checks = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits, _, _ = model(graph)

        # Training labels ONLY
        loss = F.cross_entropy(
            logits[train_t], y_t[train_t]
        )

        loss.backward()
        optimizer.step()

        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(graph)
                val_pred = (
                    val_logits[val_idx]
                    .argmax(dim=1)
                    .cpu().numpy()
                )

            val_f1 = f1_score(
                y_np[val_idx],
                val_pred,
                labels=classes,
                average="macro",
                zero_division=0
            )

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Val Macro-F1 {val_f1:.4f}",
                flush=True
            )

            if val_f1 > best_f1 + 1e-4:
                best_f1 = val_f1
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad_checks = 0
            else:
                bad_checks += 1

            if bad_checks >= PATIENCE:
                print("Early stopping", flush=True)
                break

    # Restore best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, _, _ = model(graph)
        pred = (
            logits[test_idx].argmax(dim=1)
            .cpu().numpy()
        )

    true = y_np[test_idx]

    acc = accuracy_score(true, pred)
    f1 = f1_score(
        true, pred, labels=classes,
        average="macro", zero_division=0
    )
    bacc = balanced_accuracy_score(true, pred)

    print("Best epoch:", best_epoch)
    print("Test Accuracy:", round(acc, 4))
    print("Test Macro-F1:", round(f1, 4))
    print("Test Balanced accuracy:", round(bacc, 4))

    pd.DataFrame({
        "cell_index": test_idx,
        "true": true,
        "predicted": pred
    }).to_csv(
        OUT / f"{mode}_predictions.csv",
        index=False
    )

    report = classification_report(
        true, pred, labels=classes,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        OUT / f"{mode}_class_report.csv"
    )

    torch.save(
        best_state,
        OUT / f"{mode}_best_model.pt"
    )

    results.append({
        "model": mode,
        "seed": SEED,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_f1,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "test_balanced_accuracy": bacc
    })

summary = pd.DataFrame(results)
summary.to_csv(OUT / "comparison.csv", index=False)

print("\n=== CORRECTED COMPARISON ===")
print(summary.to_string(index=False))
print("MASKED GRAPH EXPERIMENT COMPLETE")
