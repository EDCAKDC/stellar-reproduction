#!/usr/bin/env python3
"""B004 colon region holdout: supervised STELLAR Encoder vs no-neighbor-edge control.

This is NOT the full STELLAR novel-cell-type discovery algorithm.
Feature preprocessing is fit on training regions only. Validation and test
regions are disconnected from training and from each other.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, f1_score
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def make_edges(xy, k, radius):
    """Symmetrized distance-bounded kNN: no dense pairwise distance matrix."""
    n = len(xy)
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    dist, nbr = cKDTree(xy).query(
        xy, k=min(k + 1, n), distance_upper_bound=radius, workers=-1
    )
    if nbr.ndim == 1:
        nbr = nbr[:, None]
        dist = dist[:, None]
    row = np.repeat(np.arange(n), nbr.shape[1])
    col = nbr.reshape(-1)
    valid = (col < n) & (row != col) & np.isfinite(dist.reshape(-1))
    edges = torch.as_tensor(
        np.stack([row[valid], col[valid]]), dtype=torch.long
    )
    if edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return to_undirected(edges, num_nodes=n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path.home() / "Downloads/B004_training_dryad.csv")
    p.add_argument("--stellar-src", type=Path, default=Path.home() / "projects/stellar")
    p.add_argument("--out", type=Path, default=Path("results/b004_region_holdout"))
    p.add_argument("--test-region", default="reg004_CL_B004")
    p.add_argument("--val-region", default="reg003_CL_B004")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--radius", type=float, default=50.0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if not a.csv.is_file():
        p.error(f"Missing input CSV: {a.csv}")
    if not (a.stellar_src / "models" / "net.py").is_file():
        p.error(f"STELLAR source not found at {a.stellar_src}")
    if a.k < 1 or a.radius <= 0 or a.eval_every < 1 or a.patience < 1:
        p.error("k, radius, eval-every and patience must be positive")
    if a.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA not available; use --device cpu for a CPU run")

    a.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.stellar_src.resolve()))
    from models import Encoder

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    print("Reading public B004 CODEX CSV...", flush=True)
    df = pd.read_csv(a.csv, low_memory=False)
    required = {"tissue", "donor", "unique_region", "cell_type_A", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing metadata columns: {required.difference(df.columns)}")
    # Official B004 schema: 48 markers in columns 1:49, before metadata.
    markers = list(df.columns[1:49])
    if len(markers) != 48 or not all(pd.api.types.is_numeric_dtype(df[m]) for m in markers):
        raise ValueError("Unexpected 48-marker layout; inspect B004 CSV before proceeding")
    if set(markers) & required:
        raise ValueError("Metadata accidentally selected as marker features")
    df = df.loc[(df["tissue"] == "CL") & (df["donor"] == "B004")].copy()
    regions = sorted(df["unique_region"].unique().tolist())
    if len(regions) != 4:
        raise ValueError(f"Expected four colon regions; found {regions}")
    if a.test_region == a.val_region or a.test_region not in regions or a.val_region not in regions:
        raise ValueError(f"Bad val/test regions. Available: {regions}")
    train_regions = [r for r in regions if r not in {a.val_region, a.test_region}]
    subsets = {
        "train": df[df["unique_region"].isin(train_regions)].copy(),
        "val": df[df["unique_region"] == a.val_region].copy(),
        "test": df[df["unique_region"] == a.test_region].copy(),
    }
    if min(map(len, subsets.values())) == 0:
        raise ValueError("Empty train, val, or test partition")

    # The public annotated cell-type vocabulary is fixed across regions.
    # Labels from val/test are used ONLY for final validation/evaluation, not optimization.
    classes = sorted(df["cell_type_A"].dropna().astype(str).unique().tolist())
    label_id = {x: i for i, x in enumerate(classes)}
    if df["cell_type_A"].isna().any():
        raise ValueError("Missing cell type labels")
    for part, sub in subsets.items():
        print(part, len(sub), sorted(sub["unique_region"].unique().tolist()), flush=True)
    train_known = set(subsets["train"]["cell_type_A"].unique())
    for part in ["val", "test"]:
        unseen = sorted(set(subsets[part]["cell_type_A"]) - train_known)
        if unseen:
            print(f"WARNING: {part} contains cell types absent in training: {unseen}", flush=True)

    # Fit normalization ONLY on training regions to avoid leakage.
    scaler = StandardScaler().fit(subsets["train"][markers].to_numpy(dtype=np.float32))
    np.savez(a.out / "train_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    graphs = {}
    ys = {}
    counts = []
    for part, sub in subsets.items():
        raw_x = sub[markers].to_numpy(dtype=np.float32)
        xy = sub[["x", "y"]].to_numpy(dtype=np.float64)
        if not np.isfinite(raw_x).all() or not np.isfinite(xy).all():
            raise ValueError(f"{part} contains nonfinite markers/coordinates")
        features = scaler.transform(raw_x).astype(np.float32, copy=False)
        y = sub["cell_type_A"].map(label_id).to_numpy(dtype=np.int64)
        # Query neighbors separately for each region, never across regions.
        edge_lists = []
        offset = 0
        for _, region_df in sub.groupby("unique_region", sort=True):
            e = make_edges(region_df[["x", "y"]].to_numpy(dtype=np.float64), a.k, a.radius)
            if e.numel() > 0:
                edge_lists.append(e + offset)
            offset += len(region_df)
        if part == "train":
            # Reorder features/labels into the same per-region order used by edges.
            sub = pd.concat([q for _, q in sub.groupby("unique_region", sort=True)])
            raw_x = sub[markers].to_numpy(dtype=np.float32)
            features = scaler.transform(raw_x).astype(np.float32, copy=False)
            y = sub["cell_type_A"].map(label_id).to_numpy(dtype=np.int64)
        edges = torch.cat(edge_lists, dim=1) if edge_lists else torch.empty((2, 0), dtype=torch.long)
        graphs[part] = Data(x=torch.from_numpy(features), edge_index=edges)
        ys[part] = y
        print(f"{part}: nodes={len(y):,}, directed_edges={edges.shape[1]:,}", flush=True)
        counts.append(pd.DataFrame({
            "partition": part, "cell_type": classes, "count": np.bincount(y, minlength=len(classes))
        }))
    pd.concat(counts, ignore_index=True).to_csv(a.out / "class_support.csv", index=False)

    device = torch.device(a.device)
    results = []
    for mode in ["spatial", "no_edges"]:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        model = Encoder(x_dim=len(markers), num_cls=len(classes)).to(device)
        model_graphs = {}
        for part, graph in graphs.items():
            e = graph.edge_index if mode == "spatial" else torch.empty((2, 0), dtype=torch.long)
            model_graphs[part] = Data(x=graph.x, edge_index=e).to(device)
        yt = torch.tensor(ys["train"], dtype=torch.long, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-2)
        best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
        val_labels = np.unique(ys["val"])
        print(f"\n=== {mode} ===", flush=True)

        for epoch in range(1, a.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(model_graphs["train"])
            loss = F.cross_entropy(logits, yt)
            loss.backward()
            optimizer.step()
            if epoch % a.eval_every:
                continue
            model.eval()
            with torch.no_grad():
                val_logits, _, _ = model(model_graphs["val"])
                val_pred = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(ys["val"], val_pred, labels=val_labels, average="macro", zero_division=0)
            print(f"Epoch {epoch:03d} | train loss {loss.item():.4f} | val macro-F1 {val_f1:.4f}", flush=True)
            if val_f1 > best_f1 + 1e-4:
                best_f1, best_epoch, stale = float(val_f1), epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= a.patience:
                print("Early stopped based on validation only", flush=True)
                break
        if best_state is None:
            raise RuntimeError("No validation checkpoints, increase max-epochs")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_logits, _, _ = model(model_graphs["test"])
            prob = torch.softmax(test_logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        truth = ys["test"]
        test_labels = np.unique(truth)
        metrics = {
            "model": mode, "seed": a.seed, "best_epoch": best_epoch,
            "val_macro_f1": best_f1,
            "test_accuracy": accuracy_score(truth, pred),
            "test_macro_f1": f1_score(truth, pred, labels=test_labels, average="macro", zero_division=0),
            "test_balanced_accuracy": balanced_accuracy_score(truth, pred),
            "test_n": len(truth),
        }
        results.append(metrics)
        report = classification_report(
            truth, pred, labels=np.arange(len(classes)),
            target_names=classes, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(a.out / f"{mode}_class_report.csv")
        pd.DataFrame({
            "true_label": [classes[i] for i in truth],
            "pred_label": [classes[i] for i in pred],
            "confidence": prob.max(axis=1),
        }).to_csv(a.out / f"{mode}_test_predictions.csv", index=False)
        torch.save(best_state, a.out / f"{mode}_best_model.pt")
        print("Test results:", metrics, flush=True)

    result_df = pd.DataFrame(results)
    result_df.to_csv(a.out / "comparison.csv", index=False)
    (a.out / "protocol.json").write_text(json.dumps({
        "dataset": "B004 CL only, one donor",
        "train_regions": train_regions, "val_region": a.val_region,
        "test_region": a.test_region, "markers": markers,
        "label_vocabulary": classes, "seed": a.seed,
        "max_epochs": a.max_epochs, "k": a.k,
        "radius": a.radius,
        "graph_construction": "Symmetrized radius-bounded kNN, differs from original full radius graph",
        "feature_scaling": "StandardScaler fitted on training regions only",
        "task": "Supervised STELLAR Encoder; not full STELLAR novel-class discovery",
        "torch": torch.__version__,
    }, indent=2))
    print("\n=== REGION HOLDOUT COMPARISON ===", flush=True)
    print(result_df.to_string(index=False), flush=True)
    print("Saved results locally:", a.out, flush=True)


if __name__ == "__main__":
    main()
