#!/usr/bin/env python3
"""Summarize B004 region-held-out spatial vs no-edge experiments.

Reads LOCAL prediction files. Writes aggregate CSVs only; does not publish
individual-cell predictions, raw CODEX data, or model weights.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, default=Path("results"))
p.add_argument("--out", type=Path, default=Path("tables"))
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

dirs = {
    "reg001_CL_B004": a.results / "holdout_reg001_CL_B004",
    "reg002_CL_B004": a.results / "holdout_reg002_CL_B004",
    "reg003_CL_B004": a.results / "holdout_reg003_CL_B004",
    "reg004_CL_B004": a.results / "b004_region_holdout",
}
rows, paired, classes = [], [], []
missing = []
for region, root in dirs.items():
    required = [
        root / "comparison.csv", root / "class_support.csv",
        root / "spatial_test_predictions.csv",
        root / "no_edges_test_predictions.csv",
    ]
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        missing.append(region)
        print("Not complete:", region)
        continue

    comp = pd.read_csv(root / "comparison.csv")
    support = pd.read_csv(root / "class_support.csv")
    vocabulary = sorted(support["cell_type"].astype(str).unique().tolist())
    sp = pd.read_csv(root / "spatial_test_predictions.csv")
    no = pd.read_csv(root / "no_edges_test_predictions.csv")
    if len(sp) != len(no) or not sp["true_label"].equals(no["true_label"]):
        raise ValueError(f"Predictions not aligned in region {region}")
    y = sp["true_label"].astype(str).to_numpy()
    if not set(y).issubset(set(vocabulary)):
        raise ValueError(f"Unknown true labels in region {region}")
    ntest = int((support.loc[support.partition == "test", "count"]).sum())
    if ntest != len(y):
        raise ValueError(f"Test support mismatch in region {region}")
    if len(vocabulary) != 20:
        raise ValueError(f"Expected 20 colon classes; found {len(vocabulary)} in {region}")

    sp_pred = sp["pred_label"].astype(str).to_numpy()
    no_pred = no["pred_label"].astype(str).to_numpy()
    ok_sp, ok_no = sp_pred == y, no_pred == y
    paired.append({
        "test_region": region, "n_test": len(y),
        "spatial_only_correct": int((ok_sp & ~ok_no).sum()),
        "no_edges_only_correct": int((~ok_sp & ok_no).sum()),
        "net_spatial_correct": int(ok_sp.sum() - ok_no.sum()),
        "changed_predictions": int((sp_pred != no_pred).sum()),
    })
    for name, pred in [("spatial", sp_pred), ("no_edges", no_pred)]:
        c = comp.loc[comp.model == name]
        if len(c) != 1:
            raise ValueError(f"Expected one summary row per model in {region}")
        accuracy = accuracy_score(y, pred)
        if not np.isclose(float(c.iloc[0]["test_accuracy"]), accuracy, atol=1e-8):
            raise ValueError(f"Saved accuracy disagrees with predictions: {region} {name}")
        rows.append({
            "test_region": region, "model": name, "seed": int(c.iloc[0]["seed"]),
            "n_test": len(y), "n_present_test_classes": len(np.unique(y)),
            "accuracy": accuracy,
            "macro_f1_20_classes": f1_score(
                y, pred, labels=vocabulary, average="macro", zero_division=0),
            "macro_f1_present_classes": f1_score(
                y, pred, labels=np.unique(y), average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "best_epoch": int(c.iloc[0]["best_epoch"]),
            "best_val_macro_f1": float(c.iloc[0]["val_macro_f1"]),
        })
        prec, rec, f, count = precision_recall_fscore_support(
            y, pred, labels=vocabulary, zero_division=0)
        for label, n, f1 in zip(vocabulary, count, f):
            classes.append({"test_region": region, "model": name,
                            "cell_type": label, "n_test_type": int(n),
                            "f1": float(f1)})

if not rows:
    raise SystemExit("No complete held-out regions yet.")

metrics = pd.DataFrame(rows).sort_values(["test_region", "model"])
paired_df = pd.DataFrame(paired).sort_values("test_region")
class_df = pd.DataFrame(classes)
metrics.to_csv(a.out / "b004_region_metrics.csv", index=False)
paired_df.to_csv(a.out / "b004_region_paired_correct.csv", index=False)
class_wide = class_df.pivot(
    index=["test_region", "cell_type", "n_test_type"],
    columns="model", values="f1").reset_index()
class_wide["spatial_gain_f1"] = (
    class_wide["spatial"] - class_wide["no_edges"])
class_wide.to_csv(a.out / "b004_region_class_f1.csv", index=False)

diff_rows = []
for region, g in metrics.groupby("test_region"):
    sp = g.loc[g.model == "spatial"].iloc[0]
    ne = g.loc[g.model == "no_edges"].iloc[0]
    if sp.seed != ne.seed:
        raise ValueError(f"Mismatched seeds for region {region}")
    diff_rows.append({
        "test_region": region, "seed": int(sp.seed),
        "n_test": int(sp.n_test),
        "accuracy_gain_pp": 100 * (sp.accuracy - ne.accuracy),
        "macro_f1_gain_20": sp.macro_f1_20_classes - ne.macro_f1_20_classes,
        "balanced_acc_gain_pp": 100 * (
            sp.balanced_accuracy - ne.balanced_accuracy),
    })
diff_df = pd.DataFrame(diff_rows).sort_values("test_region")
diff_df.to_csv(a.out / "b004_region_effects.csv", index=False)

print("\n=== REGION-HELD-OUT EFFECTS: SPATIAL minus NO_EDGES ===")
print(diff_df.round(4).to_string(index=False))
print("\n=== PAIRED PREDICTIONS ===")
print(paired_df.to_string(index=False))
print("\n=== SELECTED CLASS F1 DIFFERENCES (ONLY WHEN >=30 TEST CELLS) ===")
for label in ["TA", "Lymphatic", "Stroma", "Macrophage", "Endothelial"]:
    sub = class_wide.loc[
        (class_wide.cell_type == label) & (class_wide.n_test_type >= 30),
        ["test_region", "n_test_type", "spatial_gain_f1"]
    ]
    if not sub.empty:
        print("\n" + label)
        print(sub.round(4).to_string(index=False))

print("\nCompleted regions:", len(diff_df), "/ 4")
if missing:
    print("Still missing:", ", ".join(missing))
if len(diff_df) == 4:
    print("\nMean of four region-level spatial minus no-edge differences:")
    print(diff_df[
        ["accuracy_gain_pp", "macro_f1_gain_20", "balanced_acc_gain_pp"]
    ].mean().round(4).to_string())
    print("These four regions come from ONE donor; not independent donors.")
print("\nAggregate CSV files saved in", a.out)
