#!/usr/bin/env python3
"""Audit B004 colon region-holdout predictions without re-training.

Expected outputs from scripts/run_b004_region_holdout.py. Reads only aggregate
class support and locally saved test predictions, and writes aggregate tables.
Does not upload cell-level records or claim biological replication.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, default=Path("results/b004_region_holdout"))
args = p.parse_args()
root = args.results
support = pd.read_csv(root / "class_support.csv")
sp = pd.read_csv(root / "spatial_test_predictions.csv")
no = pd.read_csv(root / "no_edges_test_predictions.csv")

needed = {"true_label", "pred_label", "confidence"}
assert needed.issubset(sp) and needed.issubset(no), "Prediction files have unexpected columns"
assert len(sp) == len(no), "Unequal test cell count"
assert sp["true_label"].equals(no["true_label"]), "Different test cell order or true labels"

by_class = support.pivot_table(index="cell_type", columns="partition",
                               values="count", aggfunc="sum", fill_value=0)
for part in ("train", "val", "test"):
    if part not in by_class:
        raise ValueError(f"Missing partition in class_support.csv: {part}")
by_class = by_class[["train", "val", "test"]].astype(int)
by_class.to_csv(root / "class_support_wide.csv")
print("\n=== CLASS COVERAGE ===")
print(by_class.to_string())
print("\nTest labels absent from training:",
      by_class.index[(by_class.train == 0) & (by_class.test > 0)].tolist())
print("Validation labels absent from training:",
      by_class.index[(by_class.train == 0) & (by_class.val > 0)].tolist())

labels = by_class.index.astype(str).tolist()
truth = sp["true_label"].astype(str).to_numpy()
sp_pred = sp["pred_label"].astype(str).to_numpy()
no_pred = no["pred_label"].astype(str).to_numpy()
sp_ok = sp_pred == truth
no_ok = no_pred == truth

print("\n=== PAIRED DISAGREEMENTS (same test cells) ===")
print("Both correct:", int((sp_ok & no_ok).sum()))
print("Spatial correct only:", int((sp_ok & ~no_ok).sum()))
print("No edges correct only:", int((~sp_ok & no_ok).sum()))
print("Both incorrect:", int((~sp_ok & ~no_ok).sum()))
print("Changed predicted label:", int((sp_pred != no_pred).sum()))

summary = []
class_rows = []
for mode, pred in [("spatial", sp_pred), ("no_edges", no_pred)]:
    p_cls, r_cls, f_cls, count = precision_recall_fscore_support(
        truth, pred, labels=labels, zero_division=0)
    for c, pc, rc, fc, n in zip(labels, p_cls, r_cls, f_cls, count):
        class_rows.append({"model": mode, "cell_type": c,
                           "test_support": int(n),
                           "precision": float(pc), "recall": float(rc),
                           "f1": float(fc)})
    summary.append({
        "model": mode,
        "test_accuracy": float(accuracy_score(truth, pred)),
        "test_macro_f1_all_classes": float(f1_score(
            truth, pred, labels=labels, average="macro", zero_division=0)),
        "test_macro_f1_present_classes": float(f1_score(
            truth, pred, labels=np.unique(truth),
            average="macro", zero_division=0)),
        "test_n": int(len(truth)),
    })
summary_df = pd.DataFrame(summary)
class_df = pd.DataFrame(class_rows)
summary_df.to_csv(root / "audit_metrics.csv", index=False)
class_df.to_csv(root / "audit_per_class.csv", index=False)
wide = class_df.pivot(index="cell_type", columns="model", values="f1")
wide["spatial_minus_no_edges"] = wide["spatial"] - wide["no_edges"]
wide.sort_values("spatial_minus_no_edges", ascending=False).to_csv(
    root / "audit_per_class_gain.csv")

print("\n=== CONSISTENT-LABEL METRICS ===")
print(summary_df.to_string(index=False))
print("\n=== CLASS F1 CHANGES ===")
print(wide.sort_values("spatial_minus_no_edges", ascending=False)
      .round(4).to_string())
print("\nSaved aggregate audit files to:", root)
