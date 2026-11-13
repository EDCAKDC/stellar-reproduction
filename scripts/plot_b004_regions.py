#!/usr/bin/env python3
"""Plot region-level spatial-minus-no-edge holdout differences."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--csv", type=Path, default=Path("tables/b004_region_effects.csv"))
p.add_argument("--out", type=Path, default=Path("figures/b004_region_effects.png"))
a = p.parse_args()
df = pd.read_csv(a.csv).sort_values("test_region")
required = {"test_region", "accuracy_gain_pp", "macro_f1_gain_20", "balanced_acc_gain_pp"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

# Plot each metric separately, avoiding misleading shared units.
metrics = [
    ("accuracy_gain_pp", "Accuracy difference (percentage points)", "accuracy"),
    ("macro_f1_gain_20", "Macro-F1 difference (20 colon classes)", "macro_f1"),
    ("balanced_acc_gain_pp", "Balanced accuracy difference (percentage points)", "balanced_accuracy"),
]
a.out.parent.mkdir(parents=True, exist_ok=True)
for col, ylabel, suffix in metrics:
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    vals = df[col].to_numpy()
    labels = df["test_region"].str.extract(r"(reg\d+)")[0].fillna(df["test_region"]).tolist()
    ax.bar(labels, vals)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Held-out B004 colon region")
    ax.set_title("Spatial edges minus no neighbor edges")
    ax.text(0.98, 0.98, "One donor, seed 42; exploratory", va="top", ha="right",
            transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    out = a.out if suffix == "accuracy" else a.out.with_name(a.out.stem + "_" + suffix + a.out.suffix)
    fig.savefig(out, dpi=250)
    plt.close(fig)
    print("Saved:", out)
