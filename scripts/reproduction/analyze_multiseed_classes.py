from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

seeds = [42, 7, 123, 2026]
labels = np.arange(10)
rows = []

with open("data/TonsilBE_processed_0.5.pkl", "rb") as f:
    data = pickle.load(f)

names = data[5]

for seed in seeds:
    root = Path(
        f"results/masked_transductive_seed{seed}_ep250"
    )

    for model in ["spatial", "no_edges"]:
        df = pd.read_csv(
            root / f"{model}_predictions.csv"
        )

        scores = f1_score(
            df["true"],
            df["predicted"],
            labels=labels,
            average=None,
            zero_division=0
        )

        for i, score in enumerate(scores):
            rows.append({
                "seed": seed,
                "model": model,
                "class_id": i,
                "cell_type": names[int(i)],
                "f1": float(score)
            })

df = pd.DataFrame(rows)

wide = df.pivot(
    index=["seed", "class_id", "cell_type"],
    columns="model",
    values="f1"
).reset_index()

wide["spatial_gain"] = (
    wide["spatial"] - wide["no_edges"]
)

summary = wide.groupby(
    ["class_id", "cell_type"]
).agg(
    mean_gain=("spatial_gain", "mean"),
    sd_gain=("spatial_gain", "std"),
    positive_seeds=(
        "spatial_gain",
        lambda x: int((x > 0).sum())
    )
).reset_index()

summary = summary.sort_values(
    "mean_gain", ascending=False
)

print("\n=== CLASS-SPECIFIC SPATIAL EFFECT ===")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    "results/multiseed_class_effect.csv",
    index=False
)

wide.to_csv(
    "results/multiseed_per_class_f1.csv",
    index=False
)

print("\nMULTI-SEED CLASS ANALYSIS COMPLETE")
