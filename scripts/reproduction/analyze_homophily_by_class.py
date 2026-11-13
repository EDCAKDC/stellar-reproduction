from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from datasets import GraphDataset

with open("data/TonsilBE_processed_0.5.pkl", "rb") as f:
    X, y, Xt, edges, et, names = pickle.load(f)

graph = GraphDataset(X, y, Xt, edges, et).labeled_data
truth = graph.y.cpu().numpy()
src, dst = graph.edge_index.cpu().numpy()
n = len(truth)

degree = np.bincount(dst, minlength=n)
same = np.bincount(
    dst,
    weights=(truth[src] == truth[dst]).astype(float),
    minlength=n
)

homophily = np.divide(
    same, degree,
    out=np.full(n, np.nan),
    where=degree > 0
)

region = np.full(n, "middle", dtype=object)
region[(degree > 0) & (homophily <= .25)] = "low"
region[homophily > .75] = "high"
region[degree == 0] = "isolated"

rows = []

for seed in [42, 7, 123, 2026]:
    root = Path(
        f"results/masked_transductive_seed{seed}_ep250"
    )

    sp = pd.read_csv(
        root / "spatial_predictions.csv"
    ).sort_values("cell_index").reset_index(drop=True)

    ne = pd.read_csv(
        root / "no_edges_predictions.csv"
    ).sort_values("cell_index").reset_index(drop=True)

    idx = sp.cell_index.to_numpy()
    yy = sp.true.to_numpy()

    assert np.array_equal(idx, ne.cell_index.to_numpy())
    assert np.array_equal(yy, ne.true.to_numpy())
    assert np.array_equal(yy, truth[idx])

    delta = (
        (sp.predicted.to_numpy() == yy).astype(int)
        - (ne.predicted.to_numpy() == yy).astype(int)
    )

    df = pd.DataFrame({
        "class_id": yy,
        "region": region[idx],
        "gain_pp": 100 * delta
    })

    group = (
        df[df.region.isin(["low", "high"])]
        .groupby(["class_id", "region"])
        .agg(
            n=("gain_pp", "size"),
            gain_pp=("gain_pp", "mean")
        ).reset_index()
    )

    low = group[group.region == "low"][
        ["class_id", "n", "gain_pp"]
    ].rename(columns={
        "n": "n_low",
        "gain_pp": "low_gain"
    })

    high = group[group.region == "high"][
        ["class_id", "n", "gain_pp"]
    ].rename(columns={
        "n": "n_high",
        "gain_pp": "high_gain"
    })

    paired = low.merge(high, on="class_id")
    paired["seed"] = seed
    paired["high_minus_low_pp"] = (
        paired.high_gain - paired.low_gain
    )

    rows.append(paired)

all_results = pd.concat(rows, ignore_index=True)

# Require at least 30 test cells in both groups
eligible = all_results[
    (all_results.n_low >= 30) &
    (all_results.n_high >= 30)
]

summary = eligible.groupby("class_id").agg(
    n_seeds=("seed", "nunique"),
    mean_difference_pp=("high_minus_low_pp", "mean"),
    sd_difference_pp=("high_minus_low_pp", "std"),
    positive_seeds=(
        "high_minus_low_pp",
        lambda x: int((x > 0).sum())
    )
).reset_index()

summary["cell_type"] = summary.class_id.map(
    lambda i: names[int(i)]
)

summary = summary.sort_values(
    "mean_difference_pp", ascending=False
)

out = Path("results/neighborhood_analysis")
all_results.to_csv(
    out / "class_stratified_all_seeds.csv",
    index=False
)
summary.to_csv(
    out / "class_stratified_summary.csv",
    index=False
)

print("\n=== WITHIN-CLASS HOMOPHILY EFFECT ===")
print(summary.round(3).to_string(index=False))
print("\nCLASS-STRATIFIED ANALYSIS COMPLETE")
