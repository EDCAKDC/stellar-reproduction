import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import GraphDataset

OUT = Path("results/neighborhood_analysis")
OUT.mkdir(parents=True, exist_ok=True)

with open("data/TonsilBE_processed_0.5.pkl", "rb") as f:
    X, y, Xt, edges, edgest, names = pickle.load(f)

ds = GraphDataset(X, y, Xt, edges, edgest)
graph = ds.labeled_data

y = graph.y.cpu().numpy()
src, dst = graph.edge_index.cpu().numpy()
n = len(y)

# Count incoming spatial neighbors
degree = np.bincount(dst, minlength=n)

# Fraction of neighbors sharing the cell's true label
same = (y[src] == y[dst]).astype(float)
same_count = np.bincount(
    dst, weights=same, minlength=n
)

homophily = np.full(n, np.nan)
np.divide(
    same_count, degree,
    out=homophily,
    where=degree > 0
)

# Define neighborhood groups
group = np.full(n, "isolated", dtype=object)

group[(degree > 0) & (homophily <= 0.25)] = "0-25%"
group[(homophily > 0.25) & (homophily <= 0.50)] = "25-50%"
group[(homophily > 0.50) & (homophily <= 0.75)] = "50-75%"
group[(homophily > 0.75)] = "75-100%"

info = pd.DataFrame({
    "cell_index": np.arange(n),
    "true": y,
    "cell_type": [names[int(i)] for i in y],
    "degree": degree,
    "homophily": homophily,
    "group": group
})

observed = same.mean()
src_freq = np.bincount(y[src], minlength=10) / len(src)
dst_freq = np.bincount(y[dst], minlength=10) / len(dst)
baseline = np.dot(src_freq, dst_freq)

print("Total spatial edges:", len(src))
print("Overall same-type edge fraction:", round(observed, 4))
print("Independent-endpoint baseline:", round(baseline, 4))
print("Isolated cells:", int((degree == 0).sum()))

all_rows = []

for seed in [42, 7, 123, 2026]:
    root = Path(
        f"results/masked_transductive_seed{seed}_ep250"
    )

    spatial = pd.read_csv(
        root / "spatial_predictions.csv"
    )
    no_edges = pd.read_csv(
        root / "no_edges_predictions.csv"
    )

    df = spatial.merge(
        no_edges,
        on="cell_index",
        suffixes=("_sp", "_no"),
        validate="one_to_one"
    ).merge(
        info, on="cell_index",
        validate="one_to_one"
    )

    assert np.array_equal(df.true_sp, df.true_no)
    assert np.array_equal(df.true_sp, df.true)

    df["spatial_correct"] = (
        df.predicted_sp == df.true
    ).astype(int)

    df["no_edges_correct"] = (
        df.predicted_no == df.true
    ).astype(int)

    result = df.groupby("group").agg(
        n_cells=("cell_index", "size"),
        spatial_accuracy=("spatial_correct", "mean"),
        no_edges_accuracy=("no_edges_correct", "mean")
    ).reset_index()

    result["seed"] = seed
    result["gain_pp"] = 100 * (
        result.spatial_accuracy -
        result.no_edges_accuracy
    )

    all_rows.append(result)

detail = pd.concat(all_rows, ignore_index=True)
detail.to_csv(OUT / "all_seeds.csv", index=False)

summary = detail.groupby("group").agg(
    mean_n=("n_cells", "mean"),
    mean_gain_pp=("gain_pp", "mean"),
    sd_gain_pp=("gain_pp", "std"),
    positive_seeds=("gain_pp", lambda x: int((x > 0).sum()))
).reset_index()

order = [
    "0-25%", "25-50%",
    "50-75%", "75-100%", "isolated"
]

summary["sort"] = summary.group.map(
    {k: i for i, k in enumerate(order)}
)
summary = summary.sort_values("sort").drop(columns="sort")

summary.to_csv(
    OUT / "neighborhood_summary.csv", index=False
)

print("\n=== NEIGHBORHOOD ANALYSIS ===")
print(summary.round(3).to_string(index=False))

# Plot average paired accuracy difference
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    summary.group,
    summary.mean_gain_pp,
    yerr=summary.sd_gain_pp.fillna(0),
    fmt="o",
    capsize=5
)
ax.axhline(0, linestyle="--")
ax.set_xlabel("Same-type neighbor fraction")
ax.set_ylabel("Spatial minus No edges accuracy (percentage points)")
ax.set_title("Spatial neighborhood effect across four seeds")
fig.tight_layout()
fig.savefig(
    OUT / "neighborhood_effect.png",
    dpi=300
)
plt.close(fig)

print("\nNEIGHBORHOOD ANALYSIS COMPLETE")
