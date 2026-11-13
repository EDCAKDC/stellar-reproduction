from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("results/stellar_tonsilbe")

pred = np.load(out / "predictions.npy").astype(int)
probs = np.load(out / "probabilities.npy")
meta = json.loads((out / "metadata.json").read_text())

assert len(pred) == len(probs)
assert np.isfinite(probs).all()
assert np.allclose(probs.sum(axis=1), 1, atol=1e-4)

confidence = probs.max(axis=1)
known = meta["known_classes"]

rows = []
for c in range(probs.shape[1]):
    mask = pred == c
    rows.append({
        "class_id": c,
        "cell_type": known.get(
            str(c), f"Novel candidate {c}"
        ),
        "n_cells": int(mask.sum()),
        "percentage": round(mask.mean() * 100, 2),
        "mean_confidence": (
            round(float(confidence[mask].mean()), 4)
            if mask.any() else np.nan
        )
    })

df = pd.DataFrame(rows)
df.to_csv(out / "class_summary.csv", index=False)

print(df.to_string(index=False))
print("Overall mean confidence:",
      round(float(confidence.mean()), 4))

# Figure 1: Predicted class counts
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(df["class_id"].astype(str), df["n_cells"])
ax.set_xlabel("Predicted class")
ax.set_ylabel("Number of cells")
ax.set_title("STELLAR: Predicted cell-class distribution")
fig.tight_layout()
fig.savefig(out / "class_distribution.png", dpi=300)
plt.close(fig)

# Figure 2: Prediction confidence
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(confidence, bins=40)
ax.set_xlabel("Maximum predicted probability")
ax.set_ylabel("Number of cells")
ax.set_title("STELLAR: Prediction confidence")
fig.tight_layout()
fig.savefig(out / "confidence_distribution.png", dpi=300)
plt.close(fig)

print("DIAGNOSTIC FIGURES SAVED")
