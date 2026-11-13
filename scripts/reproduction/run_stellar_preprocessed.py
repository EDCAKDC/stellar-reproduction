import pickle
import random
import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import pandas as pd
import torch

from datasets import GraphDataset
from STELLAR import STELLAR

# Reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

outdir = Path("results/stellar_tonsilbe")
outdir.mkdir(parents=True, exist_ok=True)

# Load official preprocessed graph
with open("data/TonsilBE_processed_0.5.pkl", "rb") as f:
    X_train, y_train, X_target, \
    edges_train, edges_target, inverse_dict = pickle.load(f)

dataset = GraphDataset(
    X_train, y_train, X_target,
    edges_train, edges_target
)

args = Namespace(
    device=torch.device("cuda"),
    num_heads=13,
    num_seed_class=3,
    epochs=20,
    lr=1e-3,
    wd=5e-2
)

print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("Training cells:", len(X_train), flush=True)
print("Target cells:", len(X_target), flush=True)
print("Protein markers:", X_train.shape[1], flush=True)

# Official STELLAR training
stellar = STELLAR(args, dataset)

print("STARTING STELLAR TRAINING", flush=True)
stellar.train()
print("TRAINING FINISHED", flush=True)

# Predictions
mean_uncertainty, predictions = stellar.pred()

stellar.model.eval()
with torch.no_grad():
    graph = dataset.unlabeled_data.to(args.device)
    logits, features, embeddings = stellar.model(graph)
    assert torch.isfinite(logits).all(), "Non-finite logits"
    probabilities = torch.softmax(logits, dim=1)

    confidence = probabilities.max(dim=1).values.cpu().numpy()
    embeddings = embeddings.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

# Save model
torch.save(
    stellar.model.state_dict(),
    outdir / "model_state_dict.pt"
)

# Save predictions and embeddings
np.save(outdir / "predictions.npy", predictions)
np.save(outdir / "embeddings.npy", embeddings)
np.save(outdir / "probabilities.npy", probabilities)

pd.DataFrame({
    "cell_index": np.arange(len(predictions)),
    "predicted_class": predictions,
    "confidence": confidence
}).to_csv(outdir / "cell_predictions.csv", index=False)

metadata = {
    "dataset": "TonsilBE_processed_0.5",
    "seed": seed,
    "epochs": args.epochs,
    "num_seed_class": args.num_seed_class,
    "known_classes": {
        str(k): str(v) for k, v in inverse_dict.items()
    },
    "mean_uncertainty": float(mean_uncertainty),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda
}

(outdir / "metadata.json").write_text(
    json.dumps(metadata, indent=2)
)

print("Predictions:", len(predictions))
print("Predicted class counts:")
print(pd.Series(predictions).value_counts().sort_index())
print("Mean uncertainty:", round(float(mean_uncertainty), 4))
print("GPU peak memory (MiB):",
      round(torch.cuda.max_memory_allocated()/1024**2, 1))
print("Results saved to:", outdir)
print("STELLAR TRAINING AND PREDICTION COMPLETE")
