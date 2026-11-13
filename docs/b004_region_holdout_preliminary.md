# B004 colon region-holdout: initial result (seed 42)

**Status: preliminary single-donor, single-split result.**

## Protocol
- Data: public B004 CODEX CSV, colon (CL) only; **one donor**.
- Reference markers: 48 CODEX measurements, with cell type `cell_type_A`.
- Training: `reg001_CL_B004` and `reg002_CL_B004` (55,025 cells).
- Validation: `reg003_CL_B004` (30,545 cells); used for selecting the best epoch.
- Final test: `reg004_CL_B004` (25,063 cells); withheld from training and checkpoint selection.
- Feature scaling fitted **only** on training regions.
- Region-specific, symmetrized radius-bounded k-nearest-neighbor graphs (k=8, radius=50); **not the full radius-neighbor graph of the upstream STELLAR implementation**.
- Comparison: original STELLAR **Encoder architecture** trained in ordinary supervised classification, with vs without neighbors. **Not** the complete STELLAR method for novel-class discovery.
- GPU: RTX 5090; seed 42; maximum 250 epochs; Adam; checkpoint selected by validation Macro-F1.

## One held-out region result

| Metric | Spatial edges | No neighbor edges |
|---|---:|---:|
| Validation Macro-F1 (best epoch) | 0.801164 | 0.802258 |
| Selected epoch | 215 | 245 |
| Test accuracy | 0.889518 | 0.886007 |
| Test Macro-F1 (classes present in test) | 0.798361 | 0.795145 |
| Test balanced accuracy | 0.780505 | 0.795633 |

Spatial minus no-edge test accuracy: **+0.351 percentage points**; test Macro-F1: **+0.00322**; balanced accuracy: **−1.513 percentage points**. Mixed evidence; **no claim of improvement or significance**.

## Audit caveat
The training console reported `y_pred contains classes not in y_true`. This may arise because the test region lacks some of the 21 label categories; predicting such classes is an error. The initial run computed Macro-F1 over labels **present in the test region** (so absent labels are not included in the average). For comparisons across held-out regions, a second metric should use a consistent full 21-class label list, and class support in each partition must be checked. Run:

```bash
python scripts/audit_b004_holdout.py --results results/b004_region_holdout
```

**Do not infer cross-donor generalization from these data.** Predictions on one independent region from the same donor are not an independent biological replication. Class composition, section geometry and number of unique sections can change performance. Evaluating multiple held-out regions and seeds using a pre-specified protocol is the next appropriate step.

## Reproduction
```bash
python -u scripts/run_b004_region_holdout.py \
  --csv ~/Downloads/B004_training_dryad.csv \
  --stellar-src ~/projects/stellar \
  --test-region reg004_CL_B004 \
  --val-region reg003_CL_B004 \
  --max-epochs 250 --seed 42
```

Original software: [STELLAR](https://github.com/snap-stanford/stellar), Brbić *et al.*, *Nature Methods* (2022). Input dataset: [Dryad](https://doi.org/10.5061/dryad.g4f4qrfrc).
