# B004 colon region-holdout: initial result (seed 42)

**First region-held-out run (seed 42).** The full four-region comparison is available in [the final regional analysis](b004_four_region_summary.md).

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

## Class coverage and paired predictions

The local audit script was executed against the saved predictions for seed 42:

- **20** cell types in the B004 colon subset; all test/validation labels occur in training. `Enterocyte_ITLN1p` has **zero** test cells, so the original 19-present-class Macro-F1 (0.798361 / 0.795145) differs from the **fixed 20-class** Macro-F1 (spatial **0.758443**, no edges **0.755388**). The absence of `Paneth` in this colon-only subset is expected; the complete B004 CSV contains 21 cell types across colon and small intestine.
- Paired predictions across the same 25,063 held-out cells: **21,735 both correct; 559 spatial-only correct; 471 no-edge-only correct; 2,298 both wrong**. Thus spatial produced **88** more correct labels (**+0.351 percentage points**), while **1,157** cells were assigned different labels.
- Class F1 differences (spatial minus no edges) include TA **+0.0222** (3,604 test cells), Lymphatic **+0.0206** (741), CD7_Immune **+0.1071** (only 4), B **+0.0723** (only 21), and Enterocyte_CD57p **−0.1143** (only 5). Rare-category changes are too unstable for biological conclusions.
- The class mix shifts across regions: for example, TA has **866** training cells, **5,551** validation cells and **3,604** test cells. This distribution shift matters when interpreting the holdout metrics.
- Test balanced accuracy is **0.780505 spatial vs 0.795633 no edges**; not every metric favors spatial. All findings are **exploratory**, with one donor and one held-out region, and the same test set has been examined repeatedly. No statistical significance or generalization superiority is claimed.

The other three regions were subsequently tested. The [four-region analysis](b004_four_region_summary.md) supersedes conclusions from this single fold.

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
