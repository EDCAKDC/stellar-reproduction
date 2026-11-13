# Four held-out B004 colon-region comparisons

**Exploratory single-donor result, seed 42; 13 November 2026.**

## Scope

- Dataset: [public B004 CODEX](https://doi.org/10.5061/dryad.g4f4qrfrc), colon regions only, one donor (B004).
- 110,633 distinct colon cells in four regions, 48 markers, 20 colon cell types. Each region appeared **once as the test region**; all four folds together cover the 110,633 cells, with overlapping training sets.
- For each held-out test region, one of the other three regions was used for validation, the remaining two for training. No test-region labels were used for model optimization or checkpoint selection in that fold.
- The feature scaler was fitted only on the training regions. Region-specific graphs have no cross-region edges. The graph was built with **symmetrized radius-bounded kNN (k=8, radius=50)**, which is **not** the upstream STELLAR full radius-neighbor graph.
- Both models used the original supervised STELLAR **Encoder architecture**, with spatial edges versus empty neighbor edges. This is **not** an evaluation of the full STELLAR novel-cell discovery task.
- One random seed (42), up to 250 epochs, model selected by held-out validation Macro-F1. The four folds are **not independent donors or biological replicates**.

## Effects (spatial minus no-neighbor-edges)

| Held-out test region | Test cells | Accuracy (pp) | Macro-F1 (fixed 20 classes) | Balanced accuracy (pp) |
|---|---:|---:|---:|---:|
| reg001_CL_B004 | 33,793 | -0.5741 | -0.0583 | -3.8933 |
| reg002_CL_B004 | 21,232 | +0.9467 | -0.0072 | -1.3158 |
| reg003_CL_B004 | 30,545 | -0.0688 | +0.0050 | -0.1694 |
| reg004_CL_B004 | 25,063 | +0.3511 | +0.0031 | -1.5129 |
| **Four-region unweighted mean** | | **+0.1637** | **-0.0144** | **-1.7228** |

Four-fold average treats each region equally. A separate pooled count of correct predictions sums to 74 more spatial-correct classifications across all 110,633 distinct test cells (**+0.0669 percentage points** pooled accuracy difference). These are different weighting schemes, neither establishing superiority.

### Paired test predictions

| Region | Spatial only correct | No-edge only correct | Net spatial correct |
|---|---:|---:|---:|
| reg001 | 359 | 553 | -194 |
| reg002 | 622 | 421 | +201 |
| reg003 | 608 | 629 | -21 |
| reg004 | 559 | 471 | +88 |
| **Total** | **2148** | **2074** | **+74** |

Across all four held-out regions, **balanced accuracy was lower for the spatial model in every region**. Accuracy improved in two of four regions and fixed-class Macro-F1 improved in two of four. The reg001 Macro-F1 difference of -0.0583 is especially important in the average. No inferential significance is claimed.

### Class-specific patterns

- TA F1 differences: **-0.1052, +0.0729, +0.0145, +0.0222** (reg001 through reg004). Positive in three of four, but **not stable across regions**.
- Macrophage F1 differences: **-0.0053, -0.0011, -0.0094, -0.0042**; consistently small negative values, not evidence of biological harm.
- Lymphatic F1 differences: **-0.0281, -0.0125, -0.0073, +0.0206**; mixed.
- Sparse categories remain fragile; a fixed 20-class Macro-F1 includes zero-support classes as zero to make each fold's reported definition consistent. Interpret present-class Macro-F1 alongside class support rather than comparing folds by a changing class list.

## Limitations and interpretation

These results are **hypothesis-generating**. The model is *not* uniformly helped by spatial edges, and the average balanced-accuracy difference favors removing edges. Graph construction and normalization choices may affect the finding. The first fold's test performance was examined before deciding to run the remaining held-out folds; this exploration limits the claim of a completely pristine pre-registered test.

All regions are from **one donor**, and the folds share partially overlapping training regions; neither cells nor folds can be treated as independent biological replicates. Tissue/section composition differs across regions. Comparing a spatial GNN Encoder to its no-edge configuration tests neighbor aggregation in this architecture, not the entire original STELLAR discovery pipeline or all spatial modeling approaches.

## Reproduce the summary from local predictions

```bash
python scripts/summarize_b004_regions.py --results results --out tables
python scripts/plot_b004_regions.py --csv tables/b004_region_effects.csv --out figures/b004_region_effects.png
```

Detailed script: [run_b004_region_holdout.py](../scripts/run_b004_region_holdout.py). Initial reg004 report: [b004_region_holdout_preliminary.md](b004_region_holdout_preliminary.md).

## Reasonable next checks

1. Replicate with additional random seeds using a fixed, documented protocol; do not select results based on test performance.
2. Report class supports, class-specific recall, and confusion matrices for each region.
3. Compare graph construction choices (e.g. full radius graph versus kNN) on validation splits and freeze choices before an additional independently labeled donor is evaluated.
4. **Do not** claim cross-donor generalization until another annotated donor is independently tested.
