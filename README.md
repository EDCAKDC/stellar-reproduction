# STELLAR reproduction and spatial-neighborhood ablation

**Status:** exploratory computational reproduction; **not** an independent validation of the original STELLAR paper.

## Research question

When does spatial neighborhood information improve cell-type classification from CODEX single-cell protein measurements?

## What was done

1. Adapted the original [STELLAR](https://github.com/snap-stanford/stellar) Python implementation to a recent PyTorch/PyG environment and reproduced training and predictions on the authors' preprocessed Tonsil/Barrett's esophagus graph.
2. Trained the **STELLAR Encoder only** as a supervised ten-class classifier on 86,984 labeled reference cells (44 protein markers), comparing the full spatial graph with the same encoder when neighbor edges were removed.
3. Used a **transductive** 70/10/20 train/validation/test label split on the same reference graph, selected the epoch using validation macro-F1, and repeated with random seeds 42, 7, 123, and 2026.
4. Performed exploratory class-specific and neighborhood-homophily analyses of model predictions.

The full original STELLAR method includes discovery of candidate novel cell types on an unlabeled target graph. The supervised Encoder ablation here **does not evaluate that full task**.

## Preliminary results

| Metric, four runs (mean ± sample SD) | With spatial edges | Without neighbor edges |
|---|---:|---:|
| Test accuracy | 0.9332 ± 0.0010 | 0.9344 ± 0.0029 |
| Test macro-F1 | 0.8707 ± 0.0189 | 0.8755 ± 0.0117 |
| Test balanced accuracy | 0.8539 ± 0.0291 | 0.8615 ± 0.0200 |

Spatial and no-edge classification were similar in this setting; **no general superiority is claimed**. The tests were exploratory, repeated on the same graph, and are not four independent biological replicates.

**Post hoc neighborhood analysis:** same-type neighbor fraction was 0.4572 (independent-endpoint reference 0.2510). The mean spatial-minus-no-edge accuracy difference was -0.409 percentage points for low-homophily nodes (≤25% same-type neighbors) and +0.183 percentage points for high-homophily nodes (>75%). A within-class contrast was +0.980 percentage points for innate and -0.451 for T cells, but these small differences have not been independently validated. Homophily was computed using true cell labels **after prediction, for interpretation only**.


## B004 held-out colon-region evaluation (new)

A region-disjoint **supervised STELLAR Encoder** comparison was completed on public CODEX donor B004, colon only. Two regions were used for training (55,025 cells), one for validation (30,545), and a *previously withheld* region for testing (25,063). Each region had its own distance-bounded k=8 spatial graph; train-only standardization prevented marker-scaling leakage. This is a **graph variant**, not the full-radius graph of the original paper.

| Seed 42 held-out reg004 metrics | Spatial | No edges |
|---|---:|---:|
| Test accuracy | 0.889518 | 0.886007 |
| Test Macro-F1, 19 types present in test | 0.798361 | 0.795145 |
| Test Macro-F1, all 20 colon types | 0.758443 | 0.755388 |
| Test balanced accuracy | 0.780505 | 0.795633 |

For paired predictions, spatial was correct alone on 559 cells versus no-edge correct alone on 471 cells; this is an **88-cell net difference out of 25,063**, not evidence of robust superiority. The test region contained no `Enterocyte_ITLN1p` cells, and multiple other types have very low test counts. Results are exploratory (same donor, one held-out region, hyperparameter exploration).

- Protocol and original results: [B004 region holdout report](docs/b004_region_holdout_preliminary.md)
- Train/validation/test script: [run_b004_region_holdout.py](scripts/run_b004_region_holdout.py)
- Recompute paired audit: `python scripts/audit_b004_holdout.py --results results/b004_region_holdout`

## B004 cross-region follow-up: four held-out colon regions (seed 42)

Four region-disjoint, supervised Encoder comparisons have now been completed on **110,633 colon cells from one donor**, with each region used once for final testing and the others divided into training and validation. Graphs use symmetrized radius-bounded kNN (k=8, radius=50), **not** the original full radius graph.

| Held-out region | Accuracy difference (pp) | Macro-F1 difference, fixed 20 classes | Balanced accuracy difference (pp) |
|---|---:|---:|---:|
| reg001 | -0.5741 | -0.0583 | -3.8933 |
| reg002 | +0.9467 | -0.0072 | -1.3158 |
| reg003 | -0.0688 | +0.0050 | -0.1694 |
| reg004 | +0.3511 | +0.0031 | -1.5129 |
| **Mean across regions** | **+0.1637** | **-0.0144** | **-1.7228** |

All differences are **spatial minus no-edges**. The region-unweighted accuracy improvement is small, but the Macro-F1 and balanced-accuracy averages favor the no-edge control. A pooled tally shows **74 additional correct spatial predictions out of 110,633 unique held-out cells** (a different, cell-weighted comparison). There is **no consistent performance advantage** from adding spatial edges in these exploratory runs.

The B004 experiments are **one donor, seed 42, and 4 overlapping-fold technical evaluations, not 4 biological replicates**. The results do not validate full STELLAR novel-cell-type discovery, and should not be used to claim a statistically significant or cross-donor generalization benefit.

- [Four-region methods, results, and cautions](docs/b004_four_region_summary.md)
- [Aggregate four-region summary CSV](tables/b004_region_summary_seed42.csv)
- [Summarize from local predictions](scripts/summarize_b004_regions.py)
- [Plot region differences from CSV](scripts/plot_b004_regions.py)

To regenerate plots locally:

```bash
python scripts/summarize_b004_regions.py --results results --out tables
python scripts/plot_b004_regions.py --csv tables/b004_region_effects.csv --out figures/b004_region_effects.png
```

## Figures

- `figures/macro_f1.png`: four-run macro-F1 summary, sample SD
- `figures/class_homophily_effect.png`: within-class high-minus-low homophily effect (mean ± sample SD)

## Regenerate plots from your original CSV outputs

From this folder, with your STELLAR conda environment active:

```bash
python make_figures.py --results ~/projects/stellar/results
```

The script writes plots with individual random-seed points to `figures/`. The two included figures are reconstructed from the numerical summary in the chat; the regenerated versions use the full local CSVs.

## Reproducibility

The user's original code and data remain on their Linux workstation at `~/projects/stellar`. Keep the full graph data, weights, and large cell-level prediction arrays out of the public repository. Suggested provenance:

- Original implementation: https://github.com/snap-stanford/stellar
- Paper: https://doi.org/10.1038/s41592-022-01651-8
- Preprocessed example: https://zenodo.org/records/6836373
- Original measured CODEX cells, x/y coordinates, and tissue metadata: https://doi.org/10.5061/dryad.g4f4qrfrc

**Environment used:** NVIDIA RTX 5090, PyTorch 2.10.0+cu128, PyG 2.8.1, Python 3.11, Scanpy 1.11.5. A small local Scanpy sparse-array compatibility patch and a legacy `louvain`/`setuptools` compatibility adjustment were required.

## Limitations and next steps

- These are within-reference-graph transductive tests; test-node features and graph structure were visible during training, though their labels were masked.
- Hyperparameters and analyses were inspected repeatedly; test-set conclusions should be considered hypothesis-generating.
- The original preprocessed `.pkl` lacks the accessible section/xy metadata needed to verify a true geographically held-out evaluation. **Do not label random cell splits as cross-section validation.**
- The B004 CSV was subsequently inspected and an initial held-out region evaluation completed (see section above). Four held-out B004 colon regions were subsequently evaluated with seed 42 (see above). Next: test across seeds and independently annotated donors, and quantify sensitivity to graph settings without test-set-driven tuning.

## Attribution

STELLAR: Brbić M, Cao K, Hickey JW, et al., *Nature Methods* 2022. The upstream source code is released under the MIT license; this reproduction and its preliminary results are independent exploratory work, not an official extension by the original authors.
