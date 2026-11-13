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
- Next: download original Dryad CSV, inspect tissue/region and coordinate columns, define a section-held-out protocol, fit only on training sections, choose hyperparameters on validation sections, and evaluate on untouched held-out sections.

## Attribution

STELLAR: Brbić M, Cao K, Hickey JW, et al., *Nature Methods* 2022. The upstream source code is released under the MIT license; this reproduction and its preliminary results are independent exploratory work, not an official extension by the original authors.
