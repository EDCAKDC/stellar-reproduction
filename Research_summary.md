# Evaluating Spatial Neighborhood Contributions to Graph Neural Network-Based Cell-Type Classification

**Exploratory research summary — 13 November 2026**

## Background and objective

STELLAR uses spatial graphs to annotate single-cell multiplexed imaging data and identify candidate novel cell types. This independent computational reproduction asks whether neighborhood edges improve supervised classification on a labeled CODEX reference graph, and whether any benefit differs with local label homophily. The latter is a **post hoc hypothesis**, not an independently confirmed biological discovery.

## Data and methods

The original STELLAR Tonsil/Barrett's esophagus example was run on an NVIDIA RTX 5090. The preprocessed graph contained 86,984 labeled reference cells and 45,958 unlabeled target cells, with 44 protein features per cell. The original full workflow completed and returned target predictions with 13 output classes, including candidate novel classes. The first-pass mean confidence of 0.9446 is **not** a measure of classification accuracy.

An additional *supervised Encoder ablation* was constructed on the **labeled** reference graph. Two conditions used the same GraphSAGE-based Encoder, optimization settings, and per-seed 70%/10%/20% train/validation/test mask: original spatial neighbor edges vs. no neighbor edges. Training ran for up to 250 epochs, with model selection based on validation macro-F1. The evaluation was transductive: node features and the graph were available for all cells but held-out labels were not used in training. Random seeds: 42, 7, 123, 2026.

After observing the same-graph test results, post hoc analyses measured each node's proportion of true-label-matched graph neighbors and its accuracy difference between models. The true test labels were used **only in retrospective interpretation**, not to determine predictions.

## Results

| Held-out test metric, four runs | Spatial edges, mean ± SD | No neighbor edges, mean ± SD |
|---|---:|---:|
| Accuracy | 0.9332 ± 0.0010 | 0.9344 ± 0.0029 |
| Macro-F1 | 0.8707 ± 0.0189 | 0.8755 ± 0.0117 |
| Balanced accuracy | 0.8539 ± 0.0291 | 0.8615 ± 0.0200 |

Differences were small and do not establish a benefit for either setup. Same-label spatial edges made up 45.72% of all recorded edges, compared with 25.10% under a simple independent-endpoint baseline.

| Post hoc homophily bin | Spatial minus no-edge accuracy, mean percentage points | Positive seeds |
|---|---:|---:|
| 0–25% | -0.409 | 1/4 |
| 25–50% | -0.109 | 1/4 |
| 50–75% | +0.145 | 2/4 |
| 75–100% | +0.183 | 4/4 |

The within-class high-vs-low contrast (percentage points) was +1.595 for squamous epithelial (3 eligible runs), +1.039 for B cells (4), +0.980 for innate cells (4), and -0.451 for T cells (4). These are small, descriptive effects; some groups contained few evaluation cells and these patterns were identified after examining the same data.

## Interpretation and limitations

In this single-graph transductive classification task, spatial edges contributed little to overall performance. The post hoc results are consistent with a hypothesis that neighborhood composition may moderate spatial aggregation's value; this is **not causal evidence**. Results may reflect class composition, node degree, spatial autocorrelation, training dynamics, or selection during exploratory analysis.

Crucially, this comparison concerns STELLAR's **supervised Encoder component**, not the full original novel-cell-type discovery method. It does not measure transfer to a new patient, tissue section, or donor. Four seeds are technical repetitions on the same sample graph, not four biological replicates. The original preprocessed pickle used here does not by itself establish valid specimen-level test separation.

## Next experiment

Use the authors' original Dryad CSVs, which include spatial x/y positions and annotated intestine regions. Inspect sample identifiers before defining an across-section holdout. Build graphs without connecting different held-out sections, split by independent tissue region/section (not randomly by cell), and evaluate on held-out section labels once after training and validation. Repeat across held-out sections and assess class-specific error and neighborhood heterogeneity, with no test-set-driven model changes.

## Source and credit

- Brbić M, Cao K, Hickey JW, et al. *Annotation of spatially resolved single-cell data with STELLAR*. *Nature Methods* 19, 1411–1418 (2022). https://doi.org/10.1038/s41592-022-01651-8
- Code: https://github.com/snap-stanford/stellar
- Preprocessed demo: https://zenodo.org/records/6836373
- Original CODEX data: https://doi.org/10.5061/dryad.g4f4qrfrc
