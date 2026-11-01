# Benchmarking tools and environments

The benchmarking tools were run using existing Conda environments. No new environments were created. Each tool ran in its own environment, and the resulting scores were aligned by cell barcode for AUROC evaluation. The original benchmarking code was unchanged.

## Environments and requirements

| Tool | Conda environment name | Python version | Requirements file |
|---|---|---|---|
| DeepScence | deepscence | 3.8.20 | requirements-deepscence.txt |
| SenCID | SenCID | 3.6.13 | requirements-sencid.txt |
| SenePy | senepy | 3.9.25 | requirements-senepy.txt |
| decoupler and AUROC evaluation | palantir_env | 3.10.18 | requirements-decoupler.txt |
| ICE Python backend and MAGIC | ice | 3.10.20 | requirements-ice-python.txt |

Each requirements file records the installed package versions in the existing environment actually used. These are environment inventories; fresh installations were not performed or tested.

## How the tools were run

- **DeepScence:** continuous scores were calculated in the deepscence environment using the human model with binarization enabled. DCA was already installed; no separate DCA environment was used.
- **SenCID:** all six SID models were evaluated in the SenCID environment, with denoising and binarization enabled and five threads. Denoising received raw, nonnegative integer counts. The continuous SID score specified by the benchmarking code was used for evaluation.
- **SenePy:** tissue-specific and universal signature scores were calculated in the senepy environment. Universal signatures used an automatically calculated threshold with a significance threshold of 0.05. Scoring used continuous expression rather than binarization.
- **decoupler:** marker-signature AUCell scores were calculated in the palantir_env environment after library-size normalization and logarithmic transformation. The completed comparison used decoupler 2.1.4. The unchanged original source contains the older decoupler API; execution handled this API difference separately.
- **ICE:** enrichment scores were calculated in R using normalized expression and iterative refinement enabled. MAGIC ran through reticulate using the existing ice Python environment. Seurat normalization used a scale factor of 10,000, and the ICE input matrix had genes as rows and cells as columns.
- **R AUCell:** signature AUC scores were calculated from rankings of the full normalized expression matrix.

## R environment

ICE and AUCell were run with **R 4.4.0**. Recorded R package versions are provided in **requirements-ice-r.txt**.

| R package | Version used |
|---|---|
| ICE | 1.0.0 |
| Rmagic | 2.0.3.999 |
| Seurat | 5.3.0 |
| SeuratObject | 5.1.0 |
| Matrix | 1.7-3 |
| reticulate | 1.43.0 |
| GSVA | 2.0.7 |
| AUCell | 1.28.0 |

## Input and evaluation

The initial AnnData expression matrix must contain raw counts, with cells as rows and genes as columns. Cell barcodes and gene symbols must be preserved. Methods received their required count or normalized expression representation; scores were joined by barcode before evaluation.

AUROC was calculated using continuous scores and the cohort selections and positive labels specified in the original benchmarking code.
