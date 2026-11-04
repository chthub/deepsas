# Benchmarking Tools and Environments

This directory contains the scripts and environment requirements used to benchmark DeepSAS against existing senescence detection methods.

To ensure reproducibility and avoid dependency conflicts, each benchmarking tool was executed independently in its own Conda environment. Each script loads the original AnnData dataset, performs the preprocessing required by the corresponding method, and exports continuous senescence scores as CSV files. The resulting scores are subsequently integrated by cell barcode for AUROC evaluation.

**No intermediate AnnData files are saved or required.**

## 1. Benchmarking tools

The following tools were included in the benchmarking analysis:

- **DeepScence:** https://github.com/anthony-qu/DeepScence
- **SenCID:** https://github.com/JackieHanLab/SenCID
- **SenePy:** https://github.com/jaleesr/senepy
- **decoupler:** https://pypi.org/project/decoupler/
- **ICE:** https://github.com/pxulab/ICE

## 2. Conda environments and requirements

Each tool must be installed in its corresponding Conda environment to avoid package compatibility conflicts.

| Tool | Conda environment | Python version | Requirements file |
|---|---|---|---|
| DeepScence | deepscence | 3.8.20 | requirements-deepscence.txt |
| SenCID | SenCID | 3.6.13 | requirements-sencid.txt |
| SenePy | senepy | 3.9.25 | requirements-senepy.txt |
| decoupler and AUROC evaluation | Decoupler | 3.10.18 | requirements-decoupler.txt |
| ICE Python backend and MAGIC | ice | 3.10.20 | requirements-ice-python.txt |

### Environment installation

Create each environment using its specified Python version and install the corresponding requirements.

For example, to install the DeepScence environment:

```bash
conda create -n deepscence python=3.8.20 -y
conda activate deepscence
pip install -r requirements-deepscence.txt
```

Repeat the same procedure for the remaining tools using their respective environment names, Python versions, and requirements files.

Follow the installation instructions provided in each tool's GitHub repository for any additional dependencies.

## 3. Input datasets

The benchmarking scripts evaluate six datasets:

| Dataset | Description |
|---|---|
| GSE253338 | p16-positive and p16-negative bladder cells |
| GSE233431 | uPAR-positive and uPAR-negative intestinal epithelial cells |
| PCLS | In-house precision-cut lung slices |
| GSE190889 | Human lung epithelial cells |
| GSE264648 | Human brain microglia |
| GSE172410 | Skeletal muscle satellite cells |

Input files must be in `.h5ad` format and contain raw, nonnegative integer expression counts.

The AnnData matrix must contain cells as rows and genes as columns. Cell barcodes (`adata.obs_names`) and gene symbols (`adata.var_names`) must be preserved.

The scripts use the dataset-specific file paths specified in the benchmarking code. Ensure these files are available at the expected locations before execution.

## 4. Running the benchmarking tools

The benchmarking workflow consists of four independent Python scripts:

1. `DeepScence.py`
2. `SenCID.py`
3. `SenePy.py`
4. `AUCell_Benchmarking.py`

The first three scripts can be executed independently in any order. Each produces one CSV file per dataset.

The final script combines these outputs, calculates AUCell scores, integrates existing ICE and R AUCell results, and performs AUROC evaluation.

### 4.1 DeepScence

Activate the DeepScence environment and run:

```bash
conda activate deepscence
python DeepScence.py
```

DeepScence is executed using the human model with binarization enabled.

DCA denoising is performed within the same environment. No separate DCA environment is required.

The script generates six CSV files containing continuous DeepScence scores (`ds`) and corresponding cell barcodes.

### 4.2 SenCID

Activate the SenCID environment and run:

```bash
conda activate SenCID
python SenCID.py
```

SenCID evaluates all six SID models using:

- `sidnums=[1, 2, 3, 4, 5, 6]`
- `denoising=True`
- `binarize=True`
- `threads=5`
- `savetmp=True`

SenCID independently loads the original raw-count AnnData files. This prevents preprocessing performed by DeepScence from affecting SenCID predictions.

The continuous `SID_Score` is extracted using the model specified in the original benchmarking code.

| Dataset | Selected SID model |
|---|---|
| GSE253338 | SID2 |
| GSE233431 | SID1 |
| PCLS | SID2 |
| GSE190889 | SID2 |
| GSE264648 | SID3 |
| GSE172410 | SID3 |

The script generates six CSV files containing the selected continuous `SID_Score` and corresponding cell barcodes.

### 4.3 SenePy

Activate the SenePy environment and run:

```bash
conda activate senepy
python SenePy.py
```

SenePy calculates universal and tissue-specific senescence signature scores.

The following parameters are used:

- Universal signatures are generated with `calculate_thresh=True` and `p_thres=0.05`.
- Signature scoring uses `binarize=False`.
- Human or mouse gene signatures are selected according to the original benchmarking code.

For GSE253338, GSE233431, and GSE172410, mouse gene-symbol datasets are used with mouse SenePy signatures.

For PCLS, GSE190889, and GSE264648, human gene-symbol datasets are used with human SenePy signatures.

Mouse datasets are normalized to 10,000 counts per cell, followed by logarithmic transformation. Human datasets are normalized to 100,000 counts per cell, followed by logarithmic transformation.

The script generates six CSV files containing:

- `barcode`
- `senpey_universal`
- `senpey_specific`

For GSE253338, the tissue-specific score is unavailable in the original benchmarking code and is represented by missing values.

### 4.4 AUCell and AUROC benchmarking

After completing DeepScence, SenCID, and SenePy, activate the Decoupler environment:

```bash
conda activate Decoupler
python AUCell_Benchmarking.py
```

This script performs the final benchmarking analysis.

**Step 1: Load original datasets**

Each dataset is independently loaded from its original `.h5ad` file.

**Step 2: Integrate benchmarking scores**

The DeepScence, SenCID, and SenePy CSV files are loaded and their scores are aligned to the original AnnData observations using cell barcodes.

Scores are matched by barcode rather than row position to prevent incorrect cell assignments.

**Step 3: Calculate AUCell scores**

Marker-signature AUCell scores are calculated using decoupler 2.1.4.

Expression data undergo library-size normalization and logarithmic transformation before AUCell calculation.

AUCell is performed on the full normalized expression matrix, without restricting the analysis to highly variable genes.

The evaluated signatures include CellAge, GO, and SenMayo.

**Step 4: Integrate ICE and R AUCell scores**

Previously calculated ICE and R AUCell results are loaded from their corresponding CSV files and aligned by cell barcode.

**Step 5: Calculate AUROC**

AUROC is calculated using continuous senescence scores and the dataset-specific cell selections and positive labels defined in the original benchmarking code.

The following metrics are evaluated:

| Metric | Method |
|---|---|
| `ds` | DeepScence |
| `senpey_universal` | SenePy universal |
| `senpey_specific` | SenePy tissue-specific |
| `SID_Score` | SenCID |
| `SnC scores` | DeepSAS |
| `CellAge` | AUCell |
| `GO` | AUCell |
| `SenMayo` | AUCell |
| `ICE_cellage` | ICE CellAge |
| `ICE_deepsas` | ICE DeepSAS |
| `aucell_deepsas` | R AUCell DeepSAS |

AUROC results are exported as CSV files for downstream analysis.

## 5. R environment: ICE and AUCell

ICE and R AUCell were executed using **R 4.4.0**.

The recorded package versions are provided in `requirements-ice-r.txt`.

| R package | Version |
|---|---|
| ICE | 1.0.0 |
| Rmagic | 2.0.3.999 |
| Seurat | 5.3.0 |
| SeuratObject | 5.1.0 |
| Matrix | 1.7-3 |
| reticulate | 1.43.0 |
| GSVA | 2.0.7 |
| AUCell | 1.28.0 |

ICE enrichment scores were calculated using normalized expression with iterative refinement enabled.

MAGIC was executed through reticulate using the existing `ice` Python environment.

Seurat normalization used a scale factor of 10,000. ICE received expression matrices with genes as rows and cells as columns.

R AUCell scores were calculated from gene-expression rankings using the full normalized expression matrix.

The resulting ICE and R AUCell CSV files are required for the final benchmarking integration.

## 6. Output files

Each independent benchmarking script saves one CSV file per dataset.

The output directory is:

```text
benchmark_scores/
```

Example outputs:

```text
benchmark_scores/
├── GSE253338_DeepScence.csv
├── GSE253338_SenCID.csv
├── GSE253338_SenePy.csv
├── GSE233431_DeepScence.csv
├── GSE233431_SenCID.csv
├── GSE233431_SenePy.csv
├── ...
```

The first three scripts collectively generate 18 method-specific CSV files.

Each CSV preserves the cell barcode and the corresponding continuous senescence score or scores.

The final benchmarking script combines these scores with DeepSAS, AUCell, and ICE results and exports the dataset-specific AUROC results.

**No intermediate `.h5ad` files are generated.**

## 7. Recommended execution order

Execute the following commands in sequence:

```bash
conda activate deepscence
python DeepScence.py

conda activate SenCID
python SenCID.py

conda activate senepy
python SenePy.py

conda activate Decoupler
python AUCell_Benchmarking.py
```

The final benchmarking step should only be executed after all required method-specific CSV files and external ICE/R AUCell results are available.

## 8. Reproducibility and evaluation

All methods are evaluated using the same dataset-specific cell populations and positive labels defined in the original benchmarking code.

To maintain consistency:

- Original raw-count input files are preserved.
- Each tool performs its preprocessing independently.
- Continuous scores are used for AUROC evaluation.
- Scores are aligned by cell barcode.
- Missing scores are excluded from the corresponding metric's AUROC calculation.
- AUROC is calculated only when both positive and negative classes are represented among cells with valid scores.

This workflow separates tool execution from benchmarking evaluation, reduces dependency conflicts, and allows individual methods to be rerun without repeating the entire analysis.
