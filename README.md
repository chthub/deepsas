# DeepSAS

DeepSAS (**Deep**-learning framework for cell-type-specific **S**nCs **A**nd **S**nGs) is a computational framework designed to identify senescent cells and senescence-associated genes from single-cell RNA sequencing data.

## Overview

Cellular senescence is a state of permanent cell cycle arrest that plays important roles in development, tissue homeostasis, aging, and disease. Identifying senescent cells in heterogeneous tissues is challenging due to the lack of universal markers. DeepSAS leverages graph neural networks and contrastive learning to identify senescent cells and their associated gene signatures from single-cell RNA sequencing data.

The framework integrates several key components:

1. Graph representation of cell-gene interactions
2. Graph Attention Networks (GAT) for capturing complex relationships
3. Contrastive learning with multi-level distance optimization
4. Attention mechanisms for identifying senescence-associated genes

## Features

- Identifies senescent cells in heterogeneous tissues with cell-type specificity
- Discovers senescence-associated genes specific to each cell type
- Leverages both gene expression patterns and cell-cell interactions
- Robust to batch effects and technical variations through built-in normalization
- Works across different cell types and senescence induction methods
- Scales to large datasets with optimized sampling strategies
- Provides comprehensive visualization and analysis tools

## Installation

This project is developed and tested on Linux and macOS environments.

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/chthub/deepsas.git
   cd deepsas/
   git checkout deepsas-v1
   ```
2. **Set Up a uv Environment** (recommended):
   We recommend to use uv for the environment mangement. Check this [link](https://docs.astral.sh/uv/) to install uv.

   ```bash
   uv venv --python 3.8.20
   source .venv/bin/activate
   ```
3. **Install Dependencies**:

   ```bash
   uv pip install numpy seaborn matplotlib pandas tabulate linetimer scikit-learn ipykernel 'scanpy[leiden]' tqdm gseapy
   ```

   For [Pytorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html),  ensure you select the CUDA version that best suits your system. Below is an example from our test environment:

   ```bash
   uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   uv pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   ```

## Quick Start

### Basic Usage

> **The default parameters are used only for running on the sample dataset and paper reproduce.** They have been validated on
> the bundled example dataset and are not universal settings. Before running
> DeepSAS on another dataset, adapt and validate the preprocessing, cell-type
> annotations, gene set, CCC settings, candidate thresholds, projection mode,
> optimization settings, and stopping criteria for that dataset. Do not assume
> that the example defaults or resulting SnC/SnG calls transfer unchanged.

Run the following commands from the repository directory. The first command trains on the bundled example with the default parameters; the second generates tables from its final result. On a cluster, run training on a compute node.

```bash
uv run python -u deepsas_v1.py --exp_name example --device_index 0 --retrain > ./example.log 2>&1
uv run python -u generate_3tables.py --output_dir ./outputs --exp_name example --device_index 0
```

Outputs are written to `./outputs/example/`. `--output_dir` specifies the base directory and is honored by both commands; use the same value and experiment name for training and table generation. The training log and `example_run_summary.json` report how refinement stopped.

A new end-to-end run should use `--retrain`; checkpoints made with a different projection mode are rejected explicitly. Result counts are dataset- and configuration-dependent. The outputs can be inspected with:

```bash
uv run python scripts/check_reviewer_example.py ./outputs/example
```

### Analysis Workflow

DeepSAS follows a 4-step workflow:

1. **Data Loading & Preprocessing**: Filters genes/cells and constructs the cell-gene graph
2. **Initial Embedding Generation**: Creates UMAP embeddings via scanpy
3. **Graph Attention Network Training**: Learns the graph structure with GATConv layers
4. **Contrastive Learning**: Refines embeddings to identify senescent cells and genes

### Results Analysis

After running DeepSAS, generate tables of senescence-associated cells and genes:

```bash
uv run python -u generate_3tables.py --output_dir ./outputs --exp_name example --device_index 0
```

By default, this reads `example_sencellgene-final.data` and the saved preprocessed data, preserving the training cell and gene indices. It does not assume a particular final iteration. To inspect an existing iteration explicitly, add `--epoch N`, where `N` is its zero-based iteration number. For a custom cell-type column, pass the same `--cell_type_col` used during training.

If no SnCs are detected, the cell score table still contains every processed cell, and DEG and summary gene tables retain their column definitions. Cell types without enough SnCs or control cells for differential expression are skipped with an explanation in the table-generation log.

This generates several output tables:

1. **Cell-level analysis**:

   - Cell senescence scores and binary classification (senescent/non-senescent)
   - Distribution of senescent cells across cell types
2. **Gene-level analysis**:

   - Differentially expressed genes between senescent and non-senescent cells
   - Cell-type specific senescence-associated genes with statistical measures
   - Aggregated gene information across multiple cell types
3. **Statistical measures**:

   - p-values and adjusted p-values from Wilcoxon rank-sum tests
   - Log fold-changes showing expression differences
   - Senescence-associated gene (SnG) scores derived from attention weights

For visualization and downstream analysis, follow the tutorial in [`tutorial.ipynb`](./tutorial.ipynb), which demonstrates:

- UMAP visualization of senescent cells
- Gene set enrichment analysis of identified senescence markers
- Cell-type specific senescence marker analysis and interpretation

### Large Dataset Analysis

For datasets with many cells (>50,000), use the sampling approach detailed in [Sampling_Tutorial.md](./Sampling_Tutorial.md), which provides:

- Subsampling strategies
- Batch processing scripts
- Result integration methods

### Phenotype related Analysis

To incorporate the phenotype information into the analysis, please following the tutorial in [`phenotype_analysis`](./phenotype_analysis/README.md).

## Input Data Format

DeepSAS works with h5ad format (AnnData objects from Scanpy). The input data should include:

- Gene expression matrix (cells × genes) in sparse or dense format
- Cell type annotations in `adata.obs['clusters']` or another specified column
- (Optional) Additional metadata like batch information for batch effect correction

The input expression data should be normalized counts (not log-transformed). DeepSAS handles normalization and scaling using scanpy functions. Batch correction is optional and enabled with `--batch_remove`.

## Parameters

Run `uv run python deepsas_v1.py --help` for the full command-line interface. The following settings are used by `deepsas_v1.py`.

### Input/Output Parameters

- `--input_data_count`: Input h5ad file; default `example_data/example_data.h5ad`.
- `--output_dir`: Base output directory; default `./outputs`. Run files are stored under `{output_dir}/{exp_name}/`.
- `--exp_name`: Required experiment name.
- `--device_index`: CUDA device index; default `0`. CPU is used when CUDA is unavailable.
- `--retrain`: Train the graph model and create its graph files; otherwise load existing files for the experiment.
- `--cell_type_col`: Cell-type annotation column; default `clusters`.
- `--batch_col`: Batch annotation column for optional ComBat correction; default `Sample`.
- `--batch_remove`: Enable ComBat before the cell UMAP embedding; default **off**.
- `--epoch`: Table generation only; default is the final result, or select an existing zero-based iteration number.

### Model Configuration

| Parameter                                                            | Default           | Role                                                                                                         |
| -------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `--seed`                                                           | `40`            | Random seed, including PCA and UMAP                                                                          |
| `--n_genes`                                                        | `full`          | All retained genes, or a requested number of highly variable genes plus available marker genes               |
| `--gene_set`                                                       | `full`          | Initial senescence marker lists; alternatives include`senmayo`, `fridman`, and `cellage`               |
| `--emb_size`                                                       | `12`            | Cell and gene embedding dimension                                                                            |
| `--type_specific_projections` / `--no_type_specific_projections` | enabled           | Use separate gene/cell input projections by default, or explicitly select a shared GAT projection            |
| `--ccc`                                                            | `type1`         | Binary CCC edges;`type3` omits CCC edges                                                                   |
| `--ccc_threshold`                                                  | `0.8`           | Signaling-score threshold φ for retaining CCC edges                                                         |
| `--lr_panel`                                                       | Built-in panel    | Optional ligand–receptor CSV with`ligand` and `receptor` columns                                        |

The graph uses binary connectivity: cell–gene expression presence and thresholded CCC connections are represented by graph topology. The GAT therefore accepts node features and edge indices only; the implementation does not expose an edge-feature dimension parameter. CCC probabilities are computed independently for each available ligand–receptor pair and then averaged, matching Methods 1.2.

### Training and Candidate Selection

| Parameter               | Default   | Role                                                                                       |
| ----------------------- | --------- | ------------------------------------------------------------------------------------------ |
| `--gat_epoch`         | `30`    | Graph autoencoder training epochs                                                          |
| `--gat_learning_rate` | `0.001` | Graph autoencoder Adam learning rate                                                       |
| `--gat_hidden_size`   | `32`    | GAT hidden width                                                                           |
| `--gat_dropout`       | `0.6`   | Attention dropout during GAT training; disabled for candidate scoring                      |
| `--cell_optim_epoch`  | `50`    | Cell embedding optimization epochs per outer iteration                                     |
| `--cell_hidden_size`  | `128`   | Hidden width of the cell embedding network                                                 |
| `--distance_levels`   | `0 0 4` | Initial learnable targets for the three active distance-loss terms                         |
| `--learning_rate`     | `0.01`  | Initial Adam learning rate for cell embedding optimization                                 |
| `--weight_decay`      | `0.001` | Cell optimizer weight decay                                                                |
| `--lr_decay`          | `0.85`  | Cell learning-rate multiplier after each outer iteration                                   |
| `--iqr_multiplier`    | `1.5`   | Upper-fence multiplier for SnC scores and for SnG scores in`iqr` mode                    |
| `--sng_update_mode`   | `iqr`   | SnG update rule: the default threshold-driven IQR policy or the optional`fixed10` policy |
| `--min_snc_per_type`  | `1`     | At least this many above-fence SnCs are required to retain a cell type's candidates        |
| `--max_iter`          | `10`    | Maximum number of completed outer refinement iterations                                    |
| `--convergence_tol`   | `0.9`   | Required Jaccard overlap for both candidate sets                                           |

These example-dataset defaults are exposed for reproducibility and must be adapted and validated for other datasets. Architecture and preprocessing settings include:

- By default, separate learned projections map gene and cell embeddings into the shared GAT feature space. Pass `--no_type_specific_projections` to use the shared projection path instead; switching modes requires retraining the GAT checkpoint.
- Cell embedding network: CELU activation and layer normalization. Its hidden width and the initial values of the three learned distance targets are configured by `--cell_hidden_size` and `--distance_levels`.
- Input filtering: at least `200` detected genes per cell and expression in at least `10` cells per gene.
- Initial embeddings: normalization to `10,000` counts, log transformation, scaling clipped at `10`, PCA, a neighbor graph with `10` neighbors and `40` PCs, and UMAP.
- Downstream DEG tables: a comparison requires at least `6` SnCs and `2` controls; the combined DEG/SnG table retains hits with log fold change at least `0.25`. These reporting rules do not alter the predicted SnC or SnG sets.

Obsolete options that did not affect the main pipeline have been removed. SnC candidates are determined by their score thresholds; SnG updates follow `--sng_update_mode`; initial embeddings use UMAP.

### SnG Updates and Stopping

The upper fence `Q3 + iqr_multiplier × IQR` is a score threshold. For SnGs, its quartiles are calculated over the scores of the **current unique candidate genes**, including zeros. The number of current candidates strictly above this threshold defines the requested replacement count `D2*`. Up to this many above-fence genes outside the candidate set may replace lower-scoring current candidates, pairing the highest-scoring entrants with the lowest-scoring current genes. Each replacement must strictly improve the score. The actual count satisfies `0 <= D2 <= D2*` and may be smaller when suitable new genes are unavailable. If no current candidate exceeds the upper fence, no genes are replaced. Candidate gene IDs remain unique. The working SnG candidate set is global; cell-type-specific gene scores and differential-expression tables are computed downstream.

In `iqr` mode, the convergence log records `D2*` as `sng_outlier_count` and
the accepted count `D2` as `sng_swaps`. In `fixed10` mode,
`sng_outlier_count` records the requested count of 10. The
`sng_update_mode` column identifies the active rule.

To use the alternative fixed-count SnG update policy, pass
`--sng_update_mode fixed10`. This mode requests ten replacements per iteration:
the ten highest-scoring genes outside the current candidate set replace the ten
lowest-scoring current candidates (or all available genes when either side has
fewer than ten). Candidate IDs remain unique. The default `iqr` mode uses the
threshold-driven rule described above.

The pretrained GAT embeddings are retained as fixed inputs to cell embedding optimization across outer iterations. When recomputing attention, the graph retains its original gene input features, updates only the cell rows with the optimized cell embeddings, and uses attention coefficients from the final GAT layer for candidate scoring.

After updating embeddings, attention, and genes, SnC labels are recalculated to produce a complete iteration state. Outer convergence uses `J(A, B) = |A ∩ B| / |A ∪ B|` between adjacent completed states: both SnC and SnG Jaccard values must reach `--convergence_tol`, and both previous and current sets must be nonempty. The default tolerance is `0.9`; `--convergence_tol 1` requests exact membership equality. Loss is optimized within each iteration; it is not the outer stopping criterion. In `fixed10` mode, compulsory gene turnover can keep the SnG Jaccard below `1` (ten disjoint replacements in 274 candidates give approximately `0.9296`), while `iqr` mode can make zero replacements and attain exact stability. Convergence is not guaranteed within the iteration limit.

Ideally, the model is considered to have converged if the candidate set remains unchanged between consecutive iterations. However, in practice, we have observed fluctuations where individual genes or cells repeatedly enter and exit the set. In such cases, the Jaccard similarity may fail to reach 0.99 even though the model has attained a stable state. 

## Output Files

DeepSAS generates the following files under `{output_dir}/{exp_name}/`:

- `{exp_name}_new_data.h5ad`: Processed AnnData object with filtered genes/cells
- `{exp_name}_graphnx.data`: NetworkX graph representation of cell-gene interactions
- `{exp_name}_graphpyg.data`: PyTorch Geometric graph representation for GAT model
- `{exp_name}_GAT.pt`: Trained Graph Attention Network model
- `{exp_name}_cellmodel.pt`: Cell embedding model, when cell optimization was performed
- `{exp_name}_sencellgene-epoch{epoch}.data`: Candidate state at each completed outer iteration
- `{exp_name}_sencellgene-final.data`: Final available candidate state, used by table generation
- `{exp_name}_convergence.csv`: Per-iteration candidate counts, Jaccard overlaps, loss, gene threshold and replacements, cell-type counts, and stop status
- `{exp_name}_run_summary.json`: Run configuration, final iteration, candidate counts, final-result filename, and stop status

The summary distinguishes `converged` (both nonempty candidate sets meet the overlap criterion), `max_iter` (the cap was reached without convergence), and `empty_candidates` (SnC or SnG candidates are empty). A `running` summary has not recorded a completed run and cannot be used for default final-table generation. Empty candidates are never reported as convergence.

Each iteration and final `.data` file stores the SnC dictionary, unique candidate SnG indices, attention weights, and graph edge indices. Candidate genes are stored by index, not ranked by score. The associated `{exp_name}_new_data.h5ad` defines the cell and gene ordering.

After running `generate_3tables.py`, you'll also get a folder `Senescent_Tables` in the output path. In this folder you have:

1. **Cell_Table1_SnC_scores.csv**: Information about each cell and its senescence score

   - Contains cell IDs, names, types, binary senescent indicator, and senescence scores
2. **Gene_Table2_DEG_ct_SnG_score.csv**: Differentially expressed genes between senescent and non-senescent cells

   - Includes gene names, cell types, p-values, log fold-changes, adjusted p-values, and senescence scores
3. **Gene_Table3_gene_ct_count.csv**: Gene summaries across cell types, including initial-marker membership

   - **Gene_newTable3_gene_ct_count.csv** contains the subset reported in one cell type
4. **Additional tables**:

   - **Cell_Table2_SnCs_per_ct.csv**: Counts of total cells and senescent cells per cell type
   - **Gene_Table1_SnG_scores_per_ct.csv**: Candidate gene attention scores for each cell type with SnCs
   - **table2ByCelltype.csv**: Table 2 grouped by cell type
   - **table2ByGene.csv**: Table 2 grouped by gene

For detailed explanations of each table and column, see [Senescent_Tables_Explanation.md](./Senescent_Tables_Explanation.md)

## Citation

If you use DeepSAS in your research, please cite:

```
@article{ma2025intrinsic,
  title={An Intrinsic-hoc Framework for Heterogeneous Cellular Senescence Elucidation Using Deep Graph Representation Learning and Experimental Validation},
  author={Ma, Anjun and Cheng, Hao and Vanegas, Natalia Del Pilar and Ghobashi, Ahmed and Chen, Hu and Rodriguez, Jhonny and Rosas, Lorena and Wang, Cankun and Shao, Jianming and Jiang, Yi and others},
  journal={bioRxiv},
  pages={2025--07},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
