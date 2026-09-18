# Phenotype-aware DeepSAS

This supported extension runs the current DeepSAS pipeline while also using a
phenotype annotation such as treatment or disease state. It does not maintain a
separate copy of the model: the training entry point enables phenotype mode in
`deepsas_v1.py`, so CCC construction, type-specific projections, SnG updates,
convergence, checkpoints, and final-result handling are identical to the main
pipeline.

## Input

The input AnnData must contain:

- the cell-type column selected by `--cell_type_col` (default `clusters`);
- the phenotype column selected by `--phenotype_col` (default `Condition`);
- raw or normalized, non-log-transformed counts as expected by DeepSAS.

Phenotype mode selects above-fence SnCs independently by cell type and by
phenotype, then uses the union of both selections. Both use the configured
`--iqr_multiplier`. `--min_snc_per_type` and
`--min_snc_per_phenotype` control the respective minimum retained group sizes.
SnG updating and stopping continue to use `--sng_update_mode`, `--max_iter`, and
`--convergence_tol` from the main pipeline.

## Run

Run from the repository directory:

```bash
uv run python phenotype_analysis/deepsas_v1_phenotype_v2.py \
  --input_data_count path/to/data.h5ad \
  --exp_name phenotype_run \
  --cell_type_col clusters \
  --phenotype_col Condition \
  --retrain

uv run python phenotype_analysis/generate_3tables_pheno.py \
  --input_data_count path/to/data.h5ad \
  --exp_name phenotype_run \
  --cell_type_col clusters \
  --phenotype_col Condition
```

The training wrapper automatically enables `--phenotype_aware`. The same option
can also be passed directly to `deepsas_v1.py`.

Optional phenotype settings:

| Parameter | Default | Role |
| --- | ---: | --- |
| `--phenotype_col` | `Condition` | AnnData `.obs` phenotype column |
| `--min_snc_per_phenotype` | `1` | Minimum above-fence SnCs retained for a phenotype |
| `--use_hvg_deg` | off | Retain the union of per-cell-type HVGs and configured senescence and L–R genes before DeepSAS gene selection |
| `--phenotype_hvg_count` | `1000` | HVGs requested per cell type when `--use_hvg_deg` is enabled |
| `--phenotype_zscore_threshold` | `2.0` | Reporting cutoff for phenotype-specific SnGs |

## Outputs

Training writes the same graph, model, per-iteration, `final`, convergence-log,
and run-summary files as the main pipeline. The convergence log and run summary
also include SnC counts by phenotype.

`generate_3tables_pheno.py` first creates the standard DeepSAS tables and then
writes the following under
`{output_dir}/{exp_name}/Phenotype_SnG_Results/`:

- `Gene_Table1_SnG_scores_per_ct_pheno.csv`;
- `Gene_Table1_SnG_scores_per_ct_pheno_pivot.csv`;
- `Gene_Table1_SnG_scores_per_ct_pheno_pivot_zscore.csv`;
- `Gene_Specific_SnGs_per_ct_pheno_filtered.csv`;
- `Cell_Table2_SnCs_per_ct_pheno.csv`.

Gene scores for each cell-type × phenotype group are calculated from the final
SnCs in that group. Groups without final SnCs are omitted from gene-score tables
but remain present with a zero SnC count in the cell summary.
