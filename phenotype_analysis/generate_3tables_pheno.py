"""Generate current DeepSAS tables plus phenotype-stratified summaries."""

import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import generate_3tables as base_tables
import utils


logger = logging.getLogger(__name__)
PHENOTYPE_GENE_COLUMNS = ['gene', 'cell_type', 'phenotype', 'score']
PHENOTYPE_SPECIFIC_COLUMNS = [
    'gene', 'cell_type', 'phenotype', 'z_score']


def _validate_annotations(new_data, cell_type_col, phenotype_col):
    missing = [
        column for column in (cell_type_col, phenotype_col)
        if column not in new_data.obs.columns]
    if missing:
        raise KeyError(
            f'Missing AnnData .obs columns: {missing}. '
            f'Available columns: {list(new_data.obs.columns)}')
    if new_data.obs[[cell_type_col, phenotype_col]].isna().any().any():
        raise ValueError('Cell-type and phenotype annotations must not be missing.')


def _column_zscores(frame):
    """Return finite per-column population z-scores; constants map to zero."""
    result = frame.astype(float).copy()
    for column in result.columns:
        values = result[column]
        std = values.std(ddof=0)
        result[column] = 0.0 if std == 0 or not np.isfinite(std) else (
            values - values.mean()) / std
    return result


def generate_phenotype_tables(
        new_data, sencell_dict, sen_gene_ls, attention_scores,
        edge_index_selfloop, output_path, cell_type_col='clusters',
        phenotype_col='Condition', zscore_threshold=2.0):
    """Write SnC/SnG summaries stratified by cell type and phenotype."""
    _validate_annotations(new_data, cell_type_col, phenotype_col)
    os.makedirs(output_path, exist_ok=True)

    n_genes = new_data.n_vars
    gene_cell = new_data.X.T
    snc_nodes = {int(index) for index in sencell_dict}
    candidate_names = list(new_data.var_names[sen_gene_ls])
    records = []

    group_columns = [cell_type_col, phenotype_col]
    grouped_positions = new_data.obs.groupby(
        group_columns, sort=False, observed=True).indices
    for (cell_type, phenotype), rows in grouped_positions.items():
        group_sncs = [
            n_genes + int(row) for row in rows
            if n_genes + int(row) in snc_nodes]
        if not group_sncs:
            continue
        scores = base_tables.AttentionEachGene(
            gene_cell, group_sncs, edge_index_selfloop,
            attention_scores)[sen_gene_ls].tolist()
        records.extend({
            'gene': gene,
            'cell_type': cell_type,
            'phenotype': phenotype,
            'score': score,
        } for gene, score in zip(candidate_names, scores))

    long_table = pd.DataFrame(records, columns=PHENOTYPE_GENE_COLUMNS)
    long_table.to_csv(os.path.join(
        output_path, 'Gene_Table1_SnG_scores_per_ct_pheno.csv'), index=False)

    if long_table.empty:
        raw_pivot = pd.DataFrame(index=pd.Index(candidate_names, name='gene'))
    else:
        raw_pivot = long_table.pivot(
            index='gene', columns=['cell_type', 'phenotype'], values='score')
    raw_pivot.to_csv(os.path.join(
        output_path, 'Gene_Table1_SnG_scores_per_ct_pheno_pivot.csv'))

    zscore_pivot = _column_zscores(raw_pivot)
    zscore_pivot.to_csv(os.path.join(
        output_path, 'Gene_Table1_SnG_scores_per_ct_pheno_pivot_zscore.csv'))

    specific_records = []
    for column in zscore_pivot.columns:
        selected = zscore_pivot[column][
            zscore_pivot[column] >= zscore_threshold]
        cell_type, phenotype = column
        specific_records.extend({
            'gene': gene,
            'cell_type': cell_type,
            'phenotype': phenotype,
            'z_score': zscore,
        } for gene, zscore in selected.items())
    pd.DataFrame(
        specific_records, columns=PHENOTYPE_SPECIFIC_COLUMNS).to_csv(
            os.path.join(
                output_path,
                'Gene_Specific_SnGs_per_ct_pheno_filtered.csv'),
            index=False)

    annotations = new_data.obs[group_columns].copy()
    annotations['ifSnCs'] = [
        str(int(n_genes + row in snc_nodes))
        for row in range(new_data.n_obs)]
    counts = annotations.groupby(
        group_columns, sort=False, observed=True).agg(
            number_of_cells=('ifSnCs', 'size'),
            number_of_SnCs=('ifSnCs', lambda values: (values == '1').sum()),
        ).reset_index().rename(columns={
            cell_type_col: 'cell_type', phenotype_col: 'phenotype'})
    counts.to_csv(os.path.join(
        output_path, 'Cell_Table2_SnCs_per_ct_pheno.csv'), index=False)
    logger.info('Saved phenotype tables to %s', output_path)
    return long_table, raw_pivot, zscore_pivot, counts


def main():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
    args = utils.parse_args()
    loaded = base_tables.load_results(args)
    base_tables.generate_tables(
        *loaded,
        cell_type_col=args.cell_type_col,
        deg_min_snc=args.deg_min_snc,
        deg_min_control=args.deg_min_control,
        deg_min_logfc=args.deg_min_logfc,
        normalization_target_sum=args.normalization_target_sum)

    (new_data, _initial_marker, sencell_dict, sen_gene_ls,
     attention_scores, edge_index_selfloop, standard_output) = loaded
    phenotype_output = os.path.join(
        os.path.dirname(standard_output), 'Phenotype_SnG_Results')
    generate_phenotype_tables(
        new_data, sencell_dict, sen_gene_ls, attention_scores,
        edge_index_selfloop, phenotype_output,
        cell_type_col=args.cell_type_col,
        phenotype_col=args.phenotype_col,
        zscore_threshold=args.phenotype_zscore_threshold)


if __name__ == '__main__':
    main()
