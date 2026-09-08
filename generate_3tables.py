import json
import logging
import os

import numpy as np
import pandas as pd
import scanpy as sp
import torch
import torch_scatter

import utils


logger = logging.getLogger(__name__)
DEG_COLUMNS = ['gene', 'p_val', 'logFC', 'p_val_adj']
GENE_TABLE_COLUMNS = DEG_COLUMNS + ['cell_type', 'SnG_score']


def load_results(args):
    """Load one training result and the exact cell/gene ordering it used."""
    run_dir = os.path.join(args.output_dir, args.exp_name)
    epoch = getattr(args, 'epoch', None)
    suffix = 'final' if epoch is None else f'epoch{epoch}'
    file_path = os.path.join(run_dir, f'{args.exp_name}_sencellgene-{suffix}.data')
    data_path = os.path.join(run_dir, f'{args.exp_name}_new_data.h5ad')
    for path in (file_path, data_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Training output not found: {path}. Run deepsas_v1.py first; '
                'use --epoch N only when selecting an existing iteration output.')

    summary_path = os.path.join(run_dir, f'{args.exp_name}_run_summary.json')
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path) as handle:
            summary = json.load(handle)
    if epoch is None and summary.get('status') not in {'converged', 'max_iter', 'empty_candidates'}:
        raise ValueError(
            'Final training output is incomplete or has no valid run summary. '
            'Finish the current training run before generating final tables; '
            'use --epoch N to inspect an existing iteration explicitly.')

    new_data = sp.read_h5ad(data_path)
    if args.cell_type_col not in new_data.obs.columns:
        raise KeyError(f"Cell-type column '{args.cell_type_col}' not found in saved data.")
    # These local training artifacts contain tensors and Python containers.
    loaded_data = torch.load(file_path, map_location='cpu', weights_only=False)
    sencell_dict, sen_gene_ls, attention_scores, edge_index_selfloop = loaded_data
    sen_gene_ls = list(dict.fromkeys(int(i) for i in sen_gene_ls))
    n_cells, n_genes = new_data.shape
    if any(i < 0 or i >= n_genes for i in sen_gene_ls):
        raise ValueError('Saved SnG indices do not match the processed data.')
    if any(int(i) < n_genes or int(i) >= n_genes + n_cells for i in sencell_dict):
        raise ValueError('Saved SnC indices do not match the processed data.')
    edge_index_selfloop = edge_index_selfloop.detach().cpu().long()
    attention_scores = attention_scores.detach().cpu().reshape(-1)
    if edge_index_selfloop.ndim != 2 or edge_index_selfloop.shape[0] != 2:
        raise ValueError('Saved edge indices must have shape (2, number_of_edges).')
    if attention_scores.numel() != edge_index_selfloop.shape[1]:
        raise ValueError('Expected one attention score per edge (single-head GAT).')
    if edge_index_selfloop.numel() and (
            edge_index_selfloop.min() < 0 or
            edge_index_selfloop.max() >= n_genes + n_cells):
        raise ValueError('Saved graph node indices do not match the processed data.')

    if 'initial_sng_indices' in summary:
        initial_indices = summary['initial_sng_indices']
    else:
        # Compatibility with older iteration outputs; do not preprocess again.
        markers = {g for group in utils.load_markers(args) for g in group}
        initial_indices = [i for i, name in enumerate(new_data.var_names) if name in markers]
    initial_marker = pd.DataFrame({
        'gene_idx': initial_indices,
        'gene_name': list(new_data.var_names[initial_indices]),
    })
    logger.info('Loaded %s: %d SnCs, %d SnGs', file_path, len(sencell_dict), len(sen_gene_ls))
    if summary and epoch is None:
        logger.info('Training stop status: %s; final iteration: %s',
                    summary.get('status'), summary.get('final_epoch'))
    return (new_data, initial_marker, sencell_dict, sen_gene_ls,
            attention_scores, edge_index_selfloop, os.path.join(run_dir, 'Senescent_Tables'))


def AttentionEachCell(gene_cell, sen_gene_ls, edge_index_selfloop, attention_scores):
    """Mean attention on SnG-to-cell edges for every cell."""
    n_genes, n_cells = gene_cell.shape
    gene_mask = torch.zeros(n_genes + n_cells, dtype=torch.bool)
    gene_mask[sen_gene_ls] = True
    selected = ((edge_index_selfloop[1] >= n_genes) &
                gene_mask[edge_index_selfloop[0]])
    targets = edge_index_selfloop[1, selected] - n_genes
    scores = attention_scores.reshape(-1)[selected]
    sums = torch_scatter.scatter(scores, targets, dim=0, dim_size=n_cells, reduce='sum')
    counts = torch_scatter.scatter(torch.ones_like(scores), targets,
                                   dim=0, dim_size=n_cells, reduce='sum')
    return (sums / counts.clamp_min(1)).tolist()


def AttentionEachGene(gene_cell, cell_indices, edge_index_selfloop, attention_scores):
    """Mean attention on selected SnC-to-gene edges for every gene."""
    n_genes, n_cells = gene_cell.shape
    cell_mask = torch.zeros(n_genes + n_cells, dtype=torch.bool)
    cell_mask[cell_indices] = True
    selected = ((edge_index_selfloop[1] < n_genes) &
                cell_mask[edge_index_selfloop[0]])
    targets = edge_index_selfloop[1, selected]
    scores = attention_scores.reshape(-1)[selected]
    sums = torch_scatter.scatter(scores, targets, dim=0, dim_size=n_genes, reduce='sum')
    counts = torch_scatter.scatter(torch.ones_like(scores), targets,
                                   dim=0, dim_size=n_genes, reduce='sum')
    return sums / counts.clamp_min(1)


def DEGTable(new_data, output_path, cell_type_col):
    """Run the existing Wilcoxon comparison only when both groups are usable."""
    adata_deg = new_data.copy()
    sp.pp.normalize_total(adata_deg, target_sum=1e4)
    sp.pp.log1p(adata_deg)
    results = {}
    for cell_type in adata_deg.obs[cell_type_col].unique():
        adata_sub = adata_deg[adata_deg.obs[cell_type_col] == cell_type].copy()
        n_snc = int((adata_sub.obs['ifSnCs'] == '1').sum())
        n_control = int((adata_sub.obs['ifSnCs'] == '0').sum())
        degs = pd.DataFrame(columns=DEG_COLUMNS)
        # Preserve the original >5 SnC rule; Scanpy also needs >=2 controls.
        if n_snc <= 5 or n_control < 2:
            logger.info('Skipping DEG for %s: %d SnCs, %d controls '
                        '(requires >5 SnCs and >=2 controls).', cell_type, n_snc, n_control)
        else:
            sp.tl.rank_genes_groups(adata_sub, groupby='ifSnCs', groups=['1'],
                                    reference='0', method='wilcoxon')
            ranked = adata_sub.uns['rank_genes_groups']
            degs = pd.DataFrame({
                'gene': ranked['names']['1'],
                'p_val': ranked['pvals']['1'],
                'logFC': ranked['logfoldchanges']['1'],
                'p_val_adj': ranked['pvals_adj']['1'],
            }).sort_values(by='logFC')
        results[cell_type] = degs
        save_ct_name = str(cell_type).replace(' ', '_').replace('/', '_')
        # Write an empty table too, so a previous run's results are not reused.
        degs.to_csv(os.path.join(output_path, f'{save_ct_name}_DEG_results.csv'), index=False)
    return results


def GeneTable2(ct2gene_score_df, deg_results):
    frames = []
    for cell_type, degs in deg_results.items():
        if degs.empty or cell_type not in ct2gene_score_df.columns:
            continue
        selected = degs[
            degs['gene'].isin(ct2gene_score_df.index) & (degs['logFC'] >= 0.25)
        ].copy()
        selected['cell_type'] = cell_type
        selected['SnG_score'] = selected['gene'].map(ct2gene_score_df[cell_type])
        frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=GENE_TABLE_COLUMNS)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df[all_df['SnG_score'] != 0.]


def generate_tables(new_data, initial_marker, sencell_dict, sen_gene_ls,
                    attention_scores, edge_index_selfloop, output_path, cell_type_col):
    os.makedirs(output_path, exist_ok=True)
    sencell_indices = [int(i) for i in sencell_dict]
    rows = np.asarray(sencell_indices, dtype=np.int64) - new_data.n_vars
    gene_cell = new_data.X.T
    cell_types = new_data.obs[cell_type_col]
    if_snc = np.zeros(new_data.n_obs, dtype=np.int64)
    if_snc[rows] = 1
    cell_scores = pd.DataFrame({
        'cell_id': np.arange(new_data.n_obs),
        'cell_name': new_data.obs_names,
        'cell_type': cell_types.to_numpy(),
        'ifSnCs': if_snc,
        'SnC scores': AttentionEachCell(gene_cell, sen_gene_ls, edge_index_selfloop, attention_scores),
    })
    cell_scores.to_csv(os.path.join(output_path, 'Cell_Table1_SnC_scores.csv'), index=False)
    # Assign by position without merging on possibly unrelated DataFrame indices.
    new_data.obs['ifSnCs'] = if_snc.astype(str)

    counts = pd.concat([
        cell_types.value_counts().rename('number_of_cells'),
        cell_types.iloc[rows].value_counts().rename('number_of_SnCs'),
    ], axis=1).fillna(0).astype(int)
    counts.index.name = 'cell_type'
    counts.to_csv(os.path.join(output_path, 'Cell_Table2_SnCs_per_ct.csv'))

    ct_sencell_indices = {}
    for row in rows:
        cell_type = cell_types.iloc[row]
        ct_sencell_indices.setdefault(cell_type, []).append(int(row + new_data.n_vars))
    ct2gene_score = {}
    for cell_type, indices in ct_sencell_indices.items():
        scores = AttentionEachGene(gene_cell, indices, edge_index_selfloop, attention_scores)
        ct2gene_score[cell_type] = scores[sen_gene_ls].tolist()
    ct2gene_score_df = pd.DataFrame(ct2gene_score, index=list(new_data.var_names[sen_gene_ls]))
    ct2gene_score_df.to_csv(os.path.join(output_path, 'Gene_Table1_SnG_scores_per_ct.csv'))

    deg_results = DEGTable(new_data, output_path, cell_type_col)
    total_df = GeneTable2(ct2gene_score_df, deg_results)
    total_df.to_csv(os.path.join(output_path, 'Gene_Table2_DEG_ct_SnG_score.csv'), index=False)
    grouped_df = total_df.groupby('gene').agg({
        'cell_type': lambda x: ', '.join(str(value) for value in x.unique()),
        'p_val': 'mean',
        'logFC': 'mean',
        'p_val_adj': 'mean',
        'SnG_score': 'mean',
    }).reset_index()
    grouped_df['hallmarker'] = grouped_df['gene'].isin(initial_marker['gene_name'])
    grouped_df.to_csv(os.path.join(output_path, 'Gene_Table3_gene_ct_count.csv'), index=False)
    single_type = grouped_df['cell_type'].map(lambda x: len(x.split(',')) == 1).astype(bool)
    grouped_df.loc[single_type].to_csv(
        os.path.join(output_path, 'Gene_newTable3_gene_ct_count.csv'), index=False)
    by_celltype = total_df.groupby('cell_type')['gene'].agg(
        gene_count='count', gene_list=lambda x: list(x)).reset_index()
    by_celltype.to_csv(os.path.join(output_path, 'table2ByCelltype.csv'), index=False)
    by_gene = total_df.groupby('gene').agg(
        cell_type_count=('cell_type', 'nunique'),
        cell_types=('cell_type', lambda x: list(x.unique()))).reset_index()
    by_gene.to_csv(os.path.join(output_path, 'table2ByGene.csv'), index=False)
    logger.info('Saved tables to %s', output_path)


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
    args = utils.parse_args()
    generate_tables(*load_results(args), cell_type_col=args.cell_type_col)


if __name__ == '__main__':
    main()
