"""Small regression fixtures for final-result and empty-table handling."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import scanpy as sp
import torch

import generate_3tables as tables


def example_data(cell_types):
    return sp.AnnData(
        np.asarray([[9., 1.]] * 6 + [[1., 9.]] * 2, dtype=np.float32),
        obs=pd.DataFrame({'custom_type': cell_types},
                         index=['barcode_%d' % i for i in range(8)]),
        var=pd.DataFrame(index=pd.Index(['gene_b', 'gene_a'], name='gene_symbol')),
    )


class TableRegressionTests(unittest.TestCase):
    def test_zero_single_and_multiple_selected_edges(self):
        gene_cell = np.zeros((2, 3))
        no_edges = torch.empty((2, 0), dtype=torch.long)
        empty_scores = torch.empty((0, 1))
        self.assertEqual(tables.AttentionEachCell(gene_cell, [0], no_edges, empty_scores),
                         [0., 0., 0.])
        self.assertEqual(tables.AttentionEachGene(gene_cell, [], no_edges, empty_scores).tolist(),
                         [0., 0.])
        edges = torch.tensor([[0, 1, 2], [2, 2, 0]])
        scores = torch.tensor([[0.25], [0.75], [0.5]])
        self.assertEqual(tables.AttentionEachCell(gene_cell, [0], edges, scores), [0.25, 0., 0.])
        self.assertEqual(tables.AttentionEachCell(gene_cell, [0, 1], edges, scores), [0.5, 0., 0.])
        self.assertEqual(tables.AttentionEachGene(gene_cell, [2], edges, scores).tolist(), [0.5, 0.])

    def test_final_result_and_explicit_epoch_use_saved_order_and_output_base(self):
        data = example_data(['Type A'] * 8)
        with tempfile.TemporaryDirectory() as base:
            run_dir = os.path.join(base, 'early')
            os.makedirs(run_dir)
            data.write_h5ad(os.path.join(run_dir, 'early_new_data.h5ad'))
            edges = torch.tensor([[0, 2], [2, 0]])
            scores = torch.tensor([[0.2], [0.4]])
            torch.save([{2: None}, [0], scores, edges],
                       os.path.join(run_dir, 'early_sencellgene-final.data'))
            torch.save([{3: None}, [1], scores, edges],
                       os.path.join(run_dir, 'early_sencellgene-epoch0.data'))
            with open(os.path.join(run_dir, 'early_run_summary.json'), 'w') as handle:
                json.dump({'initial_sng_indices': [1], 'status': 'max_iter',
                           'final_epoch': 0}, handle)
            args = SimpleNamespace(output_dir=base, exp_name='early', epoch=None,
                                   cell_type_col='custom_type')
            with patch.object(tables.utils, 'load_markers', side_effect=AssertionError('reloaded markers')):
                loaded = tables.load_results(args)
                self.assertEqual(list(loaded[0].var_names), ['gene_b', 'gene_a'])
                self.assertEqual(loaded[1]['gene_name'].tolist(), ['gene_a'])
                self.assertEqual(list(loaded[2]), [2])
                self.assertEqual(loaded[3], [0])
                self.assertEqual(loaded[-1], os.path.join(run_dir, 'Senescent_Tables'))
                args.epoch = 0
                loaded_epoch = tables.load_results(args)
                self.assertEqual(list(loaded_epoch[2]), [3])
                self.assertEqual(loaded_epoch[3], [1])
                for status in ('running', None, 'failed'):
                    with open(os.path.join(run_dir, 'early_run_summary.json'), 'w') as handle:
                        json.dump({'initial_sng_indices': [1], 'status': status}, handle)
                    args.epoch = None
                    with self.assertRaisesRegex(ValueError, 'incomplete'):
                        tables.load_results(args)
                    args.epoch = 0
                    self.assertEqual(list(tables.load_results(args)[2]), [3])

    def test_cell_counts_align_by_label_and_preserve_barcode_order(self):
        data = example_data(['B'] * 6 + ['A / other'] * 2)
        initial = pd.DataFrame({'gene_name': ['gene_b']})
        sncs = {2: None, 4: None, 9: None}
        edges = torch.tensor([[0, 2], [2, 0]])
        scores = torch.tensor([[0.2], [0.4]])
        with tempfile.TemporaryDirectory() as output:
            tables.generate_tables(data, initial, sncs, [0], scores, edges, output, 'custom_type')
            cells = pd.read_csv(os.path.join(output, 'Cell_Table1_SnC_scores.csv'))
            self.assertEqual(cells['cell_name'].tolist(), list(data.obs_names))
            self.assertEqual(cells['ifSnCs'].tolist(), [1, 0, 1, 0, 0, 0, 0, 1])
            counts = pd.read_csv(os.path.join(output, 'Cell_Table2_SnCs_per_ct.csv'), index_col='cell_type')
            self.assertEqual(counts.loc['B'].tolist(), [6, 2])
            self.assertEqual(counts.loc['A / other'].tolist(), [2, 1])
            self.assertTrue(os.path.isfile(os.path.join(output, 'A___other_DEG_results.csv')))
            genes = pd.read_csv(os.path.join(output, 'Gene_Table1_SnG_scores_per_ct.csv'), index_col=0)
            self.assertEqual(genes.index.tolist(), ['gene_b'])

    def test_empty_or_unusable_groups_write_empty_tables_and_ignore_stale_degs(self):
        initial = pd.DataFrame({'gene_name': ['gene_b']})
        for n_snc in (0, 7, 8):
            with self.subTest(n_snc=n_snc), tempfile.TemporaryDirectory() as output:
                data = example_data(['Type A'] * 8)
                stale_path = os.path.join(output, 'Type_A_DEG_results.csv')
                pd.DataFrame({'gene': ['gene_b'], 'p_val': [0.01], 'logFC': [2.],
                              'p_val_adj': [0.01]}).to_csv(stale_path, index=False)
                with patch.object(tables.sp.tl, 'rank_genes_groups',
                                  side_effect=AssertionError('invalid DEG comparison')):
                    tables.generate_tables(
                        data, initial, {i + 2: None for i in range(n_snc)}, [0],
                        torch.empty((0, 1)), torch.empty((2, 0), dtype=torch.long),
                        output, 'custom_type')
                self.assertTrue(pd.read_csv(stale_path).empty)
                for name in ('Gene_Table2_DEG_ct_SnG_score.csv', 'Gene_Table3_gene_ct_count.csv',
                             'Gene_newTable3_gene_ct_count.csv', 'table2ByCelltype.csv', 'table2ByGene.csv'):
                    self.assertTrue(pd.read_csv(os.path.join(output, name)).empty, name)
                result = pd.read_csv(os.path.join(output, 'Gene_Table2_DEG_ct_SnG_score.csv'))
                self.assertEqual(result.columns.tolist(), tables.GENE_TABLE_COLUMNS)
                counts = pd.read_csv(os.path.join(output, 'Cell_Table2_SnCs_per_ct.csv'))
                self.assertEqual(int(counts['number_of_SnCs'].sum()), n_snc)

    def test_eligible_group_produces_wilcoxon_and_sng_tables(self):
        data = example_data(['Type A'] * 8)
        initial = pd.DataFrame({'gene_name': ['gene_b']})
        edges = torch.tensor([[0] * 8 + list(range(2, 8)),
                              list(range(2, 10)) + [0] * 6])
        scores = torch.full((14, 1), 0.5)
        with tempfile.TemporaryDirectory() as output:
            tables.generate_tables(data, initial, {i: None for i in range(2, 8)},
                                   [0], scores, edges, output, 'custom_type')
            result = pd.read_csv(os.path.join(output, 'Gene_Table2_DEG_ct_SnG_score.csv'))
            self.assertEqual(result['gene'].tolist(), ['gene_b'])
            self.assertEqual(result['SnG_score'].tolist(), [0.5])
            self.assertGreater(float(result.iloc[0]['logFC']), 0.25)
            grouped = pd.read_csv(os.path.join(output, 'Gene_Table3_gene_ct_count.csv'))
            self.assertTrue(bool(grouped.iloc[0]['hallmarker']))


if __name__ == '__main__':
    unittest.main()
