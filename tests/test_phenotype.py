import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scanpy as sp
import torch

import utils
from phenotype_analysis import deepsas_v1_phenotype_v2 as phenotype_entry
from phenotype_analysis import generate_3tables_pheno as phenotype_tables


class PhenotypeCliTests(unittest.TestCase):
    def test_phenotype_defaults_are_explicit(self):
        with patch('sys.argv', ['deepsas_v1.py', '--exp_name', 'test']):
            args = utils.parse_args()
        self.assertFalse(args.phenotype_aware)
        self.assertEqual(args.phenotype_col, 'Condition')
        self.assertFalse(args.use_hvg_deg)
        self.assertEqual(args.phenotype_hvg_count, 1000)
        self.assertEqual(args.min_snc_per_phenotype, 1)
        self.assertEqual(args.phenotype_zscore_threshold, 2.)

    def test_entry_point_enables_current_pipeline_mode(self):
        argv = ['deepsas_v1_phenotype_v2.py', '--exp_name', 'test']
        with patch.object(sys, 'argv', argv), patch.object(
                phenotype_entry.runpy, 'run_path') as run_path:
            phenotype_entry.main()
        self.assertIn('--phenotype_aware', argv)
        self.assertEqual(run_path.call_args.kwargs['run_name'], '__main__')
        self.assertTrue(str(run_path.call_args.args[0]).endswith('deepsas_v1.py'))


class PhenotypeTableTests(unittest.TestCase):
    def _data(self):
        return sp.AnnData(
            np.ones((4, 2), dtype=np.float32),
            obs=pd.DataFrame({
                'cell_type': ['A', 'A', 'B', 'B'],
                'phenotype': ['control', 'control', 'treated', 'treated'],
            }, index=[f'cell_{i}' for i in range(4)]),
            var=pd.DataFrame(index=['gene_a', 'gene_b']))

    def test_tables_use_group_specific_sncs_and_defined_schemas(self):
        data = self._data()
        edges = torch.tensor([[2, 4], [0, 1]])
        attention = torch.tensor([[.5], [.7]])
        with tempfile.TemporaryDirectory() as output:
            long_table, raw, zscores, counts = \
                phenotype_tables.generate_phenotype_tables(
                    data, {2: None, 4: None}, [0, 1], attention, edges,
                    output, cell_type_col='cell_type',
                    phenotype_col='phenotype', zscore_threshold=.5)

            self.assertEqual(long_table.columns.tolist(),
                             phenotype_tables.PHENOTYPE_GENE_COLUMNS)
            self.assertEqual(len(long_table), 4)
            self.assertEqual(raw.shape, (2, 2))
            self.assertTrue(np.isfinite(zscores.to_numpy()).all())
            self.assertEqual(counts['number_of_cells'].tolist(), [2, 2])
            self.assertEqual(counts['number_of_SnCs'].tolist(), [1, 1])
            for filename in (
                    'Gene_Table1_SnG_scores_per_ct_pheno.csv',
                    'Gene_Table1_SnG_scores_per_ct_pheno_pivot.csv',
                    'Gene_Table1_SnG_scores_per_ct_pheno_pivot_zscore.csv',
                    'Gene_Specific_SnGs_per_ct_pheno_filtered.csv',
                    'Cell_Table2_SnCs_per_ct_pheno.csv'):
                self.assertTrue(os.path.isfile(os.path.join(output, filename)))

    def test_missing_or_null_annotations_fail_clearly(self):
        data = self._data()
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_attention = torch.empty((0, 1))
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(KeyError, 'Missing AnnData'):
                phenotype_tables.generate_phenotype_tables(
                    data, {}, [], empty_attention, empty_edges, output,
                    cell_type_col='cell_type', phenotype_col='missing')
            data.obs.loc['cell_0', 'phenotype'] = None
            with self.assertRaisesRegex(ValueError, 'must not be missing'):
                phenotype_tables.generate_phenotype_tables(
                    data, {}, [], empty_attention, empty_edges, output,
                    cell_type_col='cell_type', phenotype_col='phenotype')

    def test_empty_candidates_still_write_defined_outputs(self):
        data = self._data()
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_attention = torch.empty((0, 1))
        with tempfile.TemporaryDirectory() as output:
            long_table, raw, zscores, counts = \
                phenotype_tables.generate_phenotype_tables(
                    data, {}, [], empty_attention, empty_edges, output,
                    cell_type_col='cell_type', phenotype_col='phenotype')
            self.assertEqual(long_table.columns.tolist(),
                             phenotype_tables.PHENOTYPE_GENE_COLUMNS)
            self.assertTrue(long_table.empty)
            self.assertTrue(raw.empty)
            self.assertTrue(zscores.empty)
            self.assertEqual(int(counts['number_of_SnCs'].sum()), 0)
            filtered = pd.read_csv(os.path.join(
                output, 'Gene_Specific_SnGs_per_ct_pheno_filtered.csv'))
            self.assertEqual(filtered.columns.tolist(),
                             phenotype_tables.PHENOTYPE_SPECIFIC_COLUMNS)


if __name__ == '__main__':
    unittest.main()
