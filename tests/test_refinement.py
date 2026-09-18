import inspect
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from refinement import (candidate_convergence, mean_attention_by_target,
                        update_gene_candidates)
from model_GAT import GAEModel, require_projection_mode
from model_Sencell import Sencell
import utils


class ModelInterfaceTests(unittest.TestCase):
    def test_binary_gat_has_no_edge_feature_parameter(self):
        self.assertNotIn('edge_dim', inspect.signature(GAEModel).parameters)
        model = GAEModel(4, 4, hidden_size=8, dropout=0.1,
                         type_specific_projections=True)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        is_gene = torch.tensor([True, False])
        encoded = model.encode(torch.ones((2, 4)), edge_index, is_gene)
        self.assertEqual(encoded.shape, (2, 4))
        self.assertIn('encoder.gene_projection.weight', model.state_dict())
        self.assertIn('encoder.cell_projection.weight', model.state_dict())
        self.assertEqual(model.architecture_version, 'type-specific-projections-v1')
        for layer in (model.encoder.conv1, model.encoder.conv2):
            self.assertEqual(layer.heads, 1)
            self.assertTrue(layer.concat)
            self.assertTrue(layer.add_self_loops)
            self.assertEqual(layer.negative_slope, .2)

    def test_gene_and_cell_projections_are_distinct(self):
        model = GAEModel(2, 2, hidden_size=4, dropout=0.)
        with torch.no_grad():
            model.encoder.gene_projection.weight.zero_()
            model.encoder.gene_projection.bias.fill_(1.)
            model.encoder.cell_projection.weight.zero_()
            model.encoder.cell_projection.bias.fill_(2.)
        projected = model.encoder.project_node_types(
            torch.zeros((2, 2)), torch.tensor([True, False]))
        torch.testing.assert_close(
            projected, torch.tensor([[1., 1.], [2., 2.]]))

    def test_attention_scores_come_from_final_gat_layer(self):
        torch.manual_seed(7)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],
            [3, 3, 4, 4, 5, 5],
        ])
        node_types = torch.tensor([True, True, True, False, False, False])
        features = torch.randn((6, 4))

        for type_specific in (False, True):
            with self.subTest(type_specific=type_specific):
                model = GAEModel(
                    4, 4, hidden_size=8, dropout=0.,
                    type_specific_projections=type_specific)
                model.eval()
                data = SimpleNamespace(
                    x=features, edge_index=edge_index, y=node_types)

                with torch.no_grad():
                    projected = model.encoder.project_node_types(
                        features, node_types)
                    hidden = torch.nn.functional.elu(
                        model.encoder.conv1(projected, edge_index))
                    _, (expected_edges, expected_alpha) = model.encoder.conv2(
                        hidden, edge_index, return_attention_weights=True)
                    actual_edges, actual_alpha = model.get_attention_scores(data)

                torch.testing.assert_close(actual_edges, expected_edges)
                torch.testing.assert_close(actual_alpha, expected_alpha)
                self.assertEqual(actual_alpha.shape[0], actual_edges.shape[1])

    def test_shared_projection_mode_omits_type_specific_parameters(self):
        model = GAEModel(4, 4, hidden_size=8, dropout=0.1,
                         type_specific_projections=False)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        encoded = model.encode(torch.ones((2, 4)), edge_index)
        self.assertEqual(encoded.shape, (2, 4))
        self.assertNotIn('encoder.gene_projection.weight', model.state_dict())
        self.assertNotIn('encoder.cell_projection.weight', model.state_dict())
        self.assertEqual(model.architecture_version, 'shared-projection-v1')

    def test_cli_defaults_use_type_specific_projections(self):
        with patch('sys.argv', ['deepsas_v1.py', '--exp_name', 'test']):
            args = utils.parse_args()
        self.assertTrue(args.type_specific_projections)
        self.assertFalse(hasattr(args, 'ccc_aggregation'))
        self.assertEqual(args.sng_update_mode, 'iqr')
        self.assertEqual(args.min_snc_per_type, 1)
        self.assertEqual(args.min_genes_per_cell, 200)
        self.assertEqual(args.min_cells_per_gene, 10)
        self.assertEqual(args.normalization_target_sum, 1e4)
        self.assertEqual(args.scale_max_value, 10.)
        self.assertEqual(args.umap_n_neighbors, 10)
        self.assertEqual(args.umap_n_pcs, 40)
        self.assertEqual(args.deg_min_snc, 6)
        self.assertEqual(args.deg_min_control, 2)
        self.assertEqual(args.deg_min_logfc, .25)
        with patch('sys.argv', [
                'deepsas_v1.py', '--exp_name', 'test',
                '--no_type_specific_projections']):
            shared_args = utils.parse_args()
        self.assertFalse(shared_args.type_specific_projections)

    def test_preprocessing_and_reporting_defaults_can_be_passed_explicitly(self):
        argv = [
            'deepsas_v1.py', '--exp_name', 'test',
            '--min_genes_per_cell', '200',
            '--min_cells_per_gene', '10',
            '--normalization_target_sum', '10000',
            '--scale_max_value', '10',
            '--umap_n_neighbors', '10',
            '--umap_n_pcs', '40',
            '--deg_min_snc', '6',
            '--deg_min_control', '2',
            '--deg_min_logfc', '0.25',
        ]
        with patch('sys.argv', argv):
            explicit = utils.parse_args()
        with patch('sys.argv', ['deepsas_v1.py', '--exp_name', 'test']):
            defaults = utils.parse_args()
        names = (
            'min_genes_per_cell', 'min_cells_per_gene',
            'normalization_target_sum', 'scale_max_value',
            'umap_n_neighbors', 'umap_n_pcs', 'deg_min_snc',
            'deg_min_control', 'deg_min_logfc')
        self.assertEqual(
            {name: getattr(explicit, name) for name in names},
            {name: getattr(defaults, name) for name in names})

    def test_new_numeric_parameters_reject_invalid_values(self):
        invalid_options = (
            ('--min_genes_per_cell', '0'),
            ('--min_cells_per_gene', '0'),
            ('--normalization_target_sum', '0'),
            ('--scale_max_value', '0'),
            ('--umap_n_neighbors', '0'),
            ('--umap_n_pcs', '0'),
            ('--phenotype_hvg_count', '0'),
            ('--min_snc_per_phenotype', '0'),
            ('--phenotype_zscore_threshold', 'nan'),
            ('--deg_min_snc', '0'),
            ('--deg_min_control', '0'),
            ('--deg_min_logfc', 'nan'),
        )
        for option, value in invalid_options:
            with self.subTest(option=option), patch(
                    'sys.argv', ['deepsas_v1.py', '--exp_name', 'test',
                                 option, value]), patch('sys.stderr', new=io.StringIO()):
                with self.assertRaises(SystemExit):
                    utils.parse_args()

    def test_checkpoint_projection_mode_must_match(self):
        shared_model = SimpleNamespace(encoder=SimpleNamespace())
        type_specific_model = SimpleNamespace(encoder=SimpleNamespace(
            gene_projection=object(), cell_projection=object()))
        with self.assertRaisesRegex(RuntimeError, '--retrain'):
            require_projection_mode(shared_model, True)
        with self.assertRaisesRegex(RuntimeError, '--retrain'):
            require_projection_mode(type_specific_model, False)

    def test_cell_model_has_three_configured_distance_targets(self):
        model = Sencell(4, 8, [0., 0., 4.])
        self.assertEqual(model.levels.tolist(), [0., 0., 4.])
        with self.assertRaisesRegex(ValueError, 'three finite'):
            Sencell(4, 8, [0., 4.])


class GeneUpdateTests(unittest.TestCase):
    def test_threshold_drives_swaps_and_fixed_scores_can_stabilize(self):
        scores = np.array([0., 0., 0., 0., 0., 0., 10., 20., 30.])
        current = [0, 1, 2, 3, 4, 5, 6, 7]
        updated, info = update_gene_candidates(
            scores, current, 1.5, 'iqr')
        q1, q3 = np.percentile(scores[current], [25, 75])
        self.assertEqual(info['gene_threshold'], q3 + 1.5 * (q3 - q1))
        self.assertEqual(info['sng_outlier_count'], 2)
        self.assertEqual(info['sng_swaps'], 1)
        self.assertEqual(set(updated), {1, 2, 3, 4, 5, 6, 7, 8})
        stable, second = update_gene_candidates(
            scores, updated, 1.5, 'iqr')
        np.testing.assert_array_equal(stable, updated)
        self.assertEqual(second['sng_swaps'], 0)

    def test_order_does_not_change_identity_and_output_is_unique(self):
        scores = np.arange(30, dtype=float)
        old = np.arange(20)
        forward, info = update_gene_candidates(
            scores, old, 0, 'iqr')
        reverse, _ = update_gene_candidates(
            scores, old[::-1], 0, 'iqr')
        duplicate, _ = update_gene_candidates(
            scores, np.r_[old, old[:4]], 0, 'iqr')
        np.testing.assert_array_equal(forward, reverse)
        np.testing.assert_array_equal(forward, duplicate)
        self.assertEqual(len(forward), len(set(forward)))
        self.assertEqual(len(forward), len(old))
        self.assertGreater(info['sng_swaps'], 0)
        for added, removed in zip(info['sng_added'], info['sng_removed']):
            self.assertGreater(scores[added], scores[removed])
            self.assertGreater(scores[added], info['gene_threshold'])

    def test_above_fence_does_not_force_worse_replacement(self):
        scores = np.array([0.] * 17 + [5., 10., 20.])
        updated, info = update_gene_candidates(
            scores, [18, 19], 1.5, 'iqr')
        np.testing.assert_array_equal(updated, [18, 19])
        self.assertEqual(info['sng_swaps'], 0)

    def test_equal_scores_and_empty_initial_set_do_not_invent_genes(self):
        updated, info = update_gene_candidates(
            np.ones(12), [1, 4], 1.5, 'iqr')
        np.testing.assert_array_equal(updated, [1, 4])
        self.assertEqual(info['sng_swaps'], 0)
        no_outliers, info = update_gene_candidates(
            [1., 1., 1., 10.], [0, 1, 2], 1.5, 'iqr')
        np.testing.assert_array_equal(no_outliers, [0, 1, 2])
        self.assertEqual(info['sng_outlier_count'], 0)
        empty, info = update_gene_candidates(
            [0., 0., 0., 5.], [], 1.5, 'iqr')
        self.assertEqual(len(empty), 0)
        self.assertEqual(info['sng_swaps'], 0)

    def test_invalid_scores_fail_clearly(self):
        for scores in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                update_gene_candidates(scores, [], 1.5, 'iqr')

    def test_fixed10_mode_replaces_ten_and_keeps_unique_ids(self):
        scores = np.arange(30, dtype=float)
        current = np.arange(20)
        updated, info = update_gene_candidates(
            scores, current, 1.5, 'fixed10')
        np.testing.assert_array_equal(updated, np.arange(10, 30))
        self.assertIsNone(info['gene_threshold'])
        self.assertEqual(info['sng_outlier_count'], 10)
        self.assertEqual(info['sng_swaps'], 10)
        self.assertEqual(len(updated), len(set(updated)))

    def test_fixed10_mode_is_explicit_and_validated(self):
        scores = np.arange(8, dtype=float)
        updated, info = update_gene_candidates(
            scores, [0, 1, 2], 1.5, 'fixed10')
        self.assertEqual(info['sng_swaps'], 3)
        np.testing.assert_array_equal(updated, [5, 6, 7])
        with self.assertRaisesRegex(ValueError, 'update mode'):
            update_gene_candidates(scores, [0, 1], 1.5, 'unknown')


class ConvergenceTests(unittest.TestCase):
    def test_empty_sets_are_never_normal_convergence(self):
        status, jc, jg = candidate_convergence([], [], [1], [1], 0.99)
        self.assertEqual(status, 'empty_candidates')
        self.assertEqual((jc, jg), (1., 1.))
        self.assertEqual(candidate_convergence([2], [2], [], [], .99)[0],
                         'empty_candidates')

    def test_both_sets_must_stabilize(self):
        self.assertEqual(candidate_convergence([1, 2], [1, 2], [3], [3], .99)[0],
                         'converged')
        self.assertEqual(candidate_convergence([1, 2], [1], [3], [3], .99)[0],
                         'running')
        self.assertEqual(candidate_convergence([1], [1], [3, 4], [3], .99)[0],
                         'running')
        self.assertEqual(candidate_convergence(None, [1], None, [3], .99)[0],
                         'running')

    def test_configured_tolerance_is_used(self):
        self.assertEqual(candidate_convergence(range(200), range(199), [1], [1], .99)[0],
                         'converged')
        self.assertEqual(candidate_convergence(range(200), range(199), [1], [1], 1.)[0],
                         'running')

    def test_fixed10_overlap_passes_point_nine_but_not_point_nine_nine(self):
        previous_genes = range(274)
        current_genes = range(10, 284)
        status, _, gene_jaccard = candidate_convergence(
            [1], [1], previous_genes, current_genes, .9)
        self.assertEqual(status, 'converged')
        self.assertAlmostEqual(gene_jaccard, 264 / 284)
        self.assertEqual(candidate_convergence(
            [1], [1], previous_genes, current_genes, .99)[0], 'running')


class AttentionTests(unittest.TestCase):
    def test_means_use_only_connected_selected_sources(self):
        edge = torch.tensor([[0, 1, 2, 0], [3, 3, 3, 4]])
        alpha = torch.tensor([[.2], [.4], [.9], [.6]])
        scores, connected = mean_attention_by_target(edge, alpha, [0, 1], 6)
        torch.testing.assert_close(scores, torch.tensor([0., 0., 0., .3, .6, 0.]))
        self.assertEqual(connected.tolist(), [False, False, False, True, True, False])

    def test_empty_attention_is_well_defined(self):
        scores, connected = mean_attention_by_target(
            torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1)), [], 3)
        self.assertEqual(scores.tolist(), [0., 0., 0.])
        self.assertFalse(connected.any())


if __name__ == '__main__':
    unittest.main()
