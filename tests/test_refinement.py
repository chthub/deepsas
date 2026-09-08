import unittest

import numpy as np
import torch

from refinement import (candidate_convergence, mean_attention_by_target,
                        update_gene_candidates)


class GeneUpdateTests(unittest.TestCase):
    def test_threshold_drives_swaps_and_fixed_scores_can_stabilize(self):
        scores = np.array([0., 0., 0., 0., 0., 0., 10., 20., 30.])
        current = [0, 1, 2, 3, 4, 5, 6, 7]
        updated, info = update_gene_candidates(scores, current)
        q1, q3 = np.percentile(scores[current], [25, 75])
        self.assertEqual(info['gene_threshold'], q3 + 1.5 * (q3 - q1))
        self.assertEqual(info['sng_outlier_count'], 2)
        self.assertEqual(info['sng_swaps'], 1)
        self.assertEqual(set(updated), {1, 2, 3, 4, 5, 6, 7, 8})
        stable, second = update_gene_candidates(scores, updated)
        np.testing.assert_array_equal(stable, updated)
        self.assertEqual(second['sng_swaps'], 0)

    def test_order_does_not_change_identity_and_output_is_unique(self):
        scores = np.arange(30, dtype=float)
        old = np.arange(20)
        forward, info = update_gene_candidates(scores, old, iqr_multiplier=0)
        reverse, _ = update_gene_candidates(scores, old[::-1], iqr_multiplier=0)
        duplicate, _ = update_gene_candidates(scores, np.r_[old, old[:4]], iqr_multiplier=0)
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
        updated, info = update_gene_candidates(scores, [18, 19])
        np.testing.assert_array_equal(updated, [18, 19])
        self.assertEqual(info['sng_swaps'], 0)

    def test_equal_scores_and_empty_initial_set_do_not_invent_genes(self):
        updated, info = update_gene_candidates(np.ones(12), [1, 4])
        np.testing.assert_array_equal(updated, [1, 4])
        self.assertEqual(info['sng_swaps'], 0)
        no_outliers, info = update_gene_candidates([1., 1., 1., 10.], [0, 1, 2])
        np.testing.assert_array_equal(no_outliers, [0, 1, 2])
        self.assertEqual(info['sng_outlier_count'], 0)
        empty, info = update_gene_candidates([0., 0., 0., 5.], [])
        self.assertEqual(len(empty), 0)
        self.assertEqual(info['sng_swaps'], 0)

    def test_invalid_scores_fail_clearly(self):
        for scores in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                update_gene_candidates(scores, [])


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
