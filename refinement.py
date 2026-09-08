"""Candidate updates and stopping rules shared by the pipeline and its checks."""

import numpy as np
import torch


def mean_attention_by_target(edge_index, attention, source_indices, num_nodes):
    """Mean over connected selected sources; sum attention heads as in v1."""
    source_mask = torch.zeros(num_nodes, dtype=torch.bool)
    source_mask[torch.as_tensor(list(source_indices), dtype=torch.long)] = True
    mask = source_mask[edge_index[0]]
    weights = attention.detach()
    if weights.ndim == 2:
        weights = weights.sum(dim=1)
    targets = edge_index[1, mask]
    sums = torch.zeros(num_nodes, dtype=weights.dtype)
    counts = torch.zeros(num_nodes, dtype=weights.dtype)
    sums.index_add_(0, targets, weights[mask])
    counts.index_add_(0, targets, torch.ones_like(weights[mask]))
    return sums / counts.clamp_min(1), counts > 0


def update_gene_candidates(scores, current_genes, iqr_multiplier=1.5):
    """Replace weak candidates with strictly better, above-fence nonmembers.

    The fence and requested swap count use the current SnG candidates' scores,
    including zeros. The count is the number above that fence. Actual accepted
    replacements are bounded by this count and can be zero. Gene ID breaks
    score ties deterministically; input order never affects membership.
    """
    scores = np.asarray(scores, dtype=float)
    current = np.unique(np.asarray(current_genes, dtype=np.int64))
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError('SnG scores must be a nonempty, finite one-dimensional array')
    if not np.isfinite(iqr_multiplier) or iqr_multiplier < 0:
        raise ValueError('IQR multiplier must be finite and nonnegative')
    if current.size and (current.min() < 0 or current.max() >= scores.size):
        raise ValueError('Candidate gene index is outside the score array')
    if not current.size:
        return current, dict(gene_threshold=None, sng_outlier_count=0,
                             sng_swaps=0, sng_added=[], sng_removed=[])
    q1, q3 = np.percentile(scores[current], [25, 75])
    threshold = float(q3 + iqr_multiplier * (q3 - q1))
    requested_swaps = int(np.count_nonzero(scores[current] > threshold))
    outliers = np.flatnonzero(scores > threshold)
    entrants = np.setdiff1d(outliers, current)
    entrants = entrants[np.lexsort((entrants, -scores[entrants]))]
    outgoing = current[np.lexsort((current, scores[current]))]
    limit = min(requested_swaps, len(entrants), len(outgoing))
    accepted = scores[entrants[:limit]] > scores[outgoing[:limit]]
    added, removed = entrants[:limit][accepted], outgoing[:limit][accepted]
    updated = np.union1d(np.setdiff1d(current, removed), added)
    return updated, {
        'gene_threshold': threshold,
        'sng_outlier_count': requested_swaps,
        'sng_swaps': int(len(added)),
        'sng_added': added.tolist(),
        'sng_removed': removed.tolist(),
    }


def candidate_convergence(previous_cells, cells, previous_genes, genes, tolerance):
    """Return outer-loop status and the two adjacent-set Jaccard values."""
    cells, genes = set(cells), set(genes)

    def overlap(previous, current):
        if previous is None:
            return float('nan')
        previous = set(previous)
        union = previous | current
        return len(previous & current) / len(union) if union else 1.0

    j_cells, j_genes = overlap(previous_cells, cells), overlap(previous_genes, genes)
    if not cells or not genes:
        return 'empty_candidates', j_cells, j_genes
    if (previous_cells is not None and previous_genes is not None
            and len(previous_cells) > 0 and len(previous_genes) > 0
            and j_cells >= tolerance and j_genes >= tolerance):
        return 'converged', j_cells, j_genes
    return 'running', j_cells, j_genes
