"""
Multiple-comparison correction — Benjamini–Hochberg FDR.

The study tests many SAE features against many observables (a feature x
observable grid), plus per-observable permutation nulls. Reporting raw p-values
across such a grid inflates false positives. The draft currently reports only a
max-statistic permutation null per observable; a reviewer will additionally want
per-cell q-values under an explicit FDR procedure. These functions provide
Benjamini–Hochberg (1995) step-up q-values and a convenience wrapper for a
feature x observable p-value matrix.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg step-up FDR-adjusted q-values for a 1-D array of p-values.

    q_i is the smallest FDR level at which hypothesis i is rejected. The standard
    monotone enforcement is applied (cumulative minimum from largest to smallest
    rank) and values are clipped to [0, 1]. NaN p-values propagate as NaN q-values
    and are excluded from the count ``m``.

    Parameters
    ----------
    pvals : (K,) array of p-values in [0, 1]; NaNs allowed.

    Returns
    -------
    qvals : (K,) array of adjusted q-values, aligned to the input order.
    """
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    ranks = np.arange(1, m + 1)
    adj = ranked * m / ranks
    # enforce monotonicity: q is non-decreasing in p (min from the top rank down)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    q_valid = np.empty(m)
    q_valid[order] = adj
    q[valid] = q_valid
    return q


def bh_reject(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Boolean rejection mask at FDR level ``q`` (via adjusted q-values)."""
    return benjamini_hochberg(pvals) <= q


def fdr_grid(pval_matrix: np.ndarray) -> np.ndarray:
    """
    BH q-values over an entire (F, O) feature x observable p-value matrix, with the
    correction applied jointly across ALL F*O cells (the honest denominator for a
    grid search). NaNs (e.g. dead features) are ignored in the count.

    Returns a q-value matrix of the same shape.
    """
    M = np.asarray(pval_matrix, dtype=np.float64)
    flat_q = benjamini_hochberg(M.ravel())
    return flat_q.reshape(M.shape)
