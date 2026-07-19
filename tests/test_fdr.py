"""Unit tests for Benjamini–Hochberg FDR correction against known references."""

from __future__ import annotations

import numpy as np
import pytest

from qsae.analysis.fdr import benjamini_hochberg, bh_reject, fdr_grid


class TestBenjaminiHochberg:
    def test_matches_statsmodels_reference_values(self):
        # Reference q-values computed independently (BH step-up) for this vector.
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
        q = benjamini_hochberg(p)
        expected = np.array([0.008, 0.032, 0.0672, 0.0672, 0.0672,
                             0.08, 0.08457143, 0.205])
        assert np.allclose(q, expected, atol=1e-6)

    def test_all_equal_pvalues(self):
        p = np.full(5, 0.02)
        q = benjamini_hochberg(p)
        assert np.allclose(q, 0.02 * 5 / 5)  # m/m == 1 factor at top rank

    def test_monotonic_in_pvalue(self):
        rng = np.random.default_rng(0)
        p = np.sort(rng.uniform(size=50))
        q = benjamini_hochberg(p)
        assert np.all(np.diff(q) >= -1e-12)  # q non-decreasing with p

    def test_qvalues_bounded(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(size=100)
        q = benjamini_hochberg(p)
        assert q.min() >= 0.0 and q.max() <= 1.0

    def test_nan_propagates_and_is_excluded_from_m(self):
        p = np.array([0.01, np.nan, 0.02])
        q = benjamini_hochberg(p)
        assert np.isnan(q[1])
        # m = 2, smallest p=0.01 rank1 -> 0.01*2/1=0.02; 0.02 rank2 -> 0.02
        assert q[0] == pytest.approx(0.02)
        assert q[2] == pytest.approx(0.02)

    def test_uniform_null_controls_fdr(self):
        # Under the global null, expected number of rejections at q=0.05 is small.
        rng = np.random.default_rng(2)
        rejections = [bh_reject(rng.uniform(size=200), q=0.05).sum() for _ in range(50)]
        assert np.mean(rejections) < 5  # well-controlled

    def test_strong_signal_rejected(self):
        p = np.concatenate([np.full(10, 1e-6), np.random.default_rng(3).uniform(0.2, 1, 90)])
        rej = bh_reject(p, q=0.05)
        assert rej[:10].all()  # the 10 tiny p-values are rejected


class TestFDRGrid:
    def test_grid_shape_and_joint_correction(self):
        rng = np.random.default_rng(4)
        M = rng.uniform(size=(6, 4))
        Q = fdr_grid(M)
        assert Q.shape == (6, 4)
        # joint correction == flat BH then reshape
        assert np.allclose(Q.ravel(), benjamini_hochberg(M.ravel()))
