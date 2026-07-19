"""Analytic-case unit tests for the input-recoverability controls."""

from __future__ import annotations

import numpy as np
import pytest

from qsae.analysis.input_control import (
    bootstrap_partial_corr,
    incremental_r2,
    oof_ridge_predict,
    partial_corr_controlling,
    residualize,
)


class TestResidualize:
    def test_removes_own_control(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=200)
        a = 3.0 * z + 5.0  # perfectly linear in z
        r = residualize(a, z)
        assert np.allclose(r, 0.0, atol=1e-8)

    def test_none_control_centers(self):
        a = np.array([1.0, 2.0, 3.0])
        assert np.allclose(residualize(a, None), a - 2.0)


class TestPartialCorr:
    def test_identical_series_is_one(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=300)
        z = rng.normal(size=300)
        assert partial_corr_controlling(a, a, z) == pytest.approx(1.0, abs=1e-6)

    def test_confound_only_association_vanishes(self):
        # a and b are correlated ONLY through the shared control z; controlling
        # for z must drive the partial correlation to ~0.
        rng = np.random.default_rng(2)
        z = rng.normal(size=5000)
        a = z + 0.1 * rng.normal(size=5000)
        b = z + 0.1 * rng.normal(size=5000)
        assert abs(partial_corr_controlling(a, b, z)) < 0.05
        assert partial_corr_controlling(a, b, None) > 0.9  # raw corr is high

    def test_genuine_association_survives(self):
        # a and b share a component orthogonal to z; controlling for z keeps it.
        rng = np.random.default_rng(3)
        z = rng.normal(size=5000)
        shared = rng.normal(size=5000)
        a = z + shared + 0.1 * rng.normal(size=5000)
        b = z + shared + 0.1 * rng.normal(size=5000)
        assert partial_corr_controlling(a, b, z) > 0.9

    def test_poly_control_removes_quadratic_confound(self):
        # b depends on z^2; a linear control leaves signal, a poly-2 control kills it.
        rng = np.random.default_rng(4)
        z = rng.normal(size=5000)
        a = z**2 + 0.1 * rng.normal(size=5000)
        b = z**2 + 0.1 * rng.normal(size=5000)
        lin = partial_corr_controlling(a, b, z.reshape(-1, 1))
        quad = partial_corr_controlling(a, b, np.c_[z, z**2])
        assert lin > 0.5           # linear control cannot remove z^2 dependence
        assert abs(quad) < 0.1     # poly-2 control does


class TestIncrementalR2:
    def test_noise_repr_adds_nothing(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(600, 2))
        y = z[:, 0] + 0.1 * rng.normal(size=600)
        noise = rng.normal(size=(600, 8))
        out = incremental_r2(noise, z, y, seed=0)
        assert out["r2_control"] > 0.8
        assert out["delta"] < 0.05   # pure-noise representation adds ~nothing

    def test_informative_repr_adds_signal(self):
        rng = np.random.default_rng(6)
        z = rng.normal(size=(600, 2))
        extra = rng.normal(size=600)
        y = z[:, 0] + extra + 0.1 * rng.normal(size=600)  # 'extra' not in z
        repr_feats = extra.reshape(-1, 1) + 0.05 * rng.normal(size=(600, 1))
        out = incremental_r2(repr_feats, z, y, seed=0)
        assert out["delta"] > 0.3    # representation supplies the missing 'extra'


class TestOOFAndBootstrap:
    def test_oof_predict_shape_and_skill(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(400, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.normal(size=400)
        pred = oof_ridge_predict(X, y, seed=0)
        assert pred.shape == y.shape
        assert np.corrcoef(pred, y)[0, 1] > 0.99

    def test_bootstrap_ci_brackets_estimate(self):
        rng = np.random.default_rng(8)
        z = rng.normal(size=500)
        shared = rng.normal(size=500)
        a = z + shared + 0.1 * rng.normal(size=500)
        b = z + shared + 0.1 * rng.normal(size=500)
        out = bootstrap_partial_corr(a, b, z, n_boot=500, seed=0)
        assert out["ci_lo"] <= out["estimate"] <= out["ci_hi"]
        assert out["ci_lo"] > 0.8  # strong genuine partial correlation
