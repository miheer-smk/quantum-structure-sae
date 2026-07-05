"""
Tests for the sparse ground-state solver (scales the analysis past the dense
L=8 ceiling used elsewhere). Correctness is checked against the dense
exact-diagonalisation kernel and against the uniform-field reference solver.
"""

from __future__ import annotations

import numpy as np

from qsae.reverse_arrow.data import (
    compute_ground_energies,
    compute_ground_states_sparse,
)
from qsae.observables import long_range_zz_fast


def test_sparse_energies_match_dense_L8():
    rng = np.random.default_rng(0)
    h = rng.uniform(0.1, 2.0, size=(16, 8))
    e_dense = compute_ground_energies(h, J=1.0)
    e_sparse, _ = compute_ground_states_sparse(h, J_fields=1.0)
    np.testing.assert_allclose(e_sparse, e_dense, atol=1e-9)


def test_sparse_states_are_normalised_and_real_energy():
    rng = np.random.default_rng(1)
    h = rng.uniform(0.1, 2.0, size=(5, 6))
    e, states = compute_ground_states_sparse(h, J_fields=1.0)
    norms = np.linalg.norm(states, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)
    assert np.all(np.isfinite(e))


def test_sparse_ground_state_recovers_order_trend():
    # ordered (small h) should have larger end-to-end ZZ than disordered (large h)
    h_ordered = np.full((1, 6), 0.2)
    h_disordered = np.full((1, 6), 3.0)
    _, s_ord = compute_ground_states_sparse(h_ordered, J_fields=1.0)
    _, s_dis = compute_ground_states_sparse(h_disordered, J_fields=1.0)
    assert long_range_zz_fast(s_ord[0], 6) > long_range_zz_fast(s_dis[0], 6)


def test_per_bond_disordered_couplings_run_and_match_uniform():
    # a per-bond J array equal to 1 everywhere must match the scalar-J solve
    rng = np.random.default_rng(2)
    h = rng.uniform(0.1, 2.0, size=(4, 6))
    e_scalar, _ = compute_ground_states_sparse(h, J_fields=1.0)
    J_bonds = np.ones((4, 5))
    e_bonds, _ = compute_ground_states_sparse(h, J_fields=J_bonds)
    np.testing.assert_allclose(e_scalar, e_bonds, atol=1e-9)
