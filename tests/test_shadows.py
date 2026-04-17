"""
Tests for classical shadows against analytically known values.

The most important test: shadow-estimated <P> must converge to the true
expectation on known states (GHZ, product states, random states vs
direct inner product).
"""

from __future__ import annotations

import numpy as np
import pytest

from qsae.shadows import (
    build_feature_observables,
    compute_pauli_shadow,
    estimate_pauli_expectation,
)


def _ghz(n: int) -> np.ndarray:
    """GHZ state |0...0> + |1...1> / sqrt(2)."""
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    return psi


def _product_plus(n: int) -> np.ndarray:
    """|+>^n  — each Z_i has expectation 0, each X_i has expectation +1."""
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    out = plus
    for _ in range(n - 1):
        out = np.kron(out, plus)
    return out


def _pauli_exact(psi: np.ndarray, pauli: np.ndarray) -> float:
    """Compute exact <P> by building the Pauli operator."""
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    table = [I2, X, Y, Z]

    op = table[pauli[0]]
    for p in pauli[1:]:
        op = np.kron(op, table[p])
    return float(np.real(psi.conj() @ op @ psi))


def test_product_state_single_qubit_expectations() -> None:
    n = 3
    psi = _product_plus(n)
    rng = np.random.default_rng(42)
    bases, outcomes = compute_pauli_shadow(psi, n, n_samples=4000, rng=rng)

    # <X_0> should be ~1, <Z_0> should be ~0
    xp = np.zeros(n, dtype=np.int8); xp[0] = 1  # X_0
    zp = np.zeros(n, dtype=np.int8); zp[0] = 3  # Z_0

    x_est = estimate_pauli_expectation(xp, bases, outcomes)
    z_est = estimate_pauli_expectation(zp, bases, outcomes)
    assert abs(x_est - 1.0) < 0.15, f"<X_0> on |+>^n estimate = {x_est}"
    assert abs(z_est - 0.0) < 0.15, f"<Z_0> on |+>^n estimate = {z_est}"


def test_ghz_correlations() -> None:
    """
    GHZ state exact expectations:
      <Z_0 Z_1> = 1
      <Z_0> = 0
      <X_0 X_1 X_2> = 1
    """
    n = 3
    psi = _ghz(n)
    rng = np.random.default_rng(7)
    bases, outcomes = compute_pauli_shadow(psi, n, n_samples=6000, rng=rng)

    cases = [
        (np.array([3, 3, 0], dtype=np.int8), 1.0),  # Z_0 Z_1
        (np.array([3, 0, 0], dtype=np.int8), 0.0),  # Z_0
        (np.array([1, 1, 1], dtype=np.int8), 1.0),  # X_0 X_1 X_2
    ]
    for pauli, truth in cases:
        est = estimate_pauli_expectation(pauli, bases, outcomes)
        assert abs(est - truth) < 0.25, (
            f"pauli={pauli} truth={truth} est={est}"
        )


def test_random_state_observables_match_exact() -> None:
    """For a random pure state, shadow estimates should match exact <P>."""
    n = 3
    rng = np.random.default_rng(2024)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi = psi / np.linalg.norm(psi)

    bases, outcomes = compute_pauli_shadow(psi, n, n_samples=8000, rng=rng)

    obs = build_feature_observables(n, "paulis_weight_1_2")
    max_err = 0.0
    for k in range(obs.shape[0]):
        est = estimate_pauli_expectation(obs[k], bases, outcomes)
        truth = _pauli_exact(psi, obs[k])
        max_err = max(max_err, abs(est - truth))
    assert max_err < 0.3, f"max shadow estimation error = {max_err}"


def test_feature_observables_shape() -> None:
    n = 4
    obs1 = build_feature_observables(n, "all_weight_1")
    assert obs1.shape == (3 * n, n)
    obs2 = build_feature_observables(n, "paulis_weight_1_2")
    # 3n weight-1 + 3(n-1) nearest-neighbor same-type weight-2
    assert obs2.shape == (3 * n + 3 * (n - 1), n)
