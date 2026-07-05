"""
Tests for quantum observables against analytically known values.

Key checkpoints
---------------
* Product state |0⟩^n  : S=0, ⟨Z_i⟩=1, ⟨X_i⟩=0, ⟨Z_i Z_j⟩=1 for all i,j
* Product state |+⟩^n  : S=0, ⟨Z_i⟩=0, ⟨X_i⟩=1, ⟨Z_i Z_j⟩=0 for i≠j
* GHZ state            : S=log(2) (half-chain), ⟨Z_i Z_j⟩=1 for all i≠j
* Phase proximity      : δ=0 at h_c, δ<0 for h<h_c, δ>0 for h>h_c
* TFIM ground states   : entropy / correlators match physical intuition at
                         h=0.1 (ordered) vs h=5.0 (disordered)
"""

from __future__ import annotations

import numpy as np
import pytest

from qsae.observables import (
    compute_all_observables,
    entanglement_spectrum,
    half_chain_entanglement_entropy,
    long_range_zz,
    longitudinal_magnetization,
    nearest_neighbor_zz,
    order_parameter,
    phase_proximity,
    transverse_magnetization,
    zz_correlator,
    zz_correlator_matrix,
)


# ---------------------------------------------------------------------------
# Reference states
# ---------------------------------------------------------------------------

def _all_zeros(n: int) -> np.ndarray:
    """Computational basis state |0…0⟩."""
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def _all_plus(n: int) -> np.ndarray:
    """|+⟩^n tensor product."""
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    out = plus
    for _ in range(n - 1):
        out = np.kron(out, plus)
    return out


def _ghz(n: int) -> np.ndarray:
    """GHZ state (|0…0⟩ + |1…1⟩) / √2."""
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return psi


def _tfim_ground_state(n: int, h: float, J: float = 1.0) -> np.ndarray:
    """Exact ground state of uniform TFIM (all h_i = h) via dense ED."""
    from scipy.linalg import eigh

    dim = 1 << n
    I2 = np.eye(2, dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)

    def op_at(op, site):
        out = None
        for k in range(n):
            factor = op if k == site else I2
            out = factor if out is None else np.kron(out, factor)
        return out

    H = np.zeros((dim, dim))
    for i in range(n - 1):
        H -= J * (op_at(Z, i) @ op_at(Z, i + 1))
    for i in range(n):
        H -= h * op_at(X, i)

    _, vecs = eigh(H, subset_by_index=[0, 0])
    return vecs[:, 0].astype(np.complex128)


# ---------------------------------------------------------------------------
# Entanglement entropy
# ---------------------------------------------------------------------------

class TestEntanglementEntropy:
    def test_product_state_zero_entropy(self):
        """Product states are unentangled — S = 0."""
        for n in (2, 4, 6):
            psi = _all_zeros(n)
            S = half_chain_entanglement_entropy(psi, n)
            assert S < 1e-10, f"n={n}: S={S:.2e} should be 0 for |0…0⟩"

    def test_plus_product_state_zero_entropy(self):
        """Tensor product of |+⟩ is also unentangled."""
        n = 4
        psi = _all_plus(n)
        S = half_chain_entanglement_entropy(psi, n)
        assert S < 1e-10, f"S={S:.2e} should be 0 for |+⟩^n"

    def test_ghz_entropy_log2(self):
        """Half-chain entropy of GHZ state is log(2) (maximally entangled qubit pair)."""
        n = 4
        psi = _ghz(n)
        S = half_chain_entanglement_entropy(psi, n)
        assert abs(S - np.log(2)) < 0.01, (
            f"GHZ n={n}: S={S:.6f}, expected log(2)={np.log(2):.6f}"
        )

    def test_entropy_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            n = 4
            psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            psi /= np.linalg.norm(psi)
            S = half_chain_entanglement_entropy(psi, n)
            assert S >= -1e-12, f"entropy must be non-negative, got {S}"

    def test_entropy_critical_point_is_maximum(self):
        """
        Entanglement entropy peaks at the quantum critical point (h = h_c = 1).

        Deep disordered phase (h >> J): ground state → |+>^n (product state, S→0).
        Deep ordered phase (h << J, finite L): ground state is Z2-symmetry-broken
            GHZ-like superposition, so S → log(2).
        Critical point (h ≈ 1): entanglement is maximal, S > log(2).
        """
        n = 8
        psi_critical   = _tfim_ground_state(n, h=1.0)
        psi_disordered = _tfim_ground_state(n, h=5.0)
        S_critical   = half_chain_entanglement_entropy(psi_critical, n)
        S_disordered = half_chain_entanglement_entropy(psi_disordered, n)
        assert S_critical > S_disordered, (
            f"S_critical={S_critical:.4f} should exceed S_disordered={S_disordered:.4f}"
        )
        # Deep disordered phase should have very low entropy (near product state)
        assert S_disordered < 0.1, (
            f"Deep disordered S={S_disordered:.4f} should be near 0 (product-like)"
        )


# ---------------------------------------------------------------------------
# Entanglement spectrum
# ---------------------------------------------------------------------------

class TestEntanglementSpectrum:
    def test_product_state_single_schmidt_value(self):
        """Product state: only one nonzero Schmidt value (= 1)."""
        n = 4
        psi = _all_zeros(n)
        spec = entanglement_spectrum(psi, n)
        assert abs(spec[0] - 1.0) < 1e-10
        assert np.all(spec[1:] < 1e-10)

    def test_ghz_two_equal_schmidt_values(self):
        """GHZ state splits equally: both Schmidt values = 1/√2."""
        n = 4
        psi = _ghz(n)
        spec = entanglement_spectrum(psi, n)
        assert abs(spec[0] - 1.0 / np.sqrt(2)) < 0.01
        assert abs(spec[1] - 1.0 / np.sqrt(2)) < 0.01


# ---------------------------------------------------------------------------
# ZZ correlators
# ---------------------------------------------------------------------------

class TestZZCorrelator:
    def test_all_zeros_perfect_correlation(self):
        """In |0…0⟩, ⟨Z_i Z_j⟩ = 1 for all i,j (all spins up)."""
        n = 4
        psi = _all_zeros(n)
        for i in range(n):
            for j in range(n):
                c = zz_correlator(psi, n, i, j)
                assert abs(c - 1.0) < 1e-10, f"ZZ({i},{j})={c} ≠ 1 for |0⟩^n"

    def test_plus_state_zero_zz(self):
        """In |+⟩^n, ⟨Z_i Z_j⟩ = 0 for i ≠ j (no Z-correlations)."""
        n = 4
        psi = _all_plus(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = zz_correlator(psi, n, i, j)
                assert abs(c) < 0.01, f"ZZ({i},{j})={c:.4f} should be ~0 for |+⟩^n"

    def test_ghz_perfect_zz_correlation(self):
        """GHZ: ⟨Z_i Z_j⟩ = 1 for all i ≠ j (all-up or all-down simultaneously)."""
        n = 4
        psi = _ghz(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = zz_correlator(psi, n, i, j)
                assert abs(c - 1.0) < 0.01, (
                    f"GHZ ZZ({i},{j})={c:.4f} should be 1"
                )

    def test_zz_matrix_symmetric(self):
        rng = np.random.default_rng(42)
        n = 4
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi /= np.linalg.norm(psi)
        M = zz_correlator_matrix(psi, n)
        assert np.allclose(M, M.T, atol=1e-10), "ZZ matrix must be symmetric"

    def test_nn_zz_ordered_vs_disordered(self):
        """Ordered TFIM (h≪J) has large ⟨Z_i Z_{i+1}⟩; disordered (h≫J) has small."""
        n = 6
        psi_ord  = _tfim_ground_state(n, h=0.1)
        psi_dis  = _tfim_ground_state(n, h=5.0)
        nn_ord  = nearest_neighbor_zz(psi_ord, n).mean()
        nn_dis  = nearest_neighbor_zz(psi_dis, n).mean()
        assert nn_ord > nn_dis, (
            f"⟨ZZ⟩_ordered={nn_ord:.4f} should exceed ⟨ZZ⟩_disordered={nn_dis:.4f}"
        )


# ---------------------------------------------------------------------------
# Magnetization
# ---------------------------------------------------------------------------

class TestMagnetization:
    def test_zeros_z_magnetization(self):
        """In |0…0⟩, ⟨Z_i⟩ = 1 for all sites."""
        n = 4
        psi = _all_zeros(n)
        mag = longitudinal_magnetization(psi, n)
        assert np.allclose(mag, 1.0, atol=1e-10), f"Z mag = {mag}"

    def test_plus_x_magnetization(self):
        """In |+⟩^n, ⟨X_i⟩ = 1 and ⟨Z_i⟩ = 0 for all sites."""
        n = 4
        psi = _all_plus(n)
        x_mag = transverse_magnetization(psi, n)
        z_mag = longitudinal_magnetization(psi, n)
        assert np.allclose(x_mag, 1.0, atol=1e-10), f"X mag = {x_mag}"
        assert np.allclose(z_mag, 0.0, atol=1e-10), f"Z mag = {z_mag}"

    def test_order_parameter_ordered_vs_disordered(self):
        """Order parameter larger in ordered phase (h≪J) than disordered (h≫J)."""
        n = 6
        psi_ord = _tfim_ground_state(n, h=0.1)
        psi_dis = _tfim_ground_state(n, h=5.0)
        op_ord = order_parameter(psi_ord, n)
        op_dis = order_parameter(psi_dis, n)
        assert op_ord > op_dis, (
            f"|⟨Z⟩|_ordered={op_ord:.4f} should exceed |⟨Z⟩|_disordered={op_dis:.4f}"
        )


# ---------------------------------------------------------------------------
# Phase proximity
# ---------------------------------------------------------------------------

class TestPhaseProximity:
    def test_zero_at_critical_point(self):
        assert phase_proximity(1.0) == pytest.approx(0.0)

    def test_negative_in_ferromagnetic_phase(self):
        assert phase_proximity(0.5) < 0

    def test_positive_in_paramagnetic_phase(self):
        assert phase_proximity(2.0) > 0

    def test_array_input(self):
        h = np.array([0.5, 1.0, 1.5])
        delta = phase_proximity(h)
        assert delta.shape == (3,)
        assert delta[0] < 0
        assert delta[1] == pytest.approx(0.0)
        assert delta[2] > 0


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

class TestComputeAllObservables:
    def test_output_keys_and_shapes(self):
        n = 4
        N = 3
        rng = np.random.default_rng(7)
        states = np.array([
            (lambda p: p / np.linalg.norm(p))(
                rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            )
            for _ in range(N)
        ])
        h_vals = rng.uniform(0.5, 1.5, size=N)

        obs = compute_all_observables(states, n, h_values=h_vals)

        assert obs["entropy"].shape == (N,)
        assert obs["nn_zz"].shape == (N, n - 1)
        assert obs["mean_nn_zz"].shape == (N,)
        assert obs["transverse_mag"].shape == (N, n)
        assert obs["mean_x"].shape == (N,)
        assert obs["order_param"].shape == (N,)
        assert obs["phase_proximity"].shape == (N,)

    def test_without_h_values_no_proximity_key(self):
        n = 4
        psi = _all_zeros(n)
        obs = compute_all_observables(psi[np.newaxis], n, h_values=None)
        assert "phase_proximity" not in obs

    def test_product_state_zero_entropy_batch(self):
        n = 4
        psi = _all_zeros(n)
        obs = compute_all_observables(psi[np.newaxis], n)
        assert obs["entropy"][0] < 1e-10

    def test_zz_matrix_option(self):
        n = 4
        psi = _all_zeros(n)
        obs = compute_all_observables(psi[np.newaxis], n, compute_zz_matrix=True)
        assert "zz_matrix" in obs
        assert obs["zz_matrix"].shape == (1, n, n)


class TestLongRangeZZ:
    """End-to-end correlator ⟨Z_0 Z_{n-1}⟩ and the finite-L order proxy."""

    def test_ghz_long_range_zz_is_one(self):
        n = 6
        assert long_range_zz(_ghz(n), n) == pytest.approx(1.0, abs=1e-12)

    def test_plus_product_long_range_zz_is_zero(self):
        n = 6
        assert long_range_zz(_all_plus(n), n) == pytest.approx(0.0, abs=1e-12)

    def test_zeros_product_long_range_zz_is_one(self):
        n = 5
        assert long_range_zz(_all_zeros(n), n) == pytest.approx(1.0, abs=1e-12)

    def test_batch_keys_and_proxy_nonnegative(self):
        rng = np.random.default_rng(0)
        n, N = 4, 8
        states = np.stack([_ghz(n)] * N)
        obs = compute_all_observables(states, n)
        assert obs["long_range_zz"].shape == (N,)
        assert obs["order_param_proxy"].shape == (N,)
        # proxy = sqrt(clip(lr_zz, 0)) is always finite and >= 0
        assert np.all(obs["order_param_proxy"] >= 0.0)
        assert np.all(np.isfinite(obs["order_param_proxy"]))
        # GHZ: lr_zz = 1  ->  proxy = 1
        assert obs["order_param_proxy"] == pytest.approx(1.0, abs=1e-9)

    def test_ordered_phase_has_larger_long_range_zz(self):
        """TFIM: ordered (h≪J) should have larger ⟨Z_0 Z_{L-1}⟩ than disordered."""
        from qsae.datasets import tfim_ground_states

        n = 8
        ordered = tfim_ground_states(n=n, h_values=np.array([0.2]))[0]
        disordered = tfim_ground_states(n=n, h_values=np.array([3.0]))[0]
        assert long_range_zz(ordered, n) > long_range_zz(disordered, n)
