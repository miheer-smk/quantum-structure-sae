"""
Validation tests for the diversity-family Hamiltonian builders (Phase 0.7).

Gate before any training: the non-integrable arm is meaningless if the ANNNI
builder is subtly wrong, so we check every named limit and cross-check the XXZ
builder against an INDEPENDENT complex-Pauli construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from qsae.observables import longitudinal_magnetization_fast
from qsae.physics.hamiltonians import (
    annni_ground_energies,
    annni_ground_states,
    free_xx_ground_energy,
    xxz_ground_energies,
    xxz_ground_states,
    xxz_hamiltonian,
)
from qsae.reverse_arrow.data import compute_ground_energies


def _independent_xxz(jz: np.ndarray, L: int) -> np.ndarray:
    """XXZ built from complex Pauli X/Y/Z (different construction than the module,
    which uses real S+/S- ladder operators) -- a true cross-implementation check."""
    Xc = np.array([[0, 1], [1, 0]], dtype=complex)
    Yc = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Zc = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def at(op, i):
        out = None
        for k in range(L):
            f = op if k == i else I2
            out = f if out is None else np.kron(out, f)
        return out

    dim = 1 << L
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(L - 1):
        H += at(Xc, i) @ at(Xc, i + 1)
        H += at(Yc, i) @ at(Yc, i + 1)
        H += jz[i] * (at(Zc, i) @ at(Zc, i + 1))
    return H


class TestXXZFreeFermionLimit:
    """XXZ at Jz=0 reduces to the free XX chain."""

    @pytest.mark.parametrize("L", [2, 4, 6, 8])
    def test_jz0_matches_free_fermion(self, L):
        e = xxz_ground_energies(np.zeros((1, L - 1)))[0]
        assert e == pytest.approx(free_xx_ground_energy(L), abs=1e-9)

    def test_free_xx_reference_hand_value_L2(self):
        # XX+YY on two sites: eigenvalues {0,0,+2,-2}; ground = -2.
        assert free_xx_ground_energy(2) == pytest.approx(-2.0, abs=1e-12)


class TestXXZBuilderCorrectness:
    """Cross-check the real-ladder builder against complex Pauli, plus the
    Heisenberg-point energy."""

    def test_matches_independent_complex_pauli(self):
        rng = np.random.default_rng(0)
        for L in (4, 6):
            jz = rng.uniform(0.0, 2.0, size=L - 1)
            wm = np.linalg.eigvalsh(xxz_hamiltonian(jz))
            wi = np.linalg.eigvalsh(_independent_xxz(jz, L))
            assert np.allclose(wm, wi, atol=1e-10)

    def test_heisenberg_point_L2_exact(self):
        # XXZ at Jz=1, L=2 is the Heisenberg dimer: singlet -3, triplet +1 (Pauli).
        assert np.linalg.eigvalsh(xxz_hamiltonian(np.array([1.0])))[0] == pytest.approx(
            -3.0, abs=1e-10)

    def test_heisenberg_point_spectrum_matches_independent(self):
        L = 5
        wm = np.linalg.eigvalsh(xxz_hamiltonian(np.ones(L - 1)))
        wi = np.linalg.eigvalsh(_independent_xxz(np.ones(L - 1), L))
        assert np.allclose(wm, wi, atol=1e-10)


class TestANNNIReducesToTFIM:
    """ANNNI at J2=0 is exactly the disordered TFIM used elsewhere in the repo."""

    def test_j2_zero_matches_tfim_kernel(self):
        rng = np.random.default_rng(1)
        h = rng.uniform(0.1, 2.0, size=(24, 8))
        e_annni = annni_ground_energies(h, J1=1.0, J2=0.0)
        e_tfim = compute_ground_energies(h, J=1.0)
        assert np.allclose(e_annni, e_tfim, atol=1e-9)

    def test_frustration_changes_the_hamiltonian(self):
        rng = np.random.default_rng(2)
        h = rng.uniform(0.1, 2.0, size=(8, 8))
        assert not np.allclose(annni_ground_energies(h, J2=0.0),
                               annni_ground_energies(h, J2=0.3))


class TestGroundStateConsistency:
    """Batched eigvalsh energies agree with the eigh state-returning path."""

    def test_xxz(self):
        rng = np.random.default_rng(3)
        jz = rng.uniform(0.0, 2.0, size=(6, 7))
        assert np.allclose(xxz_ground_energies(jz), xxz_ground_states(jz)[0], atol=1e-9)

    def test_annni(self):
        rng = np.random.default_rng(4)
        h = rng.uniform(0.1, 2.0, size=(6, 8))
        assert np.allclose(annni_ground_energies(h, J2=0.3),
                           annni_ground_states(h, J2=0.3)[0], atol=1e-9)


class TestBeyondInputProtection:
    """The beyond-input claim rests on <Z_i> vanishing so the correlator cannot
    factorise. Verify numerically on real ground states (not just by symmetry)."""

    def _max_abs_z(self, states, L):
        return max(float(np.max(np.abs(longitudinal_magnetization_fast(states[k], L))))
                   for k in range(states.shape[0]))

    def test_xxz_single_site_z_vanishes(self):
        rng = np.random.default_rng(5)
        jz = rng.uniform(0.5, 2.0, size=(32, 7))
        _, states = xxz_ground_states(jz)
        assert self._max_abs_z(states, 8) < 1e-6

    def test_annni_single_site_z_vanishes(self):
        rng = np.random.default_rng(6)
        h = rng.uniform(0.1, 2.0, size=(32, 8))
        _, states = annni_ground_states(h, J2=0.3)
        assert self._max_abs_z(states, 8) < 1e-6
