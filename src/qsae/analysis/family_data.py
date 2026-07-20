"""
Data generation for the Hamiltonian-diversity families (Phase 0.7).

Each family exposes the same interface used by the diversity driver:
  * energies(input_fields)           -> (N,) ground-state energies (training)
  * states(input_fields)             -> (N,), (N, 2^L) energies + state vectors (eval)
  * sample_inputs(rng, n)            -> (N, input_dim) disorder draws
  * observables(states, L)           -> dict of per-sample beyond-input observables
The transformer input is the disordered parameter vector; only the trained
weights vary across seeds, the eval yardstick is held fixed.
"""

from __future__ import annotations

import numpy as np

from qsae.observables import (
    _z_signs,
    half_chain_entanglement_entropy,
    long_range_zz_fast,
    nearest_neighbor_zz_fast,
    transverse_magnetization_fast,
)
from qsae.physics.hamiltonians import (
    annni_ground_energies,
    annni_ground_states,
    xxz_ground_energies,
    xxz_ground_states,
)


def staggered_structure_factor(state: np.ndarray, L: int) -> float:
    """Neel structure factor m_s^2 = <(sum_i (-1)^i Z_i)^2> / L^2, built from ZZ
    correlators (beyond-input: it needs only <Z_i Z_j>, and <Z_i>=0 here)."""
    dim = state.shape[0]
    p = (state.conj() * state).real
    M = np.zeros(dim)
    for i in range(L):
        M += ((-1) ** i) * _z_signs(L, i, dim)
    return float(np.sum(p * M * M) / L ** 2)


def max_abs_single_site_z(states: np.ndarray, L: int) -> float:
    """max_k max_i |<Z_i>| over a batch of states -- the beyond-input protection
    check reported before any probe results."""
    dim = 1 << L
    out = 0.0
    for k in range(states.shape[0]):
        p = (states[k].conj() * states[k]).real
        mz = max(abs(float(np.sum(p * _z_signs(L, i, dim)))) for i in range(L))
        out = max(out, mz)
    return out


def compute_observables(states: np.ndarray, L: int) -> dict[str, np.ndarray]:
    """Beyond-input observable set shared across families."""
    N = states.shape[0]
    entropy = np.empty(N)
    nn = np.empty((N, L - 1))
    tmag = np.empty((N, L))
    lrz = np.empty(N)
    ssf = np.empty(N)
    for k in range(N):
        psi = states[k]
        entropy[k] = half_chain_entanglement_entropy(psi, L)
        nn[k] = nearest_neighbor_zz_fast(psi, L)
        tmag[k] = transverse_magnetization_fast(psi, L)
        lrz[k] = long_range_zz_fast(psi, L)
        ssf[k] = staggered_structure_factor(psi, L)
    return {"entropy": entropy, "mean_nn_zz": nn.mean(1), "mean_x": tmag.mean(1),
            "long_range_zz": lrz, "staggered_sf": ssf}


class XXZFamily:
    """XXZ with per-bond Jz disorder. Input = {Jz_i} (length L-1)."""

    name = "xxz"

    def __init__(self, L: int, jz_min: float, jz_max: float):
        self.L = L
        self.input_dim = L - 1
        self.lo, self.hi = jz_min, jz_max

    def sample_inputs(self, rng, n):
        return rng.uniform(self.lo, self.hi, size=(n, self.input_dim))

    def energies(self, jz):
        return xxz_ground_energies(jz)

    def states(self, jz):
        return xxz_ground_states(jz)


class ANNNIFamily:
    """Transverse-field ANNNI (non-integrable). Input = {h_i} (length L)."""

    name = "annni"

    def __init__(self, L: int, h_min: float, h_max: float, J1: float, J2: float):
        self.L = L
        self.input_dim = L
        self.lo, self.hi = h_min, h_max
        self.J1, self.J2 = J1, J2

    def sample_inputs(self, rng, n):
        return rng.uniform(self.lo, self.hi, size=(n, self.input_dim))

    def energies(self, h):
        return annni_ground_energies(h, J1=self.J1, J2=self.J2)

    def states(self, h):
        return annni_ground_states(h, J1=self.J1, J2=self.J2)


def make_family(cfg: dict):
    fam = cfg["family"]
    if fam == "xxz":
        return XXZFamily(cfg["L"], cfg["jz_min"], cfg["jz_max"])
    if fam == "annni":
        return ANNNIFamily(cfg["L"], cfg["h_min"], cfg["h_max"], cfg["J1"], cfg["J2"])
    raise ValueError(f"unknown family {fam!r}")
