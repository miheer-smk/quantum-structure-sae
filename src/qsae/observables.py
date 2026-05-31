"""
Quantum observables for TFIM ground states.

All functions operate on exact state vectors (pure states) obtained from
exact diagonalisation.  Inputs are (2^n,) complex128 numpy arrays.

Observables implemented
-----------------------
* half_chain_entanglement_entropy  — von Neumann entropy S(A) of the left half-chain
* entanglement_spectrum            — Schmidt values across the bipartition
* zz_correlator                    — ⟨Z_i Z_j⟩ for arbitrary site pairs
* zz_correlator_matrix             — full L×L matrix of ⟨Z_i Z_j⟩
* transverse_magnetization         — per-site ⟨X_i⟩ and total mean
* longitudinal_magnetization       — per-site ⟨Z_i⟩ and total mean
* order_parameter                  — |⟨Z⟩| (ferromagnetic order parameter)
* phase_proximity                  — normalised distance |h - h_c| / h_c
* compute_all_observables          — batch driver: returns dict of arrays

References
----------
* Sachdev, *Quantum Phase Transitions* (2nd ed.), Ch. 5 — TFIM observables.
* Calabrese & Cardy (2004) — entanglement entropy in 1+1D CFTs.
* Huang, Kueng, Preskill (2020) — classical shadow estimation of observables.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _reduce_density_matrix(state: np.ndarray, n: int, n_A: int) -> np.ndarray:
    """
    Compute the reduced density matrix ρ_A by tracing out subsystem B.

    Parameters
    ----------
    state  : (2^n,) complex128 — normalised pure state
    n      : number of qubits
    n_A    : number of qubits in subsystem A (left block, qubits 0..n_A-1)

    Returns
    -------
    rho_A  : (2^n_A, 2^n_A) complex128 density matrix
    """
    dim_A = 1 << n_A       # 2^n_A
    dim_B = 1 << (n - n_A)  # 2^(n-n_A)
    psi = state.reshape(dim_A, dim_B)
    rho_A = psi @ psi.conj().T
    return rho_A


def _pauli_expectation(state: np.ndarray, op: np.ndarray) -> float:
    """⟨ψ|op|ψ⟩ for a Hermitian operator given as a dense matrix."""
    return float(np.real(state.conj() @ (op @ state)))


def _single_qubit_op(
    op2: np.ndarray,  # (2,2) operator on single qubit
    site: int,
    n: int,
) -> np.ndarray:
    """Tensor product: I ⊗ … ⊗ op2 ⊗ … ⊗ I, op on qubit `site`."""
    I2 = np.eye(2, dtype=np.complex128)
    out = None
    for k in range(n):
        factor = op2 if k == site else I2
        out = factor if out is None else np.kron(out, factor)
    return out


def _two_qubit_op(
    opA: np.ndarray,
    opB: np.ndarray,
    siteA: int,
    siteB: int,
    n: int,
) -> np.ndarray:
    """Tensor product placing opA at siteA, opB at siteB, identity elsewhere."""
    I2 = np.eye(2, dtype=np.complex128)
    out = None
    for k in range(n):
        if k == siteA:
            factor = opA
        elif k == siteB:
            factor = opB
        else:
            factor = I2
        out = factor if out is None else np.kron(out, factor)
    return out


_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Entanglement entropy
# ---------------------------------------------------------------------------

def half_chain_entanglement_entropy(
    state: np.ndarray,
    n: int,
    cut: int | None = None,
    eps: float = 1e-14,
) -> float:
    """
    Von Neumann entropy S(A) = -Tr[ρ_A log ρ_A] across a bipartition.

    Parameters
    ----------
    state  : (2^n,) complex128 normalised pure state
    n      : number of qubits
    cut    : bipartition point; A = qubits [0, cut).  Default: n//2.
    eps    : eigenvalue floor for numerical stability in log

    Returns
    -------
    S      : float ≥ 0  (in nats; divide by log(2) for bits)
    """
    if cut is None:
        cut = n // 2
    rho_A = _reduce_density_matrix(state, n, cut)
    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > eps]
    return float(-np.sum(eigvals * np.log(eigvals)))


def entanglement_spectrum(
    state: np.ndarray,
    n: int,
    cut: int | None = None,
) -> np.ndarray:
    """
    Schmidt values (square roots of ρ_A eigenvalues) across the bipartition.

    Returns
    -------
    spectrum : (2^min(cut, n-cut),) float array in descending order
    """
    if cut is None:
        cut = n // 2
    rho_A = _reduce_density_matrix(state, n, cut)
    eigvals = np.sort(np.linalg.eigvalsh(rho_A))[::-1]
    return np.sqrt(np.clip(eigvals, 0, None))


# ---------------------------------------------------------------------------
# Two-point correlators
# ---------------------------------------------------------------------------

def zz_correlator(
    state: np.ndarray,
    n: int,
    i: int,
    j: int,
) -> float:
    """
    ⟨Z_i Z_j⟩ connected (= ⟨Z_i Z_j⟩ − ⟨Z_i⟩⟨Z_j⟩) when i ≠ j;
    returns ⟨Z_i Z_j⟩ (raw) when i == j (= ⟨I⟩ = 1 as a sanity check).

    For the TFIM order parameter the raw ⟨Z_i Z_j⟩ at large |i-j| is used.
    """
    ZZ = _two_qubit_op(_Z, _Z, i, j, n)
    return _pauli_expectation(state, ZZ)


def zz_correlator_matrix(state: np.ndarray, n: int) -> np.ndarray:
    """
    Full L×L symmetric matrix of ⟨Z_i Z_j⟩.

    Diagonal entries are 1 (⟨Z_i^2⟩ = ⟨I⟩ = 1 for qubit systems).
    """
    corr = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            c = zz_correlator(state, n, i, j)
            corr[i, j] = c
            corr[j, i] = c
    return corr


def nearest_neighbor_zz(state: np.ndarray, n: int) -> np.ndarray:
    """
    ⟨Z_i Z_{i+1}⟩ for i = 0, …, n-2.

    This is the standard ferromagnetic order parameter density for open-BC TFIM.
    """
    return np.array([zz_correlator(state, n, i, i + 1) for i in range(n - 1)])


# ---------------------------------------------------------------------------
# Magnetization
# ---------------------------------------------------------------------------

def transverse_magnetization(state: np.ndarray, n: int) -> np.ndarray:
    """
    Per-site transverse magnetization ⟨X_i⟩ for i = 0, …, n-1.

    In the TFIM, the paramagnetic phase (h ≫ J) is characterised by ⟨X_i⟩ → 1.
    """
    return np.array([
        _pauli_expectation(state, _single_qubit_op(_X, i, n))
        for i in range(n)
    ])


def longitudinal_magnetization(state: np.ndarray, n: int) -> np.ndarray:
    """
    Per-site longitudinal magnetization ⟨Z_i⟩ for i = 0, …, n-1.

    Vanishes in the thermodynamic limit for both phases of the 1D TFIM
    (Z₂ symmetry), but finite-size effects make it nonzero.
    """
    return np.array([
        _pauli_expectation(state, _single_qubit_op(_Z, i, n))
        for i in range(n)
    ])


def order_parameter(state: np.ndarray, n: int) -> float:
    """
    Ferromagnetic order parameter: mean |⟨Z_i⟩| over all sites.

    Positive in the ordered (h < h_c) phase, approaches 0 in the
    paramagnetic (h > h_c) phase.
    """
    return float(np.mean(np.abs(longitudinal_magnetization(state, n))))


# ---------------------------------------------------------------------------
# Phase proximity
# ---------------------------------------------------------------------------

def phase_proximity(h: float | np.ndarray, h_c: float = 1.0) -> float | np.ndarray:
    """
    Normalised signed distance from the quantum critical point.

        δ = (h - h_c) / h_c

    Negative in the ferromagnetic phase, positive in the paramagnetic phase.
    |δ| → 0 at criticality; |δ| → 1 is one coupling-length away.
    """
    return (np.asarray(h) - h_c) / h_c


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def compute_all_observables(
    states: np.ndarray,   # (N, 2^n) complex128
    n: int,
    h_values: np.ndarray | None = None,  # (N,) per-sample mean field
    h_c: float = 1.0,
    compute_zz_matrix: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute all implemented observables for N ground states.

    Parameters
    ----------
    states          : (N, 2^n) complex128 array of normalised pure states
    n               : number of qubits
    h_values        : (N,) float — mean transverse field per sample (optional)
    h_c             : quantum critical point field value (default 1.0)
    compute_zz_matrix : if True, also compute the full L×L ZZ matrix per sample
                        (expensive: O(N L² 4^n))

    Returns
    -------
    obs : dict with keys
        "entropy"         : (N,) von Neumann entropy (half-chain, nats)
        "nn_zz"           : (N, n-1) nearest-neighbour ZZ correlators
        "mean_nn_zz"      : (N,) mean of nn_zz over bonds
        "transverse_mag"  : (N, n) per-site ⟨X_i⟩
        "mean_x"          : (N,) site-averaged ⟨X⟩
        "order_param"     : (N,) ferromagnetic order parameter |⟨Z⟩|
        "phase_proximity" : (N,) δ = (h - h_c)/h_c  [only if h_values given]
        "zz_matrix"       : (N, n, n) full ZZ matrix [only if compute_zz_matrix]
    """
    N = states.shape[0]
    entropy       = np.empty(N)
    nn_zz         = np.empty((N, n - 1))
    transverse_m  = np.empty((N, n))
    order_param   = np.empty(N)

    for k in range(N):
        psi = states[k]
        entropy[k]      = half_chain_entanglement_entropy(psi, n)
        nn_zz[k]        = nearest_neighbor_zz(psi, n)
        transverse_m[k] = transverse_magnetization(psi, n)
        order_param[k]  = order_parameter(psi, n)

    obs: dict[str, np.ndarray] = {
        "entropy":        entropy,
        "nn_zz":          nn_zz,
        "mean_nn_zz":     nn_zz.mean(axis=1),
        "transverse_mag": transverse_m,
        "mean_x":         transverse_m.mean(axis=1),
        "order_param":    order_param,
    }

    if h_values is not None:
        obs["phase_proximity"] = phase_proximity(h_values, h_c=h_c)

    if compute_zz_matrix:
        zz_mat = np.empty((N, n, n))
        for k in range(N):
            zz_mat[k] = zz_correlator_matrix(states[k], n)
        obs["zz_matrix"] = zz_mat

    return obs
