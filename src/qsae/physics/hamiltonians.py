"""
Hamiltonian builders for the Hamiltonian-diversity study (Phase 0.7).

Two new families beyond the disordered TFIM, each chosen so that a genuinely
NON-LOCAL, beyond-input order observable exists (the single-site ``<Z_i>``
vanishes by an unbroken symmetry, so the two-point correlator cannot factorise
into input-linear pieces):

* XXZ with per-bond Jz disorder  (integrable, but U(1) instead of TFIM's Z2)::

      H = sum_i  X_i X_{i+1} + Y_i Y_{i+1} + Jz_i Z_i Z_{i+1}          (J_xy = 1)

  Input = the disordered anisotropy vector {Jz_i} (length L-1). The global
  pi-rotation about x (prod_i X_i) sends Z_i -> -Z_i and leaves H invariant, so
  a non-degenerate ground state has <Z_i> = 0.

* Transverse-field ANNNI  (genuinely non-integrable via next-nearest ZZ
  frustration, while KEEPING the Z2 spin-flip symmetry so the order proxy stays
  beyond-input -- unlike the draft's symmetry-broken mixed-field control)::

      H = -J1 sum_i Z_i Z_{i+1} - J2 sum_i Z_i Z_{i+2} - sum_i h_i X_i

  Input = the disordered transverse fields {h_i} (length L). At J2 = 0 this is
  exactly the disordered TFIM.

Conventions
-----------
Pauli operators with eigenvalues +-1 (matching ``qsae.observables``). Open
boundaries. All Hamiltonians here are real-symmetric (the XX+YY term is built
from real S^+/S^- ladder operators), so we use float64 throughout. Site 0 is the
most-significant qubit, matching the np.kron convention in ``qsae.observables``.
"""

from __future__ import annotations

import numpy as np

# single-qubit building blocks (real)
_Z2 = np.array([[1.0, 0.0], [0.0, -1.0]])
_X2 = np.array([[0.0, 1.0], [1.0, 0.0]])
_SP = np.array([[0.0, 1.0], [0.0, 0.0]])   # sigma^+ = |0><1|
_SM = np.array([[0.0, 0.0], [1.0, 0.0]])   # sigma^- = |1><0|
_I2 = np.eye(2)


def _op_at(op2: np.ndarray, site: int, L: int) -> np.ndarray:
    """Embed a single-site 2x2 operator at ``site`` in the 2^L Hilbert space."""
    out = None
    for k in range(L):
        factor = op2 if k == site else _I2
        out = factor if out is None else np.kron(out, factor)
    return out


def _two_site(opA: np.ndarray, opB: np.ndarray, i: int, j: int, L: int) -> np.ndarray:
    """Embed opA at site i and opB at site j (i != j) in the 2^L space."""
    out = None
    for k in range(L):
        factor = opA if k == i else (opB if k == j else _I2)
        out = factor if out is None else np.kron(out, factor)
    return out


# ---------------------------------------------------------------------------
# Reusable dense operator sets (built once per L)
# ---------------------------------------------------------------------------

def _z_ops(L: int) -> np.ndarray:
    """Stack of single-site Z operators, shape (L, 2^L, 2^L)."""
    return np.stack([_op_at(_Z2, i, L) for i in range(L)])


def _x_ops(L: int) -> np.ndarray:
    """Stack of single-site X operators, shape (L, 2^L, 2^L)."""
    return np.stack([_op_at(_X2, i, L) for i in range(L)])


def _zz_bonds(L: int, offset: int) -> np.ndarray:
    """Stack of Z_i Z_{i+offset} operators for i = 0 .. L-1-offset."""
    return np.stack([_two_site(_Z2, _Z2, i, i + offset, L)
                     for i in range(L - offset)])


def _xy_hopping_sum(L: int) -> np.ndarray:
    """Sum over nn bonds of (X_i X_{i+1} + Y_i Y_{i+1}) = 2(S+_i S-_{i+1} + h.c.),
    built from real ladder operators so the result is real-symmetric."""
    dim = 1 << L
    out = np.zeros((dim, dim))
    for i in range(L - 1):
        out += 2.0 * (_two_site(_SP, _SM, i, i + 1, L)
                      + _two_site(_SM, _SP, i, i + 1, L))
    return out


# ---------------------------------------------------------------------------
# XXZ with per-bond Jz disorder
# ---------------------------------------------------------------------------

def xxz_hamiltonian(jz: np.ndarray, L: int | None = None) -> np.ndarray:
    """Dense XXZ Hamiltonian for one anisotropy realisation.

    Parameters
    ----------
    jz : (L-1,) per-bond Z-anisotropies.
    L  : chain length; inferred as len(jz)+1 if omitted.
    """
    jz = np.asarray(jz, dtype=np.float64)
    if L is None:
        L = jz.shape[0] + 1
    H = _xy_hopping_sum(L)
    zz = _zz_bonds(L, 1)
    H = H + np.einsum("b,bij->ij", jz, zz)
    return H


def xxz_ground_energies(jz_fields: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Ground-state energies for N anisotropy realisations, shape (N,).

    jz_fields : (N, L-1). Uses batched dense ``eigvalsh`` (L=8 -> 256x256)."""
    jz_fields = np.asarray(jz_fields, dtype=np.float64)
    N, Lm1 = jz_fields.shape
    L = Lm1 + 1
    xy = _xy_hopping_sum(L)
    zz = _zz_bonds(L, 1)
    energies = np.empty(N)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        H = xy[None] + np.einsum("nb,bij->nij", jz_fields[s:e], zz)
        energies[s:e] = np.linalg.eigvalsh(H)[:, 0]
    return energies


def xxz_ground_states(jz_fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground energies and state vectors, shapes (N,), (N, 2^L). Dense ``eigh``."""
    jz_fields = np.asarray(jz_fields, dtype=np.float64)
    N, Lm1 = jz_fields.shape
    L = Lm1 + 1
    xy = _xy_hopping_sum(L)
    zz = _zz_bonds(L, 1)
    energies = np.empty(N)
    states = np.empty((N, 1 << L), dtype=np.complex128)
    for k in range(N):
        H = xy + np.einsum("b,bij->ij", jz_fields[k], zz)
        w, v = np.linalg.eigh(H)
        energies[k] = w[0]
        states[k] = v[:, 0]
    return energies, states


# ---------------------------------------------------------------------------
# Transverse-field ANNNI  (non-integrable via J2 frustration, Z2 preserved)
# ---------------------------------------------------------------------------

def annni_hamiltonian(h: np.ndarray, J1: float = 1.0, J2: float = 0.3) -> np.ndarray:
    """Dense transverse-field ANNNI Hamiltonian for one field realisation.

    H = -J1 sum ZZ_nn - J2 sum ZZ_nnn - sum_i h_i X_i, open BC. J2=0 -> TFIM."""
    h = np.asarray(h, dtype=np.float64)
    L = h.shape[0]
    H = -J1 * _zz_bonds(L, 1).sum(axis=0)
    if L > 2 and J2 != 0.0:
        H = H - J2 * _zz_bonds(L, 2).sum(axis=0)
    H = H - np.einsum("i,ijk->jk", h, _x_ops(L))
    return H


def annni_ground_energies(h_fields: np.ndarray, J1: float = 1.0, J2: float = 0.3,
                          chunk_size: int = 512) -> np.ndarray:
    """Ground-state energies for N field realisations, shape (N,). h_fields:(N,L)."""
    h_fields = np.asarray(h_fields, dtype=np.float64)
    N, L = h_fields.shape
    fixed = -J1 * _zz_bonds(L, 1).sum(axis=0)
    if L > 2 and J2 != 0.0:
        fixed = fixed - J2 * _zz_bonds(L, 2).sum(axis=0)
    x_ops = _x_ops(L)
    energies = np.empty(N)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        H = fixed[None] - np.einsum("ni,ijk->njk", h_fields[s:e], x_ops)
        energies[s:e] = np.linalg.eigvalsh(H)[:, 0]
    return energies


def annni_ground_states(h_fields: np.ndarray, J1: float = 1.0, J2: float = 0.3
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Ground energies and state vectors, shapes (N,), (N, 2^L). Dense ``eigh``."""
    h_fields = np.asarray(h_fields, dtype=np.float64)
    N, L = h_fields.shape
    fixed = -J1 * _zz_bonds(L, 1).sum(axis=0)
    if L > 2 and J2 != 0.0:
        fixed = fixed - J2 * _zz_bonds(L, 2).sum(axis=0)
    x_ops = _x_ops(L)
    energies = np.empty(N)
    states = np.empty((N, 1 << L), dtype=np.complex128)
    for k in range(N):
        H = fixed - np.einsum("i,ijk->jk", h_fields[k], x_ops)
        w, v = np.linalg.eigh(H)
        energies[k] = w[0]
        states[k] = v[:, 0]
    return energies, states


# ---------------------------------------------------------------------------
# Independent analytic reference (used by tests, not by the pipeline)
# ---------------------------------------------------------------------------

def free_xx_ground_energy(L: int) -> float:
    """Exact ground energy of the open XX chain (XXZ at Jz=0) via free fermions.

    XX+YY on a bond = 2(S+ S- + h.c.) -> hopping amplitude 2. The open-chain
    single-particle spectrum is eps_k = 4 cos(k pi / (L+1)), k = 1..L; the ground
    energy fills every negative mode.
    """
    k = np.arange(1, L + 1)
    eps = 4.0 * np.cos(k * np.pi / (L + 1))
    return float(eps[eps < 0].sum())
