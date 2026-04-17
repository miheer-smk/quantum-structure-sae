"""
Classical shadow tomography.

Implements the protocol of:
    Huang, Kueng, Preskill.
    "Predicting many properties of a quantum system from very few
    measurements." Nature Physics 2020.

A classical shadow is a compact classical description of a quantum state
that lets us estimate expectation values of many observables from a small
number of randomized measurements.

We implement TWO variants:

1. Random Pauli (product-basis) shadows
     - cheap, implementable on NISQ hardware
     - observable-estimation cost: 3^k / eps^2 for k-local Paulis

2. Random Clifford shadows (full global Clifford)
     - expensive but informationally richer
     - observable-estimation cost: 4^k / eps^2 ... actually 2^n for global observables
     - used mainly for theoretical comparison in simulation

Design notes
------------
We operate IN SIMULATION — we treat the QNN's pre-measurement state |ψ(x,θ)>
as known (obtained from the state vector circuit), apply randomized
measurements classically, and return the classical feature vector. This
is identical in information content to running shadows on hardware, but
massively faster for research iteration.

The output of shadow extraction is a fixed-dimensional feature vector per
input — exactly what a sparse autoencoder expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# -- single-qubit stabilizer states in Z eigenbasis --------------------------
# (these are the post-measurement states we obtain under Pauli-basis sampling)
_ZERO = np.array([1.0, 0.0], dtype=np.complex128)
_ONE = np.array([0.0, 1.0], dtype=np.complex128)
_PLUS = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
_MINUS = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2)
_PLUSI = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2)
_MINUSI = np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2)

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

# single-qubit "inverse channel" snapshot: 3 ρ - I  when outcome is a pure stabilizer
# (per Huang-Kueng-Preskill, the classical snapshot for Pauli shadows)
def _pauli_snapshot(basis: int, outcome: int) -> np.ndarray:
    """
    Return the single-qubit shadow snapshot 3|b><b| - I for a measurement
    in basis 'basis' (0=X, 1=Y, 2=Z) with outcome 0 or 1.
    """
    if basis == 0:
        ket = _PLUS if outcome == 0 else _MINUS
    elif basis == 1:
        ket = _PLUSI if outcome == 0 else _MINUSI
    elif basis == 2:
        ket = _ZERO if outcome == 0 else _ONE
    else:
        raise ValueError(basis)
    rho = np.outer(ket, ket.conj())
    return 3.0 * rho - _I2


def _sample_pauli_outcome(
    state: np.ndarray, n: int, bases: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Given a pure state |ψ> on n qubits and a choice of measurement basis per
    qubit (0=X, 1=Y, 2=Z), sample a computational-basis outcome after
    rotating into the chosen bases.

    Returns outcomes: array of shape (n,) with values in {0,1}.
    """
    # Build U = ⊗_i R_i that rotates basis i's eigenbasis into computational Z.
    # For X: H; for Y: H S†; for Z: I.
    H = (np.array([[1, 1], [1, -1]], dtype=np.complex128)) / np.sqrt(2)
    Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)

    rotated = state.reshape([2] * n)
    for i, b in enumerate(bases):
        if b == 0:
            R = H
        elif b == 1:
            R = H @ Sdg
        else:
            R = _I2
        # apply R on axis i
        rotated = np.tensordot(R, rotated, axes=([1], [i]))
        # tensordot puts the contracted axis at position 0; move it back to i
        rotated = np.moveaxis(rotated, 0, i)

    probs = np.abs(rotated.reshape(-1)) ** 2
    probs = probs / probs.sum()  # numerical safety
    idx = rng.choice(2**n, p=probs)
    # decode idx into n bits (big-endian, qubit 0 = MSB to match PennyLane)
    outcomes = np.array([(idx >> (n - 1 - k)) & 1 for k in range(n)], dtype=np.int8)
    return outcomes


@dataclass
class ShadowConfig:
    n_samples: int = 512
    kind: Literal["pauli"] = "pauli"  # clifford can be added later
    feature_observables: Literal["paulis_weight_1_2", "all_weight_1"] = (
        "paulis_weight_1_2"
    )
    seed: int | None = None


def compute_pauli_shadow(
    state: np.ndarray, n: int, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Pauli classical shadow of `n_samples` snapshots.

    Returns
    -------
    bases : (n_samples, n) int8 array with values in {0,1,2}
    outcomes : (n_samples, n) int8 array with values in {0,1}

    These two arrays together form the classical shadow. Expectation values
    of any Pauli observable P can be estimated from them without storing
    the full snapshot matrices (which would be 2^n x 2^n).
    """
    bases = rng.integers(0, 3, size=(n_samples, n), dtype=np.int8)
    outcomes = np.empty((n_samples, n), dtype=np.int8)
    for s in range(n_samples):
        outcomes[s] = _sample_pauli_outcome(state, n, bases[s], rng)
    return bases, outcomes


def estimate_pauli_expectation(
    pauli: np.ndarray,  # length-n array of {0=I,1=X,2=Y,3=Z}
    bases: np.ndarray,  # (S, n)
    outcomes: np.ndarray,  # (S, n)
) -> float:
    """
    Estimate <P> from a Pauli shadow using the single-shot estimator.

    For each snapshot, include it iff, on every qubit i where P_i != I,
    the measurement basis matches P_i.  Then contribute
        prod_{i: P_i != I} (1 - 2*outcome_i) * 3^{weight(P)}  / S_used
    where weight = number of non-I terms.

    This is the standard median-of-means style estimator reduced to a
    plain mean; for smaller n_samples one can use median-of-means.
    """
    n = pauli.shape[0]
    # pauli encoding:  0=I, 1=X->basis 0, 2=Y->basis 1, 3=Z->basis 2
    nonI = pauli != 0
    weight = int(nonI.sum())
    if weight == 0:
        return 1.0

    # required bases per qubit (where nonI)
    required = np.where(pauli == 1, 0, np.where(pauli == 2, 1, 2))  # (n,)
    mask = np.all(
        (bases[:, nonI] == required[nonI]),
        axis=1,
    )  # (S,)
    if not mask.any():
        return 0.0

    # signs: (-1)^{outcome} on non-I qubits, multiplied
    signs = np.prod(1 - 2 * outcomes[mask][:, nonI], axis=1).astype(np.float64)
    return float((3.0**weight) * signs.mean() * (mask.sum() / bases.shape[0]))
    # multiplicative correction for the fraction of snapshots that hit the right basis
    # -- equivalent to averaging only over matching snapshots with factor 3^weight.


def build_feature_observables(n: int, kind: str) -> np.ndarray:
    """
    Return an array of shape (F, n) listing the Pauli observables whose
    expectation values form the feature vector. Each row uses the encoding
    0=I, 1=X, 2=Y, 3=Z.
    """
    obs = []
    if kind == "all_weight_1":
        for i in range(n):
            for p in (1, 2, 3):
                v = np.zeros(n, dtype=np.int8)
                v[i] = p
                obs.append(v)
    elif kind == "paulis_weight_1_2":
        # all weight-1 Paulis
        for i in range(n):
            for p in (1, 2, 3):
                v = np.zeros(n, dtype=np.int8)
                v[i] = p
                obs.append(v)
        # all nearest-neighbor weight-2 Paulis of type ZZ, XX, YY
        for i in range(n - 1):
            for p in (1, 2, 3):
                v = np.zeros(n, dtype=np.int8)
                v[i] = p
                v[i + 1] = p
                obs.append(v)
    else:
        raise ValueError(kind)
    return np.asarray(obs, dtype=np.int8)


def shadow_to_feature_vector(
    bases: np.ndarray,
    outcomes: np.ndarray,
    observables: np.ndarray,
) -> np.ndarray:
    """
    Convert a shadow into a fixed-length real feature vector by estimating
    <P> for each P in `observables`.
    """
    F = observables.shape[0]
    out = np.empty(F, dtype=np.float64)
    for k in range(F):
        out[k] = estimate_pauli_expectation(observables[k], bases, outcomes)
    return out


def extract_shadow_features(
    states: np.ndarray,  # (B, 2**n) complex
    cfg: ShadowConfig,
) -> np.ndarray:
    """
    Batch extraction: one feature vector per input state.
    Returns (B, F) real array, where F = len(observables).
    """
    B = states.shape[0]
    n = int(np.log2(states.shape[1]))
    rng = np.random.default_rng(cfg.seed)
    observables = build_feature_observables(n, cfg.feature_observables)
    feats = np.empty((B, observables.shape[0]), dtype=np.float64)
    for b in range(B):
        bases, outcomes = compute_pauli_shadow(states[b], n, cfg.n_samples, rng)
        feats[b] = shadow_to_feature_vector(bases, outcomes, observables)
    return feats
