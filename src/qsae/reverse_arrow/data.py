"""
TFIMDataset — generates (h_i, E_0) pairs for the disordered 1D TFIM.

Hamiltonian  H = -J Σ_i Z_i Z_{i+1}  -  Σ_i h_i X_i
  with per-site fields h_i drawn independently from Uniform(h_min, h_max).

Ground-state energy is found via exact diagonalisation of the 2^L × 2^L
Hamiltonian.  For L=8, each matrix is 256×256; batched eigvalsh is fast on CPU.

Usage
-----
    train_ds, val_ds, test_ds = make_splits(L=8, n_train=5000, seed=0)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    for h_batch, e_batch in loader:
        ...

Caching
-------
Pass `cache_path` to `make_splits()` to persist/reload from a .pt file so
the expensive ED step runs only once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Exact-diagonalisation kernel (pure numpy, no scipy sparse)
# Dense np.linalg.eigvalsh is used instead of scipy.sparse.linalg.eigsh because
# for L=8 the Hilbert space is only 256×256 — batched dense eigvalsh on (N,256,256)
# is ~10× faster than N serial sparse eigsh calls.
# ---------------------------------------------------------------------------

def _build_zz_xx_dense(L: int, J: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
      zz_part  : (2^L, 2^L) real array  =  -J Σ Z_i Z_{i+1}
      x_ops    : (L, 2^L, 2^L) real array,  x_ops[i] = X_i ⊗ I_{rest}
    All matrices are real-symmetric float64.
    """
    dim = 1 << L  # 2^L

    # --- ZZ part (diagonal in computational basis) -----------------------
    # For each basis state |s⟩, diagonal element = -J Σ_i sign(s_i) sign(s_{i+1})
    zz_diag = np.zeros(dim, dtype=np.float64)
    for i in range(L - 1):
        # bit i and bit i+1 of each state index
        bi = (np.arange(dim) >> i) & 1       # 0 or 1
        bi1 = (np.arange(dim) >> (i + 1)) & 1
        # Z eigenvalue: 0→+1, 1→-1
        si = 1.0 - 2.0 * bi
        si1 = 1.0 - 2.0 * bi1
        zz_diag += -J * si * si1
    zz_part = np.diag(zz_diag)

    # --- X_i operators (off-diagonal: flip bit i) ------------------------
    x_ops = np.zeros((L, dim, dim), dtype=np.float64)
    for i in range(L):
        flip_mask = 1 << i
        for s in range(dim):
            s_flipped = s ^ flip_mask
            x_ops[i, s, s_flipped] = 1.0

    return zz_part, x_ops


def compute_ground_energies(
    h_fields: np.ndarray,  # (N, L) per-site fields
    J: float = 1.0,
    chunk_size: int = 512,
) -> np.ndarray:
    """
    Compute ground-state energies for N disorder realisations in h_fields.

    Parameters
    ----------
    h_fields  : (N, L) float64 array of per-site transverse fields
    J         : Ising coupling strength (default 1.0)
    chunk_size: number of Hamiltonians diagonalised per numpy batch call

    Returns
    -------
    energies : (N,) float64 array
    """
    N, L = h_fields.shape
    zz_part, x_ops = _build_zz_xx_dense(L, J=J)

    energies = np.empty(N, dtype=np.float64)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        h_chunk = h_fields[start:end]  # (chunk, L)

        # H_batch[n] = zz_part - sum_i h_i * X_i
        # einsum: (chunk, L) x (L, dim, dim) → (chunk, dim, dim)
        H_batch = (
            zz_part[np.newaxis]                          # (1, dim, dim)
            - np.einsum("nl,lij->nij", h_chunk, x_ops)  # (chunk, dim, dim)
        )  # shape: (chunk, dim, dim)

        # eigvalsh returns eigenvalues in ascending order; [0] = ground state
        energies[start:end] = np.linalg.eigvalsh(H_batch)[:, 0]

    return energies


# ---------------------------------------------------------------------------
# Sparse ground-state solver (scales past the dense L=8 ceiling)
# ---------------------------------------------------------------------------
# The dense path above materialises x_ops of shape (L, 2^L, 2^L) — 1.6 GB at
# L=12 — so it cannot reach the larger systems needed for the scaling study.
# The sparse path builds each Pauli string as a scipy.sparse operator once
# (O(2^L) nnz), then forms H = -Σ_b J_b Z_b Z_{b+1} - Σ_i h_i X_i per disorder
# realisation and finds the ground state via Lanczos (eigsh, k=1). It supports
# per-site fields h_i *and* per-bond couplings J_b (the latter for the
# disordered-coupling / non-integrable extensions).

def _sparse_pauli_ops(L: int):
    """Return (zz_bonds, x_sites): lists of sparse Z_iZ_{i+1} and X_i operators."""
    from scipy.sparse import identity, kron, csr_matrix

    I2 = identity(2, format="csr", dtype=np.float64)
    X = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    Z = csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))

    def op_at(op, site):
        out = None
        for k in range(L):
            factor = op if k == site else I2
            out = factor if out is None else kron(out, factor, format="csr")
        return out

    x_sites = [op_at(X, i) for i in range(L)]
    z_sites = [op_at(Z, i) for i in range(L)]
    zz_bonds = [(z_sites[i] @ z_sites[i + 1]).tocsr() for i in range(L - 1)]
    return zz_bonds, x_sites, z_sites


def _assemble_sparse_H(zz_bonds, x_sites, z_sites, h: np.ndarray, J, g):
    """H = -Σ_b J_b Z_b Z_{b+1} - Σ_i h_i X_i - Σ_i g_i Z_i for one realisation.

    g is the longitudinal field. g = 0 is the (integrable) TFIM; g ≠ 0 gives the
    non-integrable mixed-field Ising model.
    """
    L = len(x_sites)
    Jb = np.broadcast_to(np.asarray(J, dtype=np.float64), (L - 1,))
    gb = np.broadcast_to(np.asarray(g, dtype=np.float64), (L,))
    H = None
    for b in range(L - 1):
        term = -Jb[b] * zz_bonds[b]
        H = term if H is None else H + term
    for i in range(L):
        H = H + (-float(h[i])) * x_sites[i]
        if gb[i] != 0.0:
            H = H + (-float(gb[i])) * z_sites[i]
    return H.tocsr()


def compute_ground_states_sparse(
    h_fields: np.ndarray,          # (N, L) per-site transverse fields
    J_fields=1.0,                  # scalar, (L-1,), or (N, L-1) per-bond couplings
    g_fields=0.0,                  # scalar, (L,), or (N, L) longitudinal field(s)
    return_states: bool = True,
):
    """
    Ground-state energies (and optionally vectors) for N disorder realisations via
    sparse Lanczos — memory-safe for L up to ~14 on 16 GB RAM.

    Setting `g_fields` ≠ 0 adds a longitudinal −Σ g_i Z_i term, turning the
    integrable TFIM into the non-integrable mixed-field Ising model.

    Returns
    -------
    energies : (N,) float64
    states   : (N, 2**L) complex128   (only if return_states)
    """
    from scipy.sparse.linalg import eigsh

    N, L = h_fields.shape
    zz_bonds, x_sites, z_sites = _sparse_pauli_ops(L)
    J_arr = np.asarray(J_fields, dtype=np.float64)
    g_arr = np.asarray(g_fields, dtype=np.float64)

    energies = np.empty(N, dtype=np.float64)
    states = np.empty((N, 1 << L), dtype=np.complex128) if return_states else None
    for k in range(N):
        Jk = J_arr[k] if J_arr.ndim == 2 else J_arr
        gk = g_arr[k] if g_arr.ndim == 2 else g_arr
        H = _assemble_sparse_H(zz_bonds, x_sites, z_sites, h_fields[k], Jk, gk)
        w, v = eigsh(H, k=1, which="SA")
        energies[k] = float(w[0])
        if return_states:
            states[k] = v[:, 0]
    return (energies, states) if return_states else energies


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TFIMDataset(Dataset):
    """
    PyTorch Dataset of (h, E_0) pairs for the disordered 1D TFIM.

    Parameters
    ----------
    h_fields  : (N, L) float tensor
    energies  : (N,)   float tensor
    """

    def __init__(self, h_fields: torch.Tensor, energies: torch.Tensor) -> None:
        assert h_fields.shape[0] == energies.shape[0]
        self.h_fields = h_fields.float()
        self.energies = energies.float()

    def __len__(self) -> int:
        return self.h_fields.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.h_fields[idx], self.energies[idx]

    @property
    def L(self) -> int:
        return self.h_fields.shape[1]


# ---------------------------------------------------------------------------
# Split factory
# ---------------------------------------------------------------------------

def make_splits(
    L: int = 8,
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    h_min: float = 0.1,
    h_max: float = 2.0,
    J: float = 1.0,
    seed: int = 0,
    cache_path: Optional[Path | str] = None,
    chunk_size: int = 512,
) -> tuple[TFIMDataset, TFIMDataset, TFIMDataset]:
    """
    Generate (or load from cache) train/val/test splits.

    Returns
    -------
    train_ds, val_ds, test_ds : TFIMDataset triples
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            print(f"[data] loading cached dataset from {cache_path}")
            saved = torch.load(cache_path, weights_only=False)
            return (
                TFIMDataset(saved["h_train"], saved["e_train"]),
                TFIMDataset(saved["h_val"],   saved["e_val"]),
                TFIMDataset(saved["h_test"],  saved["e_test"]),
            )

    rng = np.random.default_rng(seed)
    N_total = n_train + n_val + n_test

    print(f"[data] generating {N_total} TFIM samples  (L={L}, J={J}, "
          f"h∈[{h_min},{h_max}])  …")
    h_fields = rng.uniform(h_min, h_max, size=(N_total, L)).astype(np.float64)

    print(f"[data] running exact diagonalisation (chunk_size={chunk_size}) …")
    energies = compute_ground_energies(h_fields, J=J, chunk_size=chunk_size)

    # Convert to tensors
    h_t = torch.from_numpy(h_fields).float()
    e_t = torch.from_numpy(energies).float()

    # Split
    h_train, h_val, h_test = (
        h_t[:n_train], h_t[n_train:n_train+n_val], h_t[n_train+n_val:]
    )
    e_train, e_val, e_test = (
        e_t[:n_train], e_t[n_train:n_train+n_val], e_t[n_train+n_val:]
    )

    print(
        f"[data] splits: train={len(h_train)}, val={len(h_val)}, test={len(h_test)}\n"
        f"[data] energy stats — "
        f"mean={e_train.mean():.4f}  std={e_train.std():.4f}  "
        f"min={e_train.min():.4f}  max={e_train.max():.4f}"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "h_train": h_train, "e_train": e_train,
                "h_val":   h_val,   "e_val":   e_val,
                "h_test":  h_test,  "e_test":  e_test,
                "meta": {"L": L, "J": J, "h_min": h_min, "h_max": h_max, "seed": seed},
            },
            cache_path,
        )
        print(f"[data] cached to {cache_path}")

    return (
        TFIMDataset(h_train, e_train),
        TFIMDataset(h_val,   e_val),
        TFIMDataset(h_test,  e_test),
    )
