"""
Datasets for QML interpretability experiments.

- Bars-and-Stripes (2x2, 3x3): classic QML benchmark, classifies whether
  a binary image consists of uniform rows ("bars") or uniform columns
  ("stripes").
- MNIST-4x4: downsampled MNIST, 16 features -> 8 or 16 qubits depending
  on encoding.
- 1D TFIM ground states: classically-generable for small n; used as the
  "genuine quantum data" benchmark.

Everything returns torch tensors.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Bars and Stripes
# ---------------------------------------------------------------------------
def bars_and_stripes(size: int = 2, n_per_class: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the full Bars-and-Stripes dataset at resolution `size`.

    Returns
    -------
    x : (N, size*size) float tensor with values in {-1, +1}
    y : (N,) long tensor with labels {0 = bars, 1 = stripes}
    """
    patterns = []
    labels = []
    # bars: each row is uniform
    for mask in range(1, 2**size):  # exclude all-zero (boring) and include all-one
        row = [(mask >> i) & 1 for i in range(size)]
        img = np.tile(np.array(row, dtype=np.float32), (size, 1))
        patterns.append(img.flatten())
        labels.append(0)
    # stripes: each column uniform
    for mask in range(1, 2**size):
        col = [(mask >> i) & 1 for i in range(size)]
        img = np.tile(np.array(col, dtype=np.float32).reshape(-1, 1), (1, size))
        patterns.append(img.flatten())
        labels.append(1)

    x = np.asarray(patterns, dtype=np.float32) * 2.0 - 1.0  # map {0,1} -> {-1,+1}
    y = np.asarray(labels, dtype=np.int64)

    # rescale to angles in [-pi/2, pi/2] for RY-style encoding
    x = x * (np.pi / 2)

    if n_per_class is not None:
        # sample with replacement to produce a balanced dataset of size 2*n_per_class
        rng = np.random.default_rng(0)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        choose0 = rng.choice(idx0, size=n_per_class, replace=True)
        choose1 = rng.choice(idx1, size=n_per_class, replace=True)
        idx = np.concatenate([choose0, choose1])
        rng.shuffle(idx)
        x, y = x[idx], y[idx]

    return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# MNIST-4x4 (requires torchvision if used; we ship a deterministic stub)
# ---------------------------------------------------------------------------
def mnist_downsampled(
    n_train: int = 500, n_test: int = 200, size: int = 4, seed: int = 0
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Downsampled MNIST (requires torchvision). Binary classification between
    digits 3 and 6 by default. Returns (train, test).

    If torchvision is not installed, raises ImportError with guidance.
    """
    try:
        import torchvision
        from torchvision.transforms import Resize, ToTensor, Compose, Lambda
    except ImportError as e:
        raise ImportError(
            "mnist_downsampled requires torchvision. Install with 'pip install torchvision', "
            "or use bars_and_stripes() for a dependency-free alternative."
        ) from e

    tr = Compose(
        [
            ToTensor(),
            Resize((size, size), antialias=True),
            Lambda(lambda t: t.flatten() * np.pi),  # scale to RY angles
        ]
    )
    tr_set = torchvision.datasets.MNIST(
        root="./data", train=True, transform=tr, download=True
    )
    te_set = torchvision.datasets.MNIST(
        root="./data", train=False, transform=tr, download=True
    )

    def _pick(ds, n):
        rng = np.random.default_rng(seed)
        keep = [(x, y) for (x, y) in ds if y in (3, 6)]
        rng.shuffle(keep)
        keep = keep[:n]
        xs = torch.stack([x for x, _ in keep])
        ys = torch.tensor([0 if y == 3 else 1 for _, y in keep], dtype=torch.long)
        return xs, ys

    return _pick(tr_set, n_train), _pick(te_set, n_test)


# ---------------------------------------------------------------------------
# 1D Transverse Field Ising Model ground states
# ---------------------------------------------------------------------------
def tfim_ground_states(
    n: int, h_values: np.ndarray, J: float = 1.0
) -> np.ndarray:
    """
    Compute ground states of the 1D transverse field Ising model
        H = -J sum_i Z_i Z_{i+1} - h sum_i X_i
    by exact diagonalization.

    Returns
    -------
    states : (len(h_values), 2**n) complex array, each row a ground state.
    """
    from scipy.sparse import csr_matrix, eye, kron
    from scipy.sparse.linalg import eigsh

    I2 = eye(2, format="csr", dtype=np.complex128)
    X = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    Z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))

    def op_at(op, i):
        out = None
        for k in range(n):
            factor = op if k == i else I2
            out = factor if out is None else kron(out, factor, format="csr")
        return out

    ZZ_sum = sum(op_at(Z, i) @ op_at(Z, i + 1) for i in range(n - 1))
    X_sum = sum(op_at(X, i) for i in range(n))

    states = np.empty((len(h_values), 2**n), dtype=np.complex128)
    for k, h in enumerate(h_values):
        H = -J * ZZ_sum - h * X_sum
        # ground state
        _, vecs = eigsh(H, k=1, which="SA")
        states[k] = vecs[:, 0]
    return states


def tfim_phase_labels(h_values: np.ndarray, h_c: float = 1.0) -> np.ndarray:
    """
    Binary labels for the TFIM phase (ferromagnetic h < h_c vs paramagnetic h > h_c).
    """
    return (h_values > h_c).astype(np.int64)
