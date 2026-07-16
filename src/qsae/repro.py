"""
Reproducibility utilities — global seeding and determinism flags.

Call `set_global_seed(cfg["seed"])` at the top of every experiment. The
returned record (what was seeded, which determinism flags took effect) is
logged by `qsae.runlog.RunLogger` so any unavoidable nondeterminism is on
the record rather than silent.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> dict:
    """
    Seed Python, NumPy, and PyTorch (CPU + all CUDA devices).

    Parameters
    ----------
    seed          : the experiment seed (from the config file).
    deterministic : if True, request deterministic torch algorithms
                    (warn_only=True: ops without a deterministic
                    implementation warn instead of crashing) and set
                    cuDNN to deterministic, non-benchmarking mode.

    Returns
    -------
    record : dict describing exactly what was set — include it in run logs.

    Notes
    -----
    Known residual nondeterminism even with these flags:
      * scipy.sparse.linalg.eigsh Lanczos starting vectors (we pass v0
        explicitly where it matters — see physics code);
      * CUDA atomics in some reduction kernels (warn_only surfaces these).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    record = {
        "seed": seed,
        "deterministic_requested": deterministic,
        "cuda_available": torch.cuda.is_available(),
    }
    if deterministic:
        # CUBLAS needs this env var for deterministic matmuls (CUDA >= 10.2).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        record["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
        record["torch_deterministic_algorithms"] = "warn_only"
    return record


def seeded_generator(seed: int) -> np.random.Generator:
    """A NumPy Generator for data sampling that is independent of global state."""
    return np.random.default_rng(seed)
