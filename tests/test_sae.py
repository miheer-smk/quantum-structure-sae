"""Sanity tests for the TopK sparse autoencoder."""

from __future__ import annotations

import pytest
import torch

from qsae.sae import SAEConfig, TopKSAE


def test_topk_sparsity_exact() -> None:
    cfg = SAEConfig(d_in=20, d_hidden=64, k=5)
    sae = TopKSAE(cfg)
    x = torch.randn(32, cfg.d_in)
    z, _ = sae.encode(x)
    # each row should have exactly k nonzero entries... unless some TopK values
    # were negative pre-ReLU; with random init many will survive though.
    # We assert <= k (k is an upper bound after ReLU).
    nnz = (z > 0).sum(dim=-1)
    assert (nnz <= cfg.k).all(), f"expected L0 <= {cfg.k}, got max {nnz.max().item()}"


@pytest.mark.slow
def test_sae_reconstructs_simple_data() -> None:
    """
    On a synthetic "sparse superposition" dataset where inputs are sums of
    k_true << d_in fixed directions, a TopK SAE with k = k_true should
    reconstruct well after training.
    """
    torch.manual_seed(0)
    d_in, k_true = 32, 4
    directions = torch.randn(8, d_in)
    directions = directions / directions.norm(dim=1, keepdim=True)

    def sample(n: int) -> torch.Tensor:
        idx = torch.randint(0, directions.shape[0], (n, k_true))
        weights = torch.rand(n, k_true)
        x = torch.zeros(n, d_in)
        for i in range(n):
            for j in range(k_true):
                x[i] += weights[i, j] * directions[idx[i, j]]
        return x

    xs = sample(1024)

    cfg = SAEConfig(d_in=d_in, d_hidden=32, k=k_true, lr=3e-3)
    sae = TopKSAE(cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    for _ in range(400):
        out = sae(xs)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        sae.post_step(out["z"])

    final_recon = out["recon_loss"].item()
    baseline = (xs - xs.mean(dim=0)).pow(2).mean().item()
    assert final_recon < 0.5 * baseline, (
        f"recon {final_recon:.4f} not much better than mean baseline {baseline:.4f}"
    )
