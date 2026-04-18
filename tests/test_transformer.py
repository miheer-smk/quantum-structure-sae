"""Tests for TFIMTransformer (reverse_arrow module)."""

from __future__ import annotations

import torch
import pytest

from qsae.reverse_arrow import TFIMTransformer, TransformerConfig


# ---------------------------------------------------------------------------
# 1. Forward-pass shape
# ---------------------------------------------------------------------------
def test_forward_pass_shape() -> None:
    """Output must be a 1-D tensor with length == batch size."""
    cfg = TransformerConfig(L=8, d_model=64, n_heads=4, n_layers=3, d_ff=256)
    model = TFIMTransformer(cfg)
    model.eval()
    batch = 4
    h = torch.randn(batch, cfg.L)
    with torch.no_grad():
        out = model(h)
    assert out.shape == (batch,), f"expected shape ({batch},), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Overfit on 4 examples
# ---------------------------------------------------------------------------
def test_overfit_4_examples() -> None:
    """
    A model with enough capacity should memorise 4 fixed (h, E) pairs
    to near-zero MSE in 200 gradient steps.
    """
    torch.manual_seed(42)
    cfg = TransformerConfig(L=8, d_model=64, n_heads=4, n_layers=3, d_ff=256, dropout=0.0)
    model = TFIMTransformer(cfg)
    model.train()

    h = torch.randn(4, cfg.L)
    targets = torch.randn(4)

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(200):
        opt.zero_grad()
        pred = model(h)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        final_mse = torch.nn.functional.mse_loss(model(h), targets).item()

    assert final_mse < 1e-3, (
        f"model failed to overfit 4 examples: final MSE={final_mse:.6f} (expected < 1e-3)"
    )


# ---------------------------------------------------------------------------
# 3. Checkpoint round-trip
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path) -> None:
    """
    Saving and reloading state_dict must produce bit-identical outputs.
    """
    torch.manual_seed(7)
    cfg = TransformerConfig(L=8, d_model=64, n_heads=4, n_layers=3, d_ff=256)
    model = TFIMTransformer(cfg)
    model.eval()

    h = torch.randn(3, cfg.L)
    with torch.no_grad():
        out1 = model(h)

    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, ckpt_path)

    # Load into a fresh model
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model2 = TFIMTransformer(checkpoint["cfg"])
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()

    with torch.no_grad():
        out2 = model2(h)

    assert torch.allclose(out1, out2, atol=1e-6), (
        f"checkpoint round-trip mismatch: max diff={( out1 - out2).abs().max().item():.2e}"
    )
