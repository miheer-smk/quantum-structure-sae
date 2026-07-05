"""
Training loops for QNN classifiers and sparse autoencoders.

Kept deliberately simple — no Hydra, no Lightning — so you can read and
modify quickly. Scale up to a proper experiment tracker once the science
is clear.
"""

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .qnn import TorchQNN
from .sae import TopKSAE


def train_qnn(
    model: TorchQNN,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.02,
    x_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    verbose: bool = True,
) -> dict:
    """Standard supervised training with Adam and cross-entropy."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )

    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        losses, correct, total = [], 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += (logits.argmax(-1) == yb).sum().item()
            total += yb.numel()

        train_loss = sum(losses) / len(losses)
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_acc = float("nan")
        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                logits, _ = model(x_val)
                val_acc = (logits.argmax(-1) == y_val).float().mean().item()
            history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"[QNN] epoch {epoch+1:3d}/{epochs}  "
                f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                + (f"val_acc={val_acc:.3f}" if not (val_acc != val_acc) else "")
            )

    return history


def train_sae(
    sae: TopKSAE,
    features: torch.Tensor,  # (N, d_in)
    *,
    epochs: int = 100,
    batch_size: int = 256,
    verbose_every: int = 10,
) -> dict:
    """Train the sparse autoencoder on the shadow feature matrix."""
    opt = torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr)
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=True)

    history = {"recon_loss": [], "dead_frac": []}
    for epoch in range(epochs):
        sae.train()
        losses = []
        for (xb,) in loader:
            out = sae(xb)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            with torch.no_grad():
                sae.post_step(out["z"])
            losses.append(out["recon_loss"].item())

        mean_recon = sum(losses) / len(losses)
        dead_frac = sae.dead_feature_fraction()
        history["recon_loss"].append(mean_recon)
        history["dead_frac"].append(dead_frac)

        if epoch % verbose_every == 0 or epoch == epochs - 1:
            print(
                f"[SAE] epoch {epoch+1:3d}/{epochs}  "
                f"recon={mean_recon:.4f}  dead_frac={dead_frac:.3f}"
            )

    return history
