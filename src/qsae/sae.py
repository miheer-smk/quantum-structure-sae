"""
TopK Sparse Autoencoder.

Implementation follows:
  * Makhzani & Frey (2013), k-sparse autoencoders.
  * Gao et al. (OpenAI, 2024), scaling and evaluating SAEs.
  * Bricken et al. (Anthropic, 2023), Towards Monosemanticity.

We use TopK (rather than L1) because it gives exact control over L0 sparsity
and avoids the shrinkage bias of L1-penalized SAEs.

Input: feature vector from classical shadows, shape (batch, d_in).
Hidden: (batch, d_hidden)  with exactly k nonzeros per row.
Output: reconstruction of the input, shape (batch, d_in).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int
    d_hidden: int
    k: int = 32  # TopK sparsity
    normalize_decoder: bool = True
    tie_init: bool = True
    lr: float = 3e-4
    aux_k: int = 0  # "aux loss" TopK for reviving dead features; 0 disables
    aux_coef: float = 1 / 32


class TopKSAE(nn.Module):
    """
    A minimal but correct TopK sparse autoencoder.

    Features
    --------
    - Exact k-sparsity via torch.topk on pre-activations.
    - Decoder column-norm constraint (keeps feature directions unit-norm,
      so activation magnitudes carry the feature magnitude).
    - Optional auxiliary "aux_k" loss on dead latents to revive them, per
      Gao et al. 2024.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.dec = nn.Linear(cfg.d_hidden, cfg.d_in, bias=True)
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))

        if cfg.tie_init:
            with torch.no_grad():
                self.dec.weight.copy_(self.enc.weight.T)
        self._init_decoder_unit_norm()

        # rolling "last fired" counter for dead-feature detection
        self.register_buffer("last_fired", torch.zeros(cfg.d_hidden, dtype=torch.long))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _init_decoder_unit_norm(self) -> None:
        norms = self.dec.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.dec.weight.div_(norms)

    def encode_preact(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x - self.pre_bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (topk activations, indices)."""
        pre = self.encode_preact(x)
        vals, idx = torch.topk(pre, self.cfg.k, dim=-1)
        vals = F.relu(vals)  # activations are nonnegative
        z = torch.zeros_like(pre).scatter_(-1, idx, vals)
        return z, pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z) + self.pre_bias

    def forward(self, x: torch.Tensor) -> dict:
        z, pre = self.encode(x)
        xhat = self.decode(z)
        recon = F.mse_loss(xhat, x)

        aux_loss = torch.tensor(0.0, device=x.device)
        if self.cfg.aux_k > 0 and self.training:
            # revive dead features: predict residual using top aux_k dead latents
            dead = (self.last_fired > 1000).float()  # haven't fired in last 1000 steps
            if dead.sum() > 0:
                residual = x - xhat.detach()
                pre_dead = pre * dead.unsqueeze(0)
                k_aux = min(self.cfg.aux_k, int(dead.sum().item()))
                if k_aux > 0:
                    vals_a, idx_a = torch.topk(pre_dead, k_aux, dim=-1)
                    vals_a = F.relu(vals_a)
                    z_aux = torch.zeros_like(pre).scatter_(-1, idx_a, vals_a)
                    xhat_aux = self.dec(z_aux)
                    aux_loss = F.mse_loss(xhat_aux, residual) * self.cfg.aux_coef

        return {
            "x_hat": xhat,
            "z": z,
            "recon_loss": recon,
            "aux_loss": aux_loss,
            "loss": recon + aux_loss,
        }

    @torch.no_grad()
    def post_step(self, z: torch.Tensor) -> None:
        """Update last_fired tracker + re-normalize decoder columns."""
        fired = (z > 0).any(dim=0)  # (d_hidden,)
        self.last_fired = torch.where(
            fired, torch.zeros_like(self.last_fired), self.last_fired + 1
        )
        self.step += 1
        if self.cfg.normalize_decoder:
            self._init_decoder_unit_norm()

    # -- Introspection utilities ----------------------------------------

    @torch.no_grad()
    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Return the k-sparse activations z for inputs x."""
        z, _ = self.encode(x)
        return z

    @torch.no_grad()
    def dead_feature_fraction(self) -> float:
        return float((self.last_fired > 1000).float().mean().item())
