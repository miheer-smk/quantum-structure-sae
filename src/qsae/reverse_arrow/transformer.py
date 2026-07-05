"""
TFIMTransformer — predicts per-site ground-state energy of the disordered
1D Transverse-Field Ising Model from the vector of local field strengths.

Input:  h ∈ R^{L}  (per-site transverse fields)
Output: E_0 ∈ R     (ground-state energy, extensive — scales ~L)

Architecture: Pre-LN Transformer encoder (norm_first=True) with learned
positional embeddings and a two-layer MLP regression head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    L: int = 8                # number of sites (= sequence length)
    d_model: int = 64         # embedding dimension
    n_heads: int = 4          # attention heads
    n_layers: int = 3         # transformer encoder layers
    d_ff: int = 256           # feed-forward hidden dim
    dropout: float = 0.0      # dropout (off by default; set >0 for regularization)

    # derived — users should not set these directly
    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class TFIMTransformer(nn.Module):
    """
    Pre-LN Transformer that maps per-site field vectors to ground-state energy.

    Parameters
    ----------
    cfg : TransformerConfig
    """

    def __init__(self, cfg: TransformerConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = TransformerConfig()
        self.cfg = cfg

        # Project each scalar h_i → d_model embedding
        self.input_proj = nn.Linear(1, cfg.d_model)

        # Learned positional embeddings (L, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.L, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Stack of Pre-LN TransformerEncoderLayers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            norm_first=True,   # Pre-LN for training stability
            batch_first=True,  # (batch, seq, d_model)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Final layer-norm before pooling
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # Regression head: d_model → d_model//2 → 1
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (batch, L) float tensor of per-site transverse fields

        Returns
        -------
        energy : (batch,) float tensor of predicted ground-state energies
        """
        # (batch, L) → (batch, L, 1) → (batch, L, d_model)
        x = self.input_proj(h.unsqueeze(-1))

        # Add positional embedding (broadcast over batch)
        x = x + self.pos_emb.unsqueeze(0)

        # Transformer encoder
        x = self.encoder(x)

        # Final norm + mean pool over sequence → (batch, d_model)
        x = self.final_norm(x)
        x = x.mean(dim=1)

        # Regression head → (batch, 1) → (batch,)
        return self.head(x).squeeze(-1)
