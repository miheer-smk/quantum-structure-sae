"""
exp_ra02_observables.py
=======================
Week 3 experiment: compute quantum observables for TFIM ground states,
extract transformer residual-stream activations, train a TopK SAE on them,
and measure Pearson correlations between discovered SAE features and the
physically-known observables.

This is the core interpretability experiment of the project.

Usage
-----
    python scripts/exp_ra02_observables.py \\
        [--ckpt runs/ra01_wide/best.pt] \\
        [--n_samples 500] \\
        [--run_dir runs/ra02_observables] \\
        [--sae_k 32] [--sae_hidden 256]

Outputs (in run_dir/)
---------------------
    correlation_heatmap.png  — SAE feature × observable Pearson-r heatmap
    top_features.json        — top 10 features per observable with |r| and p-value
    obs_stats.json           — summary statistics of computed observables
    sae_summary.json         — SAE training summary (dead fraction, recon loss)
    activations.pt           — residual-stream activations + observable arrays
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from qsae.reverse_arrow.data import make_splits, compute_ground_energies
from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.observables import compute_all_observables
from qsae.sae import SAEConfig, TopKSAE
from qsae.datasets import tfim_ground_states


# ---------------------------------------------------------------------------
# Residual-stream hook
# ---------------------------------------------------------------------------

class ResidualStreamHook:
    """
    Registers forward hooks on all TransformerEncoderLayer blocks to capture
    the residual stream (post-layer output) at each layer.

    Usage
    -----
        hook = ResidualStreamHook(model.encoder)
        with torch.no_grad():
            _ = model(h)
        acts = hook.activations  # list of (batch, L, d_model) tensors
        hook.remove()
    """

    def __init__(self, encoder: nn.TransformerEncoder) -> None:
        self.activations: list[torch.Tensor] = []
        self._handles = []
        for layer in encoder.layers:
            h = layer.register_forward_hook(self._hook)
            self._handles.append(h)

    def _hook(self, module, input, output):
        self.activations.append(output.detach().cpu())

    def remove(self):
        for h in self._handles:
            h.remove()
        self.activations.clear()


def extract_residual_stream(
    model: TFIMTransformer,
    h_fields: torch.Tensor,   # (N, L)
    batch_size: int = 256,
    layer_idx: int = -1,      # -1 = last layer
) -> torch.Tensor:
    """
    Extract mean-pooled residual stream activations from the transformer.

    Returns
    -------
    acts : (N, d_model) float tensor — mean-pooled over sequence positions
    """
    model.eval()
    all_acts = []

    for start in range(0, len(h_fields), batch_size):
        batch = h_fields[start : start + batch_size]
        hook = ResidualStreamHook(model.encoder)
        with torch.no_grad():
            _ = model(batch)
        # hook.activations[i] shape: (batch, L, d_model)
        layer_out = hook.activations[layer_idx]   # (batch, L, d_model)
        pooled = layer_out.mean(dim=1)            # (batch, d_model)
        all_acts.append(pooled)
        hook.remove()

    return torch.cat(all_acts, dim=0)   # (N, d_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pearson_matrix(
    features: np.ndarray,   # (N, F)
    observables: np.ndarray, # (N, O)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pearson r and two-sided p-value for each (feature, observable) pair.

    Returns
    -------
    r_mat : (F, O) Pearson-r matrix
    p_mat : (F, O) p-value matrix
    """
    F = features.shape[1]
    O = observables.shape[1]
    r_mat = np.empty((F, O))
    p_mat = np.empty((F, O))
    for f in range(F):
        for o in range(O):
            r, p = stats.pearsonr(features[:, f], observables[:, o])
            r_mat[f, o] = r
            p_mat[f, o] = p
    return r_mat, p_mat


def top_features_per_observable(
    r_mat: np.ndarray,       # (F, O)
    p_mat: np.ndarray,
    obs_names: list[str],
    top_k: int = 10,
) -> dict:
    """Return the top-k SAE features (by |r|) for each observable."""
    result = {}
    for o, name in enumerate(obs_names):
        col_r = r_mat[:, o]
        col_p = p_mat[:, o]
        order = np.argsort(-np.abs(col_r))[:top_k]
        result[name] = [
            {"feature_idx": int(i), "pearson_r": float(col_r[i]), "p_value": float(col_p[i])}
            for i in order
        ]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RA-02: Observable correlations")
    parser.add_argument("--ckpt",       default="runs/ra01_wide/best.pt")
    parser.add_argument("--n_samples",  type=int, default=500,
                        help="Number of TFIM samples for observable computation")
    parser.add_argument("--run_dir",    default="runs/ra02_observables")
    parser.add_argument("--sae_k",      type=int, default=32)
    parser.add_argument("--sae_hidden", type=int, default=256)
    parser.add_argument("--sae_epochs", type=int, default=200)
    parser.add_argument("--layer",      type=int, default=-1,
                        help="Which transformer layer's residual stream to use (-1=last)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load trained transformer
    # ------------------------------------------------------------------
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run exp_ra01_train_transformer.py first."
        )

    print(f"[ra02] loading transformer from {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = TFIMTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cfg = ckpt["cfg"]
    L = cfg.L
    print(f"[ra02] model: L={L}, d_model={cfg.d_model}, "
          f"test R²={ckpt.get('test_r2_unnorm', 'n/a')}")

    # ------------------------------------------------------------------
    # 2. Generate TFIM samples spanning both phases
    # ------------------------------------------------------------------
    print(f"[ra02] generating {args.n_samples} TFIM samples (L={L}) …")
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    h_fields = rng.uniform(0.1, 2.0, size=(args.n_samples, L)).astype(np.float64)
    h_mean   = h_fields.mean(axis=1)   # per-sample mean field

    # Ground states via exact diagonalisation (disordered TFIM)
    from qsae.reverse_arrow.data import compute_ground_energies
    energies = compute_ground_energies(h_fields, J=1.0)

    # Full state vectors via tfim_ground_states (uniform-h version for small n)
    # For the disordered model we build ground states column by column
    print("[ra02] computing ground-state vectors via exact diagonalisation …")
    from scipy.sparse import csr_matrix, eye, kron
    from scipy.sparse.linalg import eigsh

    I2 = np.eye(2, dtype=np.complex128)
    X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def op_at(op, site, n):
        out = None
        for k in range(n):
            factor = op if k == site else I2
            out = factor if out is None else np.kron(out, factor)
        return out

    # Build fixed ZZ part once
    zz = sum(op_at(Z2, i, L) @ op_at(Z2, i + 1, L) for i in range(L - 1))
    x_ops = np.stack([op_at(X2, i, L) for i in range(L)])  # (L, dim, dim)

    dim = 1 << L
    states = np.empty((args.n_samples, dim), dtype=np.complex128)
    for k in range(args.n_samples):
        H = -zz - np.einsum("i,ijk->jk", h_fields[k], x_ops)
        _, vecs = np.linalg.eigh(H)
        states[k] = vecs[:, 0]

    print(f"[ra02] states computed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Compute quantum observables
    # ------------------------------------------------------------------
    print("[ra02] computing quantum observables …")
    obs = compute_all_observables(states, L, h_values=h_mean)

    obs_summary = {
        k: {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max())}
        for k, v in obs.items()
        if v.ndim == 1
    }
    print("[ra02] observable summary:")
    for k, s in obs_summary.items():
        print(f"  {k:20s}  mean={s['mean']:+.4f}  std={s['std']:.4f}")

    with open(run_dir / "obs_stats.json", "w") as f:
        json.dump(obs_summary, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Extract transformer residual-stream activations
    # ------------------------------------------------------------------
    print(f"[ra02] extracting residual stream (layer {args.layer}) …")
    h_tensor = torch.from_numpy(h_fields).float()
    activations = extract_residual_stream(model, h_tensor, layer_idx=args.layer)
    print(f"[ra02] activations shape: {activations.shape}")

    # ------------------------------------------------------------------
    # 5. Train TopK SAE on activations
    # ------------------------------------------------------------------
    d_model = activations.shape[1]
    print(f"[ra02] training TopK SAE: d_in={d_model}, "
          f"d_hidden={args.sae_hidden}, k={args.sae_k} …")

    acts_t = activations.float()
    acts_t = (acts_t - acts_t.mean(0)) / (acts_t.std(0) + 1e-6)  # normalise

    sae_cfg = SAEConfig(
        d_in=d_model,
        d_hidden=args.sae_hidden,
        k=args.sae_k,
        lr=3e-4,
        aux_k=args.sae_k // 4,
    )
    sae = TopKSAE(sae_cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=sae_cfg.lr)

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(acts_t), batch_size=64, shuffle=True)

    recon_history = []
    for epoch in range(args.sae_epochs):
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
        recon_history.append(np.mean(losses))
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:4d}/{args.sae_epochs}  "
                  f"recon={recon_history[-1]:.5f}  "
                  f"dead={sae.dead_feature_fraction():.3f}")

    sae.eval()
    with torch.no_grad():
        z_all = sae.feature_activations(acts_t).numpy()  # (N, d_hidden)

    sae_summary = {
        "d_in": d_model,
        "d_hidden": args.sae_hidden,
        "k": args.sae_k,
        "final_recon_loss": float(recon_history[-1]),
        "dead_feature_fraction": sae.dead_feature_fraction(),
        "n_epochs": args.sae_epochs,
    }
    print("[ra02] SAE summary:", sae_summary)
    with open(run_dir / "sae_summary.json", "w") as f:
        json.dump(sae_summary, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Pearson correlations: SAE features × observables
    # ------------------------------------------------------------------
    print("[ra02] computing Pearson correlations …")

    scalar_obs_names = ["entropy", "mean_nn_zz", "mean_x", "order_param", "phase_proximity"]
    scalar_obs_matrix = np.stack([obs[k] for k in scalar_obs_names], axis=1)  # (N, O)

    alive_mask = (z_all > 0).any(axis=0)
    z_alive = z_all[:, alive_mask]
    print(f"[ra02] alive SAE features: {alive_mask.sum()} / {args.sae_hidden}")

    r_mat, p_mat = pearson_matrix(z_alive, scalar_obs_matrix)

    top_feats = top_features_per_observable(r_mat, p_mat, scalar_obs_names, top_k=10)
    with open(run_dir / "top_features.json", "w") as f:
        json.dump(top_feats, f, indent=2)

    print("\n[ra02] Top correlated features per observable:")
    for obs_name, entries in top_feats.items():
        best = entries[0]
        print(f"  {obs_name:20s}  best feat={best['feature_idx']:3d}  "
              f"r={best['pearson_r']:+.4f}  p={best['p_value']:.2e}")

    # ------------------------------------------------------------------
    # 7. Heatmap
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Show top-40 features by max |r| across all observables
        max_r = np.abs(r_mat).max(axis=1)
        top40 = np.argsort(-max_r)[:40]

        fig, ax = plt.subplots(figsize=(8, 10))
        im = ax.imshow(r_mat[top40].T, aspect="auto", cmap="RdBu_r",
                       vmin=-1, vmax=1)
        ax.set_xticks(range(len(top40)))
        ax.set_xticklabels([f"f{i}" for i in top40], rotation=90, fontsize=7)
        ax.set_yticks(range(len(scalar_obs_names)))
        ax.set_yticklabels(scalar_obs_names, fontsize=9)
        ax.set_title("Pearson r: SAE features × quantum observables\n(top 40 features by max |r|)")
        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()
        out = run_dir / "correlation_heatmap.png"
        fig.savefig(out, dpi=150)
        print(f"[ra02] heatmap saved to {out}")
    except ImportError:
        print("[ra02] matplotlib not available — skipping heatmap")

    # ------------------------------------------------------------------
    # 8. Save activations for further analysis
    # ------------------------------------------------------------------
    torch.save(
        {
            "activations": activations,
            "z_sae": torch.from_numpy(z_all),
            "h_fields": h_tensor,
            "obs": {k: torch.from_numpy(v.astype(np.float32))
                    for k, v in obs.items() if isinstance(v, np.ndarray)},
            "r_mat": r_mat,
            "p_mat": p_mat,
            "obs_names": scalar_obs_names,
        },
        run_dir / "activations.pt",
    )
    print(f"[ra02] activations saved to {run_dir}/activations.pt")
    print(f"[ra02] done. outputs in {run_dir}/")


if __name__ == "__main__":
    main()
