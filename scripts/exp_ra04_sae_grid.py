"""
exp_ra04_sae_grid.py
====================
Phase 1.2 — resolve the SAE cross-seed universality crack (control C5).

`exp_ra03_controls.py` found that the TopK-SAE feature basis is *not* stable
across seeds at the default config (d_hidden=256, k=32): mean matched decoder
cosine ≈ 0.37, only ~0.3% of features matching at cos > 0.7. If individual SAE
features are to carry any of the paper's weight, this needs to improve. The
RUNBOOK's own failure-mode table names the levers: widen d_hidden, shrink k,
and/or give the SAE more data/epochs.

This script sweeps (d_hidden × k), and for each cell trains `n_seeds` SAEs on the
*same* last-layer residual-stream activations, then measures cross-seed
universality via Hungarian matching of decoder directions (mean matched cosine
and the fraction matching above cos > 0.7). Because universality is a property of
the SAE decomposition of the activations — not of the quantum observables — no
exact diagonalisation is needed, so we can afford many more samples than ra03.

Outputs (run_dir/)
------------------
    grid_results.json     — full grid: universality, dead frac, recon per cell
    fig_universality.png  — heatmaps of mean matched cosine and frac(cos>0.7)
    summary.md            — table + the winning config

Usage
-----
    python scripts/exp_ra04_sae_grid.py [--ckpt runs/ra01_wide/best.pt] \\
        [--n_samples 2000] [--n_seeds 3] [--sae_epochs 200] \\
        [--hidden 256,512,1024] [--k 8,16,32]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.sae import SAEConfig, TopKSAE
from qsae.metrics import match_features, universality_score


def last_layer_activations(model: TFIMTransformer, h_fields: np.ndarray, batch=512) -> np.ndarray:
    """Mean-pooled last-encoder-layer residual stream, (N, d_model)."""
    model.eval()
    h = torch.from_numpy(h_fields).float()
    out = []
    for start in range(0, len(h), batch):
        acts = []
        handle = model.encoder.layers[-1].register_forward_hook(
            lambda m, i, o: acts.append(o.detach())
        )
        with torch.no_grad():
            _ = model(h[start:start + batch])
        handle.remove()
        out.append(acts[0].mean(dim=1))
    return torch.cat(out, 0).numpy()


def train_sae(acts: np.ndarray, k: int, hidden: int, epochs: int, seed: int) -> TopKSAE:
    torch.manual_seed(seed)
    x = torch.from_numpy(acts).float()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    cfg = SAEConfig(d_in=x.shape[1], d_hidden=hidden, k=k, lr=3e-4, aux_k=max(1, k // 4))
    sae = TopKSAE(cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True)
    last_recon = 0.0
    for _ in range(epochs):
        sae.train()
        for (xb,) in loader:
            o = sae(xb)
            opt.zero_grad()
            o["loss"].backward()
            opt.step()
            with torch.no_grad():
                sae.post_step(o["z"])
            last_recon = o["recon_loss"].item()
    sae._last_recon = last_recon  # type: ignore[attr-defined]
    return sae


def main() -> None:
    ap = argparse.ArgumentParser(description="RA-04: SAE universality grid")
    ap.add_argument("--ckpt", default="runs/ra01_wide/best.pt")
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--sae_epochs", type=int, default=200)
    ap.add_argument("--hidden", default="256,512,1024")
    ap.add_argument("--k", default="8,16,32")
    ap.add_argument("--run_dir", default="runs/ra04_sae_grid")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    hiddens = [int(x) for x in args.hidden.split(",")]
    ks = [int(x) for x in args.k.split(",")]

    ckpt = torch.load(Path(args.ckpt), weights_only=False)
    model = TFIMTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # activations only — no ED needed, so we can use many samples cheaply
    rng = np.random.default_rng(args.seed)
    h_fields = rng.uniform(0.1, 2.0, size=(args.n_samples, ckpt["cfg"].L)).astype(np.float64)
    acts = last_layer_activations(model, h_fields)
    print(f"[ra04] activations {acts.shape} from {args.n_samples} samples; "
          f"grid = hidden{hiddens} × k{ks}, {args.n_seeds} seeds each")

    results = []
    print(f"\n  {'hidden':>7s} {'k':>4s} {'meanCos':>8s} {'cos>0.7':>8s} "
          f"{'deadFrac':>8s} {'recon':>8s}")
    for hidden in hiddens:
        for k in ks:
            saes = [train_sae(acts, k, hidden, args.sae_epochs, args.seed + s)
                    for s in range(args.n_seeds)]
            sims_all = []
            for a in range(len(saes)):
                for b in range(a + 1, len(saes)):
                    _, sims = match_features(saes[a].dec.weight, saes[b].dec.weight)
                    sims_all.append(sims)
            sims_all = np.concatenate(sims_all)
            mean_cos = float(sims_all.mean())
            frac = float(universality_score(sims_all, threshold=0.7))
            dead = float(np.mean([s.dead_feature_fraction() for s in saes]))
            recon = float(np.mean([getattr(s, "_last_recon", np.nan) for s in saes]))
            results.append({"hidden": hidden, "k": k, "mean_matched_cosine": mean_cos,
                            "frac_cos_gt_0.7": frac, "dead_frac": dead, "recon": recon})
            print(f"  {hidden:7d} {k:4d} {mean_cos:8.3f} {frac:8.3f} {dead:8.3f} {recon:8.4f}")

    best = max(results, key=lambda r: r["frac_cos_gt_0.7"])
    out = {"meta": {"n_samples": args.n_samples, "n_seeds": args.n_seeds,
                    "sae_epochs": args.sae_epochs, "hiddens": hiddens, "ks": ks,
                    "baseline_ra03": {"hidden": 256, "k": 32,
                                      "mean_matched_cosine": 0.366, "frac_cos_gt_0.7": 0.003}},
           "grid": results, "best": best}
    (run_dir / "grid_results.json").write_text(json.dumps(out, indent=2))
    print(f"\n[ra04] best cell: hidden={best['hidden']}, k={best['k']}  "
          f"mean cos={best['mean_matched_cosine']:.3f}, "
          f"frac>0.7={best['frac_cos_gt_0.7']:.3f}")

    _figure(run_dir, results, hiddens, ks)
    _summary(run_dir, out)
    print(f"[ra04] outputs in {run_dir}/")


def _figure(run_dir, results, hiddens, ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grid_of(key):
        M = np.full((len(hiddens), len(ks)), np.nan)
        for r in results:
            M[hiddens.index(r["hidden"]), ks.index(r["k"])] = r[key]
        return M

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(
        axes, ["mean_matched_cosine", "frac_cos_gt_0.7"],
        ["Mean matched decoder cosine", "Fraction of features cos > 0.7"],
    ):
        M = grid_of(key)
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
        ax.set_yticks(range(len(hiddens)))
        ax.set_yticklabels(hiddens)
        ax.set_xlabel("k (TopK sparsity)")
        ax.set_ylabel("d_hidden")
        ax.set_title(title, fontsize=10)
        for i in range(len(hiddens)):
            for j in range(len(ks)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=8)
        plt.colorbar(im, ax=ax)
    fig.suptitle("C5 revisited — SAE cross-seed universality vs (d_hidden, k)")
    plt.tight_layout()
    fig.savefig(run_dir / "fig_universality.png", dpi=150)
    plt.close(fig)


def _summary(run_dir, out):
    lines = ["# RA-04 — SAE universality grid\n"]
    m = out["meta"]
    lines.append(f"{m['n_samples']} samples, {m['n_seeds']} seeds/cell, "
                 f"{m['sae_epochs']} epochs. Baseline (ra03, hidden=256,k=32): "
                 f"mean cos {m['baseline_ra03']['mean_matched_cosine']}, "
                 f"frac>0.7 {m['baseline_ra03']['frac_cos_gt_0.7']}.\n")
    lines.append("| d_hidden | k | mean matched cos | frac cos>0.7 | dead frac | recon |")
    lines.append("|---|---|---|---|---|---|")
    for r in out["grid"]:
        lines.append(f"| {r['hidden']} | {r['k']} | {r['mean_matched_cosine']:.3f} | "
                     f"{r['frac_cos_gt_0.7']:.3f} | {r['dead_frac']:.3f} | {r['recon']:.4f} |")
    b = out["best"]
    lines.append(f"\n**Best:** d_hidden={b['hidden']}, k={b['k']} — "
                 f"frac cos>0.7 = {b['frac_cos_gt_0.7']:.3f}, "
                 f"mean cos = {b['mean_matched_cosine']:.3f}.\n")
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
