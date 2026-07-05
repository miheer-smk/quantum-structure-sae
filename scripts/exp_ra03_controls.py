"""
exp_ra03_controls.py
====================
Week 3+ rigour pass — the control experiments that turn the raw
feature/observable correlations of `exp_ra02_observables.py` into a
publishable claim.

The naive result ("SAE features correlate with quantum observables, |r| up to
0.86") is not by itself evidence of learned quantum structure: every scalar
observable here is a smooth function of the transverse-field vector h, and h is
literally the transformer's input.  A random nonlinear map of h would also
produce features that correlate with the observables.  This script quantifies
how much of the signal is *learned* and how much is trivially inherited from the
input, using five controls that reviewers at npj QI / PRX Quantum / a
mech-interp venue will ask for:

  C1. Linear-probe decodability.  For each observable, cross-validated ridge R²
      predicting it from (a) the trained transformer's residual stream,
      (b) an UNTRAINED (random-weight) transformer's residual stream,
      (c) the raw field vector h, (d) degree-2 polynomial features of h,
      (e) the single scalar mean field h̄.  Learned structure ⇔ (a) > (b),(c).

  C2. Depth / layer sweep.  Probe R² as a function of transformer layer — where
      in the network does each observable become linearly decodable?

  C3. Permutation null for single-feature correlations.  For each observable we
      take the best-|r| alive SAE feature, then build a null by shuffling the
      observable N_PERM times and recomputing the max-|r| over *all* alive
      features (controls for the multiple-comparisons "winner's curse").
      Reported as an empirical p-value and a 95th-percentile |r| threshold.

  C4. Partial correlation controlling for the mean field.  partial-r(feature,
      observable | h̄).  This is the honesty check: does the feature track the
      observable beyond the trivial dependence on the mean field?

  C5. Cross-seed universality.  Train SAEs from 3 seeds on the same activations,
      Hungarian-match decoder directions, report the fraction of features that
      reappear (cos > 0.7) — evidence the features are properties of the
      representation, not of the optimisation.

Outputs (in run_dir/)
---------------------
    results.json            — every number below, machine-readable
    fig_probe_r2.png        — C1 grouped bar chart
    fig_layer_sweep.png     — C2 probe R² vs layer
    fig_null.png            — C3 permutation nulls vs observed |r|
    fig_partial.png         — C4 raw vs partial |r|
    summary.md              — human-readable summary table

Usage
-----
    python scripts/exp_ra03_controls.py \\
        [--ckpt runs/ra01_wide/best.pt] [--n_samples 800] \\
        [--run_dir runs/ra03_controls] [--n_perm 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.observables import compute_all_observables
from qsae.sae import SAEConfig, TopKSAE
from qsae.metrics import match_features, universality_score


SCALAR_OBS = ["entropy", "mean_nn_zz", "mean_x", "long_range_zz", "phase_proximity"]
OBS_LABELS = {
    "entropy":         "S(ρ_A)  half-chain entropy",
    "mean_nn_zz":      "⟨Z_i Z_{i+1}⟩  nn correlator",
    "mean_x":          "⟨X_i⟩  transverse mag.",
    "long_range_zz":   "⟨Z_0 Z_{L-1}⟩  order proxy",
    "phase_proximity": "δ = (h̄−h_c)/h_c  phase dist.",
}


# ---------------------------------------------------------------------------
# Ground-state generation (cached) + observables
# ---------------------------------------------------------------------------

def build_states_and_obs(n_samples: int, L: int, seed: int, cache: Path):
    """Generate disordered-TFIM ground states + observables, with disk cache."""
    if cache.exists():
        print(f"[ra03] loading cached states/obs from {cache}")
        d = torch.load(cache, weights_only=False)
        if d["h_fields"].shape == (n_samples, L):
            return d["h_fields"], d["states"], d["obs"], d["h_mean"]
        print("[ra03] cache shape mismatch — regenerating")

    print(f"[ra03] generating {n_samples} disordered-TFIM ground states (L={L}) …")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    h_fields = rng.uniform(0.1, 2.0, size=(n_samples, L)).astype(np.float64)
    h_mean = h_fields.mean(axis=1)

    I2 = np.eye(2, dtype=np.complex128)
    X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def op_at(op, site):
        out = None
        for k in range(L):
            factor = op if k == site else I2
            out = factor if out is None else np.kron(out, factor)
        return out

    zz = sum(op_at(Z2, i) @ op_at(Z2, i + 1) for i in range(L - 1))
    x_ops = np.stack([op_at(X2, i) for i in range(L)])

    dim = 1 << L
    states = np.empty((n_samples, dim), dtype=np.complex128)
    for k in range(n_samples):
        H = -zz - np.einsum("i,ijk->jk", h_fields[k], x_ops)
        _, vecs = np.linalg.eigh(H)
        states[k] = vecs[:, 0]
    print(f"[ra03] ED done in {time.time() - t0:.1f}s; computing observables …")

    obs = compute_all_observables(states, L, h_values=h_mean)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"h_fields": h_fields, "states": states, "obs": obs, "h_mean": h_mean}, cache)
    print(f"[ra03] cached to {cache}")
    return h_fields, states, obs, h_mean


# ---------------------------------------------------------------------------
# Residual-stream extraction (all layers, mean-pooled)
# ---------------------------------------------------------------------------

def extract_all_layers(model: TFIMTransformer, h_fields: np.ndarray, batch=256):
    """Return list over layers of (N, d_model) mean-pooled residual streams."""
    model.eval()
    h = torch.from_numpy(h_fields).float()
    per_layer = None
    for start in range(0, len(h), batch):
        acts = []

        def hook(m, i, o):
            acts.append(o.detach())

        handles = [layer.register_forward_hook(hook) for layer in model.encoder.layers]
        with torch.no_grad():
            _ = model(h[start:start + batch])
        for hd in handles:
            hd.remove()
        pooled = [a.mean(dim=1) for a in acts]  # each (b, d_model)
        if per_layer is None:
            per_layer = [[] for _ in pooled]
        for li, p in enumerate(pooled):
            per_layer[li].append(p)
    return [torch.cat(chunks, 0).numpy() for chunks in per_layer]


# ---------------------------------------------------------------------------
# Cross-validated linear-probe R²
# ---------------------------------------------------------------------------

def cv_probe_r2(X: np.ndarray, y: np.ndarray, n_splits=5, alpha=1.0, seed=0) -> float:
    """Mean k-fold CV R² of a standardized ridge regression y ~ X."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        reg = Ridge(alpha=alpha).fit(Xtr, y[tr])
        pred = reg.predict(Xte)
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2) + 1e-12
        scores.append(1.0 - ss_res / ss_tot)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# SAE training on a fixed activation matrix
# ---------------------------------------------------------------------------

def train_sae(acts: np.ndarray, k, hidden, epochs, seed) -> tuple[TopKSAE, np.ndarray]:
    torch.manual_seed(seed)
    x = torch.from_numpy(acts).float()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    cfg = SAEConfig(d_in=x.shape[1], d_hidden=hidden, k=k, lr=3e-4, aux_k=max(1, k // 4))
    sae = TopKSAE(cfg)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(x), batch_size=64, shuffle=True)
    for _ in range(epochs):
        sae.train()
        for (xb,) in loader:
            out = sae(xb)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            with torch.no_grad():
                sae.post_step(out["z"])
    sae.eval()
    with torch.no_grad():
        z = sae.feature_activations(x).numpy()
    return sae, z


def best_feature_r(z_alive: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Index (within alive set) and value of the max-|r| feature for target y."""
    rs = np.array([stats.pearsonr(z_alive[:, f], y)[0] for f in range(z_alive.shape[1])])
    rs = np.nan_to_num(rs)
    idx = int(np.argmax(np.abs(rs)))
    return idx, float(rs[idx])


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation r(x, y | z) for scalar controls z."""
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    denom = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)) + 1e-12
    return float((rxy - rxz * ryz) / denom)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RA-03: control experiments")
    ap.add_argument("--ckpt", default="runs/ra01_wide/best.pt")
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--run_dir", default="runs/ra03_controls")
    ap.add_argument("--sae_k", type=int, default=32)
    ap.add_argument("--sae_hidden", type=int, default=256)
    ap.add_argument("--sae_epochs", type=int, default=200)
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- trained transformer -------------------------------------------------
    ckpt = torch.load(Path(args.ckpt), weights_only=False)
    cfg = ckpt["cfg"]
    L = cfg.L
    trained = TFIMTransformer(cfg)
    trained.load_state_dict(ckpt["model_state_dict"])
    trained.eval()
    print(f"[ra03] trained model: L={L}, d_model={cfg.d_model}, "
          f"test R²={ckpt.get('test_r2_unnorm', 'n/a')}")

    # -- untrained (random-weight) control transformer, same architecture ----
    torch.manual_seed(args.seed + 999)
    untrained = TFIMTransformer(cfg)
    untrained.eval()

    # -- data ----------------------------------------------------------------
    cache = Path("data") / f"ra03_states_L{L}_N{args.n_samples}_s{args.seed}.pt"
    h_fields, states, obs, h_mean = build_states_and_obs(args.n_samples, L, args.seed, cache)
    Y = {name: np.asarray(obs[name], dtype=np.float64) for name in SCALAR_OBS}

    # -- activations ---------------------------------------------------------
    acts_trained = extract_all_layers(trained, h_fields)      # list[layer] (N, d)
    acts_untrained = extract_all_layers(untrained, h_fields)
    n_layers = len(acts_trained)
    last_tr = acts_trained[-1]
    last_un = acts_untrained[-1]

    # degree-2 polynomial features of h (for the poly-2 baseline probe)
    poly_h = PolynomialFeatures(degree=2, include_bias=False).fit_transform(h_fields)

    results: dict = {
        "meta": {
            "n_samples": args.n_samples, "L": L, "n_layers": n_layers,
            "sae_k": args.sae_k, "sae_hidden": args.sae_hidden,
            "sae_epochs": args.sae_epochs, "n_perm": args.n_perm, "seed": args.seed,
        }
    }

    # =====================================================================
    # C1. Linear-probe decodability (last layer) + baselines
    # =====================================================================
    print("\n[C1] Linear-probe R² (5-fold CV ridge) per observable:")
    print(f"  {'observable':16s} {'trained':>8s} {'untrained':>10s} "
          f"{'raw-h':>8s} {'poly2-h':>8s} {'mean-h':>8s}")
    c1 = {}
    for name in SCALAR_OBS:
        y = Y[name]
        r2_tr = cv_probe_r2(last_tr, y, seed=args.seed)
        r2_un = cv_probe_r2(last_un, y, seed=args.seed)
        r2_h = cv_probe_r2(h_fields, y, seed=args.seed)
        r2_p2 = cv_probe_r2(poly_h, y, seed=args.seed)
        r2_mh = cv_probe_r2(h_mean.reshape(-1, 1), y, seed=args.seed)
        c1[name] = {"trained": r2_tr, "untrained": r2_un,
                    "raw_h": r2_h, "poly2_h": r2_p2, "mean_h": r2_mh}
        print(f"  {name:16s} {r2_tr:8.3f} {r2_un:10.3f} "
              f"{r2_h:8.3f} {r2_p2:8.3f} {r2_mh:8.3f}")
    results["C1_probe_r2"] = c1

    # =====================================================================
    # C2. Layer sweep (probe R² per layer, trained model)
    # =====================================================================
    print("\n[C2] Probe R² across transformer layers (trained):")
    c2 = {name: [] for name in SCALAR_OBS}
    for li in range(n_layers):
        for name in SCALAR_OBS:
            c2[name].append(cv_probe_r2(acts_trained[li], Y[name], seed=args.seed))
    for name in SCALAR_OBS:
        print(f"  {name:16s} " + "  ".join(f"L{li}:{c2[name][li]:.3f}"
                                            for li in range(n_layers)))
    results["C2_layer_sweep"] = c2

    # =====================================================================
    # Train SAE on trained + untrained last-layer activations
    # =====================================================================
    print("\n[SAE] training TopK SAE on trained & untrained last-layer acts …")
    sae_tr, z_tr = train_sae(last_tr, args.sae_k, args.sae_hidden, args.sae_epochs, args.seed)
    sae_un, z_un = train_sae(last_un, args.sae_k, args.sae_hidden, args.sae_epochs, args.seed)
    alive_tr = (z_tr > 0).any(0)
    alive_un = (z_un > 0).any(0)
    ztr_a = z_tr[:, alive_tr]
    zun_a = z_un[:, alive_un]
    print(f"[SAE] alive features — trained {alive_tr.sum()}, untrained {alive_un.sum()}")
    results["sae"] = {
        "trained_dead_frac": sae_tr.dead_feature_fraction(),
        "untrained_dead_frac": sae_un.dead_feature_fraction(),
        "trained_alive": int(alive_tr.sum()), "untrained_alive": int(alive_un.sum()),
    }

    # =====================================================================
    # C3. Best single-feature |r| with permutation null (trained vs untrained)
    # C4. Partial correlation controlling for mean field h̄
    # =====================================================================
    print("\n[C3/C4] best-feature correlations, permutation null, partial-r(·|h̄):")
    print(f"  {'observable':16s} {'r_train':>8s} {'r_untr':>8s} "
          f"{'null95':>7s} {'p_perm':>8s} {'partial':>8s}")
    rng = np.random.default_rng(args.seed)
    c3, c4 = {}, {}
    for name in SCALAR_OBS:
        y = Y[name]
        idx_tr, r_tr = best_feature_r(ztr_a, y)
        idx_un, r_un = best_feature_r(zun_a, y)

        # permutation null: max-|r| over all alive trained features under shuffled y
        null = np.empty(args.n_perm)
        for p in range(args.n_perm):
            yp = rng.permutation(y)
            null[p] = _max_abs_pearson(ztr_a, yp)
        null95 = float(np.quantile(null, 0.95))
        p_perm = float((null >= abs(r_tr)).mean())

        # partial correlation of the best trained feature vs observable given h̄
        best_feat = ztr_a[:, idx_tr]
        pr = partial_corr(best_feat, y, h_mean)

        c3[name] = {"r_trained": r_tr, "feat_trained": idx_tr,
                    "r_untrained": r_un, "feat_untrained": idx_un,
                    "null_p95": null95, "p_perm": p_perm}
        c4[name] = {"raw_r": r_tr, "partial_r_given_meanh": pr}
        print(f"  {name:16s} {r_tr:8.3f} {r_un:8.3f} "
              f"{null95:7.3f} {p_perm:8.3g} {pr:8.3f}")
    results["C3_null"] = c3
    results["C4_partial"] = c4

    # =====================================================================
    # C5. Cross-seed SAE universality (trained last-layer acts)
    # =====================================================================
    print("\n[C5] cross-seed SAE universality (3 seeds) …")
    saes = [sae_tr]
    for s in (args.seed + 1, args.seed + 2):
        saes.append(train_sae(last_tr, args.sae_k, args.sae_hidden, args.sae_epochs, s)[0])
    sims_all = []
    for a in range(len(saes)):
        for b in range(a + 1, len(saes)):
            _, sims = match_features(saes[a].dec.weight, saes[b].dec.weight)
            sims_all.append(sims)
    sims_all = np.concatenate(sims_all)
    univ = universality_score(sims_all, threshold=0.7)
    print(f"[C5] mean matched cos={sims_all.mean():.3f}, "
          f"universality(cos>0.7)={univ:.3f}")
    results["C5_universality"] = {
        "mean_matched_cosine": float(sims_all.mean()),
        "universality_frac_cos_gt_0.7": float(univ),
        "n_seed_pairs": len(saes) * (len(saes) - 1) // 2,
    }

    # -- save + figures ------------------------------------------------------
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    _make_figures(run_dir, results, SCALAR_OBS, OBS_LABELS, n_layers)
    _write_summary(run_dir, results)
    print(f"\n[ra03] done. outputs in {run_dir}/")


def _max_abs_pearson(Z: np.ndarray, y: np.ndarray) -> float:
    """Vectorised max_f |pearson(Z[:,f], y)| over columns of Z."""
    yc = y - y.mean()
    Zc = Z - Z.mean(0)
    num = Zc.T @ yc
    den = np.sqrt((Zc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
    r = num / den
    return float(np.max(np.abs(np.nan_to_num(r))))


# ---------------------------------------------------------------------------
# Figures + summary
# ---------------------------------------------------------------------------

def _make_figures(run_dir, results, obs_names, labels, n_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # C1 grouped bar
    c1 = results["C1_probe_r2"]
    conds = ["trained", "untrained", "raw_h", "poly2_h", "mean_h"]
    cond_lbl = ["trained TF", "untrained TF", "raw h", "poly2 h", "mean h"]
    x = np.arange(len(obs_names))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, c in enumerate(conds):
        ax.bar(x + (i - 2) * w, [c1[o][c] for o in obs_names], w, label=cond_lbl[i])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[o] for o in obs_names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("5-fold CV R²")
    ax.set_title("C1 — Linear decodability of quantum observables\n"
                 "trained vs untrained transformer vs input baselines")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    plt.tight_layout()
    fig.savefig(run_dir / "fig_probe_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # C2 layer sweep
    c2 = results["C2_layer_sweep"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for o in obs_names:
        ax.plot(range(n_layers), c2[o], marker="o", label=labels[o])
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("probe R²")
    ax.set_xticks(range(n_layers))
    ax.set_title("C2 — Where observables become linearly decodable")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(run_dir / "fig_layer_sweep.png", dpi=150)
    plt.close(fig)

    # C3 null vs observed
    c3 = results["C3_null"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(obs_names))
    ax.bar(xs - 0.2, [abs(c3[o]["r_trained"]) for o in obs_names], 0.4, label="|r| best trained feat")
    ax.bar(xs + 0.2, [c3[o]["null_p95"] for o in obs_names], 0.4, label="permutation null 95th pct")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[o] for o in obs_names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r|")
    ax.set_title("C3 — Observed best-feature |r| vs multiple-comparison null")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(run_dir / "fig_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # C4 raw vs partial
    c4 = results["C4_partial"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs - 0.2, [abs(c4[o]["raw_r"]) for o in obs_names], 0.4, label="|r| raw")
    ax.bar(xs + 0.2, [abs(c4[o]["partial_r_given_meanh"]) for o in obs_names], 0.4,
           label="|partial r| given mean-h")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[o] for o in obs_names], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("|correlation|")
    ax.set_title("C4 — Feature↔observable correlation before/after\n"
                 "removing the trivial mean-field dependence")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(run_dir / "fig_partial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary(run_dir, results):
    c1 = results["C1_probe_r2"]
    c3 = results["C3_null"]
    c4 = results["C4_partial"]
    lines = ["# RA-03 control experiments — summary\n"]
    m = results["meta"]
    lines.append(f"N={m['n_samples']} states, L={m['L']}, "
                 f"SAE(k={m['sae_k']}, hidden={m['sae_hidden']}), "
                 f"{m['n_perm']} permutations.\n")
    lines.append("## C1 — Linear-probe R² (5-fold CV)\n")
    lines.append("| observable | trained TF | untrained TF | raw h | poly2 h | mean h |")
    lines.append("|---|---|---|---|---|---|")
    for o, v in c1.items():
        lines.append(f"| {o} | {v['trained']:.3f} | {v['untrained']:.3f} | "
                     f"{v['raw_h']:.3f} | {v['poly2_h']:.3f} | {v['mean_h']:.3f} |")
    lines.append("\n## C3/C4 — best feature, null, partial-r(·|h̄)\n")
    lines.append("| observable | r_trained | r_untrained | null p95 | p_perm | partial-r |")
    lines.append("|---|---|---|---|---|---|")
    for o in c1:
        lines.append(f"| {o} | {c3[o]['r_trained']:.3f} | {c3[o]['r_untrained']:.3f} | "
                     f"{c3[o]['null_p95']:.3f} | {c3[o]['p_perm']:.3g} | "
                     f"{c4[o]['partial_r_given_meanh']:.3f} |")
    u = results["C5_universality"]
    lines.append("\n## C5 — cross-seed universality\n")
    lines.append(f"mean matched cosine = {u['mean_matched_cosine']:.3f}; "
                 f"fraction cos>0.7 = {u['universality_frac_cos_gt_0.7']:.3f}\n")
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
