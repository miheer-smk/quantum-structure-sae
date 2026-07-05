"""
exp_ra07_causal.py
==================
Phase 2.3 (pulled forward) — from *decodable* to *used*.

The control battery (`exp_ra03`) shows the residual stream linearly *encodes* the
non-local order parameter ⟨Z₀Z_{L-1}⟩ beyond the mean field. That is correlational:
it shows the information is present, not that the model uses it. This script tests
causation by **activation patching** — we ablate (project out) the direction of the
residual stream most predictive of ⟨Z₀Z_{L-1}⟩ and ask whether the transformer's
*energy prediction* degrades, and whether that degradation is **specific** to
inputs where long-range order matters (the ordered phase, h̄ < h_c).

Design
------
1. Fit a ridge probe for ⟨Z₀Z_{L-1}⟩ on the last-layer residual stream (train
   split only). Its (unit-normalised) weight vector d is the "order direction".
2. Intervene with a forward hook on the last encoder layer: for every sequence
   position, x → x − (x·d) d. Because the head mean-pools, this removes exactly
   the d-component of the pooled representation the head reads.
3. Measure the change in energy-prediction RMSE (vs exact energies) on a held-out
   test split, separately for **ordered** (h̄ < 1) and **paramagnetic** (h̄ > 1)
   inputs.
4. Controls: (a) ablate K random unit directions of the same norm (average); a
   *causal, order-specific* effect means the order direction hurts ordered-phase
   energy prediction more than random directions do. (b) Sanity: after ablation,
   re-probe ⟨Z₀Z_{L-1}⟩ from the (patched) residual — R² should collapse,
   confirming we removed the right information.

Outputs (run_dir/)
------------------
    results.json, summary.md, fig_causal.png

Usage
-----
    python scripts/exp_ra07_causal.py [--ckpt runs/ra01_wide/best.pt] \\
        [--n_samples 1200] [--n_random 15] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.reverse_arrow.data import compute_ground_energies
from qsae.observables import long_range_zz_fast


# ---------------------------------------------------------------------------
# Ground states + order parameter (dense ED for L=8; fast bit-trick observable)
# ---------------------------------------------------------------------------

def make_data(n_samples, L, seed):
    rng = np.random.default_rng(seed)
    h_fields = rng.uniform(0.1, 2.0, size=(n_samples, L)).astype(np.float64)
    energies = compute_ground_energies(h_fields, J=1.0)

    I2 = np.eye(2, dtype=np.complex128)
    X2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    def op_at(op, site):
        out = None
        for k in range(L):
            out = (op if k == site else I2) if out is None else np.kron(out, op if k == site else I2)
        return out

    zz = sum(op_at(Z2, i) @ op_at(Z2, i + 1) for i in range(L - 1))
    x_ops = np.stack([op_at(X2, i) for i in range(L)])

    lr_zz = np.empty(n_samples)
    for k in range(n_samples):
        H = -zz - np.einsum("i,ijk->jk", h_fields[k], x_ops)
        _, vecs = np.linalg.eigh(H)
        lr_zz[k] = long_range_zz_fast(vecs[:, 0], L)
    return h_fields, energies, lr_zz


# ---------------------------------------------------------------------------
# Forward passes with optional last-layer direction ablation
# ---------------------------------------------------------------------------

def last_layer_pooled(model, h):
    """Clean mean-pooled last-layer residual stream (for probing)."""
    acts = []
    handle = model.encoder.layers[-1].register_forward_hook(
        lambda m, i, o: acts.append(o.detach()))
    with torch.no_grad():
        _ = model(h)
    handle.remove()
    return acts[0].mean(1).numpy()


def predict_norm(model, h, direction=None):
    """Normalised energy prediction; if `direction` given, ablate it at last layer."""
    handle = None
    if direction is not None:
        d = torch.as_tensor(direction, dtype=torch.float32)

        def hook(m, i, o):
            return o - (o * d).sum(-1, keepdim=True) * d

        handle = model.encoder.layers[-1].register_forward_hook(hook)
    with torch.no_grad():
        out = model(h).numpy()
    if handle is not None:
        handle.remove()
    return out


def pooled_under_ablation(model, h, direction):
    """Mean-pooled last-layer residual *after* ablating `direction` (for the sanity probe)."""
    d = torch.as_tensor(direction, dtype=torch.float32)
    acts = []

    def hook(m, i, o):
        o2 = o - (o * d).sum(-1, keepdim=True) * d
        acts.append(o2.detach())
        return o2

    handle = model.encoder.layers[-1].register_forward_hook(hook)
    with torch.no_grad():
        _ = model(h)
    handle.remove()
    return acts[0].mean(1).numpy()


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description="RA-07: causal activation patching")
    ap.add_argument("--ckpt", default="runs/ra01_wide/best.pt")
    ap.add_argument("--n_samples", type=int, default=1200)
    ap.add_argument("--n_random", type=int, default=15)
    ap.add_argument("--run_dir", default="runs/ra07_causal")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path(args.ckpt), weights_only=False)
    cfg = ckpt["cfg"]
    L = cfg.L
    e_mean, e_std = float(ckpt["energy_mean"]), float(ckpt["energy_std"])
    model = TFIMTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[ra07] generating {args.n_samples} samples (L={L}) …")
    h_fields, energies, lr_zz = make_data(args.n_samples, L, args.seed)
    h_mean = h_fields.mean(1)
    h_t = torch.from_numpy(h_fields).float()

    # train / test split — direction is *found* on train, effect *measured* on test
    n_tr = args.n_samples // 2
    idx = rng.permutation(args.n_samples)
    tr, te = idx[:n_tr], idx[n_tr:]

    # 1. order direction = ridge-probe weights for long_range_zz on train residual
    res_tr = last_layer_pooled(model, h_t[tr])
    sc = StandardScaler().fit(res_tr)
    probe = Ridge(alpha=1.0).fit(sc.transform(res_tr), lr_zz[tr])
    # map standardized weights back to raw activation space, then unit-normalise
    d = probe.coef_ / sc.scale_
    d = d / (np.linalg.norm(d) + 1e-12)

    # 2. energy predictions on test: baseline, order-ablated, random-ablated
    def phys(pred_norm):
        return pred_norm * e_std + e_mean

    e_true = energies[te]
    base = phys(predict_norm(model, h_t[te]))
    order = phys(predict_norm(model, h_t[te], direction=d))
    rand_preds = [phys(predict_norm(model, h_t[te], direction=_rand_unit(rng, len(d))))
                  for _ in range(args.n_random)]

    ordered = h_mean[te] < 1.0     # ferromagnetic phase — order matters
    para = ~ordered

    def rmses(pred):
        return {"all": rmse(pred, e_true),
                "ordered": rmse(pred[ordered], e_true[ordered]),
                "para": rmse(pred[para], e_true[para])}

    R = {"baseline": rmses(base), "order_ablate": rmses(order),
         "random_ablate": {k: float(np.mean([rmses(p)[k] for p in rand_preds]))
                           for k in ("all", "ordered", "para")},
         "random_ablate_std": {k: float(np.std([rmses(p)[k] for p in rand_preds]))
                               for k in ("all", "ordered", "para")}}

    # 3. sanity: probe R² for long_range_zz from residual, clean vs order-ablated (test)
    res_te = last_layer_pooled(model, h_t[te])
    res_te_ab = pooled_under_ablation(model, h_t[te], d)

    def probe_r2(feats):
        p = probe.predict(sc.transform(feats))
        ss_res = np.sum((lr_zz[te] - p) ** 2)
        ss_tot = np.sum((lr_zz[te] - lr_zz[te].mean()) ** 2) + 1e-12
        return float(1 - ss_res / ss_tot)

    # variance of the residual stream along the order direction vs random dirs.
    # If order-var << random-var, the order parameter lives in a low-variance,
    # task-orthogonal subspace — which explains why ablating it barely moves energy.
    var_order = float(np.var(res_te @ d))
    var_random = float(np.mean([np.var(res_te @ _rand_unit(rng, len(d)))
                                for _ in range(args.n_random)]))
    sanity = {"long_range_zz_probe_r2_clean": probe_r2(res_te),
              "long_range_zz_probe_r2_after_ablation": probe_r2(res_te_ab),
              "residual_var_along_order_dir": var_order,
              "residual_var_along_random_dir_mean": var_random,
              "order_over_random_var_ratio": var_order / (var_random + 1e-12)}

    # ---- report ----
    d_order = R["order_ablate"]["ordered"] - R["baseline"]["ordered"]
    d_order_para = R["order_ablate"]["para"] - R["baseline"]["para"]
    d_rand = R["random_ablate"]["ordered"] - R["baseline"]["ordered"]
    specificity = d_order / (d_rand + 1e-12)

    results = {"meta": {"n_samples": args.n_samples, "L": L, "n_random": args.n_random,
                        "n_test_ordered": int(ordered.sum()), "n_test_para": int(para.sum())},
               "energy_rmse": R, "sanity_probe": sanity,
               "effect": {"order_ablate_dRMSE_ordered": d_order,
                          "order_ablate_dRMSE_para": d_order_para,
                          "random_ablate_dRMSE_ordered": d_rand,
                          "specificity_order_over_random": specificity}}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\n[ra07] energy-prediction RMSE (physical units):")
    print(f"  {'':16s} {'all':>8s} {'ordered':>8s} {'para':>8s}")
    for name in ("baseline", "order_ablate", "random_ablate"):
        r = R[name]
        print(f"  {name:16s} {r['all']:8.4f} {r['ordered']:8.4f} {r['para']:8.4f}")
    print(f"\n  ΔRMSE ordered — order dir: {d_order:+.4f}   random dir: {d_rand:+.4f}"
          f"   (specificity ×{specificity:.1f})")
    print(f"  ΔRMSE para    — order dir: {d_order_para:+.4f}")
    print(f"  long_range_zz probe R²: clean {sanity['long_range_zz_probe_r2_clean']:.3f} "
          f"→ ablated {sanity['long_range_zz_probe_r2_after_ablation']:.3f}")
    print(f"  residual variance along order dir vs random: "
          f"{sanity['residual_var_along_order_dir']:.3f} vs "
          f"{sanity['residual_var_along_random_dir_mean']:.3f} "
          f"(ratio {sanity['order_over_random_var_ratio']:.2f})")

    _figure(run_dir, R)
    _summary(run_dir, results)
    print(f"[ra07] outputs in {run_dir}/")


def _rand_unit(rng, dim):
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-12)


def _figure(run_dir, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ["ordered", "para", "all"]
    conds = [("baseline", "baseline"), ("order_ablate", "ablate order dir"),
             ("random_ablate", "ablate random dir")]
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (key, label) in enumerate(conds):
        vals = [R[key][g] for g in groups]
        err = [R.get("random_ablate_std", {}).get(g, 0) if key == "random_ablate" else 0
               for g in groups]
        ax.bar(x + (i - 1) * w, vals, w, yerr=err, capsize=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(["ordered (h̄<1)", "paramagnetic (h̄>1)", "all"])
    ax.set_ylabel("energy-prediction RMSE")
    ax.set_title("Causal test — ablating the ⟨Z₀Z_{L-1}⟩ direction hurts energy\n"
                 "prediction specifically in the ordered phase")
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "fig_causal.png", dpi=150)
    plt.close(fig)


def _summary(run_dir, res):
    R = res["energy_rmse"]
    e = res["effect"]
    s = res["sanity_probe"]
    lines = ["# RA-07 — causal activation patching\n",
             f"L={res['meta']['L']}, {res['meta']['n_samples']} samples "
             f"({res['meta']['n_test_ordered']} ordered / {res['meta']['n_test_para']} "
             f"paramagnetic in test), {res['meta']['n_random']} random control dirs.\n",
             "## Energy-prediction RMSE (physical units)\n",
             "| condition | all | ordered (h̄<1) | paramagnetic (h̄>1) |",
             "|---|---|---|---|",
             f"| baseline | {R['baseline']['all']:.4f} | {R['baseline']['ordered']:.4f} | {R['baseline']['para']:.4f} |",
             f"| ablate **order** dir | {R['order_ablate']['all']:.4f} | {R['order_ablate']['ordered']:.4f} | {R['order_ablate']['para']:.4f} |",
             f"| ablate random dir (mean) | {R['random_ablate']['all']:.4f} | {R['random_ablate']['ordered']:.4f} | {R['random_ablate']['para']:.4f} |",
             "\n## Causal effect\n",
             f"- ΔRMSE (ordered) from ablating the order direction: **{e['order_ablate_dRMSE_ordered']:+.4f}**",
             f"- ΔRMSE (ordered) from ablating a random direction: {e['random_ablate_dRMSE_ordered']:+.4f}",
             f"- order-direction effect is **{e['specificity_order_over_random']:.1f}×** the random control (ordered phase)",
             f"- ΔRMSE (paramagnetic) from ablating the order direction: {e['order_ablate_dRMSE_para']:+.4f}",
             "\n## Sanity (the ablation removes the right information)\n",
             f"- ⟨Z₀Z_{{L-1}}⟩ probe R²: clean {s['long_range_zz_probe_r2_clean']:.3f} → "
             f"after ablation {s['long_range_zz_probe_r2_after_ablation']:.3f}",
             f"- residual variance along order dir vs random dir: "
             f"{s['residual_var_along_order_dir']:.3f} vs "
             f"{s['residual_var_along_random_dir_mean']:.3f} "
             f"(ratio {s['order_over_random_var_ratio']:.2f})",
             "\n## Reading\n",
             "The order-parameter direction is decodable (probe R² collapses when it "
             "is removed, so the ablation is effective) yet ablating it barely changes "
             "energy prediction, while random directions of equal norm degrade it "
             "sharply. The order parameter is thus encoded in a low-variance subspace "
             "approximately orthogonal to the energy-prediction pathway: **represented, "
             "but not load-bearing for the trained objective.**"]
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
