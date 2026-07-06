"""
exp_ra11_bootstrap.py
====================
P0 #2 — replace p-value theatre with effect sizes + bootstrap confidence intervals.

Reporting p = 3×10⁻⁸⁹ at N=800 is performative (correlation p-values shrink
mechanically with N). The statistically meaningful summary of the headline is the
*effect size with a confidence interval*. This script computes, for the trained
transformer's residual-stream encoding of ⟨Z₀Z_{L-1}⟩ (and the connected
correlator), a 95% bootstrap CI on:

  - the linear-probe R²  (out-of-fold ridge, representation → observable),
  - Pearson r between the probe's out-of-fold prediction and the observable,
  - the partial correlation r(prediction, observable | mean field h̄)
    — the representation-level analogue of the C4 headline.

Bootstrap is over the N test samples (percentile CIs); the out-of-fold predictions
are computed once by 5-fold CV, then resampled — a standard CI for a fixed predictor.

Outputs (run_dir/): bootstrap_results.json, summary.md

Usage
-----
    python scripts/exp_ra11_bootstrap.py [--ckpt runs/ra01_wide/best.pt] \\
        [--n_samples 800] [--n_boot 4000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.reverse_arrow.data import compute_ground_states_sparse
from qsae.observables import long_range_zz_fast, long_range_zz_connected_fast


def last_layer_pooled(model, h):
    acts = []
    hd = model.encoder.layers[-1].register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    with torch.no_grad():
        _ = model(torch.from_numpy(h).float())
    hd.remove()
    return acts[0].mean(1).numpy()


def oof_predictions(X, y, seed=0):
    """Out-of-fold ridge predictions (each sample predicted by a model not trained on it)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        pred[te] = reg.predict(sc.transform(X[te]))
    return pred


def partial_corr(a, b, z):
    rab = stats.pearsonr(a, b)[0]
    raz = stats.pearsonr(a, z)[0]
    rbz = stats.pearsonr(b, z)[0]
    return (rab - raz * rbz) / (np.sqrt((1 - raz ** 2) * (1 - rbz ** 2)) + 1e-12)


def r2_score(pred, y):
    return 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)


def boot_ci(fn, idx_arrays, n_boot, rng):
    """Percentile 95% CI of statistic fn(idx) over bootstrap resamples of the index."""
    n = len(idx_arrays[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        vals[b] = fn(s)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description="RA-11: bootstrap CIs on the headline")
    ap.add_argument("--ckpt", default="runs/ra01_wide/best.pt")
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run_dir", default="runs/ra11_bootstrap")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(Path(args.ckpt), weights_only=False)
    L = ckpt["cfg"].L
    model = TFIMTransformer(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[ra11] generating {args.n_samples} states (L={L}) …")
    hh = rng.uniform(0.1, 2.0, size=(args.n_samples, L))
    _, states = compute_ground_states_sparse(hh, J_fields=1.0, return_states=True)
    y_raw = np.array([long_range_zz_fast(states[i], L) for i in range(args.n_samples)])
    y_con = np.array([long_range_zz_connected_fast(states[i], L) for i in range(args.n_samples)])
    h_mean = hh.mean(1)
    res = last_layer_pooled(model, hh.astype(np.float32))

    results = {"meta": {"L": L, "n_samples": args.n_samples, "n_boot": args.n_boot,
                        "seed": args.seed, "ckpt": str(args.ckpt)}}
    for name, y in (("long_range_zz", y_raw), ("long_range_zz_connected", y_con)):
        pred = oof_predictions(res, y, args.seed)

        def stat_r2(s, pred=pred, y=y):
            return r2_score(pred[s], y[s])

        def stat_r(s, pred=pred, y=y):
            return stats.pearsonr(pred[s], y[s])[0]

        def stat_pr(s, pred=pred, y=y):
            return partial_corr(pred[s], y[s], h_mean[s])

        pt = {"probe_r2": r2_score(pred, y),
              "pearson_r": float(stats.pearsonr(pred, y)[0]),
              "partial_r_given_meanh": partial_corr(pred, y, h_mean)}
        ci = {"probe_r2": boot_ci(stat_r2, (pred,), args.n_boot, rng),
              "pearson_r": boot_ci(stat_r, (pred,), args.n_boot, rng),
              "partial_r_given_meanh": boot_ci(stat_pr, (pred,), args.n_boot, rng)}
        results[name] = {"point": pt, "ci95": ci}
        print(f"\n[{name}]")
        for k in pt:
            print(f"  {k:24s} {pt[k]:+.3f}   95% CI [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]")

    (run_dir / "bootstrap_results.json").write_text(json.dumps(results, indent=2))
    _summary(run_dir, results)
    print(f"\n[ra11] outputs in {run_dir}/")


def _summary(run_dir, r):
    lines = ["# RA-11 — bootstrap 95% CIs on the headline (representation-level)\n",
             f"L={r['meta']['L']}, N={r['meta']['n_samples']}, "
             f"{r['meta']['n_boot']} bootstrap resamples.\n",
             "| observable | statistic | estimate | 95% CI |",
             "|---|---|---|---|"]
    for name in ("long_range_zz", "long_range_zz_connected"):
        pt, ci = r[name]["point"], r[name]["ci95"]
        for k, lab in (("probe_r2", "probe R²"), ("pearson_r", "Pearson r"),
                       ("partial_r_given_meanh", "partial-r | h̄")):
            lines.append(f"| {name} | {lab} | {pt[k]:+.3f} | "
                         f"[{ci[k][0]:+.3f}, {ci[k][1]:+.3f}] |")
    lines.append("\n*Out-of-fold ridge predictions of the observable from the trained "
                 "residual stream; CIs are percentile bootstrap over the N test states.*")
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
