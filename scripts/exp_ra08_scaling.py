"""
exp_ra08_scaling.py
==================
Step 2 — does the learned non-local-order signal survive / strengthen with system
size?  The core finding at L=8 is that the trained transformer linearly encodes the
end-to-end correlator ⟨Z₀Z_{L-1}⟩ beyond an untrained network and beyond the mean
field.  The prediction (docs/week1_results.md) is that the mean-field baselines
weaken as L grows — so the *learned* advantage should grow.  This script trains a
transformer at each L and re-runs the key probe comparison, using the memory-safe
sparse ground-state solver so L can exceed the dense L=8 ceiling.

For each L we report, for ⟨Z₀Z_{L-1}⟩, the 5-fold CV ridge probe R² from:
  trained transformer · untrained transformer · raw h · mean h,
and the *learned gain* = R²(trained) − max(R²(untrained), R²(raw h), R²(mean h)).

Outputs (run_dir/)
------------------
    scaling_results.json, summary.md, fig_scaling.png

Usage
-----
    python scripts/exp_ra08_scaling.py --Ls 8,10,12 \\
        [--n_train 12000] [--n_test 800] [--epochs 60] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from qsae.reverse_arrow.transformer import TFIMTransformer, TransformerConfig
from qsae.reverse_arrow.data import compute_ground_states_sparse
from qsae.observables import long_range_zz_fast


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def gen_energies(n, L, seed, g=0.0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.1, 2.0, size=(n, L))
    e = compute_ground_states_sparse(h, J_fields=1.0, g_fields=g, return_states=False)
    return h.astype(np.float32), e.astype(np.float32)


def gen_states(n, L, seed, g=0.0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.1, 2.0, size=(n, L))
    e, states = compute_ground_states_sparse(h, J_fields=1.0, g_fields=g, return_states=True)
    lr = np.array([long_range_zz_fast(states[i], L) for i in range(n)])
    return h.astype(np.float32), lr, h.mean(1)


# ---------------------------------------------------------------------------
# Compact training loop (mirrors exp_ra01 hyperparameters)
# ---------------------------------------------------------------------------

def train_transformer(h, e, L, epochs, seed, batch=128):
    torch.manual_seed(seed)
    n_val = max(512, len(h) // 10)
    h_t = torch.from_numpy(h)
    e_t = torch.from_numpy(e)
    perm = torch.randperm(len(h))
    tr, va = perm[n_val:], perm[:n_val]
    e_mean, e_std = e_t[tr].mean().item(), e_t[tr].std().item()

    cfg = TransformerConfig(L=L, d_model=64, n_heads=4, n_layers=3, d_ff=256)
    model = TFIMTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def r2(pred, tgt):
        ss_res = ((pred - tgt) ** 2).sum()
        ss_tot = ((tgt - tgt.mean()) ** 2).sum()
        return 1.0 - (ss_res / ss_tot).item()

    best_r2, best_state, patience, since = -1e9, None, 12, 0
    for ep in range(epochs):
        model.train()
        idx = tr[torch.randperm(len(tr))]
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            xb, yb = h_t[b], (e_t[b] - e_mean) / e_std
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vr2 = r2(model(h_t[va]), (e_t[va] - e_mean) / e_std)
        if vr2 > best_r2 + 1e-4:
            best_r2, best_state, since = vr2, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_r2


def last_layer_pooled(model, h):
    acts = []
    handle = model.encoder.layers[-1].register_forward_hook(
        lambda m, i, o: acts.append(o.detach()))
    with torch.no_grad():
        _ = model(torch.from_numpy(h).float())
    handle.remove()
    return acts[0].mean(1).numpy()


def cv_probe_r2(X, y, seed=0):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    out = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        pred = reg.predict(sc.transform(X[te]))
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2) + 1e-12
        out.append(1 - ss_res / ss_tot)
    return float(np.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="RA-08: L-scaling of the order-parameter signal")
    ap.add_argument("--Ls", default="8,10,12")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--g", type=float, default=0.0,
                    help="longitudinal field; 0 = integrable TFIM, >0 = non-integrable mixed-field")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default="runs/ra08_scaling")
    args = ap.parse_args()

    Ls = [int(x) for x in args.Ls.split(",")]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for L in Ls:
        t0 = time.time()
        print(f"\n===== L={L} =====")
        print(f"[L{L}] generating {args.n_train} training energies (sparse, g={args.g}) …")
        h_tr, e_tr = gen_energies(args.n_train, L, args.seed + L, g=args.g)
        print(f"[L{L}] training transformer …")
        model, val_r2 = train_transformer(h_tr, e_tr, L, args.epochs, args.seed)
        torch.manual_seed(args.seed + 999)
        untrained = TFIMTransformer(TransformerConfig(L=L, d_model=64, n_heads=4,
                                                      n_layers=3, d_ff=256)).eval()
        print(f"[L{L}] energy val R²={val_r2:.4f}; generating {args.n_test} test states …")
        h_te, lr_zz, h_mean = gen_states(args.n_test, L, args.seed + 5000 + L, g=args.g)

        res_tr = last_layer_pooled(model, h_te)
        res_un = last_layer_pooled(untrained, h_te)
        r2_tr = cv_probe_r2(res_tr, lr_zz, args.seed)
        r2_un = cv_probe_r2(res_un, lr_zz, args.seed)
        r2_h = cv_probe_r2(h_te.astype(np.float64), lr_zz, args.seed)
        r2_mh = cv_probe_r2(h_mean.reshape(-1, 1), lr_zz, args.seed)
        gain = r2_tr - max(r2_un, r2_h, r2_mh)
        row = {"L": L, "energy_val_r2": val_r2, "lrzz_var": float(np.var(lr_zz)),
               "probe_r2_trained": r2_tr, "probe_r2_untrained": r2_un,
               "probe_r2_raw_h": r2_h, "probe_r2_mean_h": r2_mh,
               "learned_gain": gain, "seconds": time.time() - t0}
        rows.append(row)
        print(f"[L{L}] ⟨Z0Z_L-1⟩ probe R²: trained {r2_tr:.3f} | untrained {r2_un:.3f} "
              f"| raw-h {r2_h:.3f} | mean-h {r2_mh:.3f} | learned gain {gain:+.3f} "
              f"({row['seconds']:.0f}s)")

    out = {"meta": {"n_train": args.n_train, "n_test": args.n_test,
                    "epochs": args.epochs, "seed": args.seed, "Ls": Ls, "g": args.g},
           "rows": rows}
    (run_dir / "scaling_results.json").write_text(json.dumps(out, indent=2))
    _figure(run_dir, rows)
    _summary(run_dir, out)
    print(f"\n[ra08] outputs in {run_dir}/")


def _figure(run_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ls = [r["L"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, lab in [("probe_r2_trained", "trained TF"), ("probe_r2_untrained", "untrained TF"),
                     ("probe_r2_raw_h", "raw h"), ("probe_r2_mean_h", "mean h")]:
        ax1.plot(Ls, [r[key] for r in rows], marker="o", label=lab)
    ax1.set_xlabel("L (sites)")
    ax1.set_ylabel("⟨Z₀Z_{L-1}⟩ probe R²")
    ax1.set_title("Order-parameter decodability vs system size")
    ax1.set_xticks(Ls)
    ax1.legend(fontsize=8)
    ax2.plot(Ls, [r["learned_gain"] for r in rows], marker="s", color="crimson")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("L (sites)")
    ax2.set_ylabel("learned gain  (trained − best baseline)")
    ax2.set_title("Does the *learned* advantage grow with L?")
    ax2.set_xticks(Ls)
    plt.tight_layout()
    fig.savefig(run_dir / "fig_scaling.png", dpi=150)
    plt.close(fig)


def _summary(run_dir, out):
    lines = ["# RA-08 — L-scaling of the ⟨Z₀Z_{L-1}⟩ signal\n",
             f"n_train={out['meta']['n_train']}, n_test={out['meta']['n_test']}, "
             f"epochs={out['meta']['epochs']}.\n",
             "| L | energy val R² | trained | untrained | raw h | mean h | learned gain |",
             "|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['L']} | {r['energy_val_r2']:.4f} | {r['probe_r2_trained']:.3f} | "
                     f"{r['probe_r2_untrained']:.3f} | {r['probe_r2_raw_h']:.3f} | "
                     f"{r['probe_r2_mean_h']:.3f} | {r['learned_gain']:+.3f} |")
    lines.append("\n*Learned gain = R²(trained) − max(R² untrained, raw-h, mean-h). "
                 "The scaling prediction is that this grows with L as the mean-field "
                 "baselines weaken.*")
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
