"""
exp_ra10_connected.py
====================
P0 #1 — does the trained representation encode the genuinely *non-local*
(connected) order beyond the input, even where the raw correlator is input-trivial?

Motivation.  §3d found that in the non-integrable mixed-field model the learned
advantage on the *raw* correlator ⟨Z₀Z_{L-1}⟩ vanishes — but only because symmetry
breaking makes the raw correlator almost linear in the input (⟨Z₀Z_{L-1}⟩ ≈
⟨Z₀⟩⟨Z_{L-1}⟩).  The *connected* correlator ⟨Z₀Z_{L-1}⟩_c = ⟨Z₀Z_{L-1}⟩ −
⟨Z₀⟩⟨Z_{L-1}⟩ subtracts exactly that factorised, input-trivial part, isolating the
non-local content.  This script re-runs the probe comparison for **both** the raw
and the connected correlator, in the integrable (g=0) and non-integrable (g=0.5)
models, and asks whether the trained network's advantage survives on the connected
quantity.

For each (g, L, observable ∈ {raw, connected}) we report the 5-fold CV ridge probe R²
from the trained transformer, an untrained one, raw h, and mean h, and the learned
gain = R²(trained) − max(baselines).

Outputs (run_dir/): connected_results.json, summary.md, fig_connected.png

Usage
-----
    python scripts/exp_ra10_connected.py --Ls 8,10 --gs 0.0,0.5 \\
        [--n_train 10000] [--n_test 800] [--epochs 80] [--seed 0]
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
from qsae.observables import long_range_zz_fast, long_range_zz_connected_fast


def gen_energies(n, L, seed, g):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.1, 2.0, size=(n, L))
    e = compute_ground_states_sparse(h, J_fields=1.0, g_fields=g, return_states=False)
    return h.astype(np.float32), e.astype(np.float32)


def gen_states(n, L, seed, g):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.1, 2.0, size=(n, L))
    _, states = compute_ground_states_sparse(h, J_fields=1.0, g_fields=g, return_states=True)
    raw = np.array([long_range_zz_fast(states[i], L) for i in range(n)])
    conn = np.array([long_range_zz_connected_fast(states[i], L) for i in range(n)])
    return h.astype(np.float32), raw, conn, h.mean(1)


def train_transformer(h, e, L, epochs, seed, batch=128):
    torch.manual_seed(seed)
    n_val = max(512, len(h) // 10)
    h_t, e_t = torch.from_numpy(h), torch.from_numpy(e)
    perm = torch.randperm(len(h))
    tr, va = perm[n_val:], perm[:n_val]
    e_mean, e_std = e_t[tr].mean().item(), e_t[tr].std().item()
    cfg = TransformerConfig(L=L, d_model=64, n_heads=4, n_layers=3, d_ff=256)
    model = TFIMTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    def r2(p, t):
        return 1.0 - (((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()

    best, best_state, since = -1e9, None, 0
    for _ in range(epochs):
        model.train()
        idx = tr[torch.randperm(len(tr))]
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(h_t[b]), (e_t[b] - e_mean) / e_std)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vr2 = r2(model(h_t[va]), (e_t[va] - e_mean) / e_std)
        if vr2 > best + 1e-4:
            best, best_state, since = vr2, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= 12:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best


def pooled(model, h):
    acts = []
    hd = model.encoder.layers[-1].register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    with torch.no_grad():
        _ = model(torch.from_numpy(h).float())
    hd.remove()
    return acts[0].mean(1).numpy()


def cv_r2(X, y, seed=0):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    out = []
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
        pred = reg.predict(sc.transform(X[te]))
        out.append(1 - np.sum((y[te] - pred) ** 2) / (np.sum((y[te] - y[te].mean()) ** 2) + 1e-12))
    return float(np.mean(out))


def probe_row(res_tr, res_un, h, hmean, y, seed):
    r_tr = cv_r2(res_tr, y, seed)
    r_un = cv_r2(res_un, y, seed)
    r_h = cv_r2(h.astype(np.float64), y, seed)
    r_mh = cv_r2(hmean.reshape(-1, 1), y, seed)
    return {"trained": r_tr, "untrained": r_un, "raw_h": r_h, "mean_h": r_mh,
            "learned_gain": r_tr - max(r_un, r_h, r_mh)}


def main() -> None:
    ap = argparse.ArgumentParser(description="RA-10: raw vs connected correlator")
    ap.add_argument("--Ls", default="8,10")
    ap.add_argument("--gs", default="0.0,0.5")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_dir", default="runs/ra10_connected")
    args = ap.parse_args()

    Ls = [int(x) for x in args.Ls.split(",")]
    gs = [float(x) for x in args.gs.split(",")]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for g in gs:
        for L in Ls:
            t0 = time.time()
            print(f"\n===== g={g}, L={L} =====")
            h_tr, e_tr = gen_energies(args.n_train, L, args.seed + L, g)
            model, val = train_transformer(h_tr, e_tr, L, args.epochs, args.seed)
            torch.manual_seed(args.seed + 999)
            untr = TFIMTransformer(TransformerConfig(L=L, d_model=64, n_heads=4,
                                                     n_layers=3, d_ff=256)).eval()
            h_te, raw, conn, hmean = gen_states(args.n_test, L, args.seed + 5000 + L, g)
            res_tr, res_un = pooled(model, h_te), pooled(untr, h_te)
            raw_row = probe_row(res_tr, res_un, h_te, hmean, raw, args.seed)
            con_row = probe_row(res_tr, res_un, h_te, hmean, conn, args.seed)
            rows.append({"g": g, "L": L, "energy_val_r2": val,
                         "raw": raw_row, "connected": con_row,
                         "conn_var": float(np.var(conn)), "seconds": time.time() - t0})
            print(f"[g{g} L{L}] energy R²={val:.4f}")
            print(f"   RAW       gain={raw_row['learned_gain']:+.3f}  "
                  f"(tr {raw_row['trained']:.3f} | un {raw_row['untrained']:.3f} | "
                  f"h {raw_row['raw_h']:.3f} | mh {raw_row['mean_h']:.3f})")
            print(f"   CONNECTED gain={con_row['learned_gain']:+.3f}  "
                  f"(tr {con_row['trained']:.3f} | un {con_row['untrained']:.3f} | "
                  f"h {con_row['raw_h']:.3f} | mh {con_row['mean_h']:.3f})")

    out = {"meta": {"n_train": args.n_train, "n_test": args.n_test,
                    "epochs": args.epochs, "seed": args.seed, "Ls": Ls, "gs": gs},
           "rows": rows}
    (run_dir / "connected_results.json").write_text(json.dumps(out, indent=2))
    _figure(run_dir, rows)
    _summary(run_dir, out)
    print(f"\n[ra10] outputs in {run_dir}/")


def _figure(run_dir, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"g={r['g']}\nL={r['L']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, [r["raw"]["learned_gain"] for r in rows], w, label="raw ⟨Z₀Z_{L-1}⟩")
    ax.bar(x + w / 2, [r["connected"]["learned_gain"] for r in rows], w,
           label="connected ⟨Z₀Z_{L-1}⟩_c")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("learned gain (trained − best baseline)")
    ax.set_title("Does the learned advantage survive on the *connected* correlator?\n"
                 "(g=0 integrable · g=0.5 non-integrable mixed-field)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "fig_connected.png", dpi=150)
    plt.close(fig)


def _summary(run_dir, out):
    lines = ["# RA-10 — raw vs connected ⟨Z₀Z_{L-1}⟩ learned advantage\n",
             f"n_train={out['meta']['n_train']}, n_test={out['meta']['n_test']}, "
             f"epochs={out['meta']['epochs']}.\n",
             "| g | L | observable | trained | untrained | raw h | mean h | learned gain |",
             "|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        for key, lab in (("raw", "raw"), ("connected", "connected")):
            d = r[key]
            lines.append(f"| {r['g']} | {r['L']} | {lab} | {d['trained']:.3f} | "
                         f"{d['untrained']:.3f} | {d['raw_h']:.3f} | {d['mean_h']:.3f} | "
                         f"{d['learned_gain']:+.3f} |")
    lines.append("\n*Connected = ⟨Z₀Z_{L-1}⟩ − ⟨Z₀⟩⟨Z_{L-1}⟩ removes the factorised, "
                 "input-trivial part. If the learned gain is positive on the connected "
                 "correlator even at g=0.5, the effect survives non-integrability once the "
                 "observable carries genuine beyond-input structure.*")
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
