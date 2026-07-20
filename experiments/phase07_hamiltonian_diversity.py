"""
experiments/phase07_hamiltonian_diversity.py — diversity families under the same
10-seed + full poly-2 input control as Phase 0.6.

For one family (XXZ or ANNNI): train N_SEEDS transformers from independent seeds
(fresh init + disorder) to predict ground-state energy from the disordered
parameter vector, then run the IDENTICAL full-input recoverability control on each
against a random-init distribution. Only the trained weights vary; the eval set is
a fixed yardstick.

Reported in order (as agreed with the author):
  1. energy test R^2 per seed, up front (the representation analysis only means
     something if the transformer learned the Hamiltonian, cf. draft 5.1);
  2. max|<Z_i>| on the eval draw (beyond-input protection), before any probe;
  3. per target: ensemble std sigma_y NEXT TO the trained-vs-random sigma-separation,
     with a SEPARATION / NULL / UNDERPOWERED flag.

Usage
-----
    python experiments/phase07_hamiltonian_diversity.py --config configs/phase07_xxz.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qsae.analysis.distributions import classify, dist, p95_stability, sep_sd
from qsae.analysis.extract import build_input_controls, last_layer_pooled, r2_score
from qsae.analysis.family_data import (
    compute_observables,
    make_family,
    max_abs_single_site_z,
)
from qsae.analysis.input_control import (
    incremental_r2,
    oof_ridge_predict,
    partial_corr_controlling,
)
from qsae.config import load_config
from qsae.repro import seeded_generator, set_global_seed
from qsae.reverse_arrow.transformer import TFIMTransformer, TransformerConfig
from qsae.runlog import RunLogger


def train_one(inp: np.ndarray, energies: np.ndarray, model_cfg: TransformerConfig,
              tr_cfg: dict, seed: int) -> tuple[TFIMTransformer, float]:
    """Train a transformer on (input -> energy); return (model, test R^2).
    Same optimiser/schedule as the TFIM training script, family-agnostic."""
    torch.manual_seed(seed)
    x = torch.from_numpy(inp).float()
    y = torch.from_numpy(energies).float()
    n = len(x)
    n_test = max(1, min(n // 10, n - 2))
    n_val = max(1, min(n // 10, n - n_test - 1))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    te, va, tr = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    ym, ys = y[tr].mean(), y[tr].std()

    model = TFIMTransformer(model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"],
                            weight_decay=tr_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr_cfg["epochs"], eta_min=1e-6)
    bs = tr_cfg["batch_size"]

    best_val, best_state, since = -1e9, None, 0
    for _ in range(tr_cfg["epochs"]):
        model.train()
        idx = tr[torch.randperm(len(tr))]
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x[b]), (y[b] - ym) / ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vr = r2_score(((y[va] - ym) / ys).numpy(), model(x[va]).numpy())
        if vr > best_val + 1e-4:
            best_val, since = vr, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= tr_cfg["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_r2 = r2_score(((y[te] - ym) / ys).numpy(), model(x[te]).numpy())
    return model, float(test_r2)


def control_point(model, inp, controls, obs, obs_names, control_names, alpha, n_folds, fold_seed):
    """Point-estimate control for one model: per (obs, control), partial corr and
    incremental R^2 (no bootstrap; the seed spread is the uncertainty)."""
    R = last_layer_pooled(model, inp)
    out = {}
    for o in obs_names:
        y = np.asarray(obs[o], dtype=np.float64)
        pred = oof_ridge_predict(R, y, alpha, n_folds, fold_seed)
        cell = {"probe_r2": r2_score(y, pred), "controls": {}}
        for c in control_names:
            cell["controls"][c] = {
                "partial_corr": partial_corr_controlling(pred, y, controls[c]),
                "incremental_r2": incremental_r2(R, controls[c], y, alpha, n_folds, fold_seed)["delta"],
            }
        out[o] = cell
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    seed_record = set_global_seed(cfg["seed"])
    logger = RunLogger(f"phase07_{cfg['family']}", cfg, seed_record)

    fam = make_family(cfg)
    L = cfg["L"]
    obs_names = cfg["observables"]
    control_names = cfg["controls"]
    alpha, n_folds, fold_seed = cfg["probe"]["ridge_alpha"], cfg["probe"]["n_folds"], cfg["probe"]["fold_seed"]
    model_cfg = TransformerConfig(L=fam.input_dim, d_model=cfg["model"]["d_model"],
                                  n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"],
                                  d_ff=cfg["model"]["d_ff"], dropout=cfg["model"]["dropout"])

    # ---- fixed eval yardstick (states + observables + input controls) ----
    eval_rng = seeded_generator(cfg["eval"]["seed"])
    eval_inp = fam.sample_inputs(eval_rng, cfg["eval"]["n"])
    _, eval_states = fam.states(eval_inp)
    obs = compute_observables(eval_states, L)
    controls = build_input_controls(eval_inp)
    maxz = max_abs_single_site_z(eval_states, L)
    target_std = {o: float(np.std(obs[o])) for o in obs_names}
    logger.log({"event": "eval_ready", "n": cfg["eval"]["n"], "max_abs_Zi": maxz})

    print(f"\n[phase07:{cfg['family']}] eval yardstick: N={cfg['eval']['n']}, "
          f"max|<Z_i>|={maxz:.2e}")
    print("target ensemble std (sigma_y):  " +
          "  ".join(f"{o}={target_std[o]:.3f}" for o in obs_names))

    # ---- train seeds; energy R^2 per seed UP FRONT ----
    print("\nenergy test R^2 per trained seed:")
    trained_models, energy_r2 = [], []
    for s in cfg["train_seeds"]:
        t0 = time.time()
        rng = seeded_generator(1000 + s)
        inp = fam.sample_inputs(rng, cfg["train"]["n_train"])
        en = fam.energies(inp)
        model, r2 = train_one(inp, en, model_cfg, cfg["train"], s)
        trained_models.append(model)
        energy_r2.append(r2)
        print(f"  seed {s:2d}:  energy test R^2 = {r2:.4f}   ({time.time()-t0:.0f}s)")
        logger.log({"event": "trained", "seed": s, "energy_test_r2": r2})
    energy_dist = dist(energy_r2)
    print(f"  -> energy R^2 across seeds: {energy_dist['mean']:.4f} "
          f"[min {energy_dist['min']:.4f}, max {energy_dist['max']:.4f}]")

    # ---- controls: trained seeds + random-init pool ----
    tr_res = [control_point(m, eval_inp, controls, obs, obs_names, control_names,
                            alpha, n_folds, fold_seed) for m in trained_models]
    n_check = cfg["n_random_check"]
    rd_res = []
    for r in range(n_check):
        torch.manual_seed(5000 + r)
        rd_res.append(control_point(TFIMTransformer(model_cfg).eval(), eval_inp, controls,
                                    obs, obs_names, control_names, alpha, n_folds, fold_seed))
    logger.log({"event": "controls_done", "n_random": n_check})

    # ---- aggregate + classify (poly2 control is the headline) ----
    n_launch = cfg["n_random_init"]
    tol = cfg["p95_jumpy_tol"]
    floor = cfg["underpowered_std_floor"]
    summary = {}
    for o in obs_names:
        summary[o] = {"sigma_y": target_std[o],
                      "probe_r2_trained": dist([r[o]["probe_r2"] for r in tr_res]),
                      "controls": {}}
        for c in control_names:
            tr_pc = [r[o]["controls"][c]["partial_corr"] for r in tr_res]
            tr_ir = [r[o]["controls"][c]["incremental_r2"] for r in tr_res]
            rd_pc = [r[o]["controls"][c]["partial_corr"] for r in rd_res]
            rd_ir = [r[o]["controls"][c]["incremental_r2"] for r in rd_res]
            pc_t, pc_r = dist(tr_pc), dist(rd_pc)
            ir_t, ir_r = dist(tr_ir), dist(rd_ir)
            ir_stab = p95_stability(rd_ir, n_launch, tol)
            pc_stab = p95_stability(rd_pc, n_launch, tol)
            summary[o]["controls"][c] = {
                "partial_corr_trained": pc_t, "partial_corr_random": pc_r,
                "incremental_r2_trained": ir_t, "incremental_r2_random": ir_r,
                "partial_corr_sep_sd": sep_sd(pc_t, pc_r),
                "incremental_r2_sep_sd": sep_sd(ir_t, ir_r),
                "incremental_r2_p95_stability": ir_stab,
                "partial_corr_p95_stability": pc_stab,
                "incremental_r2_flag": classify(ir_t, ir_r, ir_stab["threshold"],
                                                target_std[o], floor),
                "partial_corr_flag": classify(pc_t, pc_r, pc_stab["threshold"],
                                              target_std[o], floor),
            }

    results = {"family": cfg["family"], "energy_test_r2": energy_dist,
               "max_abs_Zi": maxz, "target_std": target_std, "summary": summary,
               "meta": {"train_seeds": cfg["train_seeds"], "n_random": n_check,
                        "eval_n": cfg["eval"]["n"], "L": L,
                        "input_dim": fam.input_dim, "underpowered_std_floor": floor}}
    run_dir = logger.finish(results)
    md = _report_and_md(cfg, results, obs_names)
    (run_dir / "summary.md").write_text(md)
    _archive(cfg["family"], md)
    print(f"\n[phase07:{cfg['family']}] outputs in {run_dir}/ and results/phase07_{cfg['family']}.md")


def _report_and_md(cfg, results, obs_names):
    fam = cfg["family"]
    e = results["energy_test_r2"]
    lines = [f"# Phase 0.7 — Hamiltonian diversity: {fam.upper()}\n",
             f"Family `{fam}`, L={results['meta']['L']}, input_dim="
             f"{results['meta']['input_dim']}, {len(cfg['train_seeds'])} trained seeds "
             f"vs {results['meta']['n_random']} random inits, eval N={results['meta']['eval_n']}.\n",
             f"**Energy test R² across seeds: {e['mean']:.4f} "
             f"[min {e['min']:.4f}, max {e['max']:.4f}]**  ·  "
             f"**max|⟨Z_i⟩| = {results['max_abs_Zi']:.2e}** (beyond-input protection)\n",
             "## Headline: poly-2(input) control — trained vs random-init, with σ_y\n",
             "| observable | σ_y | probe R² (tr) | incr. R² trained | incr. R² random (p95) | sep | flag |",
             "|---|---:|---:|---:|---:|---:|:--|"]
    print("\n" + "=" * 92)
    print(f"PHASE 0.7 — {fam.upper()}  (poly-2 control; sigma_y next to every separation)")
    print("=" * 92)
    hdr = f"{'observable':16s} {'sigma_y':>7s} {'probeR2':>8s} {'incrR2_tr':>16s} {'incrR2_rand(p95)':>18s} {'sep':>7s}  flag"
    print(hdr)
    for o in obs_names:
        s = results["summary"][o]
        cc = s["controls"]["poly2_h"]
        it, ir = cc["incremental_r2_trained"], cc["incremental_r2_random"]
        thr = cc["incremental_r2_p95_stability"]["threshold"]
        flag = cc["incremental_r2_flag"]
        print(f"{o:16s} {s['sigma_y']:7.3f} {s['probe_r2_trained']['mean']:8.3f} "
              f"{it['mean']:.3f}±{it['sd']:.3f}[{it['min']:.3f}] "
              f"{ir['mean']:.3f}±{ir['sd']:.3f}({thr:.3f}) "
              f"{cc['incremental_r2_sep_sd']:+.1f}σ  {flag}")
        lines.append(f"| {o} | {s['sigma_y']:.3f} | {s['probe_r2_trained']['mean']:.3f} | "
                     f"{it['mean']:.3f} ± {it['sd']:.3f} [min {it['min']:.3f}] | "
                     f"{ir['mean']:.3f} ± {ir['sd']:.3f} ({thr:.3f}) | "
                     f"{cc['incremental_r2_sep_sd']:+.1f}σ | **{flag}** |")
    # partial-r view
    lines.append("\n## Partial correlation | poly-2(input) — trained vs random, with σ_y\n")
    lines.append("| observable | σ_y | partial-r trained | partial-r random (p95) | sep | flag |")
    lines.append("|---|---:|---:|---:|---:|:--|")
    for o in obs_names:
        s = results["summary"][o]
        cc = s["controls"]["poly2_h"]
        pt, pr = cc["partial_corr_trained"], cc["partial_corr_random"]
        thr = cc["partial_corr_p95_stability"]["threshold"]
        lines.append(f"| {o} | {s['sigma_y']:.3f} | {pt['mean']:.3f} ± {pt['sd']:.3f} "
                     f"[min {pt['min']:.3f}] | {pr['mean']:.3f} ± {pr['sd']:.3f} ({thr:.3f}) | "
                     f"{cc['partial_corr_sep_sd']:+.1f}σ | **{cc['partial_corr_flag']}** |")
    lines.append(f"\n*Flags: SEPARATION = min(trained) > random p95; UNDERPOWERED = "
                 f"no separation and σ_y < {results['meta']['underpowered_std_floor']} "
                 f"(target barely varies); NULL = varies enough but no trained advantage.*")
    return "\n".join(lines)


def _archive(family: str, md: str) -> None:
    dest = Path("results") / f"phase07_{family}.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(md)
    idx = Path("results") / "README.md"
    if idx.exists():
        txt = idx.read_text()
        row = (f"| 2026-07-20 | Phase 0.7 — {family.upper()} diversity | "
               f"[phase07_{family}.md](phase07_{family}.md) | trained vs random-init, "
               f"per-target sigma_y + SEPARATION/NULL/UNDERPOWERED flag |")
        if f"phase07_{family}.md" not in txt:
            sep = "|---|---|---|---|\n"
            txt = txt.replace(sep, sep + row + "\n", 1)
            idx.write_text(txt)


if __name__ == "__main__":
    main()
