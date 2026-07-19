"""
experiments/phase06_multiseed_trained.py — trained-model seed distribution of the
full-input recoverability control.

Retrains the transformer from N independent seeds (delegating to
scripts/exp_ra01_train_transformer.py so training is a single source of truth),
then runs the IDENTICAL point-estimate full-input control on each checkpoint and
reports the headline as a TRAINED DISTRIBUTION vs a RANDOM-INIT DISTRIBUTION:

  * partial correlation r(repr-pred, observable | poly2-h), and
  * incremental R^2 beyond poly2-h,

each as mean ± sd across trained seeds, alongside the random-init distribution
(mean ± sd, p95, max). The decision the manuscript will lead with: does the
trained-seed spread sit cleanly ABOVE the random-init p95 (min trained > random
p95), not just one point.

Only the trained weights vary across seeds; eval states, fold seed, control
construction, and ridge alpha are held fixed (see config comments).

Usage
-----
    python experiments/phase06_multiseed_trained.py \
        [--config configs/phase06_multiseed_trained.yaml] \
        [--skip-train]   # reuse existing runs/ms_trained/seed*/best.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from qsae.analysis.extract import build_input_controls, last_layer_pooled, r2_score
from qsae.analysis.input_control import (
    incremental_r2,
    oof_ridge_predict,
    partial_corr_controlling,
)
from qsae.config import load_config
from qsae.repro import set_global_seed
from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.runlog import RunLogger

_r2 = r2_score
build_controls = build_input_controls


def train_seed(s: int, cfg: dict) -> Path:
    """Delegate to the canonical training script; return the checkpoint path."""
    run_dir = cfg["train"]["run_dir_tmpl"].format(s=s)
    cache = cfg["train"]["cache_tmpl"].format(s=s)
    ckpt = Path(run_dir) / "best.pt"
    if ckpt.exists():
        print(f"[ms] seed {s}: checkpoint exists, reusing {ckpt}")
        return ckpt
    print(f"[ms] seed {s}: training -> {run_dir}")
    subprocess.run(
        [sys.executable, "scripts/exp_ra01_train_transformer.py",
         "--seed", str(s), "--run_dir", run_dir, "--cache_path", cache],
        check=True,
    )
    return ckpt


def load_eval_sets(cfg: dict):
    """Return list of (h_fields, obs, controls) for each fixed eval seed."""
    sets = []
    for es in cfg["eval"]["seeds"]:
        d = torch.load(Path(cfg["eval"]["cache_glob"].replace("{seed}", str(es))),
                       weights_only=False)
        h = np.asarray(d["h_fields"], dtype=np.float64)
        sets.append((h, d["obs"], build_controls(h)))
    return sets


def control_point(model, eval_sets, obs_names, control_names, alpha, n_folds, fold_seed):
    """Point-estimate control for ONE model: per (obs, control), average the
    partial correlation and incremental-R^2 over the fixed eval sets."""
    out = {o: {c: {"partial_corr": [], "incremental_r2": []} for c in control_names}
           for o in obs_names}
    probe_r2 = {o: [] for o in obs_names}
    for h, obs, controls in eval_sets:
        R = last_layer_pooled(model, h)
        for o in obs_names:
            if o not in obs:
                continue
            y = np.asarray(obs[o], dtype=np.float64)
            pred = oof_ridge_predict(R, y, alpha, n_folds, fold_seed)
            probe_r2[o].append(_r2(y, pred))
            for c in control_names:
                out[o][c]["partial_corr"].append(partial_corr_controlling(pred, y, controls[c]))
                out[o][c]["incremental_r2"].append(
                    incremental_r2(R, controls[c], y, alpha, n_folds, fold_seed)["delta"])
    # average over eval seeds
    res = {o: {"probe_r2": float(np.mean(probe_r2[o]))} for o in obs_names if probe_r2[o]}
    for o in res:
        res[o]["controls"] = {
            c: {"partial_corr": float(np.mean(out[o][c]["partial_corr"])),
                "incremental_r2": float(np.mean(out[o][c]["incremental_r2"]))}
            for c in control_names}
    return res


def dist(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()),
            "p05": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95)),
            "n": int(v.size), "values": [float(x) for x in v]}


def p95_stability(rand_vals, n_launch, tol, seed=0):
    """Assess whether p95(random) is stable at the launch sample size.

    Returns p95 at n_launch and over the full pool, a bootstrap CI of the
    n_launch p95 (resampling the first n_launch draws), and whether the launch
    p95 is 'jumpy' (|p95_launch - p95_full| > tol). The reported threshold is the
    full-pool p95 when jumpy, else the launch p95."""
    v = np.asarray(rand_vals, dtype=np.float64)
    v_launch = v[:n_launch]
    p95_launch = float(np.quantile(v_launch, 0.95))
    p95_full = float(np.quantile(v, 0.95))
    rng = np.random.default_rng(seed)
    boot = np.array([np.quantile(rng.choice(v_launch, size=v_launch.size, replace=True), 0.95)
                     for _ in range(2000)])
    jumpy = abs(p95_launch - p95_full) > tol
    return {"p95_launch": p95_launch, "p95_full": p95_full,
            "p95_launch_boot_ci": [float(np.quantile(boot, 0.025)),
                                   float(np.quantile(boot, 0.975))],
            "n_launch": int(n_launch), "n_full": int(v.size),
            "jumpy": bool(jumpy),
            "threshold": p95_full if jumpy else p95_launch,
            "threshold_source": "full_pool" if jumpy else "launch"}


def sep_sd(trained, random_d):
    """Separation in random-sd units: (mean_trained - mean_random) / sd_random."""
    if random_d is None or random_d["sd"] == 0:
        return float("nan")
    return (trained["mean"] - random_d["mean"]) / random_d["sd"]


def seed_flags(trained_values, threshold, sd_random, margin_sd):
    """Per-seed transparency: which trained seeds dip below the random threshold
    or land 'near' it (within margin_sd random-sds above). Nothing averaged away."""
    below, near = [], []
    for i, val in enumerate(trained_values):
        if val <= threshold:
            below.append({"idx": i, "value": float(val)})
        elif val <= threshold + margin_sd * sd_random:
            near.append({"idx": i, "value": float(val)})
    return {"n_below_threshold": len(below), "n_near_threshold": len(near),
            "below": below, "near": near}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase06_multiseed_trained.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed_record = set_global_seed(cfg["seed"])
    logger = RunLogger("phase06_multiseed_trained", cfg, seed_record)

    obs_names = cfg["observables"]
    control_names = cfg["controls"]
    alpha = cfg["probe"]["ridge_alpha"]
    n_folds = cfg["probe"]["n_folds"]
    fold_seed = cfg["probe"]["fold_seed"]
    eval_sets = load_eval_sets(cfg)

    # architecture from an existing checkpoint (all seeds share architecture)
    ref_cfg = None

    # ---- trained seeds ----
    trained_results = {}
    for s in cfg["train_seeds"]:
        ckpt_path = train_seed(s, cfg) if not args.skip_train else \
            Path(cfg["train"]["run_dir_tmpl"].format(s=s)) / "best.pt"
        ckpt = torch.load(ckpt_path, weights_only=False)
        ref_cfg = ckpt["cfg"]
        model = TFIMTransformer(ref_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        trained_results[s] = {
            "res": control_point(model, eval_sets, obs_names, control_names,
                                 alpha, n_folds, fold_seed),
            "energy_test_r2": float(ckpt.get("test_r2_unnorm", float("nan"))),
        }
        logger.log({"event": "trained_seed_done", "seed": s,
                    "energy_test_r2": trained_results[s]["energy_test_r2"]})

    # ---- random-init pool (computed once); size = n_random_check for stability ----
    n_launch = cfg.get("n_random_init", 16)
    n_check = max(cfg.get("n_random_check", n_launch), n_launch)
    rand_results = []
    for r in range(n_check):
        torch.manual_seed(1000 + r)
        rnet = TFIMTransformer(ref_cfg).eval()
        rand_results.append(control_point(rnet, eval_sets, obs_names, control_names,
                                          alpha, n_folds, fold_seed))
    logger.log({"event": "random_init_done", "n": len(rand_results),
                "n_launch": n_launch})

    # ---- aggregate: trained distribution vs random distribution ----
    n_launch = cfg.get("n_random_init", 16)
    tol = cfg.get("p95_jumpy_tol", 0.02)
    margin_sd = cfg.get("near_p95_margin_sd", 0.5)
    summary = {}
    for o in obs_names:
        if o not in trained_results[cfg["train_seeds"][0]]["res"]:
            continue
        summary[o] = {"probe_r2_trained": dist(
            [trained_results[s]["res"][o]["probe_r2"] for s in cfg["train_seeds"]]),
            "controls": {}}
        for c in control_names:
            tr_pc = [trained_results[s]["res"][o]["controls"][c]["partial_corr"]
                     for s in cfg["train_seeds"]]
            tr_ir = [trained_results[s]["res"][o]["controls"][c]["incremental_r2"]
                     for s in cfg["train_seeds"]]
            rd_pc = [rr[o]["controls"][c]["partial_corr"] for rr in rand_results]
            rd_ir = [rr[o]["controls"][c]["incremental_r2"] for rr in rand_results]
            pc_t, pc_r = dist(tr_pc), (dist(rd_pc) if rd_pc else None)
            ir_t, ir_r = dist(tr_ir), (dist(rd_ir) if rd_ir else None)
            pc_stab = p95_stability(rd_pc, n_launch, tol) if rd_pc else None
            ir_stab = p95_stability(rd_ir, n_launch, tol) if rd_ir else None
            cell = {
                "partial_corr_trained": pc_t, "partial_corr_random": pc_r,
                "incremental_r2_trained": ir_t, "incremental_r2_random": ir_r,
                "partial_corr_sep_sd": sep_sd(pc_t, pc_r),
                "incremental_r2_sep_sd": sep_sd(ir_t, ir_r),
                "partial_corr_p95_stability": pc_stab,
                "incremental_r2_p95_stability": ir_stab,
            }
            if pc_stab:
                cell["partial_corr_seed_flags"] = seed_flags(
                    tr_pc, pc_stab["threshold"], pc_r["sd"], margin_sd)
            if ir_stab:
                cell["incremental_r2_seed_flags"] = seed_flags(
                    tr_ir, ir_stab["threshold"], ir_r["sd"], margin_sd)
            summary[o]["controls"][c] = cell

    # decision (poly2-h): trained spread sits cleanly above the (stability-checked)
    # random threshold — min(trained) > threshold — for both partial-r and incr-R^2.
    verdict = {}
    for o in ("long_range_zz",):
        if o not in summary:
            continue
        cc = summary[o]["controls"]["poly2_h"]
        pc_thr = cc["partial_corr_p95_stability"]["threshold"]
        ir_thr = cc["incremental_r2_p95_stability"]["threshold"]
        verdict[o] = {
            "partial_corr_trained_min_gt_threshold": bool(cc["partial_corr_trained"]["min"] > pc_thr),
            "incremental_r2_trained_min_gt_threshold": bool(cc["incremental_r2_trained"]["min"] > ir_thr),
            "partial_corr_sep_sd": cc["partial_corr_sep_sd"],
            "incremental_r2_sep_sd": cc["incremental_r2_sep_sd"],
            "partial_corr_threshold": pc_thr, "incremental_r2_threshold": ir_thr,
            "partial_corr_trained": cc["partial_corr_trained"],
            "partial_corr_random": cc["partial_corr_random"],
            "incremental_r2_trained": cc["incremental_r2_trained"],
            "incremental_r2_random": cc["incremental_r2_random"],
            "partial_corr_seed_flags": cc["partial_corr_seed_flags"],
            "incremental_r2_seed_flags": cc["incremental_r2_seed_flags"],
            "partial_corr_p95_stability": cc["partial_corr_p95_stability"],
            "incremental_r2_p95_stability": cc["incremental_r2_p95_stability"],
        }

    energy = dist([trained_results[s]["energy_test_r2"] for s in cfg["train_seeds"]])
    results = {"summary": summary, "verdict": verdict,
               "energy_test_r2_distribution": energy,
               "meta": {"train_seeds": cfg["train_seeds"],
                        "n_random_init": len(rand_results),
                        "eval_seeds": cfg["eval"]["seeds"]}}
    run_dir = logger.finish(results)
    train_seeds = cfg["train_seeds"]
    _report(summary, verdict, energy, obs_names, train_seeds)
    md = _build_md(summary, verdict, energy, obs_names, train_seeds, cfg, run_dir)
    (run_dir / "summary.md").write_text(md)
    _archive_result(md)   # standing convention: back up every result to results/
    print(f"\n[phase06] outputs in {run_dir}/  and results/phase06_multiseed_trained.md")


def _d(x):
    return f"{x['mean']:.3f}±{x['sd']:.3f} [min {x['min']:.3f}, max {x['max']:.3f}]"


def _stab_str(stab):
    tag = "JUMPY→using full pool" if stab["jumpy"] else "stable"
    return (f"p95@{stab['n_launch']}={stab['p95_launch']:.3f} "
            f"(boot CI [{stab['p95_launch_boot_ci'][0]:.3f},{stab['p95_launch_boot_ci'][1]:.3f}]), "
            f"p95@{stab['n_full']}={stab['p95_full']:.3f} → threshold {stab['threshold']:.3f} [{tag}]")


def _flags_str(flags, train_seeds):
    def lab(items):
        return ", ".join(f"seed{train_seeds[it['idx']]}={it['value']:.3f}" for it in items) or "none"
    return (f"{flags['n_below_threshold']} below "
            f"({lab(flags['below'])}); {flags['n_near_threshold']} near "
            f"({lab(flags['near'])})")


def _report(summary, verdict, energy, obs_names, train_seeds):
    print("\n" + "=" * 82)
    print("PHASE 0.6 — TRAINED SEED DISTRIBUTION vs RANDOM-INIT DISTRIBUTION")
    print("=" * 82)
    print(f"energy test R^2 across trained seeds: {_d(energy)}")
    for label, key in [("partial-r | poly2-h", "partial_corr"),
                       ("incremental R^2 beyond poly2-h", "incremental_r2")]:
        print(f"\n{label}  (trained dist vs random dist, separation in sd units):")
        for o in obs_names:
            if o not in summary:
                continue
            cc = summary[o]["controls"]["poly2_h"]
            t, r = cc[f"{key}_trained"], cc[f"{key}_random"]
            sep = cc[f"{key}_sep_sd"]
            print(f"  {o:18s} trained {_d(t)}")
            if r:
                print(f"  {'':18s} random  {_d(r)}  |  separation {sep:+.2f} sd")
    print("\nVERDICT (long_range_zz, poly2-h):")
    for o, v in verdict.items():
        for key, nm in [("partial_corr", "partial-r"), ("incremental_r2", "incr R^2")]:
            t = v[f"{key}_trained"]
            print(f"  {nm}: trained min {t['min']:.3f} > threshold "
                  f"{v[f'{key}_threshold']:.3f}? {v[f'{key}_trained_min_gt_threshold']}  "
                  f"| separation {v[f'{key}_sep_sd']:+.2f} sd")
            print(f"      p95 stability: {_stab_str(v[f'{key}_p95_stability'])}")
            print(f"      per-seed flags: {_flags_str(v[f'{key}_seed_flags'], train_seeds)}")


def _build_md(summary, verdict, energy, obs_names, train_seeds, cfg, run_dir):
    L = ["# Phase 0.6 — trained-seed distribution of the full-input control\n",
         f"Run: `{run_dir}`  ·  Config: `{cfg.get('_config_path')}`\n",
         f"{len(train_seeds)} independently trained transformers (fresh init + "
         f"disorder), identical full-input control on each; only the trained "
         f"weights vary. Energy test R^2 across seeds: {_d(energy)}. Random-init "
         f"launch pool n={cfg.get('n_random_init')}, stability pool "
         f"n={cfg.get('n_random_check')}.\n"]
    for label, key, fmt in [("Partial correlation | poly2-h", "partial_corr", "{:.3f}"),
                            ("Incremental R² beyond poly2-h", "incremental_r2", "{:.4f}")]:
        L += [f"## {label} — trained vs random-init distribution\n",
              "| observable | trained (mean±sd [min,max]) | random-init (mean±sd, p95) | separation (sd) |",
              "|---|---|---|---|"]
        for o in obs_names:
            if o not in summary:
                continue
            cc = summary[o]["controls"]["poly2_h"]
            t, r = cc[f"{key}_trained"], cc[f"{key}_random"]
            rstr = (f"{fmt.format(r['mean'])}±{fmt.format(r['sd'])} "
                    f"(p95 {fmt.format(r['p95'])})") if r else "n/a"
            tstr = f"{fmt.format(t['mean'])}±{fmt.format(t['sd'])} [{fmt.format(t['min'])},{fmt.format(t['max'])}]"
            L.append(f"| {o} | {tstr} | {rstr} | {cc[f'{key}_sep_sd']:+.2f} |")
        L.append("")
    L += ["## Per-seed transparency (long_range_zz, poly2-h)\n",
          "Per-seed trained values (averaged over eval seeds), and any seed at or "
          "near the random threshold — nothing averaged away.\n"]
    for o, v in verdict.items():
        for key, nm in [("partial_corr", "partial-r"), ("incremental_r2", "incremental R²")]:
            t = v[f"{key}_trained"]
            per = ", ".join(f"s{train_seeds[i]}:{val:.3f}" for i, val in enumerate(t["values"]))
            L.append(f"**{nm}** per-seed: {per}")
            L.append(f"- threshold = {v[f'{key}_threshold']:.4f}; "
                     f"{_flags_str(v[f'{key}_seed_flags'], train_seeds)}")
            L.append(f"- p95 stability: {_stab_str(v[f'{key}_p95_stability'])}\n")
    L += ["## Verdict (long_range_zz, poly2-h)\n"]
    for o, v in verdict.items():
        for key, nm in [("partial_corr", "partial-r"), ("incremental_r2", "incremental R²")]:
            t = v[f"{key}_trained"]
            ok = v[f"{key}_trained_min_gt_threshold"]
            L.append(f"- **{nm}**: trained spread [{t['min']:.3f}, {t['max']:.3f}] vs "
                     f"threshold {v[f'{key}_threshold']:.3f} — clean separation "
                     f"(min>thr): **{ok}**; separation **{v[f'{key}_sep_sd']:+.2f} sd**.")
    return "\n".join(L)


def _archive_result(md: str) -> None:
    """Standing convention: mirror the result table into the committed results/
    archive and refresh its index row (no need to be asked each time)."""
    dest = Path("results") / "phase06_multiseed_trained.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(md)
    idx = Path("results") / "README.md"
    if idx.exists():
        txt = idx.read_text()
        if "_pending launch_" in txt:
            txt = txt.replace(
                "| 2026-07-19 | Phase 0.6 — trained-seed distribution | _pending launch_ | trained vs random-init distribution of the full-input control |",
                "| 2026-07-19 | Phase 0.6 — trained-seed distribution | [phase06_multiseed_trained.md](phase06_multiseed_trained.md) | see file — trained vs random-init distribution, sd-separation + per-seed flags |")
            idx.write_text(txt)


if __name__ == "__main__":
    main()
