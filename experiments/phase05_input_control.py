"""
experiments/phase05_input_control.py — the "kill-shot" full-input control.

Generalises the TMLR draft's mean-field partial-correlation headline to the full
site-resolved input and its degree-2 polynomial. For each observable we report,
for the trained AND an architecture-matched untrained transformer:

  * probe R^2 of the last-layer residual stream (reproduces Table 2);
  * partial correlation between the representation's OOF probe prediction and the
    observable, controlling for {mean-h, raw-h, poly2-h}, with bootstrap 95% CIs;
  * incremental R^2 = R^2([repr, control]) - R^2(control): does the representation
    add anything on top of a poly-2 input model?

Aggregated over the three cached seeds (42/43/44). No training, no ED — runs on
runs/ra01_wide/best.pt and data/ra03_states_L8_N800_s{seed}.pt.

Decision rule (pre-stated, not tuned): the representation-level headline for the
non-local order proxy SURVIVES iff, controlling for poly2-h, the trained partial
correlation CI excludes 0 AND the trained incremental R^2 CI excludes 0 AND both
exceed the untrained control. Reported honestly regardless of outcome.

Usage
-----
    python experiments/phase05_input_control.py [--config configs/phase05_input_control.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qsae.analysis.extract import build_input_controls, last_layer_pooled, r2_score
from qsae.analysis.input_control import (
    bootstrap_partial_corr,
    oof_ridge_predict,
    partial_corr_controlling,
)
from qsae.config import load_config
from qsae.repro import set_global_seed
from qsae.reverse_arrow.transformer import TFIMTransformer
from qsae.runlog import RunLogger

_r2 = r2_score
build_controls = build_input_controls


def bootstrap_incremental_r2(repr_feats, Z, y, alpha, n_folds, n_boot, seed):
    """Percentile-bootstrap CI for the incremental R^2 (fixed OOF predictions,
    resampled samples — a standard CI for a fixed predictor pair)."""
    y = np.asarray(y, dtype=np.float64)
    pred_both = oof_ridge_predict(
        np.concatenate([repr_feats, Z], axis=1) if Z is not None else repr_feats,
        y, alpha, n_folds, seed)
    pred_z = (oof_ridge_predict(Z, y, alpha, n_folds, seed)
              if Z is not None else np.full_like(y, y.mean()))
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = _r2(y[idx], pred_both[idx]) - _r2(y[idx], pred_z[idx])
    point = _r2(y, pred_both) - _r2(y, pred_z)
    return {"estimate": point,
            "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase05_input_control.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed_record = set_global_seed(cfg["seed"])
    logger = RunLogger("phase05_input_control", cfg, seed_record)

    ckpt = torch.load(Path(cfg["ckpt"]), weights_only=False)
    model_cfg = ckpt["cfg"]
    trained = TFIMTransformer(model_cfg)
    trained.load_state_dict(ckpt["model_state_dict"])
    trained.eval()
    # architecture-matched untrained control (draft convention: seed + 999)
    torch.manual_seed(cfg["seed"] + 999)
    untrained = TFIMTransformer(model_cfg)
    untrained.eval()

    obs_names = cfg["observables"]
    control_names = cfg["controls"]
    alpha = cfg["probe"]["ridge_alpha"]
    n_folds = cfg["probe"]["n_folds"]
    n_boot = cfg["bootstrap"]["n_boot"]

    n_rand = cfg.get("n_random_init", 0)

    # per-seed raw records: results[obs][control][model] -> list of dicts over seeds
    per_seed: dict = {o: {c: {"trained": [], "untrained": []} for c in control_names}
                      for o in obs_names}
    probe_r2: dict = {o: {"trained": [], "untrained": []} for o in obs_names}
    # random-init control: point partial-r per (obs, control, init seed), pooled over data seeds
    rand_pc: dict = {o: {c: [] for c in control_names} for o in obs_names}

    for seed in cfg["data"]["seeds"]:
        cache = Path(cfg["data"]["cache_glob"].replace("{seed}", str(seed)))
        d = torch.load(cache, weights_only=False)
        h_fields = np.asarray(d["h_fields"], dtype=np.float64)
        obs = d["obs"]
        controls = build_controls(h_fields)

        res_tr = last_layer_pooled(trained, h_fields)
        res_un = last_layer_pooled(untrained, h_fields)
        reps = {"trained": res_tr, "untrained": res_un}

        for o in obs_names:
            if o not in obs:
                continue
            y = np.asarray(obs[o], dtype=np.float64)
            for model_key, R in reps.items():
                oof_pred = oof_ridge_predict(R, y, alpha, n_folds, cfg["seed"])
                probe_r2[o][model_key].append(_r2(y, oof_pred))
                for c in control_names:
                    Z = controls[c]
                    pc = bootstrap_partial_corr(oof_pred, y, Z, n_boot=n_boot, seed=cfg["seed"])
                    inc = bootstrap_incremental_r2(R, Z, y, alpha, n_folds, n_boot, cfg["seed"])
                    per_seed[o][c][model_key].append({"partial_corr": pc, "incremental_r2": inc})

        # random-init distribution: many fresh untrained nets, point estimates only
        for r in range(n_rand):
            torch.manual_seed(1000 + r)
            rnet = TFIMTransformer(model_cfg).eval()
            R = last_layer_pooled(rnet, h_fields)
            for o in obs_names:
                if o not in obs:
                    continue
                y = np.asarray(obs[o], dtype=np.float64)
                oof_pred = oof_ridge_predict(R, y, alpha, n_folds, cfg["seed"])
                for c in control_names:
                    rand_pc[o][c].append(partial_corr_controlling(oof_pred, y, controls[c]))
        logger.log({"event": "seed_done", "seed": seed})

    # aggregate across seeds (mean of point estimates; CI = mean of per-seed CIs)
    def agg(dicts, stat):
        est = np.mean([x[stat]["estimate"] for x in dicts])
        lo = np.mean([x[stat]["ci_lo"] for x in dicts])
        hi = np.mean([x[stat]["ci_hi"] for x in dicts])
        return {"estimate": float(est), "ci_lo": float(lo), "ci_hi": float(hi),
                "per_seed": [float(x[stat]["estimate"]) for x in dicts]}

    summary: dict = {}
    for o in obs_names:
        if not probe_r2[o]["trained"]:
            continue
        summary[o] = {
            "probe_r2_trained": float(np.mean(probe_r2[o]["trained"])),
            "probe_r2_untrained": float(np.mean(probe_r2[o]["untrained"])),
            "controls": {},
        }
        for c in control_names:
            rvals = np.asarray(rand_pc[o][c], dtype=np.float64)
            rand_summary = ({
                "mean": float(rvals.mean()), "std": float(rvals.std(ddof=1)),
                "p05": float(np.quantile(rvals, 0.05)),
                "p95": float(np.quantile(rvals, 0.95)),
                "max": float(rvals.max()), "n_inits": int(rvals.size),
            } if rvals.size else None)
            summary[o]["controls"][c] = {
                "partial_corr_trained": agg(per_seed[o][c]["trained"], "partial_corr"),
                "partial_corr_untrained": agg(per_seed[o][c]["untrained"], "partial_corr"),
                "partial_corr_random_init": rand_summary,
                "incremental_r2_trained": agg(per_seed[o][c]["trained"], "incremental_r2"),
                "incremental_r2_untrained": agg(per_seed[o][c]["untrained"], "incremental_r2"),
            }

    # decision rule for the headline observable under the strongest control
    verdict = {}
    for o in ("long_range_zz", "long_range_zz_connected"):
        if o not in summary:
            continue
        cc = summary[o]["controls"]["poly2_h"]
        pc_t = cc["partial_corr_trained"]
        inc_t = cc["incremental_r2_trained"]
        pc_u = cc["partial_corr_untrained"]
        rand = cc["partial_corr_random_init"]
        # trained must beat the full random-init distribution (95th pct), not one draw
        beats_random = rand is None or pc_t["ci_lo"] > rand["p95"]
        survives = (pc_t["ci_lo"] > 0 and inc_t["ci_lo"] > 0
                    and pc_t["estimate"] > pc_u["estimate"] and beats_random)
        verdict[o] = {
            "survives_full_input_control": bool(survives),
            "partial_corr_trained_poly2": pc_t,
            "partial_corr_untrained_poly2": pc_u,
            "partial_corr_random_init_poly2": rand,
            "incremental_r2_trained_poly2": inc_t,
        }

    results = {"summary": summary, "verdict": verdict,
               "meta": {"seeds": cfg["data"]["seeds"], "n_boot": n_boot}}
    run_dir = logger.finish(results)
    _write_summary_md(run_dir, summary, verdict, control_names, obs_names)
    _print_report(summary, verdict, control_names, obs_names)
    print(f"\n[phase05] outputs in {run_dir}/")


def _fmt(a):
    return f"{a['estimate']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]"


def _print_report(summary, verdict, control_names, obs_names):
    print("\n" + "=" * 78)
    print("PHASE 0.5 — FULL-INPUT RECOVERABILITY CONTROL (kill-shot)")
    print("=" * 78)
    print(f"\n{'observable':26s} {'probe R2 (tr)':>13s} {'(untr)':>8s}")
    for o in obs_names:
        if o not in summary:
            continue
        s = summary[o]
        print(f"{o:26s} {s['probe_r2_trained']:13.3f} {s['probe_r2_untrained']:8.3f}")
    print("\nPartial correlation r(repr-pred, observable | control), trained "
          "[95% CI]:")
    for o in obs_names:
        if o not in summary:
            continue
        print(f"\n  {o}")
        for c in control_names:
            cc = summary[o]["controls"][c]
            print(f"    | {c:8s}  trained {_fmt(cc['partial_corr_trained'])}"
                  f"   untr {_fmt(cc['partial_corr_untrained'])}")
        cc = summary[o]["controls"]["poly2_h"]
        print(f"    incremental R2 | poly2_h: trained "
              f"{_fmt(cc['incremental_r2_trained'])}  "
              f"untr {_fmt(cc['incremental_r2_untrained'])}")
    print("\nVERDICT (headline observables, poly2-h control):")
    for o, v in verdict.items():
        tag = "SURVIVES" if v["survives_full_input_control"] else "DOES NOT SURVIVE"
        print(f"  {o}: {tag}")
        print(f"     partial-r trained  {_fmt(v['partial_corr_trained_poly2'])}")
        print(f"     partial-r untrained{_fmt(v['partial_corr_untrained_poly2'])}")
        rnd = v.get("partial_corr_random_init_poly2")
        if rnd:
            print(f"     partial-r random-init  mean {rnd['mean']:.3f} ± {rnd['std']:.3f} "
                  f"(p95 {rnd['p95']:.3f}, max {rnd['max']:.3f}, n={rnd['n_inits']})")
        print(f"     incremental R2     {_fmt(v['incremental_r2_trained_poly2'])}")


def _write_summary_md(run_dir, summary, verdict, control_names, obs_names):
    lines = ["# Phase 0.5 — full-input recoverability control\n",
             "Generalises the draft's scalar mean-field control to the full input "
             "vector and its degree-2 polynomial. Aggregated over cached seeds "
             "42/43/44. Partial correlation is between the trained residual "
             "stream's out-of-fold probe prediction and each observable, "
             "controlling for the named input features; incremental R^2 is "
             "R^2([repr, control]) − R^2(control). 95% percentile-bootstrap CIs.\n",
             "## Probe R^2 (reproduces draft Table 2)\n",
             "| observable | trained | untrained |", "|---|---|---|"]
    for o in obs_names:
        if o not in summary:
            continue
        s = summary[o]
        lines.append(f"| {o} | {s['probe_r2_trained']:.3f} | {s['probe_r2_untrained']:.3f} |")
    lines.append("\n## Partial correlation trained [95% CI], by control\n")
    lines.append("| observable | " + " | ".join(control_names) + " |")
    lines.append("|" + "---|" * (len(control_names) + 1))
    for o in obs_names:
        if o not in summary:
            continue
        row = [o] + [_fmt(summary[o]["controls"][c]["partial_corr_trained"]) for c in control_names]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## Incremental R^2 beyond poly2-h (trained) [95% CI]\n")
    lines.append("| observable | incremental R^2 |")
    lines.append("|---|---|")
    for o in obs_names:
        if o not in summary:
            continue
        lines.append(f"| {o} | {_fmt(summary[o]['controls']['poly2_h']['incremental_r2_trained'])} |")
    lines.append("\n## Verdict (headline observables, poly2-h control)\n")
    for o, v in verdict.items():
        tag = "**SURVIVES**" if v["survives_full_input_control"] else "**does NOT survive**"
        rnd = v.get("partial_corr_random_init_poly2")
        rnd_s = (f"; random-init {rnd['mean']:.3f} ± {rnd['std']:.3f} (p95 {rnd['p95']:.3f})"
                 if rnd else "")
        lines.append(f"- `{o}`: {tag} — partial-r trained "
                     f"{_fmt(v['partial_corr_trained_poly2'])} vs untrained "
                     f"{_fmt(v['partial_corr_untrained_poly2'])}{rnd_s}; incremental R^2 "
                     f"{_fmt(v['incremental_r2_trained_poly2'])}.")
    (run_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
