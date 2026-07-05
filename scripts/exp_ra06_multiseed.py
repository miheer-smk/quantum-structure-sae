"""
exp_ra06_multiseed.py
=====================
Phase 1.4 — put error bars on the headline number.

The control battery (`exp_ra03_controls.py`) reports single-run numbers for
C1–C4. The paper's one non-trivial finding is the beyond-mean-field partial
correlation for the long-range order parameter ⟨Z₀Z_{L-1}⟩; a reviewer will want
it as mean ± std over independent seeds (each seed = fresh disorder realisations
*and* a fresh SAE). This driver runs ra03 across several seeds and aggregates.

It shells out to `exp_ra03_controls.py` (so there is a single source of truth for
the controls) and collects, per seed:
  - C1 long_range_zz trained-transformer probe R²  (and untrained, raw-h)
  - C4 long_range_zz partial-r given the mean field
  - C3 long_range_zz permutation p-value

Outputs (run_dir/)
------------------
    multiseed_results.json   — per-seed values + mean/std
    summary.md               — the headline table with error bars

Usage
-----
    python scripts/exp_ra06_multiseed.py --seeds 42,43,44 \\
        [--n_samples 800] [--n_perm 200] [--sae_epochs 200]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OBS = "long_range_zz"


def main() -> None:
    ap = argparse.ArgumentParser(description="RA-06: multi-seed headline numbers")
    ap.add_argument("--ckpt", default="runs/ra01_wide/best.pt")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--sae_epochs", type=int, default=200)
    ap.add_argument("--run_dir", default="runs/ra06_multiseed")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in seeds:
        seed_dir = run_dir / f"seed{seed}"
        res_path = seed_dir / "results.json"
        if not res_path.exists():
            print(f"\n=== running ra03 for seed {seed} ===")
            cmd = [sys.executable, "scripts/exp_ra03_controls.py",
                   "--ckpt", args.ckpt, "--n_samples", str(args.n_samples),
                   "--n_perm", str(args.n_perm), "--sae_epochs", str(args.sae_epochs),
                   "--seed", str(seed), "--run_dir", str(seed_dir)]
            subprocess.run(cmd, check=True)
        else:
            print(f"=== seed {seed}: reusing {res_path} ===")
        r = json.loads(res_path.read_text())
        per_seed.append({
            "seed": seed,
            "probe_r2_trained":   r["C1_probe_r2"][OBS]["trained"],
            "probe_r2_untrained": r["C1_probe_r2"][OBS]["untrained"],
            "probe_r2_raw_h":     r["C1_probe_r2"][OBS]["raw_h"],
            "probe_r2_mean_h":    r["C1_probe_r2"][OBS]["mean_h"],
            "partial_r_given_meanh": r["C4_partial"][OBS]["partial_r_given_meanh"],
            "perm_p":             r["C3_null"][OBS]["p_perm"],
        })

    def agg(key):
        v = np.array([p[key] for p in per_seed], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
                "values": v.tolist()}

    summary = {k: agg(k) for k in
               ["probe_r2_trained", "probe_r2_untrained", "probe_r2_raw_h",
                "probe_r2_mean_h", "partial_r_given_meanh", "perm_p"]}
    out = {"observable": OBS, "seeds": seeds, "per_seed": per_seed, "aggregate": summary}
    (run_dir / "multiseed_results.json").write_text(json.dumps(out, indent=2))

    def pm(key):
        a = summary[key]
        return f"{a['mean']:.3f} ± {a['std']:.3f}"

    print(f"\n===== headline ({OBS}), {len(seeds)} seeds =====")
    print(f"  probe R² trained    : {pm('probe_r2_trained')}")
    print(f"  probe R² untrained  : {pm('probe_r2_untrained')}")
    print(f"  probe R² raw-h      : {pm('probe_r2_raw_h')}")
    print(f"  probe R² mean-h     : {pm('probe_r2_mean_h')}")
    print(f"  partial-r | mean-h  : {pm('partial_r_given_meanh')}")

    lines = [f"# RA-06 — multi-seed headline for `{OBS}`\n",
             f"Seeds: {seeds}. N={args.n_samples}, n_perm={args.n_perm}, "
             f"sae_epochs={args.sae_epochs}.\n",
             "| quantity | mean ± std |", "|---|---|",
             f"| probe R² (trained transformer) | {pm('probe_r2_trained')} |",
             f"| probe R² (untrained transformer) | {pm('probe_r2_untrained')} |",
             f"| probe R² (raw h) | {pm('probe_r2_raw_h')} |",
             f"| probe R² (mean h) | {pm('probe_r2_mean_h')} |",
             f"| **partial-r given mean-h** | **{pm('partial_r_given_meanh')}** |",
             f"\nPer-seed partial-r: "
             f"{[round(p['partial_r_given_meanh'], 3) for p in per_seed]}"]
    (run_dir / "summary.md").write_text("\n".join(lines))
    print(f"\n[ra06] outputs in {run_dir}/")


if __name__ == "__main__":
    main()
