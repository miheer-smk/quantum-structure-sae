"""
Distribution summaries and the trained-vs-random decision machinery shared by the
multi-seed experiments (Phase 0.6 onward): per-quantity distributions, separation
in random-sd units, p95 stability with auto-bump, per-seed flags, and the
SEPARATION / NULL / UNDERPOWERED classifier that keeps a low-variance-target
non-result from masquerading as a real null.
"""

from __future__ import annotations

import numpy as np


def dist(vals) -> dict:
    """Summary of a 1-D sample: mean, sd (ddof=1), min/max, p05/p95, values."""
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()),
            "p05": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95)),
            "n": int(v.size), "values": [float(x) for x in v]}


def sep_sd(trained: dict, random_d: dict | None) -> float:
    """Separation in random-sd units: (mean_trained - mean_random)/sd_random."""
    if random_d is None or random_d["sd"] == 0:
        return float("nan")
    return (trained["mean"] - random_d["mean"]) / random_d["sd"]


def p95_stability(rand_vals, n_launch: int, tol: float, seed: int = 0) -> dict:
    """Is p95(random) stable at the launch sample size? Returns p95 at n_launch and
    over the full pool, a bootstrap CI of the n_launch p95, and the threshold to use
    (full-pool p95 when 'jumpy', else launch p95)."""
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
            "n_launch": int(n_launch), "n_full": int(v.size), "jumpy": bool(jumpy),
            "threshold": p95_full if jumpy else p95_launch,
            "threshold_source": "full_pool" if jumpy else "launch"}


def seed_flags(trained_values, threshold: float, sd_random: float, margin_sd: float) -> dict:
    """Per-seed transparency: which trained seeds fall below the random threshold or
    land within ``margin_sd`` random-sds above it. Nothing averaged away."""
    below, near = [], []
    for i, val in enumerate(trained_values):
        if val <= threshold:
            below.append({"idx": i, "value": float(val)})
        elif val <= threshold + margin_sd * sd_random:
            near.append({"idx": i, "value": float(val)})
    return {"n_below_threshold": len(below), "n_near_threshold": len(near),
            "below": below, "near": near}


def classify(trained: dict, random_d: dict | None, threshold: float,
             target_std: float, std_floor: float) -> str:
    """Three-way read of a trained-vs-random comparison for one target.

    * SEPARATION  — the trained distribution sits cleanly above the random
                    threshold (min_trained > threshold).
    * UNDERPOWERED— no separation AND the target's ensemble std is below
                    ``std_floor``: the target barely varies, so a non-result
                    cannot be distinguished from a genuine null.
    * NULL        — no separation but the target varies enough that the absence
                    of a trained advantage is informative.
    """
    if random_d is not None and trained["min"] > threshold:
        return "SEPARATION"
    if target_std < std_floor:
        return "UNDERPOWERED"
    return "NULL"
