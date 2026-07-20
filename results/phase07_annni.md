# Phase 0.7 — Hamiltonian diversity: ANNNI

Family `annni`, L=8, input_dim=8, 10 trained seeds vs 64 random inits, eval N=1000.

**Energy test R² across seeds: 0.9995 [min 0.9988, max 0.9999]**  ·  **max|⟨Z_i⟩| = 1.01e-10** (beyond-input protection)

## Headline: poly-2(input) control — trained vs random-init, with σ_y

| observable | σ_y | probe R² (tr) | incr. R² trained | incr. R² random (p95) | sep | flag |
|---|---:|---:|---:|---:|---:|:--|
| entropy | 0.093 | 0.927 | 0.024 ± 0.002 [min 0.021] | 0.010 ± 0.002 (0.012) | +6.6σ | **SEPARATION** |
| staggered_sf | 0.011 | 0.981 | 0.006 ± 0.000 [min 0.005] | 0.004 ± 0.000 (0.004) | +5.6σ | **SEPARATION** |
| long_range_zz | 0.195 | 0.976 | 0.009 ± 0.001 [min 0.008] | 0.004 ± 0.001 (0.005) | +5.9σ | **SEPARATION** |
| mean_nn_zz | 0.095 | 0.995 | 0.006 ± 0.000 [min 0.005] | 0.004 ± 0.000 (0.004) | +6.2σ | **SEPARATION** |
| mean_x | 0.119 | 0.992 | 0.004 ± 0.000 [min 0.004] | 0.003 ± 0.000 (0.003) | +6.3σ | **SEPARATION** |

## Partial correlation | poly-2(input) — trained vs random, with σ_y

| observable | σ_y | partial-r trained | partial-r random (p95) | sep | flag |
|---|---:|---:|---:|---:|:--|
| entropy | 0.093 | 0.565 ± 0.056 [min 0.469] | 0.322 ± 0.061 (0.414) | +4.0σ | **SEPARATION** |
| staggered_sf | 0.011 | 0.741 ± 0.039 [min 0.688] | 0.543 ± 0.064 (0.642) | +3.1σ | **SEPARATION** |
| long_range_zz | 0.195 | 0.613 ± 0.058 [min 0.523] | 0.324 ± 0.082 (0.474) | +3.5σ | **SEPARATION** |
| mean_nn_zz | 0.095 | 0.824 ± 0.041 [min 0.762] | 0.659 ± 0.041 (0.704) | +4.0σ | **SEPARATION** |
| mean_x | 0.119 | 0.597 ± 0.095 [min 0.482] | 0.584 ± 0.041 (0.639) | +0.3σ | **NULL** |

*Flags: SEPARATION = min(trained) > random p95; UNDERPOWERED = no separation and σ_y < 0.1 (target barely varies); NULL = varies enough but no trained advantage.*