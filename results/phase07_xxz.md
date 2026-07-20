# Phase 0.7 — Hamiltonian diversity: XXZ

Family `xxz`, L=8, input_dim=7, 10 trained seeds vs 64 random inits, eval N=1000.

**Energy test R² across seeds: 0.9996 [min 0.9992, max 0.9999]**  ·  **max|⟨Z_i⟩| = 8.53e-15** (beyond-input protection)

## Headline: poly-2(input) control — trained vs random-init, with σ_y

| observable | σ_y | probe R² (tr) | incr. R² trained | incr. R² random (p95) | sep | flag |
|---|---:|---:|---:|---:|---:|:--|
| entropy | 0.174 | 0.951 | 0.000 ± 0.000 [min 0.000] | 0.000 ± 0.000 (0.001) | -0.1σ | **NULL** |
| staggered_sf | 0.043 | 0.953 | 0.002 ± 0.000 [min 0.001] | 0.001 ± 0.000 (0.002) | +2.1σ | **UNDERPOWERED** |
| long_range_zz | 0.086 | 0.910 | 0.001 ± 0.000 [min 0.000] | 0.001 ± 0.000 (0.001) | -0.1σ | **UNDERPOWERED** |
| mean_nn_zz | 0.031 | 0.970 | 0.002 ± 0.001 [min 0.001] | 0.002 ± 0.000 (0.002) | +1.4σ | **UNDERPOWERED** |

## Partial correlation | poly-2(input) — trained vs random, with σ_y

| observable | σ_y | partial-r trained | partial-r random (p95) | sep | flag |
|---|---:|---:|---:|---:|:--|
| entropy | 0.174 | 0.088 ± 0.081 [min -0.041] | 0.271 ± 0.099 (0.421) | -1.8σ | **NULL** |
| staggered_sf | 0.043 | 0.153 ± 0.089 [min 0.039] | 0.360 ± 0.062 (0.427) | -3.3σ | **UNDERPOWERED** |
| long_range_zz | 0.086 | -0.014 ± 0.064 [min -0.122] | 0.233 ± 0.090 (0.387) | -2.7σ | **UNDERPOWERED** |
| mean_nn_zz | 0.031 | 0.206 ± 0.091 [min 0.064] | 0.407 ± 0.096 (0.555) | -2.1σ | **UNDERPOWERED** |

*Flags: SEPARATION = min(trained) > random p95; UNDERPOWERED = no separation and σ_y < 0.1 (target barely varies); NULL = varies enough but no trained advantage.*