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
---

## Locked interpretation (do not drift)

**The effect transfers to the non-integrable ANNNI model — the clean
single-variable test** (transverse fields as input, exactly as in TFIM; only
integrability changed, via next-nearest-neighbour ZZ frustration κ=0.3, with Z₂
preserved so ⟨Z_i⟩=0 and the order proxy stays beyond-input). Energy R²=0.9995,
max|⟨Z_i⟩|=1.0e-10.

**Separation and magnitude are different axes; both are reported.**
- **Separation (confidence the effect is nonzero):** the non-local order parameter
  ⟨Z₀Z_{L-1}⟩ (strong-variance target, σ_y=0.195, no tuning) separates the trained
  representation from the random-init distribution on **both** incremental R²
  (0.0091 [min 0.0082] vs 0.0043, **+5.9σ**) and partial correlation (0.613 [min
  0.523] vs 0.324, **+3.5σ**), **uniform across all 10 seeds** (min trained >
  random p95 on both). Entropy, nn-ZZ, and staggered SF also separate on both cuts.
- **Magnitude (size of the effect):** the incremental R² beyond poly-2(h) is
  **~3× smaller than TFIM** (0.009 vs 0.028). The partial correlation (0.613) is
  **on par with TFIM (0.560)**. So the effect **survives but attenuates** under
  non-integrability: robustly present, yet weaker in absolute incremental terms.
  A large σ-separation does not imply a large effect.

**Non-uniform observable, reported honestly.** ⟨X_i⟩ separates on incremental R²
(+6.3σ) but **not** on partial correlation (0.597 vs 0.584, **+0.3σ, null**). Only
the order proxy, entropy, nn-ZZ, and staggered SF separate on both. The order proxy
— the one that carries the claim — is clean on both; ⟨X_i⟩'s split is recorded
rather than dropped (same discipline as reporting the seed spread).

**Resolves the §8.2 mixed-field caveat.** The earlier mixed-field diagnostic showed
the advantage vanishing, but that was an input-triviality artifact of explicit
symmetry breaking. With Z₂ preserved and the order proxy beyond-input, the effect
**persists in a genuinely non-integrable model** — evidence that it is not a
free-fermion / integrability artifact.
