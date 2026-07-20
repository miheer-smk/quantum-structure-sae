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
---

## Locked interpretation (do not drift)

**The effect does not transfer to XXZ as configured** (Jz-disorder, principled
range Jz∼U[0.1, 2.0]). The well-powered target (entropy, σ_y=0.174) is a **clean
null** — incremental R² beyond poly-2(Jz) = 0.0004 vs 0.0004 random (−0.1σ), an
~80× smaller beyond-input signal than TFIM entropy (0.031, +4.6σ). The order
targets (⟨Z₀Z_{L-1}⟩, staggered SF, nn-ZZ) are **underpowered** (σ_y 0.03–0.09,
below the 0.10 floor), so the order-transfer question cannot be adjudicated here.

This is **conservative, not decisive**. The XXZ input is the *couplings* Jz, which
directly govern the ZZ observables, so poly-2(Jz) partly **saturates** their
decodability (entropy probe R² 0.951) — the same input-triviality confound family
as the draft's §8.2 mixed-field diagnostic. The result therefore reads as *"the
effect is absent in this configuration, for reasons we cannot fully separate from
the input choice,"* **NOT** as *"U(1) symmetry kills the effect."*

**Why XXZ cannot isolate a cause.** XXZ changes **two variables at once** relative
to TFIM — the input type (transverse fields → couplings) *and* the symmetry
(Z₂ → U(1)). It therefore cannot attribute the null to either. **ANNNI is the
clean single-variable test**: it holds the input type fixed (transverse fields,
exactly as in TFIM) and changes only integrability (nnn ZZ frustration), so ANNNI
carries the real weight of the diversity section. This framing is fixed *before*
ANNNI's numbers land, so ANNNI is not retrofitted to a convenient story.

**Named missing control (future work, not run here).** A **field-disorder XXZ**
(disorder a transverse/staggered field on top of fixed couplings, so the observable
is a non-trivial function of the field as in TFIM) is the specific experiment that
would disentangle *"U(1) kills it"* from *"coupling-as-input made it trivial."* It
is parked as explicit future work; the XXZ null is reported as confounded with the
fix named.
