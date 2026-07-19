# Phase 0.5 — Full-input recoverability control (kill-shot)

Run: `runs/phase05_input_control_20260719-132315_s42/`
Config: `configs/phase05_input_control.yaml` · Narrative: `docs/phase05_input_control.md`
Date: 2026-07-19

**Question.** Does the non-local order headline survive controlling for the FULL
site-resolved input and its degree-2 polynomial (not just the scalar mean field
h̄)? Runs on the existing `runs/ra01_wide/best.pt` (one trained model) vs a
random-init distribution (12 inits × 3 eval seeds = 36 draws). Bootstrap 95% CIs.

## Probe R² (reproduces draft Table 2)

| observable | trained | untrained (1 draw) |
|---|---|---|
| entropy | 0.930 | 0.935 |
| mean_nn_zz | 0.994 | 0.991 |
| mean_x | 0.991 | 0.983 |
| long_range_zz | 0.963 | 0.911 |
| phase_proximity | 0.999 | 0.999 |

## Partial correlation r(repr-pred, observable | control), trained [95% CI]

| observable | mean_h | raw_h | poly2_h |
|---|---|---|---|
| entropy | 0.894 [0.879,0.908] | 0.723 [0.686,0.758] | 0.613 [0.536,0.673] |
| mean_nn_zz | 0.951 [0.943,0.958] | 0.911 [0.896,0.924] | 0.835 [0.794,0.863] |
| mean_x | 0.959 [0.952,0.965] | 0.948 [0.940,0.956] | 0.750 [0.690,0.792] |
| **long_range_zz** | 0.938 [0.924,0.950] | 0.916 [0.898,0.933] | **0.648 [0.576,0.710]** |
| phase_proximity | 0.000 | −0.000 | 0.000 |

## Under the strongest (poly-2) control: trained vs random-init distribution

| observable | trained partial-r [95% CI] | random-init mean±std (p95, max) | clears random p95? |
|---|---|---|---|
| **⟨Z₀Z_{L-1}⟩ (non-local order)** | **0.648 [0.576, 0.710]** | 0.301 ± 0.115 (0.478, 0.495) | **yes** (0.576 > 0.478) |
| ⟨ZᵢZ_{i+1}⟩ nn correlator | 0.835 [0.794, 0.863] | 0.405 ± 0.139 (0.609, 0.649) | yes |
| ⟨Xᵢ⟩ transverse mag. | 0.750 [0.690, 0.792] | 0.392 ± 0.120 (0.603, 0.655) | yes |
| S(ρ_A) half-chain entropy | 0.613 [0.536, 0.673] | 0.430 ± 0.080 (0.557, 0.581) | **no** (0.536 < 0.557) |
| phase proximity δ | 0.000 | 0.000 | trivial (correct) |

Incremental R² beyond poly-2(h), trained vs (single untrained draw):
⟨Z₀Z_{L-1}⟩ **0.032 [0.025, 0.039]** vs 0.009; entropy 0.034 vs 0.021;
⟨X⟩ 0.013 vs 0.008; nn-ZZ 0.010 vs 0.006; phase proximity 0.000.

## Verdict

- **Headline SURVIVES** the full poly-2 input control: trained partial-r 0.648
  exceeds all 36 random inits (max 0.495); CI lower bound 0.576 clears random p95
  0.478. Resolves draft Limitation #3; validates probe-primary framing.
- **Honest caveat:** effect is real but **smaller than the scalar-control number
  (0.71) implied** — trained 0.65 vs random 0.30 ± 0.12 (~3σ), not "vs ≈ 0". The
  single untrained draw (0.107) was a low-tail outlier the multi-init control caught.
- Under this stronger lens, nn-ZZ and ⟨X⟩ also beat random-init (draft called them
  training-neutral); entropy does NOT clearly beat it; phase proximity exactly trivial.
- **Known n=1 limitation on the trained side → Phase 0.6** retrains multiple seeds
  so both sides are distributions.
