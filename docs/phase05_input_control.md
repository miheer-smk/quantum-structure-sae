# Phase 0.5 — Full-input recoverability control (the "kill-shot")

**Question.** The TMLR draft's headline (the trained residual stream encodes the
non-local order proxy ⟨Z₀Z_{L-1}⟩ "beyond the input") is controlled only against
the **scalar mean field** h̄ (draft Table 3/4; partial-r = 0.706 ± 0.011). Draft
Limitation #3 concedes this removes "one specified scalar confound, not arbitrary
nonlinear dependence on the full field vector." Since a degree-2 polynomial of the
raw input already reaches probe R² = 0.942 vs the trained 0.961 (draft Table 2),
the real test is: **does the effect survive controlling for the full site-resolved
input and its degree-2 polynomial?**

**Design.** For each observable we compute the partial correlation between the
trained residual stream's out-of-fold ridge prediction and the observable, after
regressing out (OLS) a control set Z ∈ {mean-h, raw-h, poly2-h}, plus the
incremental R² = R²([repr, Z]) − R²(Z). Baselines: (i) an architecture-matched
untrained net (single draw, draft convention), and (ii) a **random-init
distribution** of 12 independent inits × 3 data seeds = 36 draws — so the trained
model is compared to a *distribution* of untrained nets, not one draw. 95%
percentile-bootstrap CIs (4000 resamples, control refit per replicate). Runs on
the existing `runs/ra01_wide/best.pt` and cached `data/ra03_states_L8_N800_s{42,
43,44}.pt`; no training or ED. Statistics unit-tested against analytic cases
(`tests/test_input_control.py`). At g = 0 the Z₂ symmetry gives ⟨Zᵢ⟩ = 0, so
`long_range_zz` here **is** the connected (genuinely non-local) correlator.

## Result — partial correlation under the strongest (poly-2) input control

| observable | trained partial-r [95% CI] | random-init mean ± std (p95, max) | clears random p95? |
|---|---|---|---|
| **⟨Z₀Z_{L-1}⟩ (non-local order)** | **0.648 [0.576, 0.710]** | 0.301 ± 0.115 (0.478, 0.495) | **yes** (0.576 > 0.478) |
| ⟨ZᵢZ_{i+1}⟩ nn correlator | 0.835 [0.794, 0.863] | 0.405 ± 0.139 (0.609, 0.649) | yes |
| ⟨Xᵢ⟩ transverse mag. | 0.750 [0.690, 0.792] | 0.392 ± 0.120 (0.603, 0.655) | yes |
| S(ρ_A) half-chain entropy | 0.613 [0.536, 0.673] | 0.430 ± 0.080 (0.557, 0.581) | **no** (0.536 < 0.557) |
| phase proximity δ | 0.000 [−0.000, 0.000] | 0.000 ± 0.000 (0.000) | trivial (correct) |

Incremental R² beyond poly-2(h), trained vs random-init (single untrained draw):
⟨Z₀Z_{L-1}⟩ **0.032 [0.025, 0.039]** vs 0.009; entropy 0.034 vs 0.021; ⟨X⟩ 0.013
vs 0.008; nn-ZZ 0.010 vs 0.006; phase proximity 0.000.

## Verdict — the headline SURVIVES, with an honest effect size

1. **The non-local order result holds under the strongest input control.** Against
   the full degree-2 polynomial of the input, the trained partial correlation
   (0.648, CI [0.576, 0.710]) exceeds **all 36 random inits** (max 0.495) and its
   CI lower bound clears the random-init 95th percentile (0.478). This directly
   answers draft Limitation #3 and converts it into a strength.

2. **But the effect size is smaller and the baseline higher than the scalar-control
   number implied.** The mean-field partial-r (0.71) overstated the gap: random
   transformers already reach partial-r ≈ 0.30 against poly-2 for ⟨Z₀Z_{L-1}⟩. The
   honest statement is **trained 0.65 vs random-init 0.30 ± 0.12** (~3 std), *not*
   "trained vs ≈ 0". The single untrained draw used in the draft (0.107) was on the
   low tail of the random-init distribution and should not be leaned on — this is
   exactly what the multi-init control caught.

3. **Training helps more broadly than the draft claims — but least-trivially for the
   order parameter.** Under poly-2 control, nn-ZZ and ⟨X⟩ *also* beat the random-init
   p95, which the draft (using raw probe R²) reported as training-neutral. The
   partial-correlation lens is more sensitive. Crucially, ⟨Z₀Z_{L-1}⟩ is the
   observable with the **lowest random-init baseline** (0.30 vs 0.39–0.43): it is
   where beyond-input structure is hardest to obtain without training.

4. **Entropy does not clearly beat random-init** under the strongest control
   (trained CI lower bound 0.536 vs random p95 0.557) — reported as a negative.

5. **Phase proximity is exactly trivial** under every control — the calibrated
   negative control is intact.

## Consequences for the paper

- The probe-primary framing is **validated**: the representation-level result is
  real beyond the full input, not an artifact of a weak scalar control.
- Replace the draft's mean-field-only Table 3/4 headline with this **full-input,
  distribution-baselined** version. It is a stronger, more defensible claim with a
  correctly-sized effect.
- Open follow-up (for symmetry): compare against **multiple trained seeds**, not one
  checkpoint, so both sides of the trained-vs-random comparison are distributions.
  Requires retraining the transformer at ≥3 seeds.

Artifacts: `runs/phase05_input_control_*/` (`results.json`, `summary.md`,
`config_resolved.yaml`, `meta.json` with git commit + env, `metrics.jsonl`).
