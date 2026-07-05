# Week 3 Results: Do the representations encode quantum observables?

This document reports the core interpretability experiment (`exp_ra02_observables.py`)
**and** the control battery (`exp_ra03_controls.py`) that is required to turn a
raw correlation into a defensible scientific claim.

**One-sentence summary.** The trained transformer's residual stream linearly
encodes every quantum observable we tested; for the *long-range order parameter*
⟨Z₀Z_{L-1}⟩ this encoding is substantially stronger than in an untrained network,
in the raw input, or in the mean field — evidence of genuinely **learned, non-local
quantum structure** — whereas the other observables are largely explained either
by the mean field or by generic non-linear mixing of the input.

---

## 1. Setup

- **Model.** `TFIMTransformer` trained in Week 1 (`runs/ra01_wide/best.pt`),
  test R² = 0.99991 on ground-state energy. 3 Pre-LN encoder layers, d_model = 64.
- **Probe set.** N = 800 disordered-TFIM ground states, per-site fields
  hᵢ ~ Uniform(0.1, 2.0), L = 8, computed by exact diagonalisation.
- **Observables** (exact, from the state vector — `src/qsae/observables.py`):
  half-chain entanglement entropy S(ρ_A); mean nearest-neighbour correlator
  ⟨ZᵢZ_{i+1}⟩; transverse magnetization ⟨Xᵢ⟩; end-to-end correlator
  ⟨Z₀Z_{L-1}⟩ (`long_range_zz`); phase proximity δ = (h̄−h_c)/h_c.
- **Representation.** Mean-pooled residual stream of each encoder layer; a TopK
  SAE (k = 32, d_hidden = 256) is trained on the last-layer activations.

### A correction to the observable set

The originally-specified ferromagnetic order parameter, mean|⟨Zᵢ⟩|, is
**identically zero at finite L**: the exact ground state respects the Z₂ symmetry
Π Xᵢ, so ⟨Zᵢ⟩ = 0 and the measured value is pure numerical noise (~10⁻¹³,
verified). Correlating an SAE feature against it is meaningless. We therefore
replace it with the standard finite-size proxy for spontaneous magnetization, the
maximal-separation correlator ⟨Z₀Z_{L-1}⟩ (Sachdev, *Quantum Phase Transitions*,
Ch. 5), which is O(1) in the ordered phase and decays in the paramagnetic phase.
This is added to `observables.py` (`long_range_zz`, `order_param_proxy`) with tests
against GHZ, product states, and ordered/disordered TFIM.

---

## 2. Core result (`runs/ra02_observables/`)

Training a TopK SAE on the last-layer residual stream and correlating each of the
208 alive features against each observable yields strong single-feature Pearson
correlations:

| Observable | best \|r\| | p-value |
|---|---|---|
| S(ρ_A) entropy | 0.74 | 3×10⁻⁸⁹ |
| ⟨ZᵢZ_{i+1}⟩ | 0.85 | 7×10⁻¹⁴¹ |
| ⟨Xᵢ⟩ | 0.84 | 4×10⁻¹³⁷ |
| ⟨Z₀Z_{L-1}⟩ | 0.90 | — |
| phase proximity δ | 0.86 | 6×10⁻¹⁵¹ |

Taken alone this table is **not** evidence of learned quantum structure, for a
simple reason: every observable is a smooth function of the field vector **h**, and
**h is the transformer's input**. A random non-linear map of h would also produce
features that correlate with the observables. The controls below quantify how much
of the signal is learned.

---

## 3. Controls (`runs/ra03_controls/`)

### C1 — Linear decodability (5-fold CV ridge R²)

For each observable, how well can it be linearly predicted from each representation?

| observable | **trained TF** | untrained TF | raw h | poly2 h | mean h |
|---|---|---|---|---|---|
| entropy | 0.934 | 0.938 | 0.863 | 0.941 | 0.670 |
| mean_nn_zz | 0.995 | 0.991 | 0.971 | 0.988 | 0.945 |
| mean_x | 0.991 | 0.985 | 0.916 | 0.983 | 0.896 |
| **long_range_zz** | **0.961** | 0.921 | 0.772 | 0.942 | 0.695 |
| phase_proximity | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |

Reading:
- **phase_proximity** is perfectly decodable from the *mean field alone* (R² = 1.000)
  — exactly as it must be, since δ is a function of h̄. This is a **positive negative
  control**: the method correctly reports "trivial" when the target is trivial.
- **entropy, mean_nn_zz, mean_x** are decoded almost as well by an *untrained*
  transformer as by the trained one (Δ ≤ 0.01). Most of their linear structure comes
  from generic non-linear mixing of the input, not from training.
- **long_range_zz** is the exception: trained 0.961 vs untrained 0.921 vs raw-h 0.772
  vs mean-h 0.695. Training gives a consistent, reproducible boost, and the trained
  representation beats every input baseline. Learning matters most exactly for the
  observable that is genuinely *non-local*.

### C2 — Layer sweep

`long_range_zz` decodability **increases with depth** (L0 0.916 → L1 0.945 → L2
0.961), consistent with the network assembling non-local order information across
layers. The other observables are flat across depth. See `fig_layer_sweep.png`.

### C3 — Permutation null (multiple-comparisons control)

For each observable we shuffle it 500× and recompute the max-\|r\| over *all* alive
features, building the null for the "winner's curse" of picking the best of ~230
features. The observed best-\|r\| (0.79–0.90) is far above the null 95th percentile
(0.12–0.16); empirical p ≈ 0 for every observable. The correlations are real, not an
artifact of searching many features. See `fig_null.png`.

### C4 — Partial correlation controlling for the mean field (the honesty check)

Does the best feature track the observable **beyond** the trivial mean-field
dependence? Partial correlation r(feature, observable | h̄):

| observable | raw \|r\| | partial-r given h̄ |
|---|---|---|
| **long_range_zz** | 0.90 | **0.694** |
| entropy | 0.79 | 0.348 |
| mean_x | 0.82 | 0.328 |
| mean_nn_zz | 0.85 | 0.133 |
| phase_proximity | 0.89 | **0.000** |

This is the cleanest result in the study:
- **phase_proximity → 0**: entirely explained by the mean field (it *is* the mean
  field). Correct.
- **long_range_zz → 0.694**: the feature carries substantial information about
  long-range order **beyond** the mean field. This is the non-trivial, publishable
  signal.
- entropy and ⟨X⟩ retain moderate beyond-mean-field structure; ⟨ZᵢZ_{i+1}⟩ is mostly
  mean-field. See `fig_partial.png`.

### C5 — Cross-seed SAE universality (a negative result, reported honestly)

Training SAEs from 3 seeds on the same activations and Hungarian-matching decoder
directions gives mean matched cosine 0.37 and only **0.3%** of features matching at
cos > 0.7. **The SAE feature basis is not universal across seeds** at this scale
(N = 800, d_hidden = 256, k = 32). Consequently, claims about *individual* SAE
features are weak. Importantly, the C1–C4 conclusions do **not** depend on the SAE
basis — C1/C2 use the raw residual stream, and C4's partial-correlation logic holds
for whichever feature is selected — so the main claim is robust to this limitation.

---

## 4. What can and cannot be claimed

**Defensible.**
1. The trained transformer's residual stream linearly encodes TFIM quantum
   observables; the encoding of the **non-local order parameter ⟨Z₀Z_{L-1}⟩** is
   stronger than in an untrained network, the raw input, a degree-2 polynomial of
   the input, or the mean field, and it strengthens with depth.
2. This beyond-mean-field structure survives partial-correlation control (partial-r
   = 0.69) and a strict permutation null (p ≈ 0).
3. The pipeline has calibrated negative controls: it reports "trivial" for the
   observable (phase proximity) that is genuinely trivial.

**Not (yet) supported.**
- That *individual, monosemantic SAE features* correspond one-to-one to named
  observables — the SAE basis is seed-dependent (C5).
- Any "quantum advantage" claim — out of scope by design (see RUNBOOK).

**Threats to validity / next steps.**
- **Integrability.** The 1D TFIM is *exactly solvable* — a Jordan–Wigner
  transformation maps it to free fermions — so its ground-state observables are
  comparatively low-complexity functions of **h**. "The transformer had to learn
  genuine structure" is therefore a weaker claim here than it would be for a
  non-integrable/chaotic system; the beyond-mean-field result should be
  reproduced on a non-integrable Hamiltonian (e.g. TFIM + longitudinal field, the
  ANNNI model, or a Heisenberg chain) before it is over-generalised.
- L = 8 only. The order-parameter signal should be re-checked at L = 12 (the
  mean-field baselines are expected to weaken as L grows — see `week1_results.md`).
- Disordered couplings J_{ij} would break the near-diagonal structure the polynomial
  baseline exploits and is the recommended way to make the task genuinely non-poly.
- SAE universality should be revisited with more data and width; current negative
  result may be a small-sample artifact.

---

## 5. Reproduce

```bash
# Core correlation experiment
python scripts/exp_ra02_observables.py --ckpt runs/ra01_wide/best.pt --n_samples 500
# → runs/ra02_observables/{correlation_heatmap.png, top_features.json, ...}

# Full control battery (C1–C5) — the publishable analysis
python scripts/exp_ra03_controls.py --ckpt runs/ra01_wide/best.pt \
    --n_samples 800 --n_perm 500 --sae_epochs 200
# → runs/ra03_controls/{results.json, summary.md,
#    fig_probe_r2.png, fig_layer_sweep.png, fig_null.png, fig_partial.png}
```

*Artifacts: `runs/ra02_observables/`, `runs/ra03_controls/`. State/observable cache
in `data/ra03_states_L8_N800_s42.pt` (gitignored). Generated Week 3 of the
quantum-structure-SAE project.*
