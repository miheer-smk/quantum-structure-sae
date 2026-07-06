# Classical transformers linearly encode non-local quantum order beyond the mean field

*Miheer Kulkarni. Draft extended abstract (~4 pages excl. references). Template-
agnostic Markdown — port to the target workshop's LaTeX style once the CFP is
fixed (candidate venues: NeurIPS ML4PS, ICLR workshops, TMLR). Every number is
reproducible via `bash scripts/reproduce_all.sh`; see `docs/week3_results.md`.*

> **Framing (author-approved).** Claims are made at the level of the *representation*
> — basis-independent linear probes of the residual stream — not individual sparse
> features. A d_hidden × k sweep shows the SAE feature basis is not universal across
> seeds; SAEs appear here only as an exploratory lens and that negative result is
> reported in full (§5).

## Abstract

Do classical neural networks trained on quantum data build internal
representations that reflect the underlying physics? We train a small transformer
to predict ground-state energies of the disordered 1D transverse-field Ising model
(TFIM) from its per-site fields (test R² = 0.9999) and ask whether its
residual-stream activations linearly encode standard quantum observables. The
naive answer is confounded: every observable is a function of the field vector
**h**, which is the network's input, so even an untrained network would look
"structured." Using cross-validated linear probes with an untrained-network
control, a mean-field partial-correlation control, a permutation null, and a depth
sweep, we isolate what is *learned*. We find that the **non-local order parameter**
⟨Z₀Z_{L−1}⟩ is decoded from the trained representation substantially better than
from an untrained network, the raw input, a degree-2 polynomial of the input, or
the mean field; the signal survives partial-correlation control (partial-r =
[MEAN ± STD over 3 seeds]) and a permutation null (p ≈ 0), and it strengthens with
network depth. Observables that are trivially mean-field (phase proximity) are
correctly identified as such — a calibrated negative control. This is, to our
knowledge, a clean demonstration that a classical network trained only on energies
develops a depth-assembled, non-local representation of quantum order, established
with the controls needed to rule out the trivial input-dependence explanation.
Activation patching further shows this representation is *disentangled from* the
energy-prediction pathway — it lives in a low-variance, task-orthogonal subspace —
so the network represents quantum order it does not strictly need for its objective.

## 1. Introduction

A recurring question in both scientific machine learning and mechanistic
interpretability is whether networks trained on physical data internalise known
physical structure, or merely fit input–output statistics. We study this in a
setting where the "known structure" is exactly computable: the 1D TFIM, whose
ground-state observables (entanglement entropy, spin correlators, order parameter)
follow from exact diagonalisation. Our contribution is not a single correlation
but a **controlled** answer. The central methodological point is that raw
feature↔observable correlations are confounded by the fact that the observables
are functions of the input; we show how a small battery of controls separates
learned structure from that confound, and we report honestly where the signal is
in fact trivial.

## 2. Setup

**Task and model.** Hamiltonian H = −J Σ Zᵢ Z_{i+1} − Σ hᵢ Xᵢ, open boundary,
L = 8, J = 1, per-site fields hᵢ ∼ U(0.1, 2.0). A 3-layer Pre-LN transformer
encoder (d_model = 64, 4 heads, 152k params) maps **h** → E₀, trained on 50k
exact-diagonalisation samples to test R² = 0.9999 (RMSE 0.010), a 3.5× RMSE
improvement over a degree-2 polynomial baseline (§Appendix / `docs/week1_results.md`).

**Observables** (exact, from the state vector): half-chain entanglement entropy
S(ρ_A); mean nearest-neighbour correlator ⟨ZᵢZ_{i+1}⟩; transverse magnetization
⟨Xᵢ⟩; the end-to-end correlator ⟨Z₀Z_{L−1}⟩ as the finite-size order parameter;
and phase proximity δ = (h̄−h_c)/h_c. *Note:* the single-site order parameter
mean|⟨Zᵢ⟩| vanishes identically at finite L by the Z₂ symmetry of the ground state
(measured ∼10⁻¹³), so we use the maximal-separation correlator instead.

**Representation.** Mean-pooled residual stream at each encoder layer, evaluated on
held-out disorder realisations.

## 3. Core result: linear decodability of observables

We fit 5-fold cross-validated ridge probes predicting each observable from the
last-layer residual stream, and compare against four baselines: an *untrained*
(random-weight) transformer of the same architecture, the raw input **h**, degree-2
polynomial features of **h**, and the single scalar mean field h̄.

| Observable | Trained | Untrained | Raw h | Poly-2 h | Mean h |
|---|---:|---:|---:|---:|---:|
| S(ρ_A) entropy | 0.934 | 0.938 | 0.863 | 0.941 | 0.670 |
| ⟨ZᵢZ_{i+1}⟩ | 0.995 | 0.991 | 0.971 | 0.988 | 0.945 |
| ⟨Xᵢ⟩ | 0.991 | 0.985 | 0.916 | 0.983 | 0.896 |
| **⟨Z₀Z_{L−1}⟩** | **0.961** | 0.921 | 0.772 | 0.942 | 0.695 |
| phase proximity δ | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |

Two observations. (i) Most observables are decoded nearly as well by an *untrained*
network — generic non-linear mixing of the input suffices, so training adds little.
(ii) The order parameter ⟨Z₀Z_{L−1}⟩ is the exception: the trained network beats
the untrained one and every input baseline. This is the one place where learning,
specifically, matters. The gap is stable across seeds: over 3 independent runs the
trained-network probe R² is 0.963 ± 0.002 vs 0.926 ± 0.006 untrained, 0.765 ± 0.007
raw h, and 0.689 ± 0.010 mean h.

## 4. Controls

**Depth (C2).** ⟨Z₀Z_{L−1}⟩ decodability increases monotonically with layer
(0.916 → 0.945 → 0.961), consistent with the network assembling non-local order
information across depth; other observables are flat.

**Permutation null (C3).** Shuffling each observable 500× and recomputing the best
single-feature |r| over all alive features (a multiple-comparisons control) yields a
null 95th percentile of 0.12–0.16; the observed values (0.79–0.90) give empirical
p ≈ 0.

**Mean-field partial correlation (C4).** Controlling for h̄, the representation
retains a strong beyond-mean-field association with ⟨Z₀Z_{L−1}⟩ — a whole-residual
linear probe gives partial-r = 0.934, 95% bootstrap CI [0.920, 0.948] (the
single-best-SAE-feature analogue is 0.71 ± 0.01 over 3 seeds) — whereas phase
proximity drops to 0.00, correctly identifying it as a pure mean-field quantity.
This calibrated negative control is central: it shows the method does not simply
relabel the input. (We report bootstrap CIs rather than the mechanically-tiny
N-dependent p-values.)

**Causal patching — decodable vs. used.** We ablate the residual direction most
predictive of ⟨Z₀Z_{L−1}⟩ (project it out at the last layer) and measure the effect
on the model's *energy* prediction. The ablation is effective — the order-parameter
probe R² collapses from 0.97 to −9.6 — yet energy-prediction RMSE barely moves
(0.0112 → 0.0137), while ablating random directions of equal norm degrades it ~9×
more (→ 0.100). The residual stream carries ~12× less variance along the order
direction than along a random one. **The order parameter is encoded in a
low-variance subspace approximately orthogonal to the energy-prediction pathway:
represented, but not load-bearing for the trained objective.** This is an honest
negative for the naive "the model uses order to predict energy" hypothesis, and a
more interesting positive — the network organises a disentangled order
representation it does not need for its task.

**Robustness to system size.** Retraining at L = 8, 10, 12 (energy R² = 0.9998
throughout) shows the learned gain on ⟨Z₀Z_{L−1}⟩ — R²(trained) minus the best of
{untrained, raw h, mean h} — is stable at ≈ +0.028 across sizes. The effect is not a
finite-size artifact of L = 8; the mean-field baselines weaken with L as expected,
but absolute decodability weakens in step (a fixed d_model must encode order across a
longer chain), so the advantage is preserved rather than amplified.

## 5. What can and cannot be claimed

**Supported.** (1) The trained representation linearly encodes the non-local order
parameter beyond an untrained network, the raw input, a degree-2 polynomial of it,
and the mean field, with the effect strengthening in depth and surviving both a
partial-correlation control and a permutation null. (2) The pipeline is calibrated:
it reports "trivial" for the trivial observable.

**Not supported.** (1) *Individual, monosemantic sparse features* mapping one-to-one
to named observables: a d_hidden ∈ {256,512,1024} × k ∈ {8,16,32} sweep of TopK SAEs
finds the feature basis is not universal across seeds (best cell ~6% of features
match at cos > 0.7; mean matched cosine ~0.4). We therefore make claims at the
representation level, not the feature level, and report this negative result in full.
(2) Any "quantum advantage" claim — out of scope.

## 6. Limitations and next steps

**Integrability — tested, with a caveat.** Adding a fixed longitudinal field
(non-integrable mixed-field Ising model) makes the learned advantage vanish, but for
an instructive reason: symmetry breaking polarises the ground state so ⟨Z₀Z_{L−1}⟩
becomes almost linear in **h** (raw-h probe R² 0.75 → 0.97), leaving no beyond-input
structure to learn. This clarifies *when* the effect appears — it needs an observable
with genuine beyond-input, non-local content — but it conflates non-integrability
with an input-trivial observable, so a cleaner test (disordered longitudinal field
fed to the model, or the connected correlator) is still needed.
**Scale — addressed.** Re-running the analysis at L = 8, 10, 12 (memory-safe sparse
solver) shows the learned gain is *robust* (stable ≈ +0.028) rather than a
finite-size artifact; it does not amplify at fixed model width, which may require
scaling d_model with L, or the disordered-coupling regime J_{ij}.
**Causality — addressed, with a twist.** Activation patching (above) shows the model
does *not* use the order direction for energy prediction; it is decodable but not
load-bearing. This reframes the open question from "does the model use it?" to "why
does the representation encode order it does not need?" — a question best pursued on
a non-integrable Hamiltonian and at larger L, where long-range order is more likely
to become task-relevant. These define the path from this abstract to a full paper.

## 7. Related work

**Sparse autoencoders & mechanistic interpretability.** Dictionary-learning with
sparse autoencoders was popularised by Bricken et al. (2023) and scaled to a
production model by Templeton et al. (2024); Gao et al. (2024) introduced the TopK
SAE we use and its evaluation methodology; Marks et al. (2024) build causal *feature
circuits*, motivating our decodable-vs-used intervention. Our C5 finding — that the
SAE feature basis is not seed-universal here — is why we make claims at the
representation (linear-probe) level rather than the feature level.

**Interpreting networks trained on physical data.** Iten et al. (2020, *SciNet*)
show a neural network trained on physical time-series recovers the minimal set of
physical parameters; "From Neurons to Neutrons" (2024) reverse-engineers a network
trained on nuclear data. We differ in target and method: a *quantum many-body*
ground-state task, probed for specific known observables with an explicit
untrained-network / mean-field control battery, and a causal patching test.

**Quantum many-body & measurement.** Observables and the TFIM phase structure follow
Sachdev (2011) and Calabrese & Cardy (2004); classical-shadow estimation (used in the
Bars-and-Stripes validation) follows Huang, Kueng & Preskill (2020).

## References

*Verified July 2026; re-confirm venue/pagination against the camera-ready CFP.*

1. Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With
   Dictionary Learning.* Transformer Circuits.
   https://transformer-circuits.pub/2023/monosemantic-features
2. Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable
   Features from Claude 3 Sonnet.* Transformer Circuits.
   https://transformer-circuits.pub/2024/scaling-monosemanticity
3. Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike & Wu (2024).
   *Scaling and evaluating sparse autoencoders.* arXiv:2406.04093.
4. Marks, Rager, Michaud, Belinkov, Bau & Mueller (2024). *Sparse Feature Circuits:
   Discovering and Editing Interpretable Causal Graphs in Language Models.*
   arXiv:2403.19647.
5. Iten, Metger, Wilming, del Rio & Renner (2020). *Discovering Physical Concepts
   with Neural Networks.* Phys. Rev. Lett. 124, 010508. arXiv:1807.10300.
6. *From Neurons to Neutrons: A Case Study in Interpretability* (2024).
   arXiv:2405.17425.
7. Huang, Kueng & Preskill (2020). *Predicting many properties of a quantum system
   from very few measurements.* Nature Physics 16, 1050. arXiv:2002.08953.
8. Sachdev (2011). *Quantum Phase Transitions*, 2nd ed. Cambridge University Press.
9. Calabrese & Cardy (2004). *Entanglement entropy and quantum field theory.*
   J. Stat. Mech. P06002.
