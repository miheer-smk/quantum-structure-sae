# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added (P0 — sharpen the core)
- **Integrability caveat resolved** (`exp_ra10_connected.py`, `runs/ra10_connected`).
  On the *connected* correlator the learned encoding of non-local order persists in
  the non-integrable mixed-field model (g=0.5): connected is not input-decodable
  (raw-h R² 0.26–0.37) yet the trained transformer recovers it (0.81 at L=8, 0.59 at
  L=10), positive learned gain over every baseline. The §3d "vanishing effect" was an
  input-triviality artifact of the raw correlator, not non-integrability. This
  obviates the disordered-g modeling change (former P1#4). week3_results.md §3d.
- `long_range_zz_connected` observable (⟨Z₀Z_{L-1}⟩ − ⟨Z₀⟩⟨Z_{L-1}⟩) + fast helpers
  and tests — subtracts the factorised, input-trivial part to isolate genuinely
  non-local order.
- `exp_ra11_bootstrap.py` — effect sizes with 95% bootstrap CIs for the headline
  (representation-level out-of-fold probe): ⟨Z₀Z_{L-1}⟩ probe R² 0.962 [0.951,0.970],
  partial-r given mean-h **0.934 [0.920,0.948]** (stronger and cleaner than the
  single-feature 0.71). Replaces p-value theatre. week3_results.md updated.

### Added (Steps 2–3 — scaling & non-integrable)
- **L-scaling (`runs/ra08_scaling`, `exp_ra08_scaling.py`).** Retrained the
  transformer at L = 8, 10, 12 (energy R² = 0.9998) and re-ran the ⟨Z₀Z_{L-1}⟩
  probe comparison. The learned gain over the best baseline is *robust* (≈ +0.028
  at every L) — not a finite-size artifact — but does not amplify at fixed model
  width. Written up in `week3_results.md` §3c.
- **Non-integrable mixed-field test (`runs/ra09_mixedfield`, `--g 0.5`).** Honest
  negative: breaking the Z₂ symmetry polarises the ground state so ⟨Z₀Z_{L-1}⟩
  becomes trivially input-decodable (raw-h R² 0.75 → 0.97), and the learned
  advantage disappears (gain ≈ 0 at L = 8, 10). Clarifies that the effect requires
  an observable with beyond-input structure; caveat that the test conflates
  non-integrability with an input-trivial observable. Written up in §3d.
- `docs/week3_results.md` §3c/§3d, workshop abstract, and figures updated; claims
  and limitations sections revised to match (robust across L; not universal across
  Hamiltonians).

### Added (Stage 3 infra — scaling)
- `compute_ground_states_sparse` + sparse Pauli builders in
  `reverse_arrow/data.py` — memory-safe Lanczos ground-state solver supporting
  per-site fields *and* per-bond couplings, scaling to L ≈ 14 (dense path caps at
  L=8; ~0.03 s/state at L=12). Verified against the dense kernel (`test_sparse_solver.py`,
  4 tests). Groundwork for the L-scaling and disordered-coupling studies.
- README polished to a release-quality state: multi-seed error bars, the causal
  result (§3), updated highlights/structure/reproduction, one-command repro.

### Added (Phase 2 — causal, pulled forward)
- `scripts/exp_ra07_causal.py` — activation-patching causal test. Ablating the
  ⟨Z₀Z_{L-1}⟩-predictive residual direction barely changes energy prediction
  (RMSE 0.0112 → 0.0137) while random directions degrade it ~9× more (→ 0.100),
  even though the ablation is effective (order-probe R² collapses 0.97 → −9.6).
  The order parameter is encoded in a low-variance (≈12× less than random),
  approximately task-orthogonal subspace — **represented but not load-bearing**
  for energy prediction. Honest negative for the naive "used" hypothesis; written
  up in `docs/week3_results.md` §3b and the abstract.

### Added (Phase 1 — toward workshop abstract)
- `scripts/exp_ra04_sae_grid.py` — SAE cross-seed universality sweep
  (d_hidden × k). **Robust negative result:** widening d_hidden / shrinking k
  (the RUNBOOK's suggested levers) do not resolve C5 non-universality — best cell
  (256, k=8) reaches only ~6% seed-stable features. Motivates framing the paper at
  the representation level (C1/C2/C4) rather than individual SAE features.
- `compute_all_observables_fast` + `zz_correlator_fast` /
  `transverse_magnetization_fast` / `long_range_zz_fast` / `_z_signs` in
  `observables.py` — dense-operator-free (bit-arithmetic) observables that scale to
  L ≈ 14, since the dense-kron path needs 268 MB per operator at L=12. Verified
  identical to the dense path (`TestFastObservables`, 3 new tests). `exp_ra03` now
  uses the fast path (ED+observables for N=800 in ~20 s vs ~300 s).
- `scripts/exp_ra06_multiseed.py` — runs the control battery across ≥3 seeds and
  reports the headline long-range-ZZ partial-r as mean ± std.
- `EXECUTION_PLAN.md`, `RUNBOOK.md` rewrite (+ `RUNBOOK_starter_template.md`),
  `scripts/reproduce_all.sh` (one-command regeneration).

### Added
- `scripts/exp_ra03_controls.py` — Week 3 control battery that turns the raw
  feature/observable correlations into a defensible claim: (C1) cross-validated
  linear-probe R² of each observable from the trained transformer vs an *untrained*
  transformer, raw h, degree-2 polynomial h, and the mean field; (C2) per-layer
  probe sweep; (C3) permutation null on the best-feature |r| (multiple-comparisons
  control); (C4) partial correlation controlling for the mean field; (C5) cross-seed
  SAE universality. Emits `results.json`, `summary.md`, and four figures.
- `docs/week3_results.md` — full Week 3 write-up: core result, all five controls,
  and honest limitations (mean-field confound, non-universal SAE basis). Headline:
  the residual stream encodes the non-local order parameter ⟨Z₀Z_{L-1}⟩ beyond the
  mean field (partial-r = 0.69, permutation p ≈ 0), stronger than an untrained net.
- `long_range_zz` and `order_param_proxy` observables + tests. **Fix:** the previous
  order parameter mean|⟨Zᵢ⟩| is identically 0 at finite L by Z₂ symmetry (measured
  ~10⁻¹³) and was replaced by the end-to-end correlator ⟨Z₀Z_{L-1}⟩; `exp_ra02`
  updated to use it.

- `docs/exp01_bas_results.md` — write-up of the Bars-and-Stripes QNN→shadow→SAE
  validation run (`runs/exp01/`): QNN test acc 0.979, SAE monosemantic fraction
  0.611, confirming the interpretability pipeline recovers structure on
  known-ground-truth classical data.

### Changed
- `scripts/exp_ra02_observables.py` — swapped the degenerate `order_param` for
  `long_range_zz` in the correlation set (best feature r = 0.907).

### Previously added (Week 3 core)
- `src/qsae/observables.py` — quantum observables module: von Neumann entanglement
  entropy, entanglement spectrum, ZZ correlators (single pair, nearest-neighbour,
  full matrix), transverse and longitudinal magnetization, ferromagnetic order
  parameter, phase proximity, and a batch driver `compute_all_observables`.
- `tests/test_observables.py` — 20 tests covering all observable functions against
  analytically known states (product states, GHZ, uniform TFIM ground states).
- `scripts/exp_ra02_observables.py` — Week 3 experiment: extracts transformer
  residual-stream activations, trains a TopK SAE, and computes Pearson correlations
  between SAE features and the five scalar observables (entropy, ⟨ZZ⟩, ⟨X⟩,
  order parameter, phase proximity). Outputs a correlation heatmap and JSON summary.
- `CITATION.cff` — machine-readable citation metadata (CFF 1.2.0).
- `LICENSE` — MIT license.
- `.github/workflows/ci.yml` — GitHub Actions CI: runs pytest on Python 3.10/3.11/3.12
  and ruff lint on every push and PR.
- `.github/ISSUE_TEMPLATE/` — bug report and feature request templates.
- `CONTRIBUTING.md` — contributor guide with setup instructions and conventions.
- `CHANGELOG.md` — this file.
- `notebooks/` — Jupyter notebook directory (scaffolds added).
- Exported `TFIMTransformer`, `TransformerConfig`, and all observables from top-level
  `qsae` package (`from qsae import TFIMTransformer, compute_all_observables`).
- Updated `README.md` with CI/license/Python badges, Results section with actual
  R² numbers from Week 1, and complete module layout table.
- Updated `pyproject.toml` with full metadata (authors, license, URLs, keywords).

---

## [0.1.0] — 2026-04-17 (Week 1)

### Added
- Initial research scaffold: `TFIMTransformer`, `TopKSAE`, classical shadows
  framework, TFIM data generation, interpretability metrics.
- `scripts/exp_ra01_train_transformer.py` — trains transformer to R² > 0.999 on
  TFIM ground-state energy prediction.
- `scripts/ra01_baseline_check.py` — compares transformer vs linear and poly-2
  baselines; generates pred-vs-truth plot.
- `docs/week1_results.md` — detailed Week 1 results write-up including the
  narrow-h negative result and its physical interpretation.
- 9 tests covering SAE, classical shadows, and transformer.
