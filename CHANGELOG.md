# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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
