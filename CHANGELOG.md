# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
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
