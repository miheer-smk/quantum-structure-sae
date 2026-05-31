# Contributing

This is a research project in active development. Contributions, questions, and
collaboration requests are welcome.

## Quick setup

```bash
git clone https://github.com/miheer-smk/quantum-structure-sae.git
cd quantum-structure-sae
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -v   # all tests should pass
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/qsae/` | Reusable library (SAE, observables, shadows, datasets, metrics) |
| `src/qsae/reverse_arrow/` | TFIMTransformer and disordered-TFIM data loader |
| `scripts/` | Experiment entry points (`exp_ra*.py`) |
| `tests/` | pytest test suite — run before every commit |
| `docs/` | Written results and notes |
| `notebooks/` | Exploratory Jupyter notebooks |

## Before submitting a PR

1. **Run the full test suite:** `pytest -v` — all tests must pass.
2. **Run the smoke test:** `python scripts/smoke_test.py` — should complete without error.
3. **Add tests** for any new public function in `src/qsae/`.
4. **Keep commits focused** — one logical change per commit.
5. **Do not commit** data files, model checkpoints, or secrets (see `.gitignore`).

## Experiment naming convention

Experiment scripts follow `exp_<id>_<short_description>.py`.  Add a matching
entry in `docs/` when results are ready (see `docs/week1_results.md` for style).

## Questions

Open a GitHub issue or email the author (see `CITATION.cff` for contact).
