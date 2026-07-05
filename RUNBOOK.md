# Runbook — `quantum-structure-sae`

> **Note on this file.** The original runbook (a generic QNN + classical-shadow +
> Bars-and-Stripes plan from the `qsae` starter template) has been preserved as
> [`RUNBOOK_starter_template.md`](RUNBOOK_starter_template.md). The project pivoted
> to the **`reverse_arrow` / TFIM-transformer interpretability** line, and this
> runbook describes *that* actual roadmap. The starter template is still worth
> reading for its generic advice — the failure-modes table, the "share the draft
> with one QML and one mech-interp reader" rule, and the Bricken/Marks/Templeton
> reading list all still apply.

## What this project actually is

A classical transformer is trained to regress the ground-state energy of the 1D
disordered Transverse-Field Ising Model (TFIM) from per-site fields **h**
(R² = 0.9999). We then ask whether its internal representations linearly encode
known quantum observables, using SAEs **and** linear probes, with a control
battery that separates *learned* structure from the trivial fact that the
observables are functions of the input **h**.

The Bars-and-Stripes QNN→shadow→SAE pipeline (from the starter template) is kept
as a *classical-data validation* of the interpretability stack, not as the main
experiment.

## Project stages ↔ documents (resolves the "week" numbering)

The `docs/week*.md` filenames are historical; think in **stages**:

| Stage | What | Script(s) | Write-up |
|---|---|---|---|
| **1 — Energy prediction** | Train TFIMTransformer + poly/linear baselines | `exp_ra01_train_transformer.py`, `ra01_baseline_check.py` | [`docs/week1_results.md`](docs/week1_results.md) |
| **2 — Interpretability + controls** | SAE on residual stream; C1–C5 control battery | `exp_ra02_observables.py`, `exp_ra03_controls.py` | [`docs/week3_results.md`](docs/week3_results.md) |
| **(validation)** | BAS QNN→shadow→SAE sanity check | `exp01_bas3.py` | [`docs/exp01_bas_results.md`](docs/exp01_bas_results.md) |

The starter template's "Week 2 baselines" were folded into Stage 2's control
battery (`exp_ra03_controls.py`, controls C1–C5). There is deliberately no
separate "Week 2" document.

## Roadmap (see `EXECUTION_PLAN.md` for the detailed two-phase plan)

**Phase 1 — finish the current version (target: 4-page workshop abstract).**
Housekeeping (lint/docs) ✓ · resolve the SAE cross-seed universality crack (C5) ·
extend to L=12 and disordered couplings J_ij · multi-seed the headline
long-range-ZZ partial-r · add the integrability caveat · write the draft ·
one-command reproduction.

**Phase 2 — upgrade for a top venue (target: full paper).** Break integrability
(second Hamiltonian) · scale L via ED then DMRG/tensor networks · make the
interpretability claim *causal* via activation patching, not just correlational ·
full ablation grid · verified related work · scoped theory contribution ·
effect sizes + CIs over bare p-values.

## Standing rules

1. Never invent a citation, number, or result — search and verify first.
2. Framing decisions (SAE reframe, theory-vs-conjecture, venue) are the author's
   call: present evidence and options, don't decide silently.
3. Match the `docs/week*.md` standard of stating *what can and cannot be claimed*;
   report negative results (e.g. C5 non-universality) as prominently as positive.
4. Commit incrementally; keep `CHANGELOG.md` current.

## Quick reference

```bash
pip install -e ".[dev]"
pytest -q                                                    # 37 tests
ruff check src/ tests/ scripts/ --select=E,F,W --ignore=E501,W291,W293
python scripts/smoke_test.py                                 # end-to-end sanity
bash scripts/reproduce_all.sh                                # regenerate all results
```

### Known compute ceilings
- Exact diagonalisation: dense, fine to **L ≤ 12** on 16 GB RAM; L = 13–14 needs
  sparse `eigsh` and a memory-efficient observables path; L > 14 needs
  DMRG/tensor-network ground states (new tooling, scope separately).
