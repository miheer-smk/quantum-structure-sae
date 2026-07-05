# quantum-structure-sae — Execution Plan
### Phase 1: Finish the current version → Phase 2: Upgrade for top journals/conferences

Prepared July 5, 2026, from a direct clone + test run of `main`
(HEAD `a785418`, "Week 3: SAE observable correlations + control battery, fix
order parameter"). Re-verify every number below before relying on it — the repo
moves on. The stage↔document map and known compute ceilings live in
[`RUNBOOK.md`](RUNBOOK.md).

**How to use this doc:** work Phase 0 + Phase 1 first, inside the repo root. Do
not start Phase 2 until the Phase 1 exit gate is actually met — the two phases
target different deliverables (a workshop extended abstract vs. a full paper).

---

## Verified context

- **Research question:** does a classical transformer trained on TFIM
  ground-state energies (params → energy regression) develop internal
  representations that linearly encode quantum observables?
- **Headline result** (`docs/week3_results.md`): the residual stream encodes the
  non-local order parameter `⟨Z₀Z_{L-1}⟩` beyond the mean-field confound
  (partial-r = 0.694 vs. 0.000 for the trivial control), survives a permutation
  null (p ≈ 0), and strengthens with depth (L0 0.916 → L1 0.945 → L2 0.961).
- **The crack (C5):** the SAE feature basis is *not* stable across seeds
  (mean matched cosine 0.37; only 0.3% match at cos > 0.7). The strongest
  results (C1/C2/C4) are about the raw residual stream, not individual SAE
  features — resolve this framing mismatch before scaling anything else.
- **Code health:** 37/37 tests pass; CI's exact lint command now exits 0.

---

## PHASE 0 — Orient (every session, ~5 min)

1. `git log --oneline -10` and `git status`.
2. `pip install -e ".[dev]"` then `pytest -q` — confirm test count.
3. `ruff check src/ tests/ scripts/ --select=E,F,W --ignore=E501,W291,W293`.
4. Read `docs/week3_results.md` in full before claiming "what we found."

---

## PHASE 1 — Finish the current version
**Target: a submittable 4-page workshop extended abstract.**

- **1.1 Housekeeping.** ruff to 0; `pyproject` description matches reality;
  RUNBOOK/pyproject/week-numbering reconciled.
- **1.2 SAE-universality crack (critical path).** Grid `d_hidden ∈ {256,512,1024}`,
  `k ∈ {8,16,32}`, more data/epochs; recompute C5. If it improves, re-run the
  full C1–C5 at the winner. If not, reframe the paper around the
  representation-level result (C1/C2/C4) and demote the SAE — **author's call,
  surface both outcomes.**
- **1.3 L=12 + disordered couplings J_ij.** Re-run the analysis; report whether
  the long-range-ZZ result survives.
- **1.4 Multi-seed the headline.** ≥3 seeds; report the long-range-ZZ partial-r
  as mean ± std.
- **1.5 Integrability caveat.** 1D TFIM is exactly solvable (Jordan–Wigner →
  free fermions) — note it in the limitations.
- **1.6 Workshop draft.** 4-page NeurIPS-style abstract compressing `docs/*.md`.
- **1.7 Reproducibility.** One command regenerates every number/figure.

**Phase 1 exit gate:** tests green + lint 0 · docs consistent · SAE crack fixed
or framing updated (author signed off) · L=12 & disordered-J written up ·
headline partial-r as mean ± std over ≥3 seeds · 4-page draft in a workshop
template · one-command reproduction on a clean checkout.

---

## PHASE 2 — Upgrade for top venues
**Target: full paper (NeurIPS/ICML/ICLR-caliber or a physics-ML journal).**

- **2.1** Break integrability — add a non-integrable Hamiltonian (TFIM +
  longitudinal field, ANNNI, or XY/Heisenberg); show the depth-increasing,
  beyond-mean-field signal holds (or not).
- **2.2** Scale L via ED (~14 sites), then DMRG/tensor networks (scope as its
  own tooling task).
- **2.3** Make the claim *causal*: ablate/patch the residual direction most
  predictive of `long_range_zz` and check energy predictions degrade
  specifically for long-range-order-sensitive inputs.
- **2.4** Full ablation grid (SAE k/d_hidden, transformer depth/width, n_samples).
- **2.5** Verified related work (search, do not invent).
- **2.6** Scoped theory: a real proof or an explicit empirical regularity +
  conjecture, clearly labeled.
- **2.7** Effect sizes + bootstrap CIs alongside (not instead of) the C3
  permutation null; avoid bare extreme p-values.
- **2.8** Full draft + venue logistics — **re-verify all deadlines before acting.**
- **2.9** Human review: one QML/physics reader + one mech-interp reader.

**Phase 2 exit gate:** second (non-integrable) Hamiltonian result · causal
intervention result · full ablation grid · verified related work · theory proven
or labeled conjecture · effect sizes + CIs · full draft in venue template · QML
and mech-interp readers have commented.

---

## Standing rules
1. Never invent a citation, number, or result — search and verify first.
2. Don't silently decide the paper's framing (SAE reframe 1.2, theory-vs-conjecture
   2.6, venue 2.8 are the author's calls).
3. Match the `docs/week*.md` standard of stating what can/cannot be claimed;
   report negative results as prominently as positive ones.
4. Commit incrementally; keep `CHANGELOG.md` current.
5. Re-verify every date/deadline before acting on it.
