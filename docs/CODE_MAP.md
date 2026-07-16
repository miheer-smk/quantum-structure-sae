# CODE_MAP — repository audit (Phase 0)

Audited 2026-07-16 at HEAD `4b46138` ("P0#1: connected correlator resolves the
integrability caveat"). One untracked file: `scripts/exp_ra12_ablations.py`.

**Important context for the new roadmap:** this repo is *not* greenfield. It
already contains a working TFIM-transformer → probe/SAE pipeline with a control
battery, causal patching, L-scaling, a non-integrable extension, bootstrap CIs,
multi-seed error bars, and a workshop-abstract draft. The new phase plan should
*extend and harden* this apparatus, not rebuild it. Existing results worth
preserving are summarized at the bottom.

---

## 1. Package: `src/qsae/`

### Core science modules (active line of research: "reverse arrow")

| File | What it does | Status |
|---|---|---|
| `reverse_arrow/transformer.py` | `TFIMTransformer` — Pre-LN encoder (default L=8 sites, d_model=64, 3 layers, 152k params). Input: per-site fields `h ∈ R^L` → scalar `E₀`. Mean-pool + MLP head. No built-in activation hooks; scripts attach ad-hoc `register_forward_hook`s. | Solid. Needs a proper `hooks.py`. |
| `reverse_arrow/data.py` | Disordered-TFIM data engine. Two ED kernels: (1) dense batched `eigvalsh` (fast at L=8, energies only), (2) `compute_ground_states_sparse` — sparse Lanczos (`eigsh`), supports per-site `h_i`, per-bond `J_b`, longitudinal `g_i` (non-integrable mixed-field). `TFIMDataset` + `make_splits` (uniform disorder, .pt caching). | Solid to L≈14. No XXZ/Heisenberg. Only open BC. |
| `observables.py` | Exact-state observables: half-chain von Neumann entropy (partial trace via reshape), entanglement spectrum, ⟨Z_iZ_j⟩ (dense-kron *and* fast bit-arithmetic paths), ⟨X_i⟩, ⟨Z_i⟩, `long_range_zz` = ⟨Z₀Z_{L-1}⟩ (finite-size order proxy; mean|⟨Z_i⟩| is 0 by Z₂ symmetry), `long_range_zz_connected` (subtracts ⟨Z₀⟩⟨Z_{L-1}⟩), `phase_proximity` δ=(h̄−h_c)/h_c, batch drivers `compute_all_observables{,_fast}`. | Solid, well-tested (35 tests). "distance-to-critical" exists as δ. |
| `sae.py` | `TopKSAE` (Gao et al. 2024): exact-k TopK, unit-norm decoder columns, tied init, aux-k dead-feature revival, `last_fired` tracker. | Solid. No FVU/absorption/splitting metrics yet. |
| `metrics.py` | Polysemanticity (class-entropy), cross-seed feature matching (Hungarian on decoder cosines) + `universality_score`, dead-fraction, top-activating examples. | Reusable for Phase-3 SAE metrics. |
| `training.py` | Simple `train_qnn` / `train_sae` loops. The *real* transformer training loop lives in `scripts/exp_ra01_train_transformer.py` (AdamW, cosine LR, early stop, energy normalization, R² on unnormalized energies). | Training logic is script-bound; Phase-2 should lift it into the package. |
| `datasets.py` | Bars-and-Stripes, MNIST-4×4 stubs, and `tfim_ground_states(n, h_values)` for the *clean* (uniform-h) TFIM via sparse eigsh. | Clean-TFIM path used by tests/plots. |

### Future-work modules (do not touch, per project rules)

| File | What it does |
|---|---|
| `qnn.py` | PennyLane variational circuits (`TorchQNN`) with `state_at_depth` hooks — used only by the Bars-and-Stripes validation line. |
| `shadows.py` | Classical shadow tomography (Pauli + Clifford, in simulation) → feature vectors for the SAE. Used only by the BAS line. |

Top-level `__init__.py` re-exports everything (imports pennylane at package
import time — a heavy import that any new config/logging module should avoid
triggering unnecessarily).

## 2. Scripts (`scripts/`) — one per experiment, CLI-arg driven (no configs)

| Script | Experiment | Key artifact |
|---|---|---|
| `exp_ra01_train_transformer.py` | Stage 1: train TFIMTransformer, L=8, 50k train, target R²>0.995. Hyperparams in a `BASE_CFG` dict in-file. | `runs/ra01_wide/best.pt` (test R²=0.9999) |
| `ra01_baseline_check.py` | Linear + poly-2 baselines vs transformer; pred-vs-truth figure. | `figures/ra01_wide_pred_vs_truth.png` |
| `exp_ra02_observables.py` | TopK SAE on last-layer residual; Pearson feature×observable heatmap. | `runs/ra02_observables/` |
| `exp_ra03_controls.py` | **The control battery C1–C5**: C1 probe-decodability (trained vs untrained vs raw-h vs poly2-h vs mean-h), C2 layer sweep, C3 permutation null on max-|r| (winner's-curse control), C4 partial-r given mean field, C5 cross-seed SAE universality. | `runs/ra03_controls/results.json` |
| `exp_ra04_sae_grid.py` | d_hidden×k sweep of C5 universality. **Robust negative**: best cell ~6% seed-stable features. | `runs/ra04_sae_grid/` |
| `exp_ra06_multiseed.py` | Shells out to ra03 across seeds 42/43/44; headline as mean±std. | `runs/ra06_multiseed/` |
| `exp_ra07_causal.py` | Activation patching: ablate ridge "order direction" at last layer; measure energy RMSE (ordered vs para) + sanity re-probe + variance-along-direction. Result: *decodable but not load-bearing*. | `runs/ra07_causal/` |
| `exp_ra08_scaling.py` | Retrain at L=8/10/12 (sparse solver); learned gain stable ≈+0.028. Also `--g` for mixed-field (ra09). | `runs/ra08_scaling/`, `runs/ra09_mixedfield/` |
| `exp_ra10_connected.py` | Raw vs *connected* correlator, g∈{0,0.5}: effect survives non-integrability on the connected quantity (+0.125 gain at L=8). | `runs/ra10_connected/` |
| `exp_ra11_bootstrap.py` | Percentile-bootstrap 95% CIs for the representation-level headline: partial-r 0.934 [0.920, 0.948]. | `runs/ra11_bootstrap/` |
| `exp_ra12_ablations.py` | (untracked) Capacity sweep (n_layers×d_model) + probe robustness (n, ridge α). | `runs/ra12_ablations/` |
| `exp01_bas3.py`, `smoke_test.py` | Bars-and-Stripes QNN→shadow→SAE validation line; end-to-end smoke. | `runs/exp01/` |
| `reproduce_all.sh` | One-command regeneration of everything (FAST=1 / SKIP_BAS=1 switches). | — |

**Pattern shared by all analysis scripts** (and duplicated in each): generate
disorder samples → ED → observables (cached .pt in `data/`) → extract mean-pooled
residual stream via forward hooks → 5-fold CV ridge probe R² (`cv_probe_r2`) →
partial correlation → JSON + summary.md + figure. This duplication
(`cv_probe_r2`, `partial_corr`, `pooled`, `gen_states` appear in ≥4 scripts) is
the main refactor target for the new `analysis/`/`baselines/` package modules.

## 3. Tests (`tests/`, 49 test functions)

- `test_observables.py` (35) — the physics validation core: Bell/GHZ entropy =
  log 2, product states = 0, entropy peaks near g_c for clean TFIM, correlator
  signs/limits in both phases, fast-path ≡ dense-path equivalence.
- `test_sparse_solver.py` (5) — sparse Lanczos vs dense kernel agreement,
  per-bond J, mixed-field g.
- `test_transformer.py` (3), `test_sae.py` (2), `test_shadows.py` (4).
- No slow/fast markers; suite takes ~5–10 min because shadow tests are heavy.

## 4. Docs & results

- `docs/week1_results.md` — Stage 1 (energy regression; includes a narrow-h negative result).
- `docs/week3_results.md` — **the main results document**: controls C1–C5, causal §3b, scaling §3c, mixed-field §3d + connected-correlator resolution, "what can/cannot be claimed."
- `docs/exp01_bas_results.md` — BAS validation.
- `paper/workshop_abstract.md` — 4-page draft with verified related work.
- `EXECUTION_PLAN.md`, `RUNBOOK.md` — the prior two-phase roadmap (workshop → full paper); Phase-1 exit gate essentially met.
- `runs/*/summary.md` — per-experiment human-readable summaries (committed; heavy artifacts gitignored).

## 5. Infrastructure gaps (what Phase 0 adds)

1. **No `configs/`** — every experiment is CLI-args + in-file dicts. Resolved
   configs are saved (ra01 writes `config.json`) but not versioned inputs.
2. **No unified run logger** — each script prints + writes its own JSON; no
   JSONL/W&B, no git-hash capture, no runs manifest.
3. **No global seeding utility** — each script seeds ad-hoc (some seed torch
   only); no determinism flags recorded.
4. **No fast/slow pytest markers, no Makefile** — CI (`.github/workflows/ci.yml`)
   runs full pytest + ruff on 3.10/3.11/3.12.
5. **Analysis helpers duplicated across scripts** (see §2 pattern note).
6. `partial_corr` handles only a *scalar* control (mean field). The new
   input-recoverability control requires partialling out the full `h` vector
   (regress-out-then-correlate), which is a genuine methodological upgrade, not
   just a refactor.

## 6. Established results the new phases must not regress

| # | Finding | Where |
|---|---|---|
| 1 | Energy regressor test R² = 0.9999 (L=8, 50k) | ra01 |
| 2 | ⟨Z₀Z_{L-1}⟩ decodable beyond untrained/raw-h/poly2/mean-h; strengthens with depth | ra03 C1/C2 |
| 3 | Representation-level partial-r given mean-h = 0.934 [0.920, 0.948] | ra11 |
| 4 | Phase proximity correctly reported trivial (calibrated negative control) | ra03 C4 |
| 5 | SAE feature basis NOT universal across seeds (~0.3–6% at cos>0.7) — claims made at representation level | ra03 C5 + ra04 |
| 6 | Order direction is decodable but **not load-bearing** (ablation ≈ no energy effect; lives in low-variance task-orthogonal subspace) | ra07 |
| 7 | Learned gain stable across L=8/10/12 (≈+0.028), not amplifying | ra08 |
| 8 | Mixed-field raw correlator becomes input-trivial (honest negative); connected correlator restores the effect (+0.125) | ra09/ra10 |

**Known tensions with the new roadmap:** (a) the new plan's "SAE features
recover observables" framing must contend with established result #5 — the
prior work already found SAE features non-universal and pivoted to
representation-level claims; (b) result #6 predates the new Phase-6 causal
plan and suggests steering/ablation effects on *energy* will be small by
design — the interesting causal target is the model's *implicit observable*,
not its output; (c) the existing partial-correlation control uses mean-h only —
the new full-input control is strictly stronger and may shrink headline numbers.
