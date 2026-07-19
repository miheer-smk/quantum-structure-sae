# Manuscript update — Phase 0.6 (multi-seed + full-input control)

Record of the edits applied to `tmlr_draft.md` after the Phase 0.5 (full-input
recoverability control) and Phase 0.6 (10-trained-seed distribution) results.
Source of the numbers: `results/phase05_input_control.md`,
`results/phase06_multiseed_trained.md` (10 trained seeds vs 64 random inits, full
degree-two input control, L=8).

## What changed and why

The draft's headline was controlled only against the **scalar mean field** (partial-r
`0.706 ± 0.011`) and leaned on a **single untrained draw**. Phase 0.5/0.6 replace
this with a **full degree-two input control** and a **trained-seed distribution (10)
vs random-init distribution (64)**. The effect survives and is uniform across seeds,
but the honest effect size is smaller than the scalar number implied.

## The four pieces applied

### 1. Abstract — lead with the conservative cut
Removed scalar `0.706` and the single-untrained framing. New lead: incremental R²
beyond poly-2(h) = **0.028 ± 0.003 vs 0.012 ± 0.004 random, +4.2σ, uniform over 10
seeds**; secondary sentence: beyond-input partial correlation **0.56 ± 0.05** (raw
probe R² 0.953 ± 0.006). Methods sentence updated to "full site-resolved field
vector and its degree-two polynomial" and "ten independently trained transformers
against a 64-model random-initialization distribution."

### 2. §5.2 + Table 2 — reconcile raw vs beyond-input decodability
Table 2 (raw decodability, single draw, end-to-end ZZ = **0.961** preserved) is now
explicitly labeled as raw linear access. Added **Table 2b** (beyond-input control,
10-seed vs 64-random distribution) and corrected the prose: raw decodability is
saturated/similar for several observables, but under the poly-2 control nn-ZZ
(**+7.3σ**) and ⟨X⟩ (**+4.5σ**) separate from random init, entropy separates on
incremental R² (**+4.6σ**) but only modestly in partial correlation (**+1.8σ**),
and the order proxy is distinctive by having the *lowest* random-init baseline.
Contribution 3 updated to match (no longer says "most other observables decoded
nearly as well by an untrained network" without qualification).

**Table 2b (beyond-input control, full degree-two polynomial; trained 10 seeds vs random 64):**

| Observable | Probe R² (trained) | Partial r \| poly-2 (trained) | Partial r \| poly-2 (random, p95) | Sep. | Incr. R² (trained) | Incr. R² (random) | Sep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Half-chain entropy | 0.918 ± 0.012 | 0.565 ± 0.040 | 0.404 ± 0.090 (0.525) | +1.8σ | 0.031 ± 0.002 | 0.018 ± 0.003 | +4.6σ |
| Mean nearest-neighbour ZZ | 0.993 ± 0.001 | 0.765 ± 0.039 | 0.362 ± 0.115 (0.534) | +3.5σ | 0.010 ± 0.000 | 0.005 ± 0.001 | +7.3σ |
| Mean X magnetisation | 0.987 ± 0.002 | 0.623 ± 0.066 | 0.369 ± 0.099 (0.559) | +2.6σ | 0.012 ± 0.001 | 0.007 ± 0.001 | +4.5σ |
| End-to-end ZZ | 0.953 ± 0.006 | 0.560 ± 0.046 | 0.280 ± 0.109 (0.429) | +2.6σ | 0.028 ± 0.003 | 0.012 ± 0.004 | +4.2σ |
| Phase proximity | 0.999 ± 0.000 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |

### 3. §5.5 + Table 4 — ten-seed replication (was three-seed mean-field)
Rewrote §5.5 to the 10-trained-seed vs 64-random framing with per-seed uniformity
(0 seeds below/near threshold). New Table 4:

| Quantity for end-to-end ZZ (poly-2 control) | Trained (10 seeds) | Random init (64 networks) |
|---|---:|---:|
| Beyond-input partial correlation | 0.560 ± 0.046 [min 0.503] | 0.280 ± 0.109 (p95 0.429) |
| Incremental R² beyond degree-two input | 0.028 ± 0.003 [min 0.023] | 0.012 ± 0.004 (p95 0.017) |
| Raw probe R² (trained residual) | 0.953 ± 0.006 | — |

### 4. Limitation #3 — downgraded, not deleted
Now "Restricted controls (substantially addressed)": the full site-resolved vector +
degree-two polynomial control addresses the scalar-confound caveat; residual caveat
narrowed to "does not exhaust arbitrary nonlinear dependence on the full field
vector."

## Consistency sweep (post-edit)
`0.706`, `0.107`, "three-seed" phrasing → 0 occurrences. Table 2's raw `0.961`
preserved. σ to one decimal, R² to three decimals throughout. Also updated §1
intro, §4 setup, §9 discussion, §12 conclusion, §3.6 cross-reference, and the
reproducibility command block (added `experiments/phase05_input_control.py` and
`experiments/phase06_multiseed_trained.py`).

## Not done (flagged)
- The submission PDF is built by an external toolchain (no build tooling in repo);
  these Markdown edits need a separate pandoc/LaTeX compile to reach the PDF.
- FDR (`qsae.analysis.fdr`) is implemented and tested but not yet applied to the
  SAE feature×observable grid in the paper; not claimed in the text.
