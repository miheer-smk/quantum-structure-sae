# Week 1 Results: TFIM-L8 Transformer Training

## What we trained

**Architecture — `TFIMTransformer`**

- 3 Pre-LN Transformer encoder layers (`norm_first=True`)
- d_model = 64, 4 attention heads, feedforward dim = 256 (GELU)
- Input: length-8 sequence of per-site transverse fields h_i, each projected
  from R¹ → R^{64} via a learned linear layer, then summed with learned
  positional embeddings
- Output: scalar ground-state energy via mean-pool → two-layer MLP head
- 152,833 trainable parameters

**Dataset (primary, wide-h regime)**

- Hamiltonian: H = −J Σ Z_i Z_{i+1} − Σ h_i X_i, open boundary conditions,
  L = 8, J = 1 fixed
- Per-site fields h_i ~ Uniform(0.1, 2.0) independently
- 50,000 train / 5,000 val / 5,000 test; ground-state energies via dense exact
  diagonalisation (256×256 matrices, batched `np.linalg.eigvalsh`)
- Energy targets normalized to zero mean / unit variance on train split;
  R² always reported on unnormalized (physical) energies
- Cached to `data/tfim_L8_N50k.pt` (gitignored)

**Training hyperparameters**

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ with cosine decay to 10⁻⁶ |
| Weight decay | 10⁻⁴ |
| Batch size | 128 |
| Max epochs | 200 |
| Early stop patience | 15 epochs (val R² improvement ≥ 10⁻⁴) |
| Mixed precision | torch.amp (AMP), CUDA |
| Gradient clip | 1.0 |
| Seed | 0 (unified: data + model) |

---

## Primary result (wide-h regime)

Training converged at epoch 46 (early stop at epoch 61). Wall-clock: ~6 min
(including ~2.5 min dataset generation).

| Split | R² (unnormalized) | RMSE |
|---|---|---|
| Validation (best) | **0.999903** | 0.01087 |
| Test | **0.999910** | 0.01038 |

Target R² > 0.995 comfortably achieved.

---

## Baseline comparison

To check whether the transformer solved a trivially polynomial task, we trained
two closed-form baselines on the same 50k training examples and evaluated on
the same 5k test set.

### Wide-h regime: h_i ~ Uniform(0.1, 2.0)

| Model | Features | Test R² | Test RMSE |
|---|---|---|---|
| Linear regression | 8 (raw h_i) | 0.9812 | 0.1501 |
| Poly-2 regression | 45 (degree-2 products) | 0.9989 | 0.0366 |
| **Transformer** | learned | **0.9999** | **0.0104** |

The transformer achieves a **3.5× RMSE improvement over poly-2**, confirming
the task contains structure beyond degree-2 polynomials in the wide-h range.

### Narrow-h regime: h_i ~ Uniform(0.7, 1.3) — exploratory, negative result

We also trained on a narrower field range centered on the quantum critical point
g = h/J = 1, hypothesising that near-critical correlations would make the task
harder for baselines. The result was the opposite:

| Model | Features | Test R² | Test RMSE |
|---|---|---|---|
| Linear regression | 8 (raw h_i) | 0.9984 | 0.0142 |
| Poly-2 regression | 45 | **0.99999** | 0.0013 |
| Transformer | learned | 0.9996 | 0.0069 |

Poly-2 is essentially perfect (R² = 0.99999), and the transformer *underperforms*
poly-2 — it cannot beat a 45-parameter closed-form model in this regime.

---

## Physical interpretation of the negative result

When h_i ∈ [0.7, 1.3], each per-site field varies only ±0.3J around the
coupling scale J = 1. In this window the ground-state energy is well-approximated
by a Taylor expansion in the field deviations δh_i = h_i − h̄:

E₀(h) ≈ E₀(h̄) + Σ_i (∂E₀/∂h_i) δh_i + ½ Σ_{ij} (∂²E₀/∂h_i∂h_j) δh_i δh_j + …

The dominant contributions are single-site (linear) and diagonal quadratic
(∂²E₀/∂h_i²) terms, both of which poly-2 captures exactly. The off-diagonal
cross-site terms (∂²E₀/∂h_i∂h_j for i ≠ j), which encode inter-site
entanglement structure, are O(δh²) and small relative to noise. Narrowing the
h-range therefore *suppresses* the signal we care about rather than amplifying it.

The wide-h regime, by contrast, spans both the ordered phase (h ≪ J, energy ≈
−(L−1)J) and the disordered phase (h ≫ J, energy ≈ −Σ h_i), forcing the model
to learn the nonlinear crossover region around g ≈ 1 where cross-site
correlations are maximal and polynomial approximation breaks down.

---

## Follow-up directions (week 6+)

Two stronger levers for making the task genuinely hard for polynomial baselines,
without changing L:

1. **Disordered couplings J_{ij} ~ Uniform(0.5, 1.5):** Random bond disorder
   generates a dense Hessian ∂²E₀/∂h_i∂h_j with no special structure, breaking
   the near-diagonal approximation poly-2 exploits. Even in narrow h-ranges,
   poly-2 would need O(L²) independent coefficients per disorder realization.

2. **Larger L (12 or 16):** For L = 16 the Hilbert space is 65,536-dimensional.
   Exact diagonalisation requires sparse methods (scipy `eigsh`) and is ~100×
   slower per sample. But the energy is an extensive quantity (E₀ ∝ L) with
   long-range correlations; degree-2 polynomial approximation quality degrades
   as L increases because the number of cross-site terms grows as L² while
   single-site terms grow as L.

Both changes would be made in `data.py` and `exp_ra01_train_transformer.py`
with no architecture changes needed for L ≤ 16.

---

*Generated: week 1 of quantum-structure-SAE project. Primary artifacts in
`runs/ra01_wide/` (wide-h) and `runs/ra01b_narrow/` (narrow-h exploratory).
Checkpoints and dataset caches are gitignored and must be regenerated.*
