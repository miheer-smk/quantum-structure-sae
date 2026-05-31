# quantum-structure-sae

[![CI](https://github.com/miheer-smk/quantum-structure-sae/actions/workflows/ci.yml/badge.svg)](https://github.com/miheer-smk/quantum-structure-sae/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Research question:** Do classical transformers trained on quantum ground-state energies develop internal representations that correspond to quantum observables — entanglement entropy, order parameters, and proximity to phase transitions?

**Approach:** Train a small transformer to predict 1D Transverse-Field Ising Model (TFIM) ground-state energies from per-site Hamiltonian parameters. Apply TopK sparse autoencoders to the transformer's residual-stream activations. Measure Pearson correlations between discovered SAE features and physically known quantum observables computed by exact diagonalisation.

**Status:** Week 1 complete (transformer training, baselines). Week 3 in progress (observable computation, SAE on activations).

---

## Results

### Week 1 — Transformer training (R² > 0.999 ✓)

Trained a Pre-LN Transformer (3 layers, d_model=64, 4 heads, 153k parameters) on 50,000 TFIM instances with per-site disordered fields h_i ~ Uniform(0.1, 2.0).

| Model | Test R² | Test RMSE |
|---|---|---|
| Linear regression (8 features) | 0.9812 | 0.1501 |
| Poly-2 regression (45 features) | 0.9989 | 0.0366 |
| **TFIMTransformer (ours)** | **0.9999** | **0.0104** |

The transformer achieves **3.5× RMSE improvement over the degree-2 polynomial baseline**, confirming the prediction task contains structure beyond polynomial approximation in the wide-h regime. See [`docs/week1_results.md`](docs/week1_results.md) for the full analysis including the narrow-h negative result and its physical interpretation.

---

## Quickstart

```bash
# Linux/macOS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v                            # 20+ tests, all green
python scripts/smoke_test.py         # end-to-end QNN → shadow → SAE pipeline
```

```bash
# Windows
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
pytest -v
```

### Run the main experiments

```bash
# Week 1: train the TFIM transformer
python scripts/exp_ra01_train_transformer.py
# outputs: runs/ra01_train/{best.pt, curves.png, log.txt}

# Baseline comparison (requires ra01 checkpoint + dataset cache)
python scripts/ra01_baseline_check.py --ckpt runs/ra01_train/best.pt

# Week 3: compute observables and correlate with SAE features
python scripts/exp_ra02_observables.py --ckpt runs/ra01_train/best.pt
# outputs: runs/ra02_observables/{correlation_heatmap.png, top_features.json, ...}
```

---

## Repository layout

```
src/qsae/                    # Installable Python package
  ├── sae.py                 # TopK Sparse Autoencoder (Gao et al. 2024)
  ├── observables.py         # Quantum observables: entropy, ZZ, magnetization
  ├── datasets.py            # TFIM ground states, Bars-and-Stripes, MNIST-4x4
  ├── metrics.py             # Polysemanticity, universality, dead-feature metrics
  ├── training.py            # Training loops for QNN and SAE
  ├── shadows.py             # Classical shadow tomography (Huang-Kueng-Preskill 2020)
  ├── qnn.py                 # PennyLane variational quantum circuits
  └── reverse_arrow/         # TFIM transformer sub-package
      ├── transformer.py     # TFIMTransformer (Pre-LN encoder + MLP head)
      └── data.py            # TFIMDataset, make_splits, exact diagonalisation

scripts/
  ├── exp_ra01_train_transformer.py  # Train transformer → R² > 0.999
  ├── ra01_baseline_check.py         # Linear/poly-2 baselines + pred-vs-truth plot
  ├── exp_ra02_observables.py        # Observable computation + SAE feature correlation
  ├── exp01_bas3.py                  # Bars-and-Stripes 3×3 QNN experiment
  └── smoke_test.py                  # End-to-end sanity check (~10s)

tests/
  ├── test_sae.py            # TopK SAE sparsity and reconstruction tests
  ├── test_shadows.py        # Classical shadows vs analytically known expectation values
  ├── test_transformer.py    # TFIMTransformer: shape, overfit, checkpoint round-trip
  └── test_observables.py    # Observables vs product states, GHZ, TFIM ground states

docs/
  └── week1_results.md       # Full Week 1 results + physical interpretation

notebooks/                   # Exploratory Jupyter notebooks
```

---

## Module guide

### `qsae.observables` — quantum observables

```python
from qsae.observables import compute_all_observables
from qsae.datasets import tfim_ground_states
import numpy as np

h_values = np.linspace(0.1, 2.0, 100)
states = tfim_ground_states(n=6, h_values=h_values)

obs = compute_all_observables(states, n=6, h_values=h_values.mean(keepdims=True))
# obs keys: entropy, nn_zz, mean_nn_zz, transverse_mag, mean_x, order_param,
#           phase_proximity  (when h_values provided)
```

### `qsae.sae` — TopK Sparse Autoencoder

```python
from qsae import SAEConfig, TopKSAE
import torch

cfg = SAEConfig(d_in=64, d_hidden=256, k=32)
sae = TopKSAE(cfg)
out = sae(torch.randn(32, 64))
# out: {"x_hat", "z", "recon_loss", "aux_loss", "loss"}
```

### `qsae.reverse_arrow` — TFIM Transformer

```python
from qsae import TFIMTransformer, TransformerConfig
import torch

cfg = TransformerConfig(L=8, d_model=64, n_heads=4, n_layers=3, d_ff=256)
model = TFIMTransformer(cfg)
h = torch.rand(16, 8)          # (batch, L) per-site fields
energy = model(h)              # (batch,) predicted ground-state energies
```

---

## Roadmap

| Milestone | Status |
|---|---|
| Transformer on TFIM, R² > 0.995 | ✅ Done (R² = 0.9999) |
| Baseline comparison (linear, poly-2) | ✅ Done |
| `observables.py` module with full test coverage | ✅ Done |
| TopK SAE on transformer residual stream | 🔄 In progress |
| Feature-to-observable Pearson correlation analysis | 🔄 In progress |
| Scaling to L=12, disordered couplings | ⬜ Planned |
| Workshop submission (ICLR 2026 workshop) | ⬜ Planned |
| Full paper (NeurIPS 2026 / Nature Computational Science) | ⬜ Planned |

---

## Citation

If you use this work, please cite:

```bibtex
@software{kulkarni2026qsae,
  author  = {Kulkarni, Miheer},
  title   = {quantum-structure-sae: Mechanistic Interpretability of Transformers via Sparse Autoencoders on Quantum Data},
  year    = {2026},
  url     = {https://github.com/miheer-smk/quantum-structure-sae},
  license = {MIT}
}
```

See also [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

---

## References

- Gao et al. (2024). *Scaling and evaluating sparse autoencoders.* [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
- Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* [Anthropic](https://transformer-circuits.pub/2023/monosemantic-features)
- Huang, Kueng, Preskill (2020). *Predicting many properties of a quantum system from very few measurements.* Nature Physics. [doi:10.1038/s41567-020-0932-7](https://doi.org/10.1038/s41567-020-0932-7)
- Sachdev (2011). *Quantum Phase Transitions* (2nd ed.). Cambridge University Press.

---

## Author

Miheer Kulkarni — Undergraduate Researcher, 2026.
