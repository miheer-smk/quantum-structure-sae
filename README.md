# quantum-structure-sae

**Research question:** Do classical transformers trained on quantum ground-state energies develop internal representations that correspond to quantum observables (entanglement entropy, order parameters, proximity to phase transitions)?

**Methodology:** Train a small transformer to predict TFIM (transverse-field Ising model) ground-state energies from Hamiltonian parameters. Apply TopK sparse autoencoders to the transformer's residual stream activations. Test whether SAE features correlate with known quantum observables computed from exact diagonalization.

**Status:** In progress (April 2026).

## Built on

This repo extends the `qsae` starter codebase with:
- A transformer for TFIM energy prediction
- Quantum observable computation (entanglement entropy, correlations, magnetization)
- Feature-to-observable correlation analysis

The reusable infrastructure (classical shadows, TopK SAE, TFIM data, training loops) lives in `src/qsae/`.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install pytest
pytest -v
python scripts/smoke_test.py
```

## Roadmap

- **Week 1-2:** Transformer on TFIM, target R² > 0.995 on held-out test set
- **Week 3:** Compute quantum observables per input (⟨Z_i Z_j⟩, half-chain entanglement entropy, distance to g_c)
- **Week 4:** Train TopK SAE on residual stream activations
- **Week 5:** Measure feature-to-observable Pearson correlation; go/no-go decision
- **Month 3-4:** Scaling experiments, baselines, write workshop submission
- **Month 6-9:** Full paper, target NeurIPS 2026 or Nature Computational Science

## Layout

src/qsae/            # Reusable research infrastructure (from qsae starter)
sae.py             # TopK Sparse Autoencoder
datasets.py        # Includes tfim_ground_states for data generation
metrics.py         # Polysemanticity, universality metrics
training.py        # Training loops
qnn.py             # Quantum circuits (kept for future work)
shadows.py         # Classical shadows (kept for future work)
scripts/             # Entry points for experiments
tests/               # pytest suite

## Author

Miheer Kulkarni — Undergraduate Researcher, 2026.