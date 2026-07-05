# Experiment 01 — Bars-and-Stripes QNN → shadows → SAE

The classical-data sanity experiment for the interpretability pipeline
(`scripts/exp01_bas3.py`), run end-to-end. It establishes that the
QNN → classical-shadow → TopK-SAE stack recovers monosemantic features on a
task with known structure (Bars-and-Stripes), before applying it to quantum data.

## Setup

- **Data.** Bars-and-Stripes 3×3 (9 qubits), 120 examples/class, 80/20 split
  (train = 192, test = 48).
- **QNN.** 9-qubit, 3-layer data-reuploading circuit, linear entangler, per-qubit
  Z measurements; trained 40 epochs, Adam lr = 0.03.
- **Shadows.** 600 Pauli classical-shadow snapshots per state; weight-1&2 Pauli
  observables → 21-dim shadow-feature vectors.
- **SAE.** TopK, d_hidden = 8·d_in = 168 → 408 latent, k = 8, 300 epochs.

## Results (`runs/exp01/`)

| Metric | Value | Reading |
|---|---:|---|
| QNN test accuracy | **0.979** | BAS-3×3 is learned nearly perfectly |
| SAE dead fraction | 0.174 | healthy; most latents active |
| live features | 337 / 408 | |
| polysemanticity (mean) | 0.325 | lower ⇒ more monosemantic |
| **monosemantic fraction** | **0.611** | well above the ≥ 0.2 "real signal" bar (RUNBOOK) |

The QNN converges within ~2 epochs (val acc 0.979 by epoch 2) and the SAE drives
reconstruction MSE to ~10⁻³ with no feature collapse. A monosemantic fraction of
0.61 means a majority of live features concentrate their activation on a single
Bars-or-Stripes class — the pipeline recovers interpretable structure on data
whose ground truth we know, validating it as a probe before the quantum-data
experiments (`exp_ra02`, `exp_ra03`).

Top-activating-input inspection (printed by the script) shows features that fire
almost exclusively for one class label, consistent with the monosemanticity metric.

## Reproduce

```bash
python scripts/exp01_bas3.py     # ~20–40 min CPU; writes runs/exp01/
```

Artifacts: `runs/exp01/{qnn.pt, shadow_features.npy, labels.npy, sae.pt,
metrics.json, history.json}`.
