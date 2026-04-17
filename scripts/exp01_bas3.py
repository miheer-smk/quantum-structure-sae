"""
Experiment 01 — First real run of the full pipeline.

Difference from smoke_test.py:
  * Bigger Bars-and-Stripes dataset (3x3 => 9 qubits, harder)
  * More training epochs and samples
  * Shadow sampling with more snapshots
  * Larger SAE with proper hyperparameters
  * Saves artifacts (model, features, SAE, metrics) to disk
  * Reports interpretability metrics in detail

Expected wall time on laptop CPU: 15-40 minutes depending on machine.

Usage:
    python scripts/exp01_bas3.py

Outputs (in ./runs/exp01/):
    qnn.pt                   trained QNN weights
    shadow_features.npy      (N, F) shadow-feature matrix
    labels.npy               (N,) labels
    sae.pt                   trained SAE weights
    metrics.json             final interpretability metrics
    history.json             training curves
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from qsae import (
    QNNConfig, TorchQNN,
    SAEConfig, TopKSAE,
    ShadowConfig,
    bars_and_stripes,
    train_qnn, train_sae,
    extract_shadow_features,
    feature_summary, top_activating_examples,
)


OUTDIR = Path("runs/exp01")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # --- 1. Dataset: 3x3 BAS oversampled to 120 examples per class -------
    print("=" * 72)
    print("STEP 1 — dataset (Bars-and-Stripes 3x3)")
    print("=" * 72)
    x_all, y_all = bars_and_stripes(size=3, n_per_class=120)

    # stratified split
    perm = torch.randperm(len(x_all))
    x_all, y_all = x_all[perm], y_all[perm]
    n_tr = int(0.8 * len(x_all))
    x_tr, y_tr = x_all[:n_tr], y_all[:n_tr]
    x_te, y_te = x_all[n_tr:], y_all[n_tr:]
    print(f"train={len(x_tr)}  test={len(x_te)}  features={x_tr.shape[1]}")

    # --- 2. QNN ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2 — train QNN (9 qubits, 3 layers)")
    print("=" * 72)
    cfg_q = QNNConfig(
        n_qubits=9, n_layers=3,
        encoding="reupload", entangler="linear",
        measurement="z_each",
    )
    model = TorchQNN(cfg_q, n_classes=2)

    t0 = time.time()
    history_q = train_qnn(
        model, x_tr, y_tr,
        epochs=40, batch_size=16, lr=0.03,
        x_val=x_te, y_val=y_te, verbose=True,
    )
    print(f"QNN training: {time.time()-t0:.1f}s")

    with torch.no_grad():
        logits, _ = model(x_te)
        test_acc = (logits.argmax(-1) == y_te).float().mean().item()
    print(f"final test accuracy: {test_acc:.3f}")

    torch.save(
        {"state_dict": model.state_dict(), "cfg": cfg_q.__dict__, "test_acc": test_acc},
        OUTDIR / "qnn.pt",
    )

    # --- 3. Shadow extraction --------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3 — extract classical shadows")
    print("=" * 72)
    t0 = time.time()
    # Use the full dataset so the SAE has enough data
    states = model.latent_states(x_all).numpy()
    cfg_s = ShadowConfig(
        n_samples=600, kind="pauli",
        feature_observables="paulis_weight_1_2", seed=0,
    )
    feats = extract_shadow_features(states, cfg_s)
    print(f"shadow features: shape={feats.shape}  time={time.time()-t0:.1f}s")
    np.save(OUTDIR / "shadow_features.npy", feats)
    np.save(OUTDIR / "labels.npy", y_all.numpy())

    # --- 4. SAE training -------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4 — train TopK SAE")
    print("=" * 72)
    feats_t = torch.from_numpy(feats).float()
    mu, sigma = feats_t.mean(0), feats_t.std(0) + 1e-6
    feats_t = (feats_t - mu) / sigma

    d_in = feats_t.shape[1]
    cfg_a = SAEConfig(
        d_in=d_in, d_hidden=8 * d_in, k=8,
        lr=3e-3, aux_k=4,
    )
    sae = TopKSAE(cfg_a)
    t0 = time.time()
    history_s = train_sae(sae, feats_t, epochs=300, batch_size=32, verbose_every=25)
    print(f"SAE training: {time.time()-t0:.1f}s")
    torch.save(
        {"state_dict": sae.state_dict(), "cfg": cfg_a.__dict__,
         "mu": mu, "sigma": sigma},
        OUTDIR / "sae.pt",
    )

    # --- 5. Metrics -------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 5 — interpretability metrics")
    print("=" * 72)
    with torch.no_grad():
        z = sae.feature_activations(feats_t)
    summary = feature_summary(z, y_all)
    for k, v in summary.items():
        print(f"  {k:28s} = {v}")

    # top activating examples per feature (non-dead only)
    live = (z > 0).any(dim=0)
    print(f"\n  live features: {int(live.sum())}/{cfg_a.d_hidden}")
    print("  top activating examples for first 5 live features:")
    live_idx = torch.where(live)[0][:5].tolist()
    for f in live_idx:
        idx, vals = top_activating_examples(z, f, top_k=5)
        labels_of_top = y_all[idx].tolist()
        print(f"    feature {f:4d}: activations={vals.round(3)}  "
              f"top-input-labels={labels_of_top}")

    # --- save everything ---
    metrics = {**summary, "test_acc": test_acc}
    with open(OUTDIR / "metrics.json", "w") as f:
        json.dump({k: float(v) if not isinstance(v, str) else v
                   for k, v in metrics.items()}, f, indent=2)
    with open(OUTDIR / "history.json", "w") as f:
        json.dump({"qnn": history_q, "sae": history_s}, f, indent=2)

    print(f"\nDONE. Artifacts written to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
