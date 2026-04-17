"""
End-to-end smoke test.

Runs the full research pipeline on a tiny Bars-and-Stripes problem:

    1. Train a small QNN classifier  (4 qubits, 2x2 bars-and-stripes)
    2. Extract classical shadows of the pre-measurement state for each input
    3. Train a TopK sparse autoencoder on the shadow features
    4. Report interpretability metrics (polysemanticity, dead fraction, etc.)

If this runs end-to-end and prints reasonable metrics, the stack works.
Expected wall-clock time on a laptop CPU: ~2-5 minutes.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from qsae import (
    QNNConfig, TorchQNN,
    SAEConfig, TopKSAE,
    ShadowConfig,
    bars_and_stripes,
    train_qnn, train_sae,
    extract_shadow_features,
    feature_summary,
)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    # --- 1. Dataset ----------------------------------------------------
    print("== dataset: Bars-and-Stripes, 2x2 ==")
    x, y = bars_and_stripes(size=2)
    print(f"n_samples={len(x)}  n_features={x.shape[1]}  n_classes={int(y.max())+1}")

    # 2x2 BAS lives on 4 qubits
    cfg_q = QNNConfig(n_qubits=4, n_layers=2, encoding="reupload", entangler="linear")

    # --- 2. QNN training ----------------------------------------------
    print("\n== training QNN ==")
    t0 = time.time()
    model = TorchQNN(cfg_q, n_classes=2)
    train_qnn(
        model, x, y,
        epochs=20, batch_size=4, lr=0.05,
        x_val=x, y_val=y, verbose=True,
    )
    print(f"QNN training: {time.time()-t0:.1f}s")

    with torch.no_grad():
        logits, _ = model(x)
        acc = (logits.argmax(-1) == y).float().mean().item()
    print(f"final accuracy (train == test here): {acc:.3f}")

    # --- 3. Classical shadow extraction -------------------------------
    print("\n== extracting classical shadows ==")
    t0 = time.time()
    states = model.latent_states(x).numpy()
    cfg_s = ShadowConfig(n_samples=400, kind="pauli",
                         feature_observables="paulis_weight_1_2", seed=0)
    feats = extract_shadow_features(states, cfg_s)
    print(f"shadow features: shape={feats.shape}  time={time.time()-t0:.1f}s")

    # --- 4. SAE training -----------------------------------------------
    print("\n== training TopK SAE ==")
    feats_t = torch.from_numpy(feats).float()
    # normalize per-feature to unit variance for stable training
    feats_t = (feats_t - feats_t.mean(0)) / (feats_t.std(0) + 1e-6)

    cfg_a = SAEConfig(d_in=feats_t.shape[1], d_hidden=32, k=4,
                      lr=3e-3, aux_k=2)
    sae = TopKSAE(cfg_a)
    t0 = time.time()
    train_sae(sae, feats_t, epochs=60, batch_size=8, verbose_every=10)
    print(f"SAE training: {time.time()-t0:.1f}s")

    # --- 5. Interpretability metrics -----------------------------------
    print("\n== interpretability summary ==")
    with torch.no_grad():
        z = sae.feature_activations(feats_t)
    summary = feature_summary(z, y)
    for k, v in summary.items():
        print(f"  {k:28s} = {v}")

    print("\nSMOKE TEST OK -- pipeline runs end-to-end.")


if __name__ == "__main__":
    main()
