"""
Shared extraction/control-construction helpers for the analysis pipeline.

Previously duplicated across experiment scripts (see docs/CODE_MAP.md §2). Kept
here so every experiment reads the residual stream and builds input controls the
same way.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures


def last_layer_pooled(model, h_fields: np.ndarray, batch: int = 256) -> np.ndarray:
    """Mean-pooled last-layer residual stream, shape (N, d_model).

    Registers a forward hook on the final encoder block, runs the model in
    eval/no-grad, and mean-pools over sites — matching the representation the
    regression head reads."""
    model.eval()
    h = torch.from_numpy(np.asarray(h_fields)).float()
    chunks = []
    for start in range(0, len(h), batch):
        acts: list[torch.Tensor] = []
        handle = model.encoder.layers[-1].register_forward_hook(
            lambda m, i, o: acts.append(o.detach()))
        with torch.no_grad():
            model(h[start:start + batch])
        handle.remove()
        chunks.append(acts[0].mean(dim=1))
    return torch.cat(chunks, 0).numpy()


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination R^2 of predictions against targets."""
    y = np.asarray(y, dtype=np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def build_input_controls(h_fields: np.ndarray) -> dict[str, np.ndarray]:
    """The three input control sets partialled out in the recoverability analysis:
    scalar mean field, the full site-resolved input, and its degree-2 polynomial."""
    h = np.asarray(h_fields, dtype=np.float64)
    return {
        "mean_h": h.mean(axis=1, keepdims=True),
        "raw_h": h,
        "poly2_h": PolynomialFeatures(degree=2, include_bias=False).fit_transform(h),
    }
