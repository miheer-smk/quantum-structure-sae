"""
Metrics for interpretability evaluation.

- Polysemanticity score (entropy of activation over semantic classes).
- Feature universality (cosine similarity of matched features across seeds).
- Dead/ultra-dense feature fractions.
- Feature-activation statistics for human inspection.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Polysemanticity
# ---------------------------------------------------------------------------
def polysemanticity(
    activations: torch.Tensor,  # (N, d_hidden)
    labels: torch.Tensor,  # (N,) class labels
    threshold: float = 1e-4,
) -> torch.Tensor:
    """
    Per-feature polysemanticity score in [0, 1], defined as the normalized
    entropy of the feature's activation distribution across classes:

        p_c = sum_{i: y_i = c} a_{i,f}  /  sum_i a_{i,f}
        poly_f = H(p) / log(C)

    A value near 0 means the feature is monosemantic (fires for one class
    only); near 1 means uniform across classes (polysemantic).

    Features that never fire get score NaN.
    """
    C = int(labels.max().item()) + 1
    act = activations.clamp(min=0).double()
    # aggregate activations per class
    per_class = torch.zeros(C, act.shape[1], dtype=torch.float64)
    for c in range(C):
        mask = labels == c
        if mask.any():
            per_class[c] = act[mask].sum(dim=0)

    totals = per_class.sum(dim=0)
    probs = per_class / totals.clamp(min=threshold)  # (C, d_hidden)
    # replace NaN columns (dead features) with uniform to avoid log(0); we'll mask
    probs = torch.where(
        totals.unsqueeze(0) > threshold, probs, torch.full_like(probs, 1.0 / C)
    )
    entropy = -(probs * (probs.clamp(min=1e-12).log())).sum(dim=0)
    score = entropy / np.log(C)
    score = torch.where(totals > threshold, score, torch.full_like(score, float("nan")))
    return score


def fraction_monosemantic(
    poly_scores: torch.Tensor, threshold: float = 0.3
) -> float:
    """Fraction of non-dead features whose polysemanticity score is below threshold."""
    alive = ~torch.isnan(poly_scores)
    if alive.sum() == 0:
        return 0.0
    return float((poly_scores[alive] < threshold).float().mean().item())


# ---------------------------------------------------------------------------
# Feature universality across seeds
# ---------------------------------------------------------------------------
def match_features(
    dec_weights_a: torch.Tensor,  # (d_in, d_hidden)
    dec_weights_b: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find optimal one-to-one matching between features of two SAEs via the
    Hungarian algorithm on decoder-direction cosine similarity.

    Returns
    -------
    matching : (d_hidden,) int array where matching[i] is the feature in B
               matched to feature i in A.
    similarities : (d_hidden,) float array of cosine similarities at matching.
    """
    A = dec_weights_a.detach().cpu().numpy()
    B = dec_weights_b.detach().cpu().numpy()
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    # cost = -similarity, solve assignment
    cos = A.T @ B  # (d_hidden, d_hidden)
    row_ind, col_ind = linear_sum_assignment(-cos)
    matching = np.full(A.shape[1], -1, dtype=int)
    matching[row_ind] = col_ind
    sims = cos[row_ind, col_ind]
    return matching, sims


def universality_score(similarities: np.ndarray, threshold: float = 0.7) -> float:
    """
    Fraction of features whose best cross-seed match exceeds `threshold`
    cosine similarity. A classical-mech-interp benchmark; values >0.5 are
    typical for well-trained SAEs on classical LLMs.
    """
    return float((similarities > threshold).mean())


# ---------------------------------------------------------------------------
# Feature-level statistics for human inspection / auto-interp
# ---------------------------------------------------------------------------
def top_activating_examples(
    activations: torch.Tensor,  # (N, d_hidden)
    feature_idx: int,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return indices and activation values of top-k inputs for a given feature."""
    col = activations[:, feature_idx]
    vals, idx = torch.topk(col, k=min(top_k, col.shape[0]))
    return idx.cpu().numpy(), vals.cpu().numpy()


def feature_summary(
    activations: torch.Tensor, labels: torch.Tensor
) -> dict:
    """
    Quick summary of feature population: dead fraction, density distribution,
    polysemanticity distribution.
    """
    fired = (activations > 0).float()
    density = fired.mean(dim=0)
    alive_mask = density > 0
    dead_frac = float(1 - alive_mask.float().mean().item())
    poly = polysemanticity(activations, labels)
    mono_frac = fraction_monosemantic(poly)
    return {
        "dead_fraction": dead_frac,
        "density_mean": float(density[alive_mask].mean().item() if alive_mask.any() else 0),
        "density_median": float(density[alive_mask].median().item() if alive_mask.any() else 0),
        "polysemanticity_mean": float(
            poly[~torch.isnan(poly)].mean().item() if (~torch.isnan(poly)).any() else float("nan")
        ),
        "monosemantic_fraction": mono_frac,
    }
