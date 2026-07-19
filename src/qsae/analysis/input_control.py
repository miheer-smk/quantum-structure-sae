"""
Input-recoverability controls — the confound analysis at the heart of the paper.

Every observable in this study is a deterministic function of the Hamiltonian
input ``h``. A correlation between a representation and an observable can
therefore be *inherited from the input* rather than *computed by training*. The
TMLR draft controls only for the scalar mean field ``h-bar``. The functions here
generalise that control to the full site-resolved input vector and its degree-2
polynomial expansion, isolating what the trained representation carries *beyond*
everything linearly/quadratically available in the raw input.

Two complementary statistics, both computed out-of-fold to avoid leakage:

* ``partial_corr_controlling`` — partial correlation between the representation's
  out-of-fold probe prediction and the observable, after removing (by OLS) the
  contribution of a control matrix ``Z`` (mean field / raw h / poly-2 h). This
  is the vector-control generalisation of the draft's Table-3/Table-4 statistic.

* ``incremental_r2`` — the increase in out-of-fold ridge R^2 from *adding* the
  representation to a probe that already has the input control ``Z``. Directly
  answers: does the representation add anything on top of a poly-2 input model?

All array shapes are documented; units are dimensionless (correlations, R^2).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def oof_ridge_predict(
    X: np.ndarray,        # (N, d) features
    y: np.ndarray,        # (N,) target
    alpha: float = 1.0,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Out-of-fold ridge predictions of ``y`` from ``X`` (each row predicted by a
    model that never saw it). Features standardised on the training fold only."""
    y = np.asarray(y, dtype=np.float64)
    pred = np.empty_like(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        reg = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred[te] = reg.predict(sc.transform(X[te]))
    return pred


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def residualize(a: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """Return the OLS residual of ``a`` (N,) after regressing out control ``Z``
    (N, k) with an intercept. ``Z=None`` returns ``a`` mean-centred. This removes
    exactly the part of ``a`` linearly explained by the columns of ``Z``
    (Frisch–Waugh–Lovell); pass polynomial features in ``Z`` to remove nonlinear
    input dependence."""
    a = np.asarray(a, dtype=np.float64)
    if Z is None:
        return a - a.mean()
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    reg = LinearRegression().fit(Z, a)
    return a - reg.predict(Z)


def partial_corr_controlling(
    a: np.ndarray,        # (N,) e.g. representation OOF prediction
    b: np.ndarray,        # (N,) observable
    Z: np.ndarray | None,  # (N, k) control features, or None
) -> float:
    """Partial correlation r(a, b | Z): Pearson correlation of the parts of ``a``
    and ``b`` not linearly explained by ``Z``. Reduces to ordinary correlation
    when ``Z=None``."""
    ra = residualize(a, Z)
    rb = residualize(b, Z)
    denom = np.sqrt(float(np.sum(ra**2)) * float(np.sum(rb**2))) + 1e-12
    return float(np.sum(ra * rb) / denom)


def incremental_r2(
    repr_feats: np.ndarray,  # (N, d) representation
    Z: np.ndarray | None,    # (N, k) input control, or None
    y: np.ndarray,           # (N,) observable
    alpha: float = 1.0,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Out-of-fold ridge R^2 for the observable from (i) the control ``Z`` alone,
    (ii) ``Z`` plus the representation (standardised, concatenated), and their
    difference. ``delta`` > 0 means the representation carries information about
    ``y`` beyond what ``Z`` already provides."""
    y = np.asarray(y, dtype=np.float64)
    R = np.asarray(repr_feats, dtype=np.float64)
    if Z is None:
        r2_z = 0.0
        both = R
    else:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        r2_z = _r2(y, oof_ridge_predict(Z, y, alpha, n_folds, seed))
        both = np.concatenate([R, Z], axis=1)
    r2_both = _r2(y, oof_ridge_predict(both, y, alpha, n_folds, seed))
    return {"r2_control": r2_z, "r2_control_plus_repr": r2_both,
            "delta": r2_both - r2_z}


def bootstrap_partial_corr(
    a: np.ndarray,
    b: np.ndarray,
    Z: np.ndarray | None,
    n_boot: int = 4000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """Percentile bootstrap CI for ``partial_corr_controlling(a, b | Z)`` by
    resampling the N samples. The residualisation is refit inside each bootstrap
    replicate so the CI reflects uncertainty in the control fit as well."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    Z_arr = None if Z is None else np.asarray(Z, dtype=np.float64)
    if Z_arr is not None and Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    n = len(a)
    rng = np.random.default_rng(seed)
    point = partial_corr_controlling(a, b, Z_arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Zi = None if Z_arr is None else Z_arr[idx]
        boots[i] = partial_corr_controlling(a[idx], b[idx], Zi)
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return {"estimate": point, "ci_lo": lo, "ci_hi": hi}
