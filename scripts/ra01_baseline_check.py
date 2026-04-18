"""
ra01_baseline_check.py
Sanity-check baselines against the trained TFIMTransformer.

Usage:
  python scripts/ra01_baseline_check.py [--cache CACHE_PT] [--ckpt CKPT_PT] [--plot_dir DIR]

Outputs:
  <plot_dir>/pred_vs_truth.png  — transformer predictions vs truth (500 test points)
  stdout                        — R² for linear, poly-2, and transformer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--cache",    default="data/tfim_L8_N50k.pt")
parser.add_argument("--ckpt",     default="runs/ra01_train_wide/best.pt")
parser.add_argument("--plot_dir", default="runs/ra01_train_wide")
args = parser.parse_args()

CACHE    = Path(args.cache)
CKPT     = Path(args.ckpt)
PLOT_DIR = Path(args.plot_dir)

if not CACHE.exists():
    sys.exit(f"[error] dataset cache not found: {CACHE}")
if not CKPT.exists():
    sys.exit(f"[error] checkpoint not found: {CKPT}")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[check] loading dataset from {CACHE}")
saved = torch.load(CACHE, weights_only=False)

h_train = saved["h_train"].numpy().astype(np.float64)  # (50000, 8)
e_train = saved["e_train"].numpy().astype(np.float64)  # (50000,)
h_test  = saved["h_test"].numpy().astype(np.float64)   # (5000, 8)
e_test  = saved["e_test"].numpy().astype(np.float64)   # (5000,)

print(f"[check] train={len(h_train)}, test={len(h_test)}")

# ---------------------------------------------------------------------------
# Baseline 1: Linear regression
# ---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

print("\n[check] fitting linear regression …")
lr = LinearRegression()
lr.fit(h_train, e_train)
e_pred_linear = lr.predict(h_test)
r2_linear = r2_score(e_test, e_pred_linear)
rmse_linear = float(np.sqrt(np.mean((e_pred_linear - e_test) ** 2)))
print(f"  Linear      R²={r2_linear:.6f}  RMSE={rmse_linear:.5f}")

# ---------------------------------------------------------------------------
# Baseline 2: Degree-2 polynomial regression
# ---------------------------------------------------------------------------
print("[check] fitting degree-2 polynomial regression …")
poly = PolynomialFeatures(degree=2, include_bias=True)
h_train_poly = poly.fit_transform(h_train)
h_test_poly  = poly.transform(h_test)
print(f"  poly features: {h_train_poly.shape[1]}")

pr = LinearRegression(fit_intercept=False)   # bias already in poly features
pr.fit(h_train_poly, e_train)
e_pred_poly = pr.predict(h_test_poly)
r2_poly = r2_score(e_test, e_pred_poly)
rmse_poly = float(np.sqrt(np.mean((e_pred_poly - e_test) ** 2)))
print(f"  Poly(deg=2)  R²={r2_poly:.6f}  RMSE={rmse_poly:.5f}")

# ---------------------------------------------------------------------------
# Transformer predictions
# ---------------------------------------------------------------------------
print("\n[check] loading transformer checkpoint …")
sys.path.insert(0, "src")
from qsae.reverse_arrow.transformer import TFIMTransformer

ckpt = torch.load(CKPT, weights_only=False)
energy_mean = ckpt["energy_mean"]
energy_std  = ckpt["energy_std"]
model = TFIMTransformer(ckpt["cfg"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

h_test_t = torch.from_numpy(h_test).float()
with torch.no_grad():
    e_pred_norm = model(h_test_t).numpy()
e_pred_transformer = e_pred_norm * energy_std + energy_mean

r2_transformer = float(r2_score(e_test, e_pred_transformer))
rmse_transformer = float(np.sqrt(np.mean((e_pred_transformer - e_test) ** 2)))
print(f"  Transformer  R²={r2_transformer:.6f}  RMSE={rmse_transformer:.5f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*55)
print(f"{'Model':<20} {'Test R²':>10} {'Test RMSE':>12}")
print("-"*55)
print(f"{'Linear (8 feat)':<20} {r2_linear:>10.6f} {rmse_linear:>12.5f}")
print(f"{'Poly-2 (36 feat)':<20} {r2_poly:>10.6f} {rmse_poly:>12.5f}")
print(f"{'Transformer':<20} {r2_transformer:>10.6f} {rmse_transformer:>12.5f}")
print("="*55)

gap = r2_transformer - r2_poly
print(f"\nTransformer gain over Poly-2: {gap:+.6f}")

if r2_linear > 0.99:
    print("\n[!] WARNING: linear R² > 0.99 — task may be too easy (near-linear regime).")
elif r2_poly > 0.999:
    print("\n[!] NOTE: poly-2 already achieves R² > 0.999 — transformer advantage is marginal.")
else:
    print("\n[ok] Task has real nonlinearity; transformer R² is meaningful.")

# ---------------------------------------------------------------------------
# Pred vs truth plot (500 random test examples)
# ---------------------------------------------------------------------------
print("\n[check] generating pred-vs-truth plot …")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.choice(len(h_test), size=500, replace=False)
    e_true_sample  = e_test[idx]
    e_pred_sample  = e_pred_transformer[idx]
    residuals      = e_pred_sample - e_true_sample

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # -- left: pred vs truth --
    ax = axes[0]
    ax.scatter(e_true_sample, e_pred_sample, s=8, alpha=0.5, color="steelblue")
    lo = min(e_true_sample.min(), e_pred_sample.min()) - 0.1
    hi = max(e_true_sample.max(), e_pred_sample.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="ideal y=x")
    ax.set_xlabel("True E₀")
    ax.set_ylabel("Predicted E₀")
    ax.set_title(f"Transformer: pred vs truth  (R²={r2_transformer:.5f})")
    ax.legend(fontsize=8)

    # -- right: residuals --
    ax2 = axes[1]
    ax2.scatter(e_true_sample, residuals, s=8, alpha=0.5, color="darkorange")
    ax2.axhline(0, color="r", lw=1, linestyle="--")
    ax2.set_xlabel("True E₀")
    ax2.set_ylabel("Residual (pred − true)")
    ax2.set_title(f"Residuals  RMSE={rmse_transformer:.5f}")

    fig.tight_layout()
    out_path = PLOT_DIR / "pred_vs_truth.png"
    fig.savefig(out_path, dpi=150)
    print(f"[check] plot saved to {out_path}")

except ImportError:
    print("[check] matplotlib not available — skipping plot")
