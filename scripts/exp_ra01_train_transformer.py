"""
exp_ra01_train_transformer.py
Train a TFIMTransformer to predict ground-state energies of the disordered
1D TFIM from per-site field vectors.

Outputs (written to runs/ra01_train/):
  best.pt       — model state dict, config, energy normalization (mean/std)
  curves.png    — train/val loss and R² curves
  config.json   — all hyper-parameters
  log.txt       — epoch-by-epoch training log

Usage
-----
    python scripts/exp_ra01_train_transformer.py [--seed SEED]

R² is reported on UN-normalized energies (physically meaningful).
The model trains on normalized targets (zero mean, unit variance).

Stop conditions (per spec):
  - OOM: caught and re-raised with a clear message; do NOT silently reduce batch.
  - val R² < 0.90 after epoch 50: prints WARNING, keeps training.
  - val R² stagnates for `patience` epochs: early stop.
  - val R² < 0.995 at end: print WARNING and exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.amp import GradScaler, autocast
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False

from qsae.reverse_arrow.data import make_splits
from qsae.reverse_arrow.transformer import TFIMTransformer, TransformerConfig

# ---------------------------------------------------------------------------
# Hyper-parameters (overridable only via --seed for now)
# ---------------------------------------------------------------------------
BASE_CFG = dict(
    # Data
    L=8,
    n_train=50_000,
    n_val=5_000,
    n_test=5_000,
    h_min=0.1,
    h_max=2.0,
    J=1.0,
    cache_path="data/tfim_L8_N50k.npz",

    # Model
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    dropout=0.0,

    # Training
    batch_size=128,
    epochs=200,
    lr=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    patience=15,
    min_delta=1e-4,

    # Output
    run_dir="runs/ra01_train",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    energy_mean: float,
    energy_std: float,
) -> tuple[float, float, float, float]:
    """Return (norm_mse, norm_r2, unnorm_mse, unnorm_r2)."""
    model.eval()
    preds_norm, targets_norm = [], []
    with torch.no_grad():
        for h, e in loader:
            h, e = h.to(device), e.to(device)
            e_norm = (e - energy_mean) / energy_std
            preds_norm.append(model(h))
            targets_norm.append(e_norm)

    preds_norm = torch.cat(preds_norm)
    targets_norm = torch.cat(targets_norm)

    norm_mse = F.mse_loss(preds_norm, targets_norm).item()
    norm_r2 = r2_score(preds_norm, targets_norm)

    # Invert normalization for physically meaningful metrics
    preds_unnorm = preds_norm * energy_std + energy_mean
    targets_unnorm = targets_norm * energy_std + energy_mean
    unnorm_mse = F.mse_loss(preds_unnorm, targets_unnorm).item()
    unnorm_r2 = r2_score(preds_unnorm, targets_unnorm)

    return norm_mse, norm_r2, unnorm_mse, unnorm_r2


def log(path: Path, msg: str) -> None:
    print(f"[train] {msg}")
    with open(path, "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="Global seed for torch, numpy, random, and data generation")
    args = parser.parse_args()

    CFG = {**BASE_CFG, "seed": args.seed}

    set_seeds(CFG["seed"])

    run_dir = Path(CFG["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    log_path.unlink(missing_ok=True)   # fresh log each run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(log_path, f"device={device}  seed={CFG['seed']}")
    if device.type == "cuda":
        log(log_path, f"GPU: {torch.cuda.get_device_name(0)}  "
                      f"free={torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    with open(run_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    # cache_path uses .npz suffix per spec; make_splits saves as .pt internally
    # so we use a .pt cache (same data, different extension in CFG is cosmetic)
    cache_pt = Path(CFG["cache_path"]).with_suffix(".pt")
    cache_pt.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_ds, val_ds, test_ds = make_splits(
        L=CFG["L"],
        n_train=CFG["n_train"],
        n_val=CFG["n_val"],
        n_test=CFG["n_test"],
        h_min=CFG["h_min"],
        h_max=CFG["h_max"],
        J=CFG["J"],
        seed=CFG["seed"],
        cache_path=cache_pt,
    )
    t_data = time.time() - t0
    log(log_path,
        f"dataset ready in {t_data:.1f}s — "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Energy normalization: compute from training set only
    e_train_all = train_ds.energies          # (n_train,) float32 tensor
    energy_mean = float(e_train_all.mean())
    energy_std  = float(e_train_all.std())
    log(log_path,
        f"energy stats (train) — mean={energy_mean:.4f}  std={energy_std:.4f}  "
        f"min={float(e_train_all.min()):.4f}  max={float(e_train_all.max()):.4f}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model_cfg = TransformerConfig(
        L=CFG["L"],
        d_model=CFG["d_model"],
        n_heads=CFG["n_heads"],
        n_layers=CFG["n_layers"],
        d_ff=CFG["d_ff"],
        dropout=CFG["dropout"],
    )
    model = TFIMTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(log_path, f"model params: {n_params:,}")

    # -----------------------------------------------------------------------
    # Optimiser + scheduler + AMP
    # -----------------------------------------------------------------------
    opt = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["epochs"], eta_min=1e-6
    )
    use_amp = _AMP_AVAILABLE and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None
    log(log_path, f"mixed precision AMP: {use_amp}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_r2_unnorm = -float("inf")
    patience_counter   = 0
    history: dict[str, list] = {
        "train_loss_norm": [], "val_loss_norm": [],
        "val_r2_unnorm": [], "lr": [],
    }

    log(log_path, f"starting {CFG['epochs']}-epoch run …")
    t_start = time.time()

    try:
        for epoch in range(1, CFG["epochs"] + 1):
            model.train()
            train_losses = []

            for h, e in train_loader:
                h, e = h.to(device), e.to(device)
                e_norm = (e - energy_mean) / energy_std   # train on normalized targets

                opt.zero_grad()
                if use_amp:
                    with autocast(device_type="cuda"):
                        pred = model(h)
                        loss = F.mse_loss(pred, e_norm)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = model(h)
                    loss = F.mse_loss(pred, e_norm)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    opt.step()

                train_losses.append(loss.item())

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            train_loss_norm = sum(train_losses) / len(train_losses)
            _, _, val_mse_unnorm, val_r2_unnorm = evaluate(
                model, val_loader, device, energy_mean, energy_std
            )
            val_rmse_unnorm = val_mse_unnorm ** 0.5

            history["train_loss_norm"].append(train_loss_norm)
            history["val_loss_norm"].append(val_mse_unnorm / energy_std**2)  # approx
            history["val_r2_unnorm"].append(val_r2_unnorm)
            history["lr"].append(current_lr)

            elapsed = time.time() - t_start
            line = (
                f"epoch {epoch:4d}/{CFG['epochs']}  "
                f"train_loss(norm)={train_loss_norm:.6f}  "
                f"val_R2={val_r2_unnorm:.6f}  "
                f"val_RMSE={val_rmse_unnorm:.5f}  "
                f"lr={current_lr:.2e}  elapsed={elapsed:.0f}s"
            )
            log(log_path, line)

            if epoch == 50 and val_r2_unnorm < 0.90:
                log(log_path,
                    f"WARNING: val R²={val_r2_unnorm:.4f} < 0.90 at epoch 50 — "
                    "training may be stuck.")

            # Checkpoint best model (keyed on unnormalized R²)
            if val_r2_unnorm > best_val_r2_unnorm + CFG["min_delta"]:
                best_val_r2_unnorm = val_r2_unnorm
                patience_counter   = 0
                torch.save(
                    {
                        "epoch":            epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "cfg":              model_cfg,
                        "train_cfg":        CFG,
                        "energy_mean":      energy_mean,
                        "energy_std":       energy_std,
                        "val_r2_unnorm":    val_r2_unnorm,
                        "val_rmse_unnorm":  val_rmse_unnorm,
                    },
                    run_dir / "best.pt",
                )
            else:
                patience_counter += 1

            if patience_counter >= CFG["patience"]:
                log(log_path,
                    f"early stop at epoch {epoch} "
                    f"(no improvement in {CFG['patience']} epochs, "
                    f"best val R²={best_val_r2_unnorm:.6f})")
                break

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(
                "\n[STOP] CUDA out-of-memory. "
                "batch_size has NOT been reduced. "
                "Free GPU memory or reduce BASE_CFG['batch_size'] manually.\n"
            )
        raise

    # -----------------------------------------------------------------------
    # Test evaluation on best checkpoint
    # -----------------------------------------------------------------------
    log(log_path, "evaluating best checkpoint on test set …")
    ckpt = torch.load(run_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, test_mse_unnorm, test_r2_unnorm = evaluate(
        model, test_loader, device, energy_mean, energy_std
    )
    test_rmse_unnorm = test_mse_unnorm ** 0.5

    # Patch final metrics into checkpoint
    ckpt["test_r2_unnorm"]   = test_r2_unnorm
    ckpt["test_rmse_unnorm"] = test_rmse_unnorm
    torch.save(ckpt, run_dir / "best.pt")

    wall_time = time.time() - t_start
    summary = (
        f"\n{'='*60}\n"
        f"TRAINING COMPLETE\n"
        f"  best val R²  (unnorm): {best_val_r2_unnorm:.6f}\n"
        f"  best val RMSE(unnorm): {ckpt['val_rmse_unnorm']:.6f} eV\n"
        f"  test R²      (unnorm): {test_r2_unnorm:.6f}\n"
        f"  test RMSE    (unnorm): {test_rmse_unnorm:.6f} eV\n"
        f"  early stop epoch:      {ckpt['epoch']}\n"
        f"  wall-clock time:       {wall_time:.0f}s\n"
        f"{'='*60}"
    )
    log(log_path, summary)

    if test_r2_unnorm < 0.995:
        log(log_path,
            f"WARNING: test R²={test_r2_unnorm:.4f} is BELOW target 0.995. "
            "Consider more data, longer training, or tuned hyperparameters.")
    else:
        log(log_path, "Target val R² > 0.995 achieved.")

    # -----------------------------------------------------------------------
    # Curves
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_axis = list(range(1, len(history["val_r2_unnorm"]) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.semilogy(epochs_axis, history["train_loss_norm"], label="train loss (norm)")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("MSE loss (normalized, log scale)")
        ax1.legend()
        ax1.set_title("Loss curves")

        ax2.plot(epochs_axis, history["val_r2_unnorm"], label="val R² (unnorm)")
        ax2.axhline(0.995, color="r", linestyle="--", label="target R²=0.995")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("val R² (unnormalized energies)")
        ax2.legend()
        ax2.set_title("Validation R²")

        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=150)
        log(log_path, f"curves saved to {run_dir}/curves.png")
    except ImportError:
        log(log_path, "matplotlib not installed — skipping curves plot")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    log(log_path, f"outputs in {run_dir}/")


if __name__ == "__main__":
    main()
