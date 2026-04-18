"""
exp_ra01_train_transformer.py
Train a TFIMTransformer to predict ground-state energies of the disordered
1D TFIM from per-site field vectors.

Outputs (written to runs/ra01_train/):
  best.pt       — model state dict + config at best val R²
  curves.png    — train/val R² and loss curves
  config.json   — all hyper-parameters
  log.txt       — epoch-by-epoch training log

Usage
-----
    python scripts/exp_ra01_train_transformer.py

Stop conditions (per user spec):
  - OOM: caught and re-raised with a clear message; do NOT silently reduce batch.
  - val R² < 0.90 after epoch 50: prints WARNING, keeps training.
  - val R² stagnates for `patience` epochs: early stop.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# AMP scaler — only used when CUDA is available
try:
    from torch.amp import GradScaler, autocast
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False

from qsae.reverse_arrow.data import make_splits
from qsae.reverse_arrow.transformer import TFIMTransformer, TransformerConfig

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
CFG = dict(
    # Data
    L=8,
    n_train=5000,
    n_val=1000,
    n_test=1000,
    h_min=0.0,
    h_max=2.0,
    J=1.0,
    data_seed=0,
    cache_path="data/tfim_L8.pt",

    # Model
    d_model=64,
    n_heads=4,
    n_layers=3,
    d_ff=256,
    dropout=0.0,

    # Training
    batch_size=256,
    epochs=200,
    lr=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    patience=15,          # early-stop patience (val R² improvement)
    min_delta=1e-4,       # minimum improvement to reset patience counter

    # Output
    run_dir="runs/ra01_train",
    seed=1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for h, e in loader:
            h, e = h.to(device), e.to(device)
            preds.append(model(h))
            targets.append(e)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    loss = F.mse_loss(preds, targets).item()
    r2 = r2_score(preds, targets)
    return loss, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(CFG["seed"])
    run_dir = Path(CFG["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}  "
              f"free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_ds, val_ds, test_ds = make_splits(
        L=CFG["L"],
        n_train=CFG["n_train"],
        n_val=CFG["n_val"],
        n_test=CFG["n_test"],
        h_min=CFG["h_min"],
        h_max=CFG["h_max"],
        J=CFG["J"],
        seed=CFG["data_seed"],
        cache_path=Path(CFG["cache_path"]),
    )

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
    print(f"[train] model params: {n_params:,}")

    # -----------------------------------------------------------------------
    # Optimiser + scheduler
    # -----------------------------------------------------------------------
    opt = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["epochs"], eta_min=1e-6
    )
    use_amp = _AMP_AVAILABLE and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    amp_ctx = lambda: autocast(device_type="cuda") if use_amp else torch.no_grad.__class__()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    log_path = run_dir / "log.txt"
    best_val_r2 = -float("inf")
    patience_counter = 0
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_r2": [], "lr": []}

    print(f"[train] starting {CFG['epochs']}-epoch run …")
    t_start = time.time()

    try:
        for epoch in range(1, CFG["epochs"] + 1):
            model.train()
            train_losses = []

            for h, e in train_loader:
                h, e = h.to(device), e.to(device)
                opt.zero_grad()

                if use_amp:
                    with autocast(device_type="cuda"):
                        pred = model(h)
                        loss = F.mse_loss(pred, e)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    scaler.step(opt)
                    scaler.update()
                else:
                    pred = model(h)
                    loss = F.mse_loss(pred, e)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    opt.step()

                train_losses.append(loss.item())

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            train_loss = sum(train_losses) / len(train_losses)
            val_loss, val_r2 = evaluate(model, val_loader, device)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_r2"].append(val_r2)
            history["lr"].append(current_lr)

            # Log
            elapsed = time.time() - t_start
            line = (
                f"epoch {epoch:4d}/{CFG['epochs']}  "
                f"train_loss={train_loss:.6f}  "
                f"val_loss={val_loss:.6f}  "
                f"val_r2={val_r2:.6f}  "
                f"lr={current_lr:.2e}  "
                f"elapsed={elapsed:.0f}s"
            )
            print(f"[train] {line}")
            with open(log_path, "a") as f:
                f.write(line + "\n")

            # Warn early if training is not converging
            if epoch == 50 and val_r2 < 0.90:
                msg = (
                    f"WARNING: val R²={val_r2:.4f} < 0.90 at epoch 50. "
                    "Training may be slow or stuck."
                )
                print(f"[train] {msg}")
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

            # Checkpoint best model
            if val_r2 > best_val_r2 + CFG["min_delta"]:
                best_val_r2 = val_r2
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "cfg": model_cfg,
                        "val_r2": val_r2,
                        "val_loss": val_loss,
                    },
                    run_dir / "best.pt",
                )
            else:
                patience_counter += 1

            if patience_counter >= CFG["patience"]:
                print(
                    f"[train] early stop at epoch {epoch} "
                    f"(no improvement in {CFG['patience']} epochs, "
                    f"best val R²={best_val_r2:.6f})"
                )
                break

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(
                "\n[STOP] CUDA out-of-memory error. "
                "batch_size has NOT been reduced. "
                "Please free GPU memory or reduce CFG['batch_size'] manually.\n"
            )
        raise

    # -----------------------------------------------------------------------
    # Test evaluation on best checkpoint
    # -----------------------------------------------------------------------
    print("\n[train] evaluating best checkpoint on test set …")
    ckpt = torch.load(run_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_r2 = evaluate(model, test_loader, device)
    print(f"[train] test_loss={test_loss:.6f}  test_r2={test_r2:.6f}")
    print(f"[train] best val R² achieved: {best_val_r2:.6f} (epoch {ckpt['epoch']})")

    if test_r2 < 0.995:
        print(
            f"[train] WARNING: test R²={test_r2:.4f} is below target of 0.995. "
            "Consider training longer, tuning hyper-parameters, or using more data."
        )
    else:
        print(f"[train] target val R² > 0.995 achieved.")

    # -----------------------------------------------------------------------
    # Curves plot
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs_axis = list(range(1, len(history["val_r2"]) + 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.semilogy(epochs_axis, history["train_loss"], label="train loss")
        ax1.semilogy(epochs_axis, history["val_loss"],   label="val loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("MSE loss (log scale)")
        ax1.legend()
        ax1.set_title("Loss curves")

        ax2.plot(epochs_axis, history["val_r2"])
        ax2.axhline(0.995, color="r", linestyle="--", label="target R²=0.995")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("val R²")
        ax2.legend()
        ax2.set_title("Validation R²")

        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=150)
        print(f"[train] curves saved to {run_dir}/curves.png")
    except ImportError:
        print("[train] matplotlib not installed — skipping curves plot")

    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    print(f"\n[train] DONE — outputs in {run_dir}/")


if __name__ == "__main__":
    main()
