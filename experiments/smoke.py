"""
experiments/smoke.py — tiny end-to-end pass of the reproducibility harness.

Exercises: YAML config → global seeding → disordered-TFIM data (sparse ED) →
transformer training → residual-stream extraction → TopK SAE → JSONL run log
+ manifest entry. Finishes in well under a minute on CPU. Not a science run —
its only job is to prove the harness works end-to-end.

Usage
-----
    python experiments/smoke.py [--config configs/smoke.yaml] [key=value ...]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from qsae.config import load_config
from qsae.repro import set_global_seed, seeded_generator
from qsae.runlog import RunLogger
from qsae.reverse_arrow.data import compute_ground_states_sparse
from qsae.reverse_arrow.transformer import TFIMTransformer, TransformerConfig
from qsae.sae import SAEConfig, TopKSAE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    seed_record = set_global_seed(cfg["seed"])
    logger = RunLogger("smoke", cfg, seed_record)

    # ---- data: disordered TFIM, sparse ED --------------------------------
    d = cfg["data"]
    rng = seeded_generator(cfg["seed"])
    n_total = d["n_train"] + d["n_val"]
    h = rng.uniform(d["h_min"], d["h_max"], size=(n_total, d["L"]))
    energies = compute_ground_states_sparse(h, J_fields=d["J"], return_states=False)
    logger.log({"event": "data_done", "n": n_total, "L": d["L"],
                "e_mean": float(energies.mean())})

    h_t = torch.from_numpy(h).float()
    e_t = torch.from_numpy(energies).float()
    e_mean, e_std = e_t[: d["n_train"]].mean(), e_t[: d["n_train"]].std()
    e_norm = (e_t - e_mean) / e_std
    va = slice(d["n_train"], n_total)

    # ---- transformer: a few epochs ---------------------------------------
    m = cfg["model"]
    model = TFIMTransformer(TransformerConfig(
        L=d["L"], d_model=m["d_model"], n_heads=m["n_heads"],
        n_layers=m["n_layers"], d_ff=m["d_ff"], dropout=m["dropout"],
    ))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    bs = cfg["train"]["batch_size"]
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        perm = torch.randperm(d["n_train"])
        for s in range(0, d["n_train"], bs):
            b = perm[s: s + bs]
            opt.zero_grad()
            F.mse_loss(model(h_t[b]), e_norm[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(h_t[va])
            val_r2 = 1 - float(((pred - e_norm[va]) ** 2).sum()
                               / ((e_norm[va] - e_norm[va].mean()) ** 2).sum())
        logger.log({"epoch": epoch, "val_r2": val_r2})

    # ---- residual-stream extraction + SAE --------------------------------
    acts: list[torch.Tensor] = []
    handle = model.encoder.layers[-1].register_forward_hook(
        lambda mod, i, o: acts.append(o.detach()))
    with torch.no_grad():
        model(h_t)
    handle.remove()
    pooled = acts[0].mean(dim=1)  # (N, d_model)

    s = cfg["sae"]
    sae = TopKSAE(SAEConfig(d_in=pooled.shape[1], d_hidden=s["d_hidden"], k=s["k"]))
    sae_opt = torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr)
    x = (pooled - pooled.mean(0)) / (pooled.std(0) + 1e-6)
    for epoch in range(s["epochs"]):
        sae.train()
        perm = torch.randperm(len(x))
        losses = []
        for st in range(0, len(x), s["batch_size"]):
            out = sae(x[perm[st: st + s["batch_size"]]])
            sae_opt.zero_grad()
            out["loss"].backward()
            sae_opt.step()
            with torch.no_grad():
                sae.post_step(out["z"])
            losses.append(out["recon_loss"].item())
        logger.log({"sae_epoch": epoch, "sae_recon": float(np.mean(losses))})

    run_dir = logger.finish({
        "val_r2": val_r2,
        "sae_recon_final": float(np.mean(losses)),
        "sae_dead_frac": sae.dead_feature_fraction(),
        "status": "ok",
    })
    print(f"[smoke] OK — val_r2={val_r2:.3f}, outputs in {run_dir}/")


if __name__ == "__main__":
    main()
