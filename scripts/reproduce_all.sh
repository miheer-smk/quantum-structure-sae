#!/usr/bin/env bash
# reproduce_all.sh — regenerate every result and figure from a clean checkout.
#
# Usage:
#   bash scripts/reproduce_all.sh            # full run
#   FAST=1 bash scripts/reproduce_all.sh     # smaller/quicker (smoke-scale) run
#   SKIP_BAS=1 bash scripts/reproduce_all.sh # skip the ~20-40 min BAS QNN run
#
# Idempotent: the transformer checkpoint and cached datasets are reused if present.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
PY="${PYTHON:-python}"

if [[ "${FAST:-0}" == "1" ]]; then
  N_OBS=200; N_CTRL=200; N_PERM=100; SAE_EPOCHS=60
else
  N_OBS=500; N_CTRL=800; N_PERM=500; SAE_EPOCHS=200
fi
CKPT="runs/ra01_wide/best.pt"

echo "==============================================================="
echo " reproduce_all.sh   FAST=${FAST:-0}  SKIP_BAS=${SKIP_BAS:-0}"
echo "==============================================================="

echo "[0/5] environment + tests"
$PY -m pytest -q
ruff check src/ tests/ scripts/ --select=E,F,W --ignore=E501,W291,W293

echo "[1/5] Stage 1 — train transformer (skipped if checkpoint exists)"
if [[ -f "$CKPT" ]]; then
  echo "    found $CKPT — reusing"
else
  $PY scripts/exp_ra01_train_transformer.py
fi
$PY scripts/ra01_baseline_check.py --ckpt "$CKPT"

echo "[2/5] Stage 2 — SAE feature <-> observable correlations"
$PY scripts/exp_ra02_observables.py --ckpt "$CKPT" --n_samples "$N_OBS" --sae_epochs "$SAE_EPOCHS"

echo "[3/5] Stage 2 — control battery C1-C5"
$PY scripts/exp_ra03_controls.py --ckpt "$CKPT" \
    --n_samples "$N_CTRL" --n_perm "$N_PERM" --sae_epochs "$SAE_EPOCHS"

echo "[3b] SAE cross-seed universality grid (C5 revisited)"
$PY scripts/exp_ra04_sae_grid.py --ckpt "$CKPT" \
    --n_samples $((N_CTRL + 1200)) --sae_epochs "$SAE_EPOCHS"

echo "[3c] multi-seed headline (long-range-ZZ partial-r, mean +/- std)"
$PY scripts/exp_ra06_multiseed.py --ckpt "$CKPT" --seeds 42,43,44 \
    --n_samples "$N_CTRL" --n_perm "$N_PERM" --sae_epochs "$SAE_EPOCHS"

echo "[3d] causal activation patching (decodable vs used)"
$PY scripts/exp_ra07_causal.py --ckpt "$CKPT" --n_samples "$N_CTRL"

echo "[4/5] classical-data validation (Bars-and-Stripes QNN -> shadow -> SAE)"
if [[ "${SKIP_BAS:-0}" == "1" ]]; then
  echo "    SKIP_BAS=1 — skipping"
else
  $PY scripts/exp01_bas3.py
fi

echo "[5/5] done. Artifacts in runs/{ra01_wide,ra02_observables,ra03_controls,exp01}/"
echo "Key outputs:"
echo "  runs/ra03_controls/summary.md          (C1-C5 tables)"
echo "  runs/ra03_controls/fig_probe_r2.png    (headline figure)"
echo "  runs/ra02_observables/top_features.json"
