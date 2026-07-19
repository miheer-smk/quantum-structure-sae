# Phase 0.6 — trained-seed distribution of the full-input control

Run: `runs/phase06_multiseed_trained_20260719-142825_s42`  ·  Config: `configs/phase06_multiseed_trained.yaml`

10 independently trained transformers (fresh init + disorder), identical full-input control on each; only the trained weights vary. Energy test R^2 across seeds: 1.000±0.000 [min 0.999, max 1.000]. Random-init launch pool n=16, stability pool n=64.

## Partial correlation | poly2-h — trained vs random-init distribution

| observable | trained (mean±sd [min,max]) | random-init (mean±sd, p95) | separation (sd) |
|---|---|---|---|
| entropy | 0.565±0.040 [0.509,0.628] | 0.404±0.090 (p95 0.525) | +1.79 |
| mean_nn_zz | 0.765±0.039 [0.713,0.848] | 0.362±0.115 (p95 0.534) | +3.51 |
| mean_x | 0.623±0.066 [0.500,0.738] | 0.369±0.099 (p95 0.559) | +2.55 |
| long_range_zz | 0.560±0.046 [0.503,0.626] | 0.280±0.109 (p95 0.429) | +2.57 |
| phase_proximity | 0.000±0.000 [-0.000,0.000] | 0.000±0.000 (p95 0.000) | +1.56 |

## Incremental R² beyond poly2-h — trained vs random-init distribution

| observable | trained (mean±sd [min,max]) | random-init (mean±sd, p95) | separation (sd) |
|---|---|---|---|
| entropy | 0.0305±0.0023 [0.0268,0.0342] | 0.0185±0.0026 (p95 0.0218) | +4.57 |
| mean_nn_zz | 0.0097±0.0004 [0.0092,0.0104] | 0.0050±0.0006 (p95 0.0059) | +7.26 |
| mean_x | 0.0120±0.0008 [0.0107,0.0132] | 0.0071±0.0011 (p95 0.0087) | +4.52 |
| long_range_zz | 0.0283±0.0030 [0.0231,0.0320] | 0.0121±0.0039 (p95 0.0172) | +4.20 |
| phase_proximity | -0.0000±0.0000 [-0.0000,0.0001] | 0.0001±0.0000 (p95 0.0001) | -6.66 |

## Per-seed transparency (long_range_zz, poly2-h)

Per-seed trained values (averaged over eval seeds), and any seed at or near the random threshold — nothing averaged away.

**partial-r** per-seed: s1:0.512, s2:0.565, s3:0.503, s4:0.626, s5:0.525, s6:0.603, s7:0.510, s8:0.620, s9:0.556, s10:0.580
- threshold = 0.4214; 0 below (none); 0 near (none)
- p95 stability: p95@16=0.421 (boot CI [0.378,0.426]), p95@64=0.429 → threshold 0.421 [stable]

**incremental R²** per-seed: s1:0.023, s2:0.029, s3:0.025, s4:0.032, s5:0.027, s6:0.031, s7:0.026, s8:0.032, s9:0.029, s10:0.030
- threshold = 0.0166; 0 below (none); 0 near (none)
- p95 stability: p95@16=0.017 (boot CI [0.015,0.017]), p95@64=0.017 → threshold 0.017 [stable]

## Verdict (long_range_zz, poly2-h)

- **partial-r**: trained spread [0.503, 0.626] vs threshold 0.421 — clean separation (min>thr): **True**; separation **+2.57 sd**.
- **incremental R²**: trained spread [0.023, 0.032] vs threshold 0.017 — clean separation (min>thr): **True**; separation **+4.20 sd**.