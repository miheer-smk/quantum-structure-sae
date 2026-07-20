# Results archive

Curated, committed backup of **every experiment result** — the exact tables and
numbers reported in-session — one markdown file per experiment. This is the
durable record: `runs/` holds machine JSON (gitignored) and `docs/` holds
narrative write-ups, but this folder is the single place where the headline
tables for every test live together, in the form they were reported.

**Convention (standing):** after every experiment, its result table is written
here automatically (experiment drivers write directly to `results/`), and this
index gets one new row. No result is reported in chat without a copy landing
here first.

## Index (newest first)

| Date | Experiment | File | One-line result |
|---|---|---|---|
| 2026-07-20 | Phase 0.7 — ANNNI diversity | [phase07_annni.md](phase07_annni.md) | trained vs random-init, per-target sigma_y + SEPARATION/NULL/UNDERPOWERED flag |
| 2026-07-20 | Phase 0.7 — XXZ diversity | [phase07_xxz.md](phase07_xxz.md) | trained vs random-init, per-target sigma_y + SEPARATION/NULL/UNDERPOWERED flag |
| 2026-07-19 | Phase 0.6 — trained-seed distribution | [phase06_multiseed_trained.md](phase06_multiseed_trained.md) | see file — trained vs random-init distribution, sd-separation + per-seed flags |
| 2026-07-19 | Phase 0.5 — full-input control (kill-shot) | [phase05_input_control.md](phase05_input_control.md) | Non-local order headline **survives** full poly-2(h) control: trained partial-r 0.648 [0.576,0.710] vs random 0.301±0.115; incr-R² 0.032 vs 0.009 |
