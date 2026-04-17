# Runbook — First 90 Days

A concrete, realistic plan for driving this project from empty repo to an
arXiv submission. Every item is something you can do on a laptop.

## Before you start

- [ ] Unpack the tarball, `cd qsae`, verify everything works:
  ```bash
  pip install -e .
  pytest -v          # all 6 tests should pass in ~25s
  python scripts/smoke_test.py   # end-to-end in ~10s
  ```
- [ ] Set up a Python environment you like (conda, uv, venv — any is fine).
- [ ] Create a private GitHub repo, push the scaffold, commit often. Keep a
      `notes.md` that you update daily.
- [ ] Optional: register for free Weights & Biases to track sweeps later.

---

## Week 1 — Run the first real experiment

**Goal.** Reproduce `scripts/exp01_bas3.py` on your machine and *understand* every
number it prints. If you can't explain why a number is what it is, stop and dig in.

**Do.**
- [ ] Run `python scripts/exp01_bas3.py`. Wait ~30 min.
- [ ] Look at `runs/exp01/metrics.json`. Expected ballpark:
      - `test_acc` in 0.85–1.00 (3x3 BAS is a tractable problem for a 9-qubit QNN)
      - `dead_fraction` 0.3–0.7 is normal at first
      - `monosemantic_fraction` should be positive; a non-trivial number
        (say > 0.2) is your *first real research signal*.
- [ ] Open `notebooks/` and write a small exploration notebook:
      load the SAE, plot the decoder column norms, plot activation histograms
      per feature, and *look at* the top-activating inputs for each feature.
      **This is the step where you start forming hypotheses.**

**Don't.**
- Don't jump to more qubits yet. Understand 9 qubits first.
- Don't worry about "quantum advantage" arguments yet. Just get the
  interpretability to *work*.

**Deliverable.** A one-page write-up in `notes.md`: "What does the SAE find
on Bars-and-Stripes, and does it line up with my intuition about the bars
vs. stripes distinction?" If yes → you have the germ of a paper.

---

## Week 2 — Baselines & sanity checks

The biggest risk in any interpretability paper is that your "features" are
reading random noise. Multiple baselines kill that fear.

**Do.**
- [ ] **Random-QNN baseline.** Train a fresh SAE on shadows from an
      *untrained* (random-weights) QNN. Compare metrics. If monosemantic
      fractions are similar, your trained QNN isn't learning useful
      structure — re-examine the QNN training first.
- [ ] **Shuffled-labels baseline.** Train the SAE the same way, but compute
      polysemanticity against randomly permuted labels. Expect
      `monosemantic_fraction` ≈ chance. This verifies your metric is picking
      up *actual* class structure.
- [ ] **Different seeds.** Train 3 QNNs from different seeds to similar
      accuracy, train an SAE on each, then use `match_features` +
      `universality_score`. Features that show up across seeds are
      meaningful; features that don't are likely artifacts.
- [ ] **Shadow-sample budget sweep.** Vary `n_samples ∈ {100, 200, 400,
      800, 1600}` and track final reconstruction and monosemantic fraction.
      Bad shadows = noisy features = polysemantic SAE. There's a sweet spot.

**Deliverable.** A plot panel (3–4 subplots) showing how metrics compare
across the four baseline conditions. This panel will be Figure 2 of your
paper. Start it now.

---

## Week 3 — The quantum-data experiment

This is the experiment that matters for the quantum-advantage framing:
does the SAE still find interpretable features when the QNN is doing
something that's *actually quantum*?

**Do.**
- [ ] Write `scripts/exp02_tfim.py`. Use `tfim_ground_states(n=8, ...)` to
      generate, say, 200 ground states at h-values spanning 0 to 2, label
      them by phase (`h_c=1`).
- [ ] Train a QNN to classify the ground state by phase *from shadow
      features of the ground state itself* (not from h). This is the
      Preskill-style "learning on quantum data" regime where quantum
      advantage is established.
- [ ] Train an SAE on its shadow features. Compute monosemantic fraction
      and visualize top-activating inputs.
- [ ] The important question: **do the SAE features correspond to
      physically meaningful observables?** Look at each feature's decoder
      direction — it's a linear combination of Pauli observables. If the
      monosemantic features line up with things like the order parameter
      `<Z_i Z_j>` or the transverse magnetization `<X_i>`, you have a
      concrete, publishable result.

**Deliverable.** A claim of the form: "SAE features trained on a QNN
classifier of TFIM phases recover order-parameter-like Pauli combinations
with cosine similarity ≥ X against the true order parameter."

---

## Week 4 — Polysemanticity theorem (the theory piece)

You need at least one theoretical contribution to land in npj QI / PRX
Quantum. Here's a tractable one:

**Claim to try to prove.** *The number of features a QNN of n qubits and depth
L can represent in k-sparse superposition is bounded above by a function of
the entanglement entropy of its layer-L state, not by the Hilbert space
dimension 2^n.*

**Why this is tractable.** The quantum analog of Johnson-Lindenstrauss is
well-studied (random quantum states live near the Haar measure, etc.). The
missing piece is connecting near-orthogonality of decoder directions to
entanglement structure of the underlying quantum state. Take one of:

- The **compressed sensing** approach: L-layer states lie in a
  low-entanglement manifold (for polynomial L); project the decoder into
  that manifold and count directions that fit.
- The **classical-shadow information-theoretic** approach: shadows of
  n-qubit states compress to poly(n) bits; the SAE can only extract
  polynomially-many independent features from polynomially-many bits.
  (This is almost a triviality, but it's also almost certainly the
  correct statement.)

**Deliverable.** A LaTeX note (5-10 pages) with a theorem statement and
either a proof or a clean empirical demonstration plus conjecture.

---

## Weeks 5-8 — Iterate, scale, write

Now you have three pieces: baselines (Week 2), TFIM result (Week 3), a
theoretical claim (Week 4). Time to turn them into a paper.

**Do.**
- [ ] Scale experiments: 12-qubit QNNs for MNIST-4x4 (use torchvision via
      `mnist_downsampled`). This is the "real data" experiment reviewers
      ask for.
- [ ] Write `scripts/exp03_universality.py` that runs 5 seeds x 3
      architectures and produces the universality heatmap. This becomes
      Figure 3.
- [ ] **Auto-interp.** Write a small script that, for each live SAE
      feature, (1) finds top-activating inputs, (2) passes them to Claude
      API with a prompt asking for a concept description, (3) scores the
      description by checking if it predicts activations on held-out
      inputs. This is your "the features are actually human-interpretable"
      evidence.
- [ ] Start the LaTeX paper. Target: 12 pages main + appendices. Structure:
      1. Introduction (why QNN interpretability matters; the Cerezo-crisis
         framing sells well)
      2. Background (shadows, SAEs, superposition hypothesis)
      3. Method (your QSAE architecture, algorithm boxes)
      4. Empirical results (the three figure panels from Weeks 2-3)
      5. Theoretical analysis (Week 4 theorem)
      6. Related work and limitations
      7. Conclusion

---

## Weeks 9-12 — Polish, ablate, submit

**Do.**
- [ ] Write the ablation table nobody will read but reviewer 2 will insist
      on: performance vs. k (SAE sparsity), vs. d_hidden, vs. shadow
      sample count, vs. QNN depth.
- [ ] Run a 16-qubit experiment on a cloud GPU (~$5 budget on Vast.ai);
      use `lightning.gpu` via the `qsae[gpu]` extra.
- [ ] Share the draft with 2-3 people whose opinion you trust: ideally one
      QML person (for the physics) and one mech-interp person (for the ML).
      They will find different problems.
- [ ] arXiv submission. Aim for end of Week 12.

---

## Failure modes to watch for

| Symptom | Likely cause | Fix |
|---|---|---|
| `dead_fraction` > 0.95 | k too small, lr too high, not enough aux-revival | bump `aux_k`, drop lr, lengthen training |
| `monosemantic_fraction` near 1/C (chance) | SAE reading noise, not signal | more shadow samples, retrain QNN, check QNN actually learned |
| SAE features not universal across seeds | d_hidden too small, TopK k too large | widen SAE, reduce k |
| QNN stuck at 50% on BAS-2x2 | barren plateau or bad initialization | reduce depth, smaller init scale, more samples per class |
| Shadow estimates noisy | too few samples | increase `n_samples`; 2000+ is safe for 8-10 qubits |
| `scipy.sparse.eigsh` fails on TFIM | n too big (memory) | n ≤ 14 on 16GB RAM; use ED libraries for n > 14 |

---

## What to avoid

- Don't chase more qubits until 8-12 qubit experiments are rock solid.
- Don't use deep unitary QNNs for the interpretability paper — Cerezo has
  shown they're mostly dequantizable. Keep the architecture simple so
  reviewers can't attack on that front.
- Don't claim "quantum advantage" in this paper. Claim "a novel interpretability
  technique for QNNs, demonstrated on classical and quantum data." The
  advantage question is for a different paper.
- Don't skip the baselines in Week 2. They're boring but they're also 40%
  of your paper's credibility.

---

## When to ask for help

- Stuck on a PennyLane bug for more than an afternoon → PennyLane forum
  (they're fast and nice).
- Stuck on the theorem → post the statement on MathOverflow, or email one
  of the authors of the Cerezo paper.
- Stuck on the narrative → read the introduction of three top-tier
  interpretability papers (Bricken et al., Marks et al., Templeton et al.)
  and copy their structure.

Good luck.
