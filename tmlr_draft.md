# Learned Representations of Non-Local Quantum Order in Energy-Predicting Transformers

**Miheer Kulkarni**

## Abstract

Neural networks are increasingly used to model quantum many-body systems, but
predictive accuracy alone does not establish what physical information their
internal representations contain or use. We study this question in a controlled
scientific-machine-learning setting. A small transformer is trained to predict
ground-state energies of disordered one-dimensional transverse-field Ising model
(TFIM) instances from their site-resolved transverse fields. We then evaluate
whether its residual-stream representations encode observables computed from exact
ground states. The central methodological challenge is that every target
observable is itself a function of the model input. We therefore compare a trained
transformer with an architecture-matched untrained transformer, the raw input, a
degree-two polynomial input representation, and the mean field. We also use
cross-validated linear probes, a layer sweep, a permutation null for
feature--observable correlations, partial correlations that control for the full
site-resolved field vector and its degree-two polynomial, and a comparison of ten
independently trained transformers against a 64-model random-initialization
distribution.

Beyond the full degree-two polynomial of the input, the trained representation
adds a reproducible increment in linear decodability of the finite-size non-local
order proxy \(\langle Z_0 Z_{L-1}\rangle\): an incremental cross-validated \(R^2\)
of \(0.028\pm0.003\), versus \(0.012\pm0.004\) for random-initialized networks --
a \(4.2\sigma\) separation that is uniform across all ten independently trained
seeds (every seed exceeds the random-initialization 95th percentile). The
corresponding beyond-input partial correlation is \(0.56\pm0.05\) (raw probe
\(R^2=0.953\pm0.006\) at \(L=8\)). This advantage increases across the three
transformer layers and persists at \(L\in\{8,10,12\}\) with a learned gain of
approximately \(0.028\). By contrast,
activation patching shows that removing a probe-defined order direction strongly
destroys its decodability while only modestly changing energy prediction; random
directions of the same norm are much more disruptive. Sparse autoencoders (SAEs)
find correlated directions, but their individual feature bases are not stable
across seeds. Within this setting, our results distinguish *representation* of a
physical observable from its *causal utilization* by the trained prediction head.

## 1. Introduction

Scientific machine learning often uses neural networks as accurate surrogates for
physical quantities. A complementary question is whether a model trained on one
physical target develops internal representations of other meaningful properties
of the system. This question is especially tractable for small quantum many-body
systems: exact diagonalisation can provide ground states, energies, and
observables, so candidate physical variables are known rather than inferred
post hoc.

We investigate this question for a classical transformer trained to predict the
ground-state energy of the one-dimensional transverse-field Ising model (TFIM).
The model receives only the vector of local transverse fields and is supervised
only on energy. After training, we ask whether its residual stream encodes
entanglement, local correlations, transverse magnetisation, and a non-local
finite-size proxy for ferromagnetic order.

This question has an important confound. The observables are deterministic
functions of the Hamiltonian parameters, and the Hamiltonian parameters are the
model input. A correlation between an activation and an observable could therefore
be inherited from the input, or could occur in a random nonlinear map of that
input. A successful probe alone is not evidence that training produced a
distinctive physical representation. Likewise, a decodable direction need not be
a direction that the model uses for its prediction.

We address both issues with a control-oriented analysis. For each observable, we
compare linear decodability from the trained residual stream with decodability from
an untrained transformer of identical architecture, the raw input, degree-two
polynomial features of the input, and the mean field. We supplement this comparison
with a layer sweep, a multiple-comparisons-aware permutation null for SAE feature
correlations, partial correlations conditional on the full degree-two polynomial of
the input, and a ten-seed replication that compares independently trained
transformers against a random-initialization distribution. Finally, we perform a
directional intervention: we project a probe-defined order direction out of the
last residual layer and measure the consequence for energy prediction.

Our contributions are empirical and deliberately scoped:

1. We provide a controlled probe analysis showing that, in this TFIM energy
   regression setting, the trained residual stream has a reproducible linear
   advantage for the end-to-end correlator \(\langle Z_0Z_{L-1}\rangle\) beyond
   raw-input and untrained-network baselines.
2. We separate the evidence for learned structure from evidence for causal use.
   The order direction is readily decodable and can be effectively ablated, yet
   its removal has a much smaller effect on energy prediction than removal of
   random directions of equal norm.
3. We report results that constrain and sharpen the interpretation: at the level
   of raw decodability several observables are captured nearly as well by an
   untrained network, but under the full degree-two input control the end-to-end
   correlator, nearest-neighbour correlation, and transverse magnetisation each
   separate the trained representation from a random-initialization distribution;
   SAE features are not universal across seeds; and a fixed longitudinal field
   makes the selected order proxy nearly input-trivial in a mixed-field control.
4. We release an implementation with exact and sparse ground-state solvers,
   observable tests, experiment scripts, and a documented reproduction path.

The appropriate conclusion is not that the model has discovered a universal
description of quantum matter, nor that it uses non-local order to solve the
energy-regression task. Rather, within this controlled problem, training is
associated with a residual representation in which a non-local observable is more
linearly accessible than in the tested baselines. The intervention results show
why representational evidence and causal-mechanistic evidence should be reported
separately.

## 2. Related Work

### Machine learning representations in physics

SciNet was designed to study whether neural networks can discover compact,
physically meaningful variables from prediction tasks [Iten et al., 2018].
Related work formalised operationally meaningful representations of physical
systems, including quantum examples [Nautrup et al., 2020]. These studies motivate
evaluating a learned representation rather than treating predictive performance as
an explanation. Our setting differs in two respects: the input is a Hamiltonian
description of a quantum many-body system, and the quantities evaluated in the
representation are exact ground-state observables.

Neural quantum states use neural networks as variational representations of
wavefunctions and have become a major approach to many-body computation [Carleo &
Troyer, 2017]. Our transformer is not a neural quantum state and does not represent
a wavefunction. It is a classical regressor trained on exact energy labels. The
question here is therefore not whether a neural ansatz can approximate a quantum
state, but which physically interpretable variables are accessible in a network
trained for a different supervised target.

World-model work similarly studies learned latent states that support downstream
prediction and control [Ha & Schmidhuber, 2018]. The present work is closer to
scientific representation analysis than to model-based reinforcement learning:
there is no temporal latent dynamics or policy. The shared methodological concern
is that useful predictive latents need not expose all of their structure through
the task output alone.

### Probes and mechanistic interpretability

Linear classifier probes provide a standard way to measure how accessible a target
is from intermediate activations [Alain & Bengio, 2016]. Such probes are
descriptive: they establish information accessibility under a chosen probe class,
not whether the base model uses that information. Mechanistic-interpretability
work makes this distinction central, seeking causal accounts of model computation
through components, connections, and interventions [Olah et al., 2020; Elhage et
al., 2021]. We use probes as a representation-level measurement and pair them with
an activation intervention specifically to avoid treating decodability as
mechanistic necessity.

### Sparse autoencoders

Sparse autoencoders decompose activations into sparse learned dictionaries and are
used as a potential route to more interpretable feature spaces [Bricken et al.,
2023]. Top-\(k\) SAEs control activation sparsity directly and address practical
issues such as dead latents [Gao et al., 2024]. We use a Top-\(k\) SAE as an
exploratory lens on the residual stream. Importantly, cross-seed matching in our
experiments does not support stable, one-to-one semantic interpretations of
individual SAE features. The main claims of this paper therefore concern the
representation and its linear subspaces, not a universal SAE feature basis.

## 3. Problem Setting and Methods

### 3.1 Background and Dataset Generation

We consider the open-boundary one-dimensional TFIM

\[
H(\mathbf h) = -J\sum_{i=0}^{L-2} Z_iZ_{i+1}
               -\sum_{i=0}^{L-1} h_iX_i,
\]

where \(J=1\), \(X_i\) and \(Z_i\) are Pauli operators, and
\(\mathbf h=(h_0,\ldots,h_{L-1})\) is a vector of independently sampled transverse
fields. The primary task uses \(L=8\) and
\(h_i\sim\mathrm{Uniform}(0.1,2.0)\). For each field configuration we compute the
ground-state energy \(E_0(\mathbf h)\) by exact diagonalisation.

The primary energy-regression dataset contains 50,000 training, 5,000 validation,
and 5,000 test samples. Training, validation, and test field vectors are
independently drawn from the same distribution. The code uses batched dense
eigensolvers for the \(L=8\) dataset. The scaling experiments use a sparse Lanczos
solver, validated against the dense solver at \(L=8\), to obtain ground states and
energies at \(L=8,10,12\).

### 3.2 Physical Background: Quantum Observables

For fresh sets of exact ground states, we evaluate:

- half-chain von Neumann entropy \(S(\rho_A)\);
- the mean nearest-neighbour correlator
  \(\frac{1}{L-1}\sum_i\langle Z_iZ_{i+1}\rangle\);
- mean transverse magnetisation
  \(\frac{1}{L}\sum_i\langle X_i\rangle\);
- the end-to-end correlator \(\langle Z_0Z_{L-1}\rangle\);
- phase proximity \(\delta=(\bar h-h_c)/h_c\), with
  \(\bar h=L^{-1}\sum_i h_i\) and \(h_c=1\).

At finite system size, the exact TFIM ground state preserves the global
\(Z_2\) symmetry. Consequently, the single-site quantity
\(\frac{1}{L}\sum_i|\langle Z_i\rangle|\) is numerically near zero and is not a
useful order target. We therefore use the end-to-end correlator as a standard
finite-size proxy for ferromagnetic order. We refer to it as a *non-local order
proxy*, not as a thermodynamic-limit order parameter.

The repository implements dense observable routines and bit-arithmetic
implementations for correlators and magnetisation. Tests compare the fast and
dense paths on random states and TFIM ground states. Entanglement entropy is
computed from the reduced density matrix of a half-chain bipartition.

### 3.3 Transformer Architecture

The predictor maps \(\mathbf h\in\mathbb R^L\) to a scalar energy. Each scalar
\(h_i\) is linearly embedded, summed with a learned positional embedding, and
processed by a three-layer, pre-layer-normalised transformer encoder. The model
uses \(d_{\mathrm{model}}=64\), four attention heads, feed-forward width 256, no
dropout, mean pooling over sites, a final layer normalisation, and a two-layer
GELU regression head. It contains 152,833 trainable parameters.

### 3.4 Training Procedure

Targets are standardised using training-set energy statistics. We train with
AdamW (learning rate \(3\times10^{-4}\), weight decay \(10^{-4}\)), cosine learning
rate decay, gradient-norm clipping at 1.0, batch size 128, and early stopping with
patience 15. Reported physical-unit metrics are obtained after inverting the target
normalisation. The best validation checkpoint is selected by unnormalised
validation \(R^2\).

### 3.5 Interpretability Methods: Representation Extraction and Linear Probes

We register forward hooks on the transformer encoder blocks. At each layer, the
post-block residual stream has shape \((N,L,d_{\mathrm{model}})\); we mean-pool over
sites to obtain an \((N,d_{\mathrm{model}})\) representation. All representation
measurements use held-out disorder realisations rather than the energy-training
examples.

For each scalar observable, we fit a ridge regressor and report mean five-fold
cross-validated \(R^2\). Within each fold, the input representation is standardised
using only the training fold. We compare five representations:

1. the trained transformer’s last-layer residual stream;
2. an untrained transformer with the same architecture;
3. raw \(\mathbf h\);
4. degree-two polynomial features of \(\mathbf h\);
5. the scalar mean field \(\bar h\).

The untrained control tests whether generic nonlinear mixing from a randomly
initialised architecture is sufficient. The input baselines test whether the
trained representation improves linear access beyond directly available input
statistics.

### 3.6 Interpretability Methods: SAE Analysis and Correlation Controls

We train a Top-\(k\) SAE on standardised last-layer residual activations. The
default SAE has \(d_{\mathrm{in}}=64\), \(d_{\mathrm{hidden}}=256\), \(k=32\), and
an auxiliary dead-latent loss. For each observable, we compute Pearson
correlations with alive SAE features.

The raw correlation screen is not treated as a scientific result by itself. We use
three additional controls. First, for a permutation null, we repeatedly permute
the target and recompute the maximum absolute correlation over all alive features.
This null matches the search procedure used to choose the best feature and
controls the multiple-comparisons effect of reporting the maximum. Second, for the
selected best feature \(z\), we compute the partial correlation
\(r(z,y\mid\bar h)\) with observable \(y\) conditional on mean field. This is a
restricted confound control at the level of individual SAE features: it removes
linear mean-field dependence only. The representation-level analysis (Section 5.2)
strengthens this to a control for the full site-resolved field vector and its
degree-two polynomial. Third, we
train three SAEs on the same residual activation matrix with different seeds,
Hungarian-match decoder directions by cosine similarity, and report mean matched
cosine and the fraction above 0.7.

### 3.7 Activation Patching

Probing does not establish causal use. We therefore fit a ridge probe for
\(\langle Z_0Z_{L-1}\rangle\) on a training half of 1,200 fresh TFIM samples. Its
standardised coefficient is mapped back to the raw residual coordinates and
normalised to a unit vector \(\mathbf d\). On the held-out half, a hook at the final
encoder layer applies

\[
\mathbf x_i\leftarrow\mathbf x_i
-(\mathbf x_i^{\mathsf T}\mathbf d)\mathbf d
\]

at every site \(i\). Since the model subsequently mean-pools across sites, this
removes the \(\mathbf d\) component from the pooled representation read by the
head. We compare physical-unit energy RMSE with no intervention, with this
order-direction ablation, and with the mean over 15 independently sampled random
unit-direction ablations. As a sanity check, we re-evaluate order-probe \(R^2\)
after the order-direction ablation.

This is a targeted test of one linear direction at one layer. It does not recover a
complete circuit or rule out nonlinear and redundant representations of the same
observable elsewhere in the model.

## 4. Experimental Setup

The main control battery uses 800 fresh \(L=8\) ground states, 500 permutations,
and 200 SAE epochs. For the trained-seed analysis, we retrain the transformer from
ten independent seeds (fresh weight initialisation and fresh disorder draws) and
run the identical full degree-two input control on each, holding the evaluation
states fixed so that only the trained weights vary; we compare the resulting
trained distribution against a random-initialization distribution of 64 untrained
networks of the same architecture. The random-initialization 95th percentile used
as the decision threshold is verified for stability against a 16-network launch
pool. The SAE cross-seed universality analysis separately uses seeds 42, 43, and
44 on a fixed checkpoint.

For the SAE hyperparameter study, we use 2,000 residual activations and train
three SAEs for each cell in
\(d_{\mathrm{hidden}}\in\{256,512,1024\}\) and
\(k\in\{8,16,32\}\). For system-size scaling, we train a new transformer at each
of \(L=8,10,12\), using 15,000 energy samples per size and a fixed transformer
width. We then measure order-proxy probe performance on fresh test states.

We also run a mixed-field diagnostic in which a fixed longitudinal field
\(g=0.5\) adds \(-g\sum_i Z_i\) to the Hamiltonian. This breaks integrability and
the TFIM’s \(Z_2\) symmetry simultaneously. It is therefore informative about the
selected observable and input confound, but is not an isolated test of
integrability.

## 5. Results

### 5.1 Energy prediction is accurate but does not by itself establish structure

The transformer predicts held-out \(L=8\) ground-state energies accurately and
outperforms the tested linear and degree-two polynomial baselines (Table 1). This
establishes a useful predictor for the representation analysis; it does not imply
that every physically meaningful observable is represented or used.

| Model | Test \(R^2\) | Test RMSE |
|---|---:|---:|
| Linear regression | 0.9812 | 0.1501 |
| Degree-two polynomial regression | 0.9989 | 0.0366 |
| Transformer | **0.9999** | **0.0104** |

*Table 1: Energy-regression performance at \(L=8\). The transformer is trained on
50,000 field--energy pairs and evaluated on 5,000 held-out pairs.*

Figure 1 reuses the repository’s predicted-versus-true and residual plots.

<p align="center">
  <img src="../figures/ra01_wide_pred_vs_truth.png" width="720"
       alt="Transformer energy predictions and residuals"/>
</p>

*Figure 1: Held-out TFIM energy predictions and residuals for the trained
transformer.*

### 5.2 The trained residual stream has a distinct linear advantage for the non-local order proxy

Table 2 reports raw linear decodability, without confound control. At this raw
level several observables are highly and comparably decodable from the trained and
untrained representations: entropy, nearest-neighbour correlation, and transverse
magnetisation are decoded almost as well from an untrained transformer or from
degree-two input features, because their raw decodability is near-saturated. Phase
proximity is perfectly decoded from the mean field, as expected from its
definition. The end-to-end correlator is the exception even at this raw level: its
trained-representation score is 0.961, compared with 0.921 for the untrained
transformer, 0.772 for raw fields, 0.942 for degree-two input features, and 0.695
for the mean field (single architecture-matched draws; the distributional version
follows).

| Observable | Trained residual | Untrained residual | Raw \(\mathbf h\) | Poly-2 \(\mathbf h\) | Mean \(\bar h\) |
|---|---:|---:|---:|---:|---:|
| Half-chain entropy | 0.934 | 0.938 | 0.863 | 0.941 | 0.670 |
| Mean nearest-neighbour \(ZZ\) | 0.995 | 0.991 | 0.971 | 0.988 | 0.945 |
| Mean \(X\) magnetisation | 0.991 | 0.985 | 0.916 | 0.983 | 0.896 |
| End-to-end \(ZZ\) | **0.961** | 0.921 | 0.772 | 0.942 | 0.695 |
| Phase proximity | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |

*Table 2: Mean five-fold cross-validated ridge \(R^2\) for \(N=800\) fresh TFIM
instances. The end-to-end correlator is the finite-size non-local order proxy.*

<p align="center">
  <img src="../figures/ra03_probe_r2.png" width="680"
       alt="Cross-validated probe results"/>
</p>

*Figure 2: Linear-probe comparison between the trained residual stream,
architecture-matched untrained residual stream, and input baselines.*

Raw decodability, however, conflates information the network computes with
information already present in the input. To isolate the former we partial out the
full site-resolved field vector and its degree-two polynomial, and compare ten
independently trained transformers against a 64-network random-initialization
distribution (Table 2b). We report two measures: the partial correlation between
the representation's out-of-fold probe prediction and the observable given the
degree-two input, and the incremental \(R^2\) that the representation adds beyond
that input. Under this stronger control the picture is sharper than the raw
comparison suggests. For the end-to-end correlator the trained incremental \(R^2\)
is \(0.028\pm0.003\) versus \(0.012\pm0.004\) for random initialisations
(\(+4.2\sigma\)), uniform across all ten seeds; the beyond-input partial
correlation is \(0.560\pm0.046\) against a random-initialization 95th percentile of
0.429. Nearest-neighbour correlation and transverse magnetisation, which look
training-neutral at the raw level, also separate from random initialisations under
this control (incremental-\(R^2\) separations of \(+7.3\sigma\) and \(+4.5\sigma\)).
Half-chain entropy separates clearly on the incremental measure (\(+4.6\sigma\)) but
only modestly in partial correlation (\(+1.8\sigma\)). Phase proximity remains
trivially input-explained, as it must be. The end-to-end correlator is distinctive
not because it is the only observable that training helps, but because it has the
lowest random-initialization baseline -- beyond-input non-local structure is
hardest to obtain without training.

| Observable | Probe \(R^2\) (trained) | Partial \(r\mid\)poly-2 (trained) | Partial \(r\mid\)poly-2 (random, p95) | Sep. | Incr. \(R^2\) (trained) | Incr. \(R^2\) (random) | Sep. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Half-chain entropy | 0.918 ± 0.012 | 0.565 ± 0.040 | 0.404 ± 0.090 (0.525) | +1.8σ | 0.031 ± 0.002 | 0.018 ± 0.003 | +4.6σ |
| Mean nearest-neighbour \(ZZ\) | 0.993 ± 0.001 | 0.765 ± 0.039 | 0.362 ± 0.115 (0.534) | +3.5σ | 0.010 ± 0.000 | 0.005 ± 0.001 | +7.3σ |
| Mean \(X\) magnetisation | 0.987 ± 0.002 | 0.623 ± 0.066 | 0.369 ± 0.099 (0.559) | +2.6σ | 0.012 ± 0.001 | 0.007 ± 0.001 | +4.5σ |
| End-to-end \(ZZ\) | **0.953 ± 0.006** | **0.560 ± 0.046** | 0.280 ± 0.109 (0.429) | +2.6σ | **0.028 ± 0.003** | 0.012 ± 0.004 | +4.2σ |
| Phase proximity | 0.999 ± 0.000 | 0.000 | 0.000 | — | 0.000 | 0.000 | — |

*Table 2b: Beyond-input control (full degree-two polynomial of the field).
Trained-seed distribution (ten independently trained transformers, mean ± sd)
versus a random-initialization distribution (64 networks), at \(L=8\). Partial
correlation is between the representation's out-of-fold probe prediction and the
observable, conditional on the degree-two input; incremental \(R^2\) is the increase
in out-of-fold ridge \(R^2\) from adding the representation to that input.
Separation is (mean\(_\text{trained}\) − mean\(_\text{random}\))/sd\(_\text{random}\).
All ten trained seeds exceed the random-initialization 95th percentile for the
end-to-end correlator on both measures; the threshold is stable at the 16-network
launch pool (bootstrap CI [0.378, 0.426]) and at 64 networks (0.429). Energy test
\(R^2=0.9996\pm0.0003\) across the ten trained seeds.*

### 5.3 The advantage increases through transformer depth

The layer sweep localises the probe result within the representation. For
\(\langle Z_0Z_{L-1}\rangle\), probe \(R^2\) rises from 0.916 after the first
encoder layer to 0.945 after the second and 0.961 after the third. The other
observables are comparatively flat across depth. This trajectory is consistent
with, but does not by itself prove, progressive construction of a non-local
summary across layers.

<p align="center">
  <img src="../figures/ra03_layer_sweep.png" width="600"
       alt="Probe scores across transformer layers"/>
</p>

*Figure 3: Probe \(R^2\) across layers of the trained transformer. The
end-to-end \(ZZ\) score increases across depth.*

### 5.4 Correlation controls reject the simple feature-selection null

The default SAE produces 208 alive features on the primary correlation run.
The strongest raw feature--observable correlations range from 0.74 for entropy
to 0.90 for the end-to-end correlator. These values are descriptive only, because
they result from selecting the best feature from a dictionary.

The permutation null addresses this selection procedure. Across the five
observables, the 95th percentiles of the null maximum absolute correlations are
0.12--0.16, whereas the observed best-feature magnitudes are 0.79--0.90. None of
the 500 permutations exceeds the observed magnitude in the reported runs
(empirical \(p\) reported as approximately zero at this resolution). This shows
that the selected correlations are not consistent with a target-permutation null;
it does not establish that a particular feature is a unique physical factor.

Partial correlations clarify which of these associations persist after removing
the linear contribution of \(\bar h\) (Table 3). The phase-proximity result
appropriately disappears because phase proximity is defined from \(\bar h\).
The end-to-end correlator retains the largest partial correlation.

| Observable | Best-feature \(|r|\) | Partial \(r(\mathrm{feature},y\mid\bar h)\) |
|---|---:|---:|
| Entropy | 0.79 | 0.348 |
| Mean nearest-neighbour \(ZZ\) | 0.85 | 0.133 |
| Mean \(X\) magnetisation | 0.82 | 0.328 |
| End-to-end \(ZZ\) | 0.90 | **0.694** |
| Phase proximity | 0.89 | 0.000 |

*Table 3: Best SAE-feature correlations before and after controlling for the mean
field in the main \(N=800\) run.*

<p align="center">
  <img src="../figures/ra03_partial_corr.png" width="600"
       alt="Raw and partial correlations"/>
</p>

*Figure 4: Raw best-feature correlations and correlations partialled on the mean
field. The phase-proximity control is fully explained by the mean field.*

### 5.5 The order result replicates across independently trained seeds

To rule out that the effect reflects a single lucky training run, we retrain the
transformer from ten independent seeds (fresh weight initialisation and fresh
disorder draws) and repeat the full degree-two input control on each, holding the
evaluation set fixed so that only the trained weights vary. Table 4 summarises the
resulting distributions for the end-to-end correlator against the 64-network
random-initialization pool. The trained incremental \(R^2\) beyond the degree-two
input is \(0.028\pm0.003\), with every one of the ten seeds exceeding the
random-initialization 95th percentile (0.017); the beyond-input partial correlation
is \(0.560\pm0.046\), again with all ten seeds above the random 95th percentile
(0.429). No trained seed falls into or near the random range on either measure, so
the effect is uniform across seeds rather than driven by a few strong runs. These
results support robustness over the tested training seeds, not a claim of
universality over architectures, Hamiltonians, or datasets.

| Quantity for end-to-end \(ZZ\) (poly-2 control) | Trained (10 seeds) | Random init (64 networks) |
|---|---:|---:|
| Beyond-input partial correlation | \(0.560\pm0.046\) [min 0.503] | \(0.280\pm0.109\) (p95 0.429) |
| Incremental \(R^2\) beyond degree-two input | \(0.028\pm0.003\) [min 0.023] | \(0.012\pm0.004\) (p95 0.017) |
| Raw probe \(R^2\) (trained residual) | \(0.953\pm0.006\) | — |

*Table 4: Ten-seed replication for the non-local order proxy under the full
degree-two input control, against a 64-network random-initialization distribution.
Bracketed values are the minimum over the ten trained seeds. Energy test
\(R^2=0.9996\pm0.0003\) across the ten trained seeds.*

## 6. Activation Patching: Representation Is Not Causal Utilization

We next test whether the order information detected by probing is important for
the trained energy prediction. Table 5 reports held-out energy RMSE after removing
the probe-defined direction and after removing random unit directions. The order
ablation modestly increases all-sample RMSE from 0.0112 to 0.0137. In contrast,
random unit-direction ablations increase RMSE to 0.1001 on average. The
order-defined direction has roughly twelve times lower residual variance than a
random direction (0.006 versus 0.074).

| Intervention | RMSE, all | RMSE, \(\bar h<1\) | RMSE, \(\bar h>1\) |
|---|---:|---:|---:|
| None | 0.0112 | 0.0115 | 0.0110 |
| Remove order direction | 0.0137 | 0.0125 | 0.0145 |
| Remove random direction (mean of 15) | 0.1001 | 0.0966 | 0.1006 |

*Table 5: Held-out energy RMSE after final-layer directional interventions.*

The ablation is effective as a representational intervention: a probe trained on
the clean residual stream has order-proxy \(R^2\) of approximately 0.97 on the
test half, which drops to \(-9.6\) after removal of the direction. The small
energy effect is therefore not explained by failure to remove the linearly probed
component.

These observations are consistent with the order proxy being represented in a
low-variance subspace that is comparatively weakly coupled to the energy-prediction
pathway. They do not imply that the model has no causal dependence on non-local
physics; the intervention examines one probe-defined linear direction at one layer.
They do show that a strong probe result is insufficient evidence that the
corresponding direction is load-bearing for the supervised output.

<p align="center">
  <img src="../figures/ra07_causal.png" width="600"
       alt="Energy RMSE under directional ablation"/>
</p>

*Figure 5: Directional intervention results. The order-direction ablation is
effective at removing probe decodability but is substantially less disruptive to
energy prediction than random-direction ablations.*

## 7. SAE Analysis

The SAE analysis is useful for identifying directions that correlate with
observables and for constructing the permutation and partial-correlation
experiments above. It does not support a stable individual-feature account.

At the default configuration (\(d_{\mathrm{hidden}}=256,k=32\)), Hungarian
matching across three SAE seeds yields a mean matched decoder cosine of
approximately 0.37 and only 0.3% of matches above cosine 0.7. We tested whether
this was an easily resolved hyperparameter choice by sweeping
\(d_{\mathrm{hidden}}\in\{256,512,1024\}\) and \(k\in\{8,16,32\}\). The best cell,
\(d_{\mathrm{hidden}}=256,k=8\), reaches about 5.9% matches above 0.7; widening
the dictionary does not resolve the instability. Mean matched cosine remains
approximately 0.37--0.42 over the grid.

<p align="center">
  <img src="../figures/ra04_sae_universality.png" width="620"
       alt="SAE cross-seed universality grid"/>
</p>

*Figure 6: Cross-seed SAE decoder matching as a function of dictionary width and
Top-\(k\) sparsity. The improvement at smaller \(k\) does not yield a
seed-universal feature basis.*

Accordingly, our main result does not rely on assigning a fixed semantic label to
an SAE unit. The representation-level probe results use the raw residual stream,
and the SAE results are reported as an exploratory decomposition with a clear
negative finding.

As a separate pipeline-validation experiment, the repository includes a
Bars-and-Stripes quantum-neural-network to classical-shadow to SAE run. On that
small classical-data task, the QNN attains 0.979 test accuracy and the SAE has a
reported monosemantic fraction of 0.611. This validates that the implementation can
recover structured sparse features in a known-label setting; it is not evidence for
the TFIM representation claim and is therefore treated as supplementary
validation.

## 8. System-Size and Mixed-Field Diagnostics

### 8.1 Scaling from \(L=8\) to \(L=12\)

To check whether the principal result is restricted to the \(L=8\) dense-solver
setting, we retrain a transformer at \(L=8,10,12\) and use the sparse solver for
data generation. Energy validation \(R^2\) is approximately 0.9998 at each size.
Table 6 reports the non-local-proxy probe comparison. Absolute decodability
declines with \(L\) for all representations at fixed model width, but the learned
gain over the best tested baseline remains close to 0.028.

| \(L\) | Trained | Untrained | Raw \(\mathbf h\) | Mean \(\bar h\) | Learned gain |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.950 | 0.920 | 0.753 | 0.667 | +0.029 |
| 10 | 0.927 | 0.899 | 0.736 | 0.636 | +0.028 |
| 12 | 0.894 | 0.867 | 0.719 | 0.600 | +0.027 |

*Table 6: End-to-end \(ZZ\) probe \(R^2\) across system size. Learned gain is the
trained score minus the largest of the untrained, raw-field, and mean-field scores.*

<p align="center">
  <img src="../figures/ra08_scaling.png" width="720"
       alt="Order probe scores across system sizes"/>
</p>

*Figure 7: Scaling diagnostic. At fixed model width, the learned advantage is
robust over the tested sizes but does not increase with \(L\).*

This experiment provides evidence against an \(L=8\)-only explanation over the
tested range. It is not a demonstration of asymptotic scaling, and it remains
within the exact-diagonalisation/Lanczos regime rather than using tensor-network
methods for substantially larger chains.

### 8.2 Mixed-field diagnostic

Adding a fixed longitudinal field \(g=0.5\) creates a mixed-field Ising model. In
this experiment, the learned advantage for the end-to-end correlator disappears:

| \(L\) | Trained | Untrained | Raw \(\mathbf h\) | Mean \(\bar h\) | Learned gain |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.962 | 0.979 | 0.969 | 0.340 | -0.018 |
| 10 | 0.962 | 0.964 | 0.969 | 0.276 | -0.007 |

The important interpretation is not that non-integrability eliminates learned
representations. The longitudinal field explicitly breaks the TFIM’s \(Z_2\)
symmetry and polarises the state, making the end-to-end correlator almost
input-linearly decodable (raw-field \(R^2\approx0.97\)). This control therefore
conflates non-integrability with an observable that has become input-trivial. It
shows that the reported advantage depends on the target retaining beyond-input
structure; it does not isolate integrability as a causal factor.

### 8.3 Hamiltonian diversity: a non-integrable model and a cross-symmetry control

The §8.2 diagnostic is confounded because it changes the observable's
input-triviality at the same time as integrability. We therefore run two further
families, each under the full multi-seed protocol used for the headline (ten
independently trained transformers versus a 64-network random-initialization
distribution, the full degree-two input control, energy R² and the beyond-input
protection ⟨Z_i⟩→0 verified before any probe). For each observable we report its
ensemble standard deviation σ_y next to the separation, because a target that
barely varies cannot support a conclusion; a target is flagged *underpowered* when
it shows no separation and σ_y is below a fixed floor.

**A clean single-variable test — transverse-field ANNNI (non-integrable).** Adding
a next-nearest-neighbour coupling −J₂∑ZᵢZᵢ₊₂ with κ=J₂/J₁=0.3 breaks integrability
(Jordan–Wigner maps J₂ to a four-fermion interaction) while **preserving** the Z₂
spin-flip symmetry, so ⟨Z_i⟩=0 and the order proxy stays beyond-input. The input
is the transverse-field vector, exactly as in the primary TFIM setting, so this
varies **only integrability**. Energy test R² = 0.9995 ± 0.0004 over ten seeds;
max|⟨Z_i⟩| = 1.0×10⁻¹⁰. The result (Table 8) is a clear transfer: the non-local
order proxy ⟨Z₀Z_{L−1}⟩ (σ_y = 0.195, well powered) separates the trained
representation from the random-initialization distribution on **both** the
incremental R² beyond the degree-two input (0.009 vs 0.004, +5.9σ) and the
beyond-input partial correlation (0.613 vs 0.324, +3.5σ), uniformly across all ten
seeds. Entropy, the nearest-neighbour correlator, and the staggered structure
factor separate on both cuts as well; transverse magnetisation ⟨X_i⟩ separates on
the incremental measure (+6.3σ) but **not** on partial correlation (0.597 vs 0.584,
+0.3σ), and we report that split rather than dropping it.

**Separation and magnitude are distinct axes.** The large σ-separations are
statements of *confidence that the effect is nonzero*, not of its size. In absolute
terms the effect **attenuates** under non-integrability: the order-proxy incremental
R² is about three times smaller than in the TFIM (0.009 versus 0.028), while the
beyond-input partial correlation (0.613) is on par with the TFIM value (0.560). The
honest reading is therefore that the effect **survives but weakens** — robustly
present and uniform over seeds, yet smaller in absolute incremental decodability.

| Observable | σ_y | Incr. R² trained [min] | Incr. R² random (p95) | sep | Partial-r trained | sep |
|---|---:|---:|---:|---:|---:|---:|
| End-to-end ⟨Z₀Z_{L−1}⟩ | 0.195 | 0.009 [0.008] | 0.004 (0.006) | +5.9σ | 0.613 | +3.5σ |
| Half-chain entropy | 0.093 | 0.024 [0.021] | 0.010 (0.014) | +6.6σ | 0.565 | +4.0σ |
| Mean nearest-neighbour ⟨ZZ⟩ | 0.095 | 0.006 [0.005] | 0.004 (0.004) | +6.2σ | 0.824 | +4.0σ |
| Mean ⟨X⟩ magnetisation | 0.119 | 0.004 [0.004] | 0.003 (0.003) | +6.3σ | 0.597 | +0.3σ |
| Staggered structure factor | 0.011 | 0.006 [0.005] | 0.004 (0.004) | +5.6σ | 0.741 | +3.1σ |

*Table 8: Transverse-field ANNNI (κ=0.3, non-integrable), ten trained seeds vs 64
random inits, poly-2(h) control. The order proxy separates on both measures,
uniformly over seeds; magnitudes are smaller than TFIM (order-proxy incremental R²
0.009 vs 0.028) while the partial correlation is comparable (0.613 vs 0.560).*

**A cross-symmetry test that cannot isolate a cause — XXZ (integrable, U(1)).**
Disordering the per-bond anisotropy Jz of the XXZ chain gives an integrable model
with U(1) rather than Z₂ symmetry. At the principled TFIM-analog range
Jz∼U(0.1,2.0), the well-powered target is entanglement entropy (σ_y=0.174): it is a
**clean null** — incremental R² beyond poly-2 = 0.0004, versus 0.031 in the TFIM
(an ~80× smaller beyond-input signal), with the trained distribution not exceeding
the random one. The ZZ-based order observables are **underpowered** at this range
(σ_y = 0.03–0.09), so the order-transfer question cannot be adjudicated. This test
is **conservative, not decisive**: XXZ changes **two variables at once** — the input
type (transverse fields → couplings) *and* the symmetry (Z₂ → U(1)). Because the
input is the couplings that directly govern the ZZ observables, a degree-two
polynomial of the input partly saturates their decodability (entropy probe R² =
0.951), the same input-triviality confound family as §8.2. The result therefore
reads as *"the effect is absent in this configuration, for reasons we cannot fully
separate from the input choice,"* not as *"U(1) symmetry removes the effect."* The
specific control that would disentangle these — a field-disorder XXZ, in which a
field rather than the couplings is disordered so the observable is a non-trivial
function of the input — is named in the future work and not run here.

**What the diversity experiments establish.** The effect is **not specific to the
integrable TFIM**: in the single-variable test that isolates integrability (ANNNI),
the trained representation's beyond-input encoding of non-local order **survives**,
uniformly over ten seeds, though it **attenuates** in absolute terms. The
cross-symmetry test (XXZ) is confounded by an input-type change and returns an
honest, underpowered null that we do not over-read.

## 9. Discussion

The experiments support a narrow but useful conclusion. When trained to predict
TFIM ground-state energies, this transformer develops a representation from which
a finite-size non-local order proxy is more linearly accessible than from an
architecture-matched untrained transformer and the tested input baselines. The
effect has a coherent depth profile, survives a partial-correlation check against
the full degree-two polynomial of the input, exceeds a permutation null for
selected SAE correlations, and replicates uniformly across ten independently
trained seeds relative to a random-initialization distribution. It is also not an
artifact of integrability: in a non-integrable transverse-field ANNNI model
(Section 8.3) the beyond-input encoding of non-local order survives, though it
attenuates in absolute magnitude; a cross-symmetry XXZ test is confounded by an
input-type change and left as an honest, underpowered null.

The causal experiment changes the interpretation of this result. The
order-defined direction is present and probe-accessible, yet removing it is not
comparably harmful to the energy output. The representation may carry physical
information that is incidental, redundant, or weakly coupled to the particular
prediction head. Thus, a neural representation can encode a scientifically
meaningful variable without that variable being a load-bearing mechanism for the
task.

This distinction matters in scientific ML. A decoder can establish that a
representation contains an observable under the decoder’s function class; it
cannot, alone, establish that the network computes with that observable. Conversely,
a modest output change from a single directional ablation does not show that the
observable is irrelevant everywhere in the model. Reporting both measurements
makes the evidential status of the conclusion clearer.

The result also calibrates the role of sparse autoencoders. The SAE reveals strong
correlating directions but does not yield a stable cross-seed coordinate system.
For this reason, linear properties of the raw residual representation are a more
appropriate basis for the paper’s central claims than labels assigned to individual
SAE latents.

## 10. Limitations

This work has several substantial limitations.

1. **Limited Hamiltonian coverage.** The primary result concerns disordered,
   open-boundary TFIM. We add one clean single-variable generalisation — a
   non-integrable transverse-field ANNNI model (Section 8.3), where the effect
   survives but attenuates — and one cross-symmetry test, XXZ, which is confounded
   by an input-type change (couplings rather than fields as input) and returns an
   underpowered null. We therefore establish that the effect is not an integrability
   artifact, but we do not establish universality across symmetries, boundary
   conditions, or a broad Hamiltonian family; in particular the cross-symmetry
   (Z₂ vs U(1)) question remains open pending the field-disorder XXZ control.
2. **Finite sizes.** The scaling study reaches \(L=12\) using sparse
   exact-diagonalisation methods. It does not establish behaviour at system sizes
   requiring DMRG or other tensor-network solvers.
3. **Restricted controls (substantially addressed).** Our confound control now
   partials out the full site-resolved field vector and its degree-two polynomial,
   not only the scalar mean field, and the trained representation retains a
   separation from a 64-network random-initialization distribution under this
   stronger control, uniform across ten trained seeds (Tables 2b, 4). The remaining
   caveat is narrower: ordinary-least-squares residualisation on degree-two
   features does not exhaust arbitrary nonlinear dependence on the full field
   vector, since higher-order or non-polynomial functions of \(\mathbf h\) could in
   principle still account for part of the residual association.
4. **No complete circuit explanation.** The intervention targets a single
   probe-defined direction in the final layer. We do not identify attention heads,
   MLP pathways, or a complete mechanism that creates or uses the order signal.
5. **SAE non-universality.** Individual Top-\(k\) SAE features are not stable
   enough across the tested seeds to support one-to-one feature-level physical
   claims.
6. **Task specificity.** Energy is a relatively local/extensive target. The
   conclusion that the order direction is not load-bearing applies to this model,
   direction, and output task, not to all tasks involving the same Hamiltonian.
7. **No hardware claim.** The primary TFIM analysis uses classically computed
   ground states. The classical-shadow code is validated in a separate
   simulation-based pipeline, not used as hardware evidence for the main result.

## 11. Future Work

Several extensions would make the question more stringent.

- **Hamiltonian diversity.** XXZ, Heisenberg, and cluster-Ising models would test
  whether similar representational effects occur for different symmetries and
  correlation structures. A non-integrable setup should retain an observable that
  is not trivially decodable from the supplied inputs; connected correlators or
  disorder in couplings are promising candidates. A methodological caution applies
  when choosing what to disorder: feeding the *couplings* as input (e.g. per-bond
  XXZ anisotropies) tends to make the correlation observables partly
  input-decodable — the same input-triviality confound as the fixed-longitudinal
  mixed-field control (Section 8.2) — so it changes the input type and the symmetry
  at once and cannot isolate either. The single-variable comparison holds the input
  type fixed (transverse fields, as in the primary TFIM setting) and varies only
  integrability. The specific missing control that would disentangle a symmetry
  effect from an input-triviality effect is a **field-disorder XXZ**: disorder a
  transverse or staggered field on top of fixed couplings, so the order observable
  is a non-trivial function of the input as it is in the TFIM.
- **Larger systems.** DMRG or other tensor-network ground-state solvers would
  permit a study beyond the exact-diagonalisation scale and could test scaling
  model width and depth with \(L\).
- **Alternative tasks.** Predicting gaps, correlators, response functions,
  dynamics, or phase-sensitive objectives may make non-local structure more
  causally relevant to the supervised task.
- **Mechanistic localisation.** Attention-pattern analysis, path patching, and
  layer- or head-specific interventions could identify how the order signal is
  constructed and whether it is redundantly represented.
- **Representation decompositions.** More data, alternative SAE objectives, and
  subspace-level stability tests may clarify whether the observed SAE instability
  reflects non-identifiability of the dictionary or limited data/model scale.
- **Quantum data acquisition.** Classical-shadow measurements on quantum hardware
  could connect the supplementary shadow pipeline to experimentally obtained
  states, subject to measurement noise and sampling-cost analyses.

These are future research directions, not completed experiments in this work.

## 12. Conclusion

We studied internal representations of a transformer trained only to predict TFIM
ground-state energies. A carefully controlled probe analysis finds that the
trained residual stream makes a non-local finite-size order proxy more linearly
accessible than an untrained network and the tested input representations.
Ten-seed, permutation, partial-correlation, depth, and system-size analyses
support this restricted representation-level observation.

Activation patching provides the complementary result: the probe-defined order
direction can be removed effectively without a commensurate degradation in the
energy prediction. Within this setting, non-local quantum order is represented but
is not evidently load-bearing for the trained energy-regression pathway. This
separation between representation and causal utilisation is the central empirical
lesson of the study.

## Reproducibility Statement

The repository contains the model, exact and sparse ground-state solvers,
observable routines, tests, experiment scripts, result figures, and documented
hyperparameters used in this manuscript. The primary experiment entry points are:

~~~bash
# install the package and development dependencies
pip install -e ".[dev]"

# train with the checkpoint location expected by downstream scripts
python scripts/exp_ra01_train_transformer.py --run_dir runs/ra01_wide
python scripts/ra01_baseline_check.py --ckpt runs/ra01_wide/best.pt

# representation analysis and controls
python scripts/exp_ra02_observables.py --ckpt runs/ra01_wide/best.pt --n_samples 500
python scripts/exp_ra03_controls.py --ckpt runs/ra01_wide/best.pt \
    --n_samples 800 --n_perm 500 --sae_epochs 200
python scripts/exp_ra04_sae_grid.py --ckpt runs/ra01_wide/best.pt
python scripts/exp_ra06_multiseed.py --ckpt runs/ra01_wide/best.pt --seeds 42,43,44
python scripts/exp_ra07_causal.py --ckpt runs/ra01_wide/best.pt

# full degree-two input control (representation-level headline, Tables 2b/4)
python experiments/phase05_input_control.py --config configs/phase05_input_control.yaml
# ten independently trained seeds vs a 64-network random-initialization distribution
python experiments/phase06_multiseed_trained.py --config configs/phase06_multiseed_trained.yaml

# sparse-solver scaling and mixed-field diagnostic
python scripts/exp_ra08_scaling.py --Ls 8,10,12 --n_train 15000 --epochs 100
python scripts/exp_ra08_scaling.py --Ls 8,10 --g 0.5 --n_train 15000 --epochs 100 \
    --run_dir runs/ra09_mixedfield
~~~

The shell workflow in scripts/reproduce_all.sh documents a broader reproduction
path, including an optional Bars-and-Stripes QNN--shadow--SAE validation. The
explicit run-directory flag above aligns a fresh training run with the checkpoint
convention used by the downstream analysis scripts. Random seeds are arguments to
the relevant scripts, and machine-readable results are written as JSON alongside
figures. Unit tests cover the transformer, sparse solver, quantum observables, SAE
behaviour, and classical-shadow estimators.

## Appendix A. Additional Methodological Details

### A.1 Exact and sparse solvers

For \(L=8\), the data generator builds the dense \(ZZ\) term and single-site
\(X_i\) operators once, constructs batches of Hamiltonians, and obtains each
lowest eigenvalue through a symmetric eigensolver. The sparse path builds Pauli
operators as sparse matrices and uses Lanczos (eigsh) for the lowest eigenpair.
It accepts scalar or per-bond \(J\) fields and scalar or per-site longitudinal
fields. Tests compare sparse energies with the dense kernel at \(L=8\), verify
state normalisation, check the expected order trend, and test longitudinal-field
symmetry breaking.

### A.2 Definition of the empirical permutation result

For a fixed trained SAE matrix \(Z\) and observable \(y\), the observed statistic is

\[
T_{\mathrm{obs}}=\max_f |\operatorname{corr}(Z_f,y)|.
\]

For each permutation \(\pi\), the null statistic is

\[
T_\pi=\max_f |\operatorname{corr}(Z_f,\pi(y))|.
\]

The script reports the 95th percentile of \(\{T_\pi\}\) and the observed fraction
of null statistics at least as large as \(T_{\mathrm{obs}}\). With 500
permutations, a displayed value of zero means no sampled null exceeded the
observation; it should be read as a finite-resolution empirical result rather
than an exact probability of zero.

### A.3 Interpretation of the intervention

The direction \(\mathbf d\) is learned from a ridge probe and is therefore a
direction optimised for linear prediction of the selected observable. Projection
removes that component from every sequence position at one layer. The intervention
is well-suited to testing whether this particular linear code is necessary for the
output, but it need not remove all information about order: information may be
represented in correlated directions, encoded nonlinearly, reconstructed by later
computation, or present in earlier layers. The result should therefore be read as
evidence against the necessity of the selected last-layer direction, not as a
complete causal attribution.

## Appendix B. Summary of Existing Figures and Artifacts

The figures used in this draft are committed under the figures directory:

- ra01_wide_pred_vs_truth.png: primary energy-regression diagnostic;
- ra03_probe_r2.png: trained, untrained, and input-baseline probes;
- ra03_layer_sweep.png: depth-dependent probe results;
- ra03_partial_corr.png: raw versus mean-field-partialled feature correlations;
- ra04_sae_universality.png: SAE seed-stability hyperparameter grid;
- ra07_causal.png: directional-ablation energy RMSE;
- ra08_scaling.png: scaling diagnostic;
- ra09_mixedfield.png: mixed-field diagnostic.

The machine-readable output contracts are defined in the associated experiment
scripts: results.json for the control and causal runs, grid_results.json for the
SAE sweep, multiseed_results.json for seed aggregation, and scaling_results.json
for system-size runs.

## References

Alain, G. & Bengio, Y. (2016). Understanding intermediate layers using linear
classifier probes. *arXiv:1610.01644*.

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., et al.
(2023). Towards monosemanticity: Decomposing language models with dictionary
learning. *Transformer Circuits Thread*.

Carleo, G. & Troyer, M. (2017). Solving the quantum many-body problem with
artificial neural networks. *Science*, 355(6325), 602--606.
https://doi.org/10.1126/science.aag2302

Elhage, N., Nanda, N., Olsson, C., Elshowk, S., Henighan, T., Joseph, N., et al.
(2021). A mathematical framework for transformer circuits. *Transformer Circuits
Thread*.

Gao, L., Dupré la Tour, T., Tillman, H., Goh, G., Troll, R., Radford, A.,
Sutskever, I., Leike, J., & Wu, J. (2024). Scaling and evaluating sparse
autoencoders. *arXiv:2406.04093*.

Ha, D. & Schmidhuber, J. (2018). World models. *arXiv:1803.10122*.

Huang, H.-Y., Kueng, R., & Preskill, J. (2020). Predicting many properties of a
quantum system from very few measurements. *Nature Physics*, 16, 1050--1057.
https://doi.org/10.1038/s41567-020-0932-7

Iten, R., Metger, T., Wilming, H., del Rio, L., & Renner, R. (2018). Discovering
physical concepts with neural networks. *arXiv:1807.10300*.

Makhzani, A. & Frey, B. (2013). \(k\)-sparse autoencoders.
*arXiv:1312.5663*.

Nautrup, H. P., Metger, T., Iten, R., Jerbi, S., Trenkwalder, L. M., Wilming, H.,
Briegel, H. J., & Renner, R. (2020). Operationally meaningful representations of
physical systems in neural networks. *arXiv:2001.00593*.

Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020).
Zoom in: An introduction to circuits. *Distill*.

Sachdev, S. (2011). *Quantum Phase Transitions* (2nd ed.). Cambridge University
Press.
