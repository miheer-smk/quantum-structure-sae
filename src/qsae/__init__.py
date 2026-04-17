"""qsae — quantum sparse autoencoders for mechanistic interpretability of QNNs."""

from .qnn import QNNConfig, TorchQNN, build_circuit, build_state_circuit
from .sae import SAEConfig, TopKSAE
from .shadows import (
    ShadowConfig,
    build_feature_observables,
    compute_pauli_shadow,
    estimate_pauli_expectation,
    extract_shadow_features,
    shadow_to_feature_vector,
)
from .datasets import bars_and_stripes, mnist_downsampled, tfim_ground_states, tfim_phase_labels
from .metrics import (
    feature_summary,
    fraction_monosemantic,
    match_features,
    polysemanticity,
    top_activating_examples,
    universality_score,
)
from .training import train_qnn, train_sae

__all__ = [
    "QNNConfig", "TorchQNN", "build_circuit", "build_state_circuit",
    "SAEConfig", "TopKSAE",
    "ShadowConfig", "compute_pauli_shadow", "estimate_pauli_expectation",
    "build_feature_observables", "shadow_to_feature_vector", "extract_shadow_features",
    "bars_and_stripes", "mnist_downsampled", "tfim_ground_states", "tfim_phase_labels",
    "polysemanticity", "fraction_monosemantic", "match_features",
    "universality_score", "top_activating_examples", "feature_summary",
    "train_qnn", "train_sae",
]
