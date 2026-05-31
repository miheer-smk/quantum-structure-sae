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
from .observables import (
    half_chain_entanglement_entropy,
    entanglement_spectrum,
    zz_correlator,
    zz_correlator_matrix,
    nearest_neighbor_zz,
    transverse_magnetization,
    longitudinal_magnetization,
    order_parameter,
    phase_proximity,
    compute_all_observables,
)
from .reverse_arrow import TFIMTransformer, TransformerConfig

__all__ = [
    # QNN
    "QNNConfig", "TorchQNN", "build_circuit", "build_state_circuit",
    # SAE
    "SAEConfig", "TopKSAE",
    # Classical shadows
    "ShadowConfig", "compute_pauli_shadow", "estimate_pauli_expectation",
    "build_feature_observables", "shadow_to_feature_vector", "extract_shadow_features",
    # Datasets
    "bars_and_stripes", "mnist_downsampled", "tfim_ground_states", "tfim_phase_labels",
    # Interpretability metrics
    "polysemanticity", "fraction_monosemantic", "match_features",
    "universality_score", "top_activating_examples", "feature_summary",
    # Training
    "train_qnn", "train_sae",
    # Quantum observables
    "half_chain_entanglement_entropy", "entanglement_spectrum",
    "zz_correlator", "zz_correlator_matrix", "nearest_neighbor_zz",
    "transverse_magnetization", "longitudinal_magnetization",
    "order_parameter", "phase_proximity", "compute_all_observables",
    # Transformer (reverse_arrow)
    "TFIMTransformer", "TransformerConfig",
]
