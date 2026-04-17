"""
Parametrized Quantum Circuits for QML.

Provides:
  * QNN: a PennyLane-based variational quantum classifier with
    configurable data encoding and ansatz depth.
  * TorchQNN: a thin PyTorch wrapper for end-to-end training
    with standard optimizers and loss functions.

Design notes
------------
We deliberately expose *layer-wise hooks* so that classical shadows can be
extracted at arbitrary circuit depths. The standard PennyLane pattern
returns a single measurement; we instead expose a `state_at_depth(x, L)`
helper that returns the full state vector after L variational layers (in
simulation). On real hardware you would replace this with shadow sampling
at that layer.

References
----------
- Schuld, Sweke, Meyer (2021) on data re-uploading and Fourier expressivity.
- Cerezo et al. (2021) variational quantum algorithms review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


@dataclass
class QNNConfig:
    n_qubits: int = 8
    n_layers: int = 3
    encoding: Literal["angle", "amplitude", "reupload"] = "reupload"
    entangler: Literal["linear", "circular", "all_to_all"] = "linear"
    measurement: Literal["z_each", "pauli_string"] = "z_each"
    device_name: str = "lightning.qubit"

    def __post_init__(self) -> None:
        if self.n_qubits < 2:
            raise ValueError("need at least 2 qubits")
        if self.n_layers < 1:
            raise ValueError("need at least 1 layer")


def _entangle(wires: list[int], kind: str) -> None:
    """Apply a fixed entangling sub-layer."""
    n = len(wires)
    if kind == "linear":
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
    elif kind == "circular":
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
    elif kind == "all_to_all":
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[wires[i], wires[j]])
    else:
        raise ValueError(f"unknown entangler: {kind}")


def _angle_encode(x: np.ndarray | torch.Tensor, wires: list[int]) -> None:
    """RY rotations encoding x into qubits (len(x) == len(wires) expected)."""
    for i, w in enumerate(wires):
        qml.RY(x[i], wires=w)


def _variational_block(weights: np.ndarray | torch.Tensor, wires: list[int]) -> None:
    """One variational block: RY + RZ per qubit, with shape (n_qubits, 2)."""
    for i, w in enumerate(wires):
        qml.RY(weights[i, 0], wires=w)
        qml.RZ(weights[i, 1], wires=w)


def build_circuit(cfg: QNNConfig) -> Callable:
    """
    Construct a PennyLane QNode implementing the QNN forward pass.

    Returns
    -------
    qnode : Callable(x, weights) -> list[float]
        Returns <Z_i> for each qubit (when measurement == 'z_each').
    """
    dev = qml.device(cfg.device_name, wires=cfg.n_qubits)
    wires = list(range(cfg.n_qubits))

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(x, weights):
        # weights shape: (n_layers, n_qubits, 2)
        for L in range(cfg.n_layers):
            # data re-uploading: inject data at every layer (Perez-Salinas et al.)
            if cfg.encoding in ("angle", "reupload"):
                _angle_encode(x, wires)
            elif cfg.encoding == "amplitude" and L == 0:
                qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)

            _variational_block(weights[L], wires)
            _entangle(wires, cfg.entangler)

        if cfg.measurement == "z_each":
            return [qml.expval(qml.PauliZ(w)) for w in wires]
        elif cfg.measurement == "pauli_string":
            obs = qml.PauliZ(0)
            for w in wires[1:]:
                obs = obs @ qml.PauliZ(w)
            return qml.expval(obs)
        else:
            raise ValueError(f"unknown measurement: {cfg.measurement}")

    return circuit


def build_state_circuit(cfg: QNNConfig):
    """
    A sibling QNode that returns the full pre-measurement state vector.

    Intended ONLY for shadow extraction in simulation — we sample
    Pauli/Clifford measurements on top of this state classically.
    Do not use for differentiation.
    """
    dev = qml.device(cfg.device_name, wires=cfg.n_qubits)
    wires = list(range(cfg.n_qubits))

    @qml.qnode(dev, interface=None)
    def state_circuit(x, weights):
        for L in range(cfg.n_layers):
            if cfg.encoding in ("angle", "reupload"):
                _angle_encode(x, wires)
            elif cfg.encoding == "amplitude" and L == 0:
                qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
            _variational_block(weights[L], wires)
            _entangle(wires, cfg.entangler)
        return qml.state()

    return state_circuit


class TorchQNN(nn.Module):
    """
    Wraps a PennyLane QNode as a torch.nn.Module with a small classical
    head for multi-class classification.

    Weights shape: (n_layers, n_qubits, 2).
    """

    def __init__(self, cfg: QNNConfig, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.circuit = build_circuit(cfg)
        self.q_weights = nn.Parameter(
            0.1 * torch.randn(cfg.n_layers, cfg.n_qubits, 2)
        )
        # small classical head mapping n_qubits expectations -> n_classes
        self.head = nn.Linear(cfg.n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_qubits) real-valued features (already normalized).
        returns logits : (batch, n_classes).
        """
        # PennyLane with torch interface requires per-sample calls; vmap via loop.
        outs = []
        for xi in x:
            zs = self.circuit(xi, self.q_weights)
            outs.append(torch.stack([z.to(torch.float32) for z in zs]))
        feats = torch.stack(outs)  # (batch, n_qubits)
        return self.head(feats), feats

    @torch.no_grad()
    def latent_states(self, x: torch.Tensor) -> torch.Tensor:
        """Return full pre-measurement state vectors, shape (batch, 2**n)."""
        sc = build_state_circuit(self.cfg)
        weights_np = self.q_weights.detach().cpu().numpy()
        xs = x.detach().cpu().numpy()
        states = np.stack([np.asarray(sc(xi, weights_np)) for xi in xs])
        return torch.from_numpy(states)
