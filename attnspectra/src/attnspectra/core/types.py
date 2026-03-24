"""
Tipos centrales que atraviesan todo el pipeline de attnspectra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

AttentionTensor = torch.Tensor
LayerCache      = dict[str, torch.Tensor]
CacheList       = list[LayerCache | None]

@dataclass
class ModelInfo:
    """Metadatos del modelo origen del que se capturó la atención."""
    name:         str
    architecture: str
    n_layers:     int
    n_heads:      int
    d_model:      int
    extra:        dict[str, Any] = field(default_factory=dict)


@dataclass
class CapturedRun:
    """
    Resultado de un forward pass con captura de atención.

    Shapes canónicas:
      - ``attentions[layer]``:  (B, H, Q, K)  — probabilidades post-softmax
      - ``scores[layer]``:      (B, H, Q, K)  — logits Q·K^T pre-softmax
    """
    input_ids:   torch.Tensor
    attentions:  list[torch.Tensor | None]
    scores:      list[torch.Tensor | None] | None
    token_strs:  list[str]
    model_info:  ModelInfo
    style_idx:   int | None = None
    extra:       dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.attentions)

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def get_attention(self, layer: int, head: int) -> torch.Tensor:
        A = self.attentions[layer]
        if A is None:
            raise ValueError(f"La capa {layer} no tiene atención capturada.")
        return A[0, head]

    def get_scores(self, layer: int, head: int) -> torch.Tensor:
        if self.scores is None or self.scores[layer] is None:
            raise ValueError(f"No se capturaron scores para la capa {layer}.")
        return self.scores[layer][0, head]  # type: ignore[index]


@dataclass
class HeadMetrics:
    """
    Métricas espectrales y de atención por capa y cabeza.

    Cada array tiene shape ``(n_layers, n_heads)``.
    ``nan`` indica que esa capa no fue capturada, o que los datos
    necesarios no estaban disponibles (caso de métricas Sc_*).
    """

    # Familia A
    A_attn_entropy:        np.ndarray
    A_effective_rank:      np.ndarray
    A_top_singular:        np.ndarray
    A_spectral_decay_rate: np.ndarray
    A_anisotropy_index:    np.ndarray
    A_gini:                np.ndarray
    A_attn_distance:       np.ndarray

    # Familia S
    S_effective_rank:      np.ndarray
    S_top_singular:        np.ndarray

    # Familia Sc
    Sc_effective_rank:      np.ndarray
    Sc_top_singular:        np.ndarray
    Sc_spectral_decay_rate: np.ndarray
    Sc_anisotropy_index:    np.ndarray
    Sc_gini:                np.ndarray

    # Escalares
    n_layers: int
    n_heads:  int
    seq_len:  int

    def as_dict(self) -> dict[str, np.ndarray]:
        """Devuelve todos los arrays de métricas como diccionario."""
        return {
            # A
            "A_attn_entropy":        self.A_attn_entropy,
            "A_effective_rank":      self.A_effective_rank,
            "A_top_singular":        self.A_top_singular,
            "A_spectral_decay_rate": self.A_spectral_decay_rate,
            "A_anisotropy_index":    self.A_anisotropy_index,
            "A_gini":                self.A_gini,
            "A_attn_distance":       self.A_attn_distance,
            # S
            "S_effective_rank":      self.S_effective_rank,
            "S_top_singular":        self.S_top_singular,
            # Sc
            "Sc_effective_rank":      self.Sc_effective_rank,
            "Sc_top_singular":        self.Sc_top_singular,
            "Sc_spectral_decay_rate": self.Sc_spectral_decay_rate,
            "Sc_anisotropy_index":    self.Sc_anisotropy_index,
            "Sc_gini":                self.Sc_gini,
        }

    def has_scores_metrics(self) -> bool:
        """Devuelve True si los campos Sc_* contienen datos reales (no solo NaN)."""
        return not np.all(np.isnan(self.Sc_effective_rank))