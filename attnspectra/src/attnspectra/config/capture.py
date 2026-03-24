"""
Dataclasses de configuración para el proceso de captura.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    """
    Configuración de qué capturar durante el forward pass.

    Attributes
    ----------
    target_layers:
        Lista de índices de capa a capturar. ``None`` = todas las capas.
        Acepta índices negativos (``-1`` = última capa).
    capture_attn_pre:
        Capturar atención post-softmax (antes de dropout). Recomendado.
    capture_attn_post:
        Capturar atención post-dropout.
    capture_scores:
        Capturar logits pre-softmax (scores).
    which_attentions:
        Para modelos encoder-decoder: ``"encoder"``, ``"decoder"`` o ``"cross"``.
    max_seq_len:
        Si se especifica, trunca la secuencia a este número de tokens.
    """

    target_layers: list[int] | None = None
    capture_attn_pre: bool = True
    capture_attn_post: bool = False
    capture_scores: bool = False
    which_attentions: str = "encoder"     # relevante solo para encoder-decoder
    max_seq_len: int | None = None


@dataclass
class TargetSpec:
    """
    Especificación de una cabeza concreta para análisis focalizado.
    """
    layer: int
    head: int
    label: str = ""