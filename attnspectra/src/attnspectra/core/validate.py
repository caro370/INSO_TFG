"""
Funciones de validación de shapes, NaNs e invariantes del pipeline.
Todas lanzan excepciones descriptivas para facilitar el debugging.
"""

from __future__ import annotations

import torch
import numpy as np


def check_attention_tensor(A: torch.Tensor, name: str = "A") -> None:
    """
    Valida que un tensor de atención tenga shape (B, H, Q, K) y
    no contenga NaN ni Inf.
    """
    if A.dim() != 4:
        raise ValueError(
            f"[validate] {name} debe tener 4 dims (B, H, Q, K), "
            f"pero tiene shape {tuple(A.shape)}."
        )
    if torch.isnan(A).any():
        raise ValueError(f"[validate] {name} contiene NaN.")
    if torch.isinf(A).any():
        raise ValueError(f"[validate] {name} contiene Inf.")


def check_captured_run(run: object) -> None:
    """Valida la consistencia interna de un CapturedRun."""
    from attnspectra.core.types import CapturedRun

    if not isinstance(run, CapturedRun):
        raise TypeError(f"Se esperaba CapturedRun, se recibió {type(run)}.")

    B, T = run.input_ids.shape
    assert len(run.token_strs) == T, (
        f"token_strs tiene {len(run.token_strs)} elementos pero seq_len={T}."
    )
    for i, A in enumerate(run.attentions):
        if A is None:
            continue
        if A.shape[0] != B:
            raise ValueError(
                f"attentions[{i}] tiene batch={A.shape[0]} pero input_ids tiene batch={B}."
            )
        _, _, Q, K = A.shape
        if Q != T or K != T:
            raise ValueError(
                f"attentions[{i}] tiene shape (B,H,{Q},{K}) "
                f"pero seq_len={T}. Se esperaba (B,H,{T},{T})."
            )

    # Validar scores si están presentes
    if run.scores is not None:
        if len(run.scores) != len(run.attentions):
            raise ValueError(
                f"scores tiene {len(run.scores)} capas pero attentions tiene "
                f"{len(run.attentions)}. Deben tener la misma longitud."
            )
        for i, S in enumerate(run.scores):
            if S is None:
                continue
            if S.shape[0] != B:
                raise ValueError(
                    f"scores[{i}] tiene batch={S.shape[0]} pero input_ids tiene batch={B}."
                )
            _, _, Q, K = S.shape
            if Q != T or K != T:
                raise ValueError(
                    f"scores[{i}] tiene shape (B,H,{Q},{K}) "
                    f"pero seq_len={T}. Se esperaba (B,H,{T},{T})."
                )


def check_head_metrics_array(arr: np.ndarray, n_layers: int, n_heads: int, name: str = "") -> None:
    """Valida que un array de métricas tenga shape (n_layers, n_heads)."""
    if arr.shape != (n_layers, n_heads):
        raise ValueError(
            f"[validate] {name} tiene shape {arr.shape}, "
            f"se esperaba ({n_layers}, {n_heads})."
        )