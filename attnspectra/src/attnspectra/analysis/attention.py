"""
Funciones sobre matrices de atención: normalización, entropía y distancia.
"""

from __future__ import annotations

import torch


def normalize_attn(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Normaliza filas de la matriz de atención para que sumen 1.

    Reemplaza NaN / Inf antes de normalizar para no propagar valores inválidos.

    Parameters
    ----------
    A:
        Tensor ``(..., Q, K)``.

    Returns
    -------
    Tensor de la misma shape con filas que suman 1.
    """
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    Z = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / Z


def attention_entropy(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Entropía de Shannon de cada fila (posición Query), promediada sobre Q.

    Parameters
    ----------
    A:
        Tensor ``(B, H, Q, K)`` con probabilidades de atención (post-softmax).

    Returns
    -------
    Tensor ``(B, H)`` con la entropía media por cabeza.
    """
    return -(A * (A + eps).log()).sum(dim=-1).mean(dim=-1)


def per_query_entropy(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Entropía de cada posición Query individualmente (sin promediar).

    Parameters
    ----------
    A:
        Tensor ``(B, H, Q, K)``.

    Returns
    -------
    Tensor ``(B, H, Q)`` con la entropía por posición.
    """
    return -(A * (A + eps).log()).sum(dim=-1)


def attention_distance(
    A: torch.Tensor,
    causal: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Distancia media de atención ponderada.

    Fórmula para una sola secuencia:

        D̄ = Σᵢ Σⱼ α_{i,j} · |i - j| / Σᵢ Σⱼ α_{i,j}

    En modo causal (decoder-only) solo se considera la parte triangular
    inferior (j ≤ i), ya que los tokens futuros están enmascarados.
    En modo bidireccional (encoder) se usa |i - j| sobre toda la matriz.

    Parameters
    ----------
    A:
        Tensor ``(B, H, Q, K)`` con probabilidades de atención post-softmax.
        Q = K = T para atención causal.
    causal:
        Si True (por defecto), solo se considera la parte triangular inferior
        (j ≤ i). Usar False para modelos bidireccionales (BERT, etc.).
    eps:
        Mínimo para el denominador.

    Returns
    -------
    Tensor ``(B, H)`` con la distancia media en número de tokens.
    Rango teórico: [0, T-1].

    """
    B, H, Q, K = A.shape

    # Matriz de distancias |i - j|, shape (Q, K)
    i = torch.arange(Q, dtype=A.dtype, device=A.device).unsqueeze(1)  # (Q, 1)
    j = torch.arange(K, dtype=A.dtype, device=A.device).unsqueeze(0)  # (1, K)
    dist = (i - j).abs()                                               # (Q, K)

    if causal:
        # Máscara triangular inferior: solo j <= i
        mask = (j <= i).float()                                        # (Q, K)
        A_masked = A * mask.unsqueeze(0).unsqueeze(0)                  # (B,H,Q,K)
    else:
        A_masked = A

    # Numerador: Σᵢ Σⱼ α_{i,j} · |i-j|
    numerator   = (A_masked * dist.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B,H)

    # Denominador: Σᵢ Σⱼ α_{i,j}
    denominator = A_masked.sum(dim=(-2, -1)).clamp_min(eps)                       # (B,H)

    return numerator / denominator