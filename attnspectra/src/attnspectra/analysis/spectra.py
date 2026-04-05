"""
Funciones puras para análisis espectral de matrices de atención.
"""

from __future__ import annotations

import torch


def singular_values(M: torch.Tensor) -> torch.Tensor:
    """
    Calcula los valores singulares de una o varias matrices.

    Parameters
    ----------
    M:
        Tensor de shape ``(..., Q, K)``.

    Returns
    -------
    Tensor ``(..., min(Q,K))`` con los sv ordenados de mayor a menor.
    """
    return torch.linalg.svdvals(M.float())


def effective_rank(sv: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    exp(H(p))

    Versión continua y diferenciable.

    Parameters
    ----------
    sv:
        Tensor ``(..., K)`` de valores singulares.

    Returns
    -------
    Tensor ``(...)`` con el rango efectivo en [1, K].
    """
    sv = sv.clamp_min(eps)
    p = sv / sv.sum(dim=-1, keepdim=True)
    H = -(p * (p + eps).log()).sum(dim=-1)
    return H.exp()

def top_singular_value(sv: torch.Tensor) -> torch.Tensor:
    """
    Valor singular dominante σ₁ (radio espectral).

    Parameters
    ----------
    sv:
        Tensor ``(..., K)`` ordenado de mayor a menor.

    Returns
    -------
    Tensor ``(...)`` con σ₁.
    """
    return sv[..., 0]


def spectral_decay_rate(sv: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Spectral Decay Rate (SDR): pendiente α del ajuste log-lineal
    log(σᵢ) = -α·i + β.

    Parameters
    ----------
    sv:
        Tensor ``(..., K)`` de valores singulares.
    eps:
        Mínimo antes de aplicar logaritmo.

    Returns
    -------
    Tensor ``(...)`` con α ≥ 0 (nats por índice).
    """
    K = sv.shape[-1]
    batch_shape = sv.shape[:-1]

    log_sv = sv.clamp_min(eps).log()                          # (..., K)

    # Índices 0..K-1 y diseño de regresión [i, 1]
    i    = torch.arange(K, dtype=sv.dtype, device=sv.device)
    ones = torch.ones_like(i)
    X    = torch.stack([i, ones], dim=-1)                    # (K, 2)

    flat = log_sv.reshape(-1, K)                             # (N, K)
    N    = flat.shape[0]

    X_exp = X.unsqueeze(0).expand(N, -1, -1)                 # (N, K, 2)
    y_exp = flat.unsqueeze(-1)                               # (N, K, 1)

    result   = torch.linalg.lstsq(X_exp, y_exp)
    alpha    = -result.solution[:, 0, 0]                     # pendiente negada

    return alpha.reshape(batch_shape)

def anisotropy_index(sv: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Anisotropy Index (AI): (σmax − σmin) / σmean.

    Parameters
    ----------
    sv:
        Tensor ``(..., K)`` de valores singulares.
    eps:
        Mínimo para σmean en el denominador.

    Returns
    -------
    Tensor ``(...)`` con AI ≥ 0.
    """
    sigma_max = sv.max(dim=-1).values
    sigma_min = sv.min(dim=-1).values
    sigma_mean = sv.mean(dim=-1).clamp_min(eps)
    return (sigma_max - sigma_min) / sigma_mean

def gini_coefficient(sv: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Coeficiente de Gini sobre los valores singulares.

    Fórmula: G = (2·Σᵢ i·σᵢ) / (n·Σᵢ σᵢ) − (n+1)/n
    con σᵢ ordenados de menor a mayor e i empezando en 1.

    Parameters
    ----------
    sv:
        Tensor ``(..., K)`` de valores singulares (cualquier orden).
    eps:
        Mínimo para la suma total en el denominador.

    Returns
    -------
    Tensor ``(...)`` con G ∈ [0, 1).
    """
    K = sv.shape[-1]

    sv_sorted, _ = sv.sort(dim=-1)                    # ascendente
    sv_sorted    = sv_sorted.clamp_min(0.0)

    i           = torch.arange(1, K + 1, dtype=sv.dtype, device=sv.device)
    numerator   = 2.0 * (i * sv_sorted).sum(dim=-1)
    denominator = (K * sv_sorted.sum(dim=-1)).clamp_min(eps)

    return numerator / denominator - (K + 1.0) / K
