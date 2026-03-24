"""
Funciones para agregar métricas calculadas sobre conjuntos de textos.
"""

from __future__ import annotations

import numpy as np

from attnspectra.core.types import HeadMetrics


def stack_metric(metrics: list[HeadMetrics], key: str) -> np.ndarray:
    """
    Apila un campo de métricas sobre una lista de ``HeadMetrics``.

    Parameters
    ----------
    metrics:
        Lista de N ``HeadMetrics`` (uno por texto).
    key:
        Nombre del campo a extraer (p.ej. ``"A_attn_entropy"``).

    Returns
    -------
    Array de shape ``(N, L, H)``.
    """
    arrays = [getattr(m, key) for m in metrics]
    return np.stack(arrays, axis=0)


def mean_over_texts(stack: np.ndarray) -> np.ndarray:
    """
    Promedia sobre la dimensión de textos (eje 0).

    Parameters
    ----------
    stack:
        Array ``(N, L, H)``.

    Returns
    -------
    Array ``(L, H)`` con la media ignorando NaN.
    """
    return np.nanmean(stack, axis=0)


def top_sensitive_heads(
    metrics_a: list[HeadMetrics] | np.ndarray,
    metrics_b: list[HeadMetrics] | np.ndarray,
    key: str = "A_attn_entropy",
    topk: int = 10,
) -> list[tuple[int, int, float, float, float]]:
    """
    Devuelve las cabezas con mayor diferencia absoluta entre dos condiciones.

    Parameters
    ----------
    metrics_a, metrics_b:
        Listas de ``HeadMetrics`` o arrays ``(N, L, H)`` ya apilados.
    key:
        Campo a comparar.
    topk:
        Número de cabezas a devolver.

    Returns
    -------
    Lista de tuplas ``(layer, head, |Δ|, mean_a, mean_b)`` ordenada de mayor
    a menor diferencia.
    """
    if isinstance(metrics_a, list):
        stack_a = stack_metric(metrics_a, key)
    else:
        stack_a = metrics_a

    if isinstance(metrics_b, list):
        stack_b = stack_metric(metrics_b, key)
    else:
        stack_b = metrics_b

    mean_a = np.nanmean(stack_a, axis=0)   # (L, H)
    mean_b = np.nanmean(stack_b, axis=0)   # (L, H)

    D = np.abs(mean_a - mean_b)            # (L, H)
    L, H = D.shape
    flat = D.reshape(-1)
    idx = np.argsort(flat)[::-1][:topk]

    result = []
    for j in idx:
        li = j // H
        hi = j % H
        result.append((int(li), int(hi), float(D[li, hi]), float(mean_a[li, hi]), float(mean_b[li, hi])))
    return result