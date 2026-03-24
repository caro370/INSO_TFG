"""
Helpers para extraer tensores concretos de un ``CapturedRun`` de forma
cómoda, sin tener que indexar manualmente los arrays.

Todas las funciones son puras (no modifican el CapturedRun) y devuelven
tensores o arrays numpy listos para análisis o visualización.
"""

from __future__ import annotations

import numpy as np
import torch

from attnspectra.core.types import CapturedRun, HeadMetrics


# Selectores sobre CapturedRun
def get_layer_head(
    run: CapturedRun,
    layer: int,
    head: int,
    batch: int = 0,
) -> torch.Tensor:
    """
    Extrae la matriz de atención ``(Q, K)`` de una capa y cabeza concretas.

    Equivalente a ``run.get_attention(layer, head)`` pero con validación
    explícita de índices y soporte para batch distinto de 0.

    Parameters
    ----------
    run:
        CapturedRun del que extraer la atención.
    layer:
        Índice de capa. Acepta negativos (``-1`` = última capa).
    head:
        Índice de cabeza. Acepta negativos.
    batch:
        Índice de secuencia dentro del batch (por defecto 0).

    Returns
    -------
    Tensor ``(Q, K)`` con las probabilidades de atención.

    Raises
    ------
    IndexError:
        Si la capa o cabeza están fuera de rango.
    ValueError:
        Si la capa solicitada no fue capturada (es None).
    """
    n_layers = run.n_layers
    layer = layer % n_layers   # soporta índices negativos

    A_full = run.attentions[layer]
    if A_full is None:
        raise ValueError(
            f"La capa {layer} no fue capturada. "
            f"Comprueba CaptureConfig.target_layers."
        )

    n_heads = A_full.shape[1]
    head = head % n_heads

    return A_full[batch, head]   # (Q, K)


def pick_tokens(
    run: CapturedRun,
    indices: list[int] | slice,
) -> list[str]:
    """
    Selecciona un subconjunto de token_strs por índice o slice.

    Parameters
    ----------
    run:
        CapturedRun del que extraer los tokens.
    indices:
        Lista de índices enteros o un slice.

    Returns
    -------
    Lista de strings con los tokens seleccionados.
    """
    if isinstance(indices, slice):
        return run.token_strs[indices]
    return [run.token_strs[i] for i in indices]


def get_content_attention(
    run: CapturedRun,
    layer: int,
    head: int,
    prefix_len: int = 3,
    batch: int = 0,
) -> tuple[torch.Tensor, list[str]]:
    """
    Extrae la submatriz de atención correspondiente solo a los tokens de
    contenido, eliminando los tokens de prefijo (``<bos>``, style, ``<sep>``).

    Parameters
    ----------
    run:
        CapturedRun del que extraer.
    layer, head:
        Capa y cabeza a extraer.
    prefix_len:
        Número de tokens de prefijo a descartar (por defecto 3:
        ``<bos>``, ``<wiki|poem>``, ``<sep>``).
    batch:
        Índice de batch.

    Returns
    -------
    Tupla ``(A_content, token_strs_content)`` donde:
      - ``A_content``: Tensor ``(Q', K')`` sin las filas/columnas del prefijo
      - ``token_strs_content``: lista de strings de los tokens de contenido
    """
    A = get_layer_head(run, layer, head, batch)    # (Q, K)
    A_content = A[prefix_len:, prefix_len:]        # (Q', K')
    tokens = run.token_strs[prefix_len:]
    return A_content, tokens


# Selectores sobre HeadMetrics

def get_metric_layer(
    metrics: HeadMetrics,
    key: str,
    layer: int,
) -> np.ndarray:
    """
    Extrae los valores de una métrica para todas las cabezas de una capa.

    Parameters
    ----------
    metrics:
        HeadMetrics del que extraer.
    key:
        Nombre del campo (p.ej. ``"A_attn_entropy"``).
    layer:
        Índice de capa. Acepta negativos.

    Returns
    -------
    Array ``(n_heads,)`` con los valores de la métrica.
    """
    arr = getattr(metrics, key)
    layer = layer % metrics.n_layers
    return arr[layer]


def get_metric_head(
    metrics: HeadMetrics,
    key: str,
    head: int,
) -> np.ndarray:
    """
    Extrae los valores de una métrica para todas las capas de una cabeza.

    Parameters
    ----------
    metrics:
        HeadMetrics del que extraer.
    key:
        Nombre del campo (p.ej. ``"A_effective_rank"``).
    head:
        Índice de cabeza. Acepta negativos.

    Returns
    -------
    Array ``(n_layers,)`` con la evolución de la métrica a lo largo de capas.
    """
    arr = getattr(metrics, key)
    head = head % metrics.n_heads
    return arr[:, head]


def most_attended_tokens(
    run: CapturedRun,
    layer: int,
    head: int,
    topk: int = 5,
    batch: int = 0,
) -> list[tuple[str, float]]:
    """
    Devuelve los tokens más atendidos globalmente (promedio sobre queries).

    Útil para identificar qué posiciones actúan como "anclas" de atención.

    Parameters
    ----------
    run:
        CapturedRun del que extraer.
    layer, head:
        Capa y cabeza a analizar.
    topk:
        Número de tokens a devolver.
    batch:
        Índice de batch.

    Returns
    -------
    Lista de tuplas ``(token_str, attn_score_medio)`` ordenada de mayor a menor.
    """
    A = get_layer_head(run, layer, head, batch)   # (Q, K)
    mean_attn = A.mean(dim=0).numpy()             # (K,) — media sobre queries

    # topk mayores
    indices = np.argsort(mean_attn)[::-1][:topk]
    return [(run.token_strs[int(i)], float(mean_attn[i])) for i in indices]