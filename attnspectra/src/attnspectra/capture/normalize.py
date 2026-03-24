"""
Convierte los outputs de atención de distintos formatos al shape canónico:

    List[Tensor(B, H, Q, K) | None]   — una entrada por capa

Esta capa de normalización es lo que permite que ``analysis/`` sea
completamente agnóstico del modelo origen.
"""

from __future__ import annotations

import torch

from attnspectra.config.capture import CaptureConfig


def normalize_hf_attentions(
    raw: tuple[torch.Tensor, ...],
    config: CaptureConfig,
) -> list[torch.Tensor | None]:
    """
    Normaliza la tupla de atenciones que devuelven los modelos HuggingFace.

    Los modelos HF devuelven una tupla donde cada elemento tiene shape
    ``(B, H, Q, K)`` — ya en el formato correcto. Aquí aplicamos el filtro
    de capas especificado en ``config.target_layers``.

    Parameters
    ----------
    raw:
        Tupla de tensores ``(B, H, Q, K)``, uno por capa.
    config:
        CaptureConfig; si ``target_layers`` es None se devuelven todas.

    Returns
    -------
    Lista de tensores ``(B, H, Q, K)`` o ``None`` (capas no capturadas).
    """
    n_layers = len(raw)
    target = _resolve_layers(config, n_layers)

    result: list[torch.Tensor | None] = []
    for i in range(n_layers):
        if target is None or i in target:
            result.append(raw[i].detach().cpu())
        else:
            result.append(None)
    return result


def normalize_cache_list(
    caches: list[dict[str, torch.Tensor] | None],
    which: str = "attn_pre",
    config: CaptureConfig | None = None,
) -> list[torch.Tensor | None]:
    """
    Normaliza la lista de caches del modelo GPT custom al formato canónico.

    Parameters
    ----------
    caches:
        Lista de dicts de cache, una por capa. Cada dict puede tener claves
        ``attn_pre``, ``attn_post``, ``scores`` con shape ``(B, H, T, T)``.
    which:
        Qué tensor extraer de cada cache (``"attn_pre"`` por defecto).
    config:
        Si se pasa, aplica el filtro ``target_layers``.

    Returns
    -------
    Lista de tensores ``(B, H, T, T)`` o ``None``.
    """
    n_layers = len(caches)
    target = _resolve_layers(config, n_layers) if config is not None else None

    result: list[torch.Tensor | None] = []
    for i, cache in enumerate(caches):
        if target is not None and i not in target:
            result.append(None)
            continue
        if cache is None or which not in cache:
            result.append(None)
        else:
            result.append(cache[which].detach().cpu())
    return result


def _resolve_layers(config: CaptureConfig, n_layers: int) -> set[int] | None:
    """
    Devuelve el conjunto de índices de capa a capturar, o None si son todas.
    Acepta índices negativos al estilo Python.
    """
    target = getattr(config, "target_layers", None)
    if target is None:
        return None
    resolved = set()
    for idx in target:
        resolved.add(idx % n_layers)
    return resolved