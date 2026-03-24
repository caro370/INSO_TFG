"""
Orquestador de alto nivel que combina adapter + config + métricas en
llamadas simples.

En lugar de encadenar manualmente adapter → capture → compute_head_metrics,
estas funciones hacen todo el pipeline en una sola llamada, lo que simplifica
el código de los notebooks y scripts de experimentos.

Los parámetros de ``MetricConfig`` (``eps``, ``er_threshold``, ``causal``)
se propagan automáticamente a ``compute_head_metrics`` en todas las funciones.

Funciones principales
---------------------
capture_and_compute   — captura atención y calcula las 21 métricas en una llamada
capture_batch         — procesa lista de textos/ids y devuelve listas
run_degradation_exp   — pipeline completo del EXP1 (degradación de señal)
run_style_comparison  — pipeline completo del EXP3 (comparación de estilos)
"""

from __future__ import annotations

from typing import Sequence

import torch

from attnspectra.adapters.base import BaseAdapter
from attnspectra.analysis.metrics import compute_head_metrics
from attnspectra.config.capture import CaptureConfig
from attnspectra.config.metrics import MetricConfig
from attnspectra.core.types import CapturedRun, HeadMetrics
from attnspectra.transforms.token_perturbations import make_variants


# Pipeline principal

def capture_and_compute(
    adapter: BaseAdapter,
    input_ids: torch.Tensor,
    capture_cfg: CaptureConfig | None = None,
    metric_cfg:  MetricConfig  | None = None,
    style_idx: int | None = None,
) -> tuple[CapturedRun, HeadMetrics]:
    """
    Captura la atención y calcula las métricas en una sola llamada.

    Equivale a::

        run     = adapter.capture_from_ids(input_ids, capture_cfg, style_idx=style_idx)
        metrics = compute_head_metrics(
                      run,
                      eps=metric_cfg.eps,
                      er_threshold=metric_cfg.er_threshold,
                      causal=metric_cfg.causal,
                  )

    Parameters
    ----------
    adapter:
        Adapter del modelo (CustomGPTAdapter, HFTransformerAdapter, ...).
    input_ids:
        Tensor ``(B, T)`` con los ids de entrada.
    capture_cfg:
        Configuración de captura. Si es None usa los valores por defecto
        (``capture_attn_pre=True``, todas las capas).
    metric_cfg:
        Configuración de métricas. Si es None calcula todas las métricas.
    style_idx:
        Índice de estilo para modelos con style embedding.

    Returns
    -------
    Tupla ``(CapturedRun, HeadMetrics)``.

    Examples
    --------
    >>> run, metrics = capture_and_compute(adapter, input_ids, style_idx=0)
    >>> print(metrics.A_attn_entropy.shape)   # (n_layers, n_heads)
    """
    if capture_cfg is None:
        capture_cfg = CaptureConfig()
    if metric_cfg is None:
        metric_cfg = MetricConfig()

    kwargs = {} if style_idx is None else {"style_idx": style_idx}
    run = adapter.capture_from_ids(input_ids, capture_cfg, **kwargs)
    metrics = compute_head_metrics(
        run,
        eps=metric_cfg.eps,
        er_threshold=metric_cfg.er_threshold,
        causal=metric_cfg.causal,
    )

    return run, metrics


def capture_batch(
    adapter: BaseAdapter,
    ids_list: Sequence[torch.Tensor],
    capture_cfg: CaptureConfig | None = None,
    metric_cfg:  MetricConfig  | None = None,
    style_idx: int | None = None,
    verbose: bool = False,
) -> tuple[list[CapturedRun], list[HeadMetrics]]:
    """
    Procesa una lista de secuencias de ids y devuelve listas de resultados.

    Cada tensor en ``ids_list`` se procesa independientemente (batch_size=1
    por llamada), lo que permite secuencias de longitud variable.

    Parameters
    ----------
    adapter:
        Adapter del modelo.
    ids_list:
        Lista de tensores ``(1, T_i)`` o ``(T_i,)`` (se añade dimensión batch
        automáticamente si falta).
    capture_cfg:
        Configuración de captura compartida para todos los textos.
    metric_cfg:
        Configuración de métricas compartida.
    style_idx:
        Índice de estilo aplicado a todos los textos. Para estilos distintos
        por texto usa ``capture_and_compute`` directamente.
    verbose:
        Si True, imprime el progreso (índice actual / total).

    Returns
    -------
    Tupla ``(runs, metrics_list)`` con una entrada por texto.

    Examples
    --------
    >>> ids_list = [torch.tensor([prefix + enc]) for enc in encodings]
    >>> runs, metrics_list = capture_batch(adapter, ids_list, style_idx=0)
    >>> # Apilar métricas para comparar (21 campos disponibles)
    >>> stack = aspec.stack_metric(metrics_list, "A_attn_entropy")  # (N, L, H)
    >>> stack_dist = aspec.stack_metric(metrics_list, "A_attn_distance")
    """
    if capture_cfg is None:
        capture_cfg = CaptureConfig()
    if metric_cfg is None:
        metric_cfg = MetricConfig()

    runs: list[CapturedRun]    = []
    metrics_list: list[HeadMetrics] = []

    for i, ids in enumerate(ids_list):
        if verbose:
            print(f"  [{i+1}/{len(ids_list)}] seq_len={ids.shape[-1]}", end="\r")

        # Asegurar dimensión batch
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        run, metrics = capture_and_compute(
            adapter, ids,
            capture_cfg=capture_cfg,
            metric_cfg=metric_cfg,
            style_idx=style_idx,
        )
        runs.append(run)
        metrics_list.append(metrics)

    if verbose:
        print(f"  Completado: {len(ids_list)} textos procesados.  ")

    return runs, metrics_list


# Pipelines de experimento

def run_degradation_exp(
    adapter: BaseAdapter,
    base_ids: list[int],
    vocab_size: int,
    prefix_len: int = 3,
    seed: int = 42,
    capture_cfg: CaptureConfig | None = None,
    metric_cfg:  MetricConfig  | None = None,
    style_idx: int | None = None,
    verbose: bool = True,
) -> dict[str, list[HeadMetrics]]:
    """
    Pipeline completo del EXP1: genera variantes perturbadas y captura
    las métricas de cada una.

    Genera las 6 variantes estándar (clean, shuffle, replace_10%,
    replace_30%, replace_50%, random_iid) y devuelve un dict listo
    para pasar a ``plot_delta_lines_by_condition``.

    Parameters
    ----------
    adapter:
        Adapter del modelo.
    base_ids:
        Lista de ids de la secuencia limpia completa (incluido el prefijo).
    vocab_size:
        Tamaño del vocabulario (para generar ids aleatorios válidos).
    prefix_len:
        Número de tokens de prefijo que no se perturban.
    seed:
        Semilla para reproducibilidad de las perturbaciones.
    capture_cfg:
        Configuración de captura.
    metric_cfg:
        Configuración de métricas.
    style_idx:
        Índice de estilo.
    verbose:
        Si True, imprime el nombre de cada variante al procesarla.

    Returns
    -------
    Dict ``{nombre_variante: [HeadMetrics]}`` listo para
    ``plot_delta_lines_by_condition``.

    Examples
    --------
    >>> per_cond = run_degradation_exp(
    ...     adapter, base_ids,
    ...     vocab_size=cfg["vocab_size"],
    ...     prefix_len=3,
    ...     style_idx=0,
    ... )
    >>> fig = aspec.plot_delta_lines_by_condition(
    ...     per_cond, metric_key="A_attn_entropy"
    ... )
    """
    if capture_cfg is None:
        capture_cfg = CaptureConfig()
    if metric_cfg is None:
        metric_cfg = MetricConfig()

    variants = make_variants(base_ids, vocab_size=vocab_size,
                             prefix_len=prefix_len, seed=seed)

    per_cond: dict[str, list[HeadMetrics]] = {}

    for name, variant_ids in variants.items():
        if verbose:
            print(f"  variante '{name}'...")

        ids_tensor = torch.tensor([variant_ids], dtype=torch.long)
        _, metrics = capture_and_compute(
            adapter, ids_tensor,
            capture_cfg=capture_cfg,
            metric_cfg=metric_cfg,
            style_idx=style_idx,
        )
        per_cond[name] = [metrics]

    if verbose:
        print(f"  EXP1 completado: {len(per_cond)} variantes.")

    return per_cond


def run_style_comparison(
    adapter: BaseAdapter,
    ids_list: Sequence[torch.Tensor],
    style_a: int,
    style_b: int,
    style_a_name: str = "wiki",
    style_b_name: str = "poem",
    capture_cfg: CaptureConfig | None = None,
    metric_cfg:  MetricConfig  | None = None,
    verbose: bool = True,
) -> tuple[list[HeadMetrics], list[HeadMetrics]]:
    """
    Pipeline del EXP3: procesa los mismos textos con dos estilos distintos.

    Parameters
    ----------
    adapter:
        Adapter del modelo.
    ids_list:
        Lista de tensores de ids (contenido sin style token — el adapter
        debe insertar el style token según ``style_idx``).
    style_a, style_b:
        Índices de los dos estilos a comparar.
    style_a_name, style_b_name:
        Nombres para los mensajes de progreso.
    capture_cfg:
        Configuración de captura.
    metric_cfg:
        Configuración de métricas.
    verbose:
        Si True, imprime el progreso.

    Returns
    -------
    Tupla ``(metrics_a, metrics_b)`` — listas de HeadMetrics para cada estilo,
    listas para pasar a ``top_sensitive_heads`` o ``heatmap_delta``.

    Examples
    --------
    >>> metrics_wiki, metrics_poem = run_style_comparison(
    ...     adapter, ids_list, style_a=0, style_b=1
    ... )
    >>> heads = aspec.top_sensitive_heads(metrics_wiki, metrics_poem, topk=10)
    """
    if verbose:
        print(f"  Procesando estilo '{style_a_name}' ({len(ids_list)} textos)...")
    _, metrics_a = capture_batch(
        adapter, ids_list,
        capture_cfg=capture_cfg,
        metric_cfg=metric_cfg,
        style_idx=style_a,
        verbose=False,
    )

    if verbose:
        print(f"  Procesando estilo '{style_b_name}' ({len(ids_list)} textos)...")
    _, metrics_b = capture_batch(
        adapter, ids_list,
        capture_cfg=capture_cfg,
        metric_cfg=metric_cfg,
        style_idx=style_b,
        verbose=False,
    )

    if verbose:
        print(f"  EXP3 completado.")

    return metrics_a, metrics_b