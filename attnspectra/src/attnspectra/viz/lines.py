"""
Gráficos de líneas para experimentos de degradación y comparación entre condiciones.

Funciones:
  plot_delta_lines_by_condition  — Δ(condición − base) vs capa
  plot_metric_by_layer           — media ± std de una métrica vs capa
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from attnspectra.core.types import HeadMetrics
from attnspectra.analysis.aggregation import stack_metric

def plot_delta_lines_by_condition(
    per_condition: dict[str, list[HeadMetrics]],
    metric_key: str = "A_attn_entropy",
    base: str = "clean",
    title: str = "Δ metric vs layer",
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 4),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Traza ``Δ(condición − base)`` promediado sobre textos y cabezas, vs capa.

    Parameters
    ----------
    per_condition:
        Dict ``{condición: [HeadMetrics, ...]}``
    metric_key:
        Campo de ``HeadMetrics`` a comparar.
    base:
        Condición de referencia (p.ej. ``"clean"``).
    """
    if base not in per_condition:
        raise KeyError(f"La condición base '{base}' no está en per_condition.")

    base_stack = stack_metric(per_condition[base], metric_key)   # (N,L,H)
    H_base = np.nanmean(base_stack, axis=(0, 2))                 # (L,)
    L = H_base.shape[0]
    layers = np.arange(L)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for cond, metrics in per_condition.items():
        if cond == base:
            continue
        stack = stack_metric(metrics, metric_key)
        H_cond = np.nanmean(stack, axis=(0, 2))
        delta = H_cond - H_base
        ax.plot(layers, delta, marker="o", label=cond)

    ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel or f"Δ {metric_key}")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_metric_by_layer(
    metrics_list: list[HeadMetrics],
    metric_key: str = "A_attn_entropy",
    label: str = "",
    color: str | None = None,
    title: str = "Metric vs layer",
    figsize: tuple[float, float] = (8, 4),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Traza la media ± std de una métrica vs capa sobre una lista de HeadMetrics.
    """
    stack = stack_metric(metrics_list, metric_key)   # (N,L,H)
    mu = np.nanmean(stack, axis=(0, 2))              # (L,)
    sd = np.nanstd(stack, axis=(0, 2))               # (L,)
    layers = np.arange(len(mu))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    kw = {} if color is None else {"color": color}
    ax.plot(layers, mu, marker="o", label=label, **kw)
    ax.fill_between(layers, mu - sd, mu + sd, alpha=0.2, **kw)
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    if label:
        ax.legend()
    plt.tight_layout()
    return fig