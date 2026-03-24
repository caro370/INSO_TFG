"""
Heatmaps de métricas por capa × cabeza.

Funciones:
  heatmap_delta        — diferencia A-B con colormap divergente
  heatmap_metric       — métrica absoluta con colormap secuencial

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def heatmap_delta(
    A: np.ndarray,
    B: np.ndarray,
    title: str = "Δ (A − B)",
    label: str = "Δ",
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (10, 4),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Heatmap de la diferencia ``A - B`` con shape ``(L, H)``.

    Parameters
    ----------
    A, B:
        Arrays ``(n_layers, n_heads)``.
    title:
        Título del gráfico.
    label:
        Etiqueta de la colorbar.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shapes distintos: {A.shape} vs {B.shape}.")

    D = A - B
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmax = np.nanmax(np.abs(D))
    im = ax.imshow(D, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(D.shape[0]))
    ax.set_title(title)
    plt.tight_layout()
    return fig


def heatmap_metric(
    M: np.ndarray,
    title: str = "Metric",
    label: str = "",
    cmap: str = "plasma",
    figsize: tuple[float, float] = (10, 4),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Heatmap de una métrica por capa × cabeza (sin diferencia).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(M, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(M.shape[0]))
    ax.set_title(title)
    plt.tight_layout()
    return fig