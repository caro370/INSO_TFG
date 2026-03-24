"""
Visualización de matrices de atención individuales.

Funcines:
  plot_attention_matrix             — heatmap de una cabeza
  plot_attention_matrix_interactive — heatmap con sliders de capa y cabeza
"""

from __future__ import annotations

from typing import Sequence

from attnspectra.core.types import CapturedRun
import matplotlib.pyplot as plt
import numpy as np
import plotly
import torch

from attnspectra.viz.tokens import format_token_labels

def plot_attention_matrix_interactive(
    run: "CapturedRun",
    max_tokens: int = 50,
    title: str = "Attention Matrix",
    colorscale: str = "Viridis",
    width: int = 750,
    height: int = 650,
) -> "plotly.graph_objects.Figure":
    try:
        import plotly.graph_objects as go
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        raise ImportError("Requiere plotly e ipywidgets")

    available_layers = [li for li, A in enumerate(run.attentions) if A is not None]
    if not available_layers:
        raise ValueError("El CapturedRun no tiene ninguna capa de atención capturada.")

    n_heads = run.attentions[available_layers[0]].shape[1]
    T = min(run.seq_len, max_tokens)
    tokens = format_token_labels(list(run.token_strs[:T]))

    def get_matrix(layer_idx: int, head_idx: int) -> np.ndarray:
        return run.attentions[layer_idx][0, head_idx, :T, :T].float().cpu().numpy()

    # Una sola traza con la matriz inicial
    mat = get_matrix(available_layers[0], 0)
    fig = go.FigureWidget(go.Heatmap(
        z=mat,
        x=tokens,
        y=tokens,
        colorscale=colorscale,
        zmin=0,
        zmax=1.0,
        colorbar=dict(title="Atención", thickness=14),
        hovertemplate="Q: %{y}  →  K: %{x}<br>Atención: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{title} — Capa {available_layers[0]} · Cabeza 0", font=dict(size=13)),
        xaxis=dict(title="Keys (K)", tickangle=-60, tickfont=dict(size=9)),
        # scaleanchor garantiza matriz cuadrada
        yaxis=dict(
            title="Queries (Q)",
            tickfont=dict(size=9),
            autorange="reversed",
            scaleanchor="x",
            scaleratio=1,
        ),
        width=width,
        height=height,
        margin=dict(l=80, r=80, t=60, b=60),
        plot_bgcolor="white",
    )

    # Sliders independientes con ipywidgets — sin solapamiento posible
    layer_slider = widgets.IntSlider(
        value=available_layers[0],
        min=available_layers[0],
        max=available_layers[-1],
        step=1,
        description="Capa:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="400px"),
    )
    head_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=n_heads - 1,
        step=1,
        description="Cabeza:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="400px"),
    )

    def on_change(change):
        li = layer_slider.value
        h = head_slider.value
        new_mat = get_matrix(li, h)
        with fig.batch_update():
            fig.data[0].z = new_mat
            fig.data[0].zmax = float(new_mat.max()) if new_mat.max() > 0 else 1.0
            fig.layout.title.text = f"{title} — Capa {li} · Cabeza {h}"

    layer_slider.observe(on_change, names="value")
    head_slider.observe(on_change, names="value")

    display(widgets.VBox([layer_slider, head_slider, fig]))
    return None

def plot_attention_matrix(
    A: torch.Tensor | np.ndarray,
    token_strs: Sequence[str],
    max_tokens: int = 60,
    title: str = "Attention matrix",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (10, 8),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Dibuja la matriz de atención (Q, K) de una capa y cabeza concretas.

    Parameters
    ----------
    A:
        Tensor o array ``(Q, K)`` con probabilidades de atención.
    token_strs:
        Lista de strings con los tokens (usada para etiquetar ejes).
    max_tokens:
        Máximo número de tokens a mostrar (para legibilidad).
    title:
        Título del gráfico.
    cmap:
        Colormap de matplotlib.
    figsize:
        Tamaño de la figura en pulgadas.
    ax:
        Si se pasa, dibuja en ese ``Axes`` en lugar de crear una figura nueva.

    Returns
    -------
    ``Figure`` de matplotlib.
    """
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()

    T = min(A.shape[0], max_tokens)
    A = A[:T, :T]
    labels = format_token_labels(list(token_strs[:T]))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(A, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(T))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(T))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Keys (K)")
    ax.set_ylabel("Queries (Q)")
    ax.set_title(title)
    plt.tight_layout()
    return fig