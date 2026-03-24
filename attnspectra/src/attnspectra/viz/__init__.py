"""
Visualización de matrices de atención y métricas espectrales.

Todas las funciones devuelven un ``matplotlib.figure.Figure`` y aceptan
un ``ax`` opcional para incrustarlas en figuras compuestas.

Exports
----------------------------
plot_attention_matrix             — heatmap (Q, K) para una capa y cabeza
plot_attention_matrix_interactive —  heatmap con sliders de capa y cabeza
heatmap_delta                     — Δ(A − B) con colormap divergente
heatmap_metric                    — métrica absoluta por capa × cabeza
plot_delta_lines_by_condition     — Δ vs capa para cada condición de degradación
plot_metric_by_layer              — media ± std de una métrica vs capa
format_token_labels               — escape y truncado de etiquetas de tokens
"""

from attnspectra.viz.attention_maps import plot_attention_matrix, plot_attention_matrix_interactive
from attnspectra.viz.heatmaps import heatmap_delta, heatmap_metric
from attnspectra.viz.lines import plot_delta_lines_by_condition, plot_metric_by_layer
from attnspectra.viz.tokens import format_token_labels


__all__ = [
    "plot_attention_matrix",
    "plot_attention_matrix_interactive",
    "heatmap_delta",
    "heatmap_metric",
    "plot_delta_lines_by_condition",
    "plot_metric_by_layer",
    "format_token_labels",
]