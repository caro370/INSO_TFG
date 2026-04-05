"""
Dataclass de configuración para las funciones de visualización de ``viz/``.

Centraliza todos los parámetros estéticos en un único objeto que se puede
pasar a cualquier función de plotting, facilitando mantener un estilo
consistente en todos los gráficos de un experimento.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Paletas disponibles por tipo de gráfico
_VALID_DIVERGING_CMAPS  = {"RdBu_r", "coolwarm", "bwr", "seismic", "PiYG"}


@dataclass
class PlotConfig:
    """
    Configuración estética para las visualizaciones de attnspectra.

    Se puede pasar a cualquier función de ``viz/`` que acepte parámetros
    de estilo.

    Attributes
    ----------
    figsize_heatmap:
        Tamaño de figura para heatmaps de capa × cabeza.
    figsize_lines:
        Tamaño de figura para gráficos de líneas (Δ vs capa).
    figsize_attn_matrix:
        Tamaño de figura para matrices de atención individuales.
    cmap_diverging:
        Colormap para heatmaps de diferencias (Δ). Debe ser divergente.
    cmap_sequential:
        Colormap para heatmaps de métricas absolutas.
    dpi:
        Resolución al guardar figuras con ``fig.savefig()``.
    max_tokens_attn:
        Número máximo de tokens a mostrar en plot_attention_matrix.
    show_colorbar:
        Si True, muestra la colorbar en los heatmaps.
    line_marker:
        Marcador para las líneas en plot_delta_lines_by_condition.
    alpha_fill:
        Opacidad de la banda ± std en plot_metric_by_layer.
    font_size_ticks:
        Tamaño de fuente de los labels de los ejes.
    """

    # Tamaños de figura
    figsize_heatmap:      tuple[float, float] = (10, 4)
    figsize_lines:        tuple[float, float] = (8, 4)
    figsize_attn_matrix:  tuple[float, float] = (10, 8)

    # Colores y estilos
    cmap_diverging:  str = "RdBu_r"   # para heatmap_delta
    cmap_sequential: str = "plasma"   # para heatmap_metric y attention_maps

    # Salida
    dpi: int = 100

    # Opciones de visualización
    max_tokens_attn: int   = 50
    show_colorbar:   bool  = True
    line_marker:     str   = "o"
    alpha_fill:      float = 0.2
    font_size_ticks: int   = 8

    def __post_init__(self) -> None:
        if self.cmap_diverging not in _VALID_DIVERGING_CMAPS:
            pass
        if self.alpha_fill < 0.0 or self.alpha_fill > 1.0:
            raise ValueError(
                f"PlotConfig.alpha_fill debe estar en [0, 1], recibido: {self.alpha_fill}"
            )
        if self.dpi < 50:
            raise ValueError(f"PlotConfig.dpi debe ser ≥ 50, recibido: {self.dpi}")
        if self.max_tokens_attn < 2:
            raise ValueError(
                f"PlotConfig.max_tokens_attn debe ser ≥ 2, recibido: {self.max_tokens_attn}"
            )

    def heatmap_kwargs(self, diverging: bool = True) -> dict:
        """
        Devuelve kwargs listos para pasar a ``heatmap_delta`` o ``heatmap_metric``.

        Parameters
        ----------
        diverging:
            True para usar ``cmap_diverging``, False para ``cmap_sequential``.
        """
        return {
            "cmap": self.cmap_diverging if diverging else self.cmap_sequential,
            "figsize": self.figsize_heatmap,
        }

    def lines_kwargs(self) -> dict:
        """Devuelve kwargs para ``plot_delta_lines_by_condition``."""
        return {"figsize": self.figsize_lines}

    def attn_matrix_kwargs(self) -> dict:
        """Devuelve kwargs para ``plot_attention_matrix``."""
        return {
            "figsize": self.figsize_attn_matrix,
            "max_tokens": self.max_tokens_attn,
            "cmap": self.cmap_sequential,
        }