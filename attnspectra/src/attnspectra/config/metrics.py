"""
Dataclass de configuración para ``compute_head_metrics``.

Permite seleccionar qué familias de métricas calcular y ajustar parámetros
numéricos, útil para reducir el coste computacional cuando solo se necesita
un subconjunto de métricas en un experimento concreto.
"""

from __future__ import annotations

from dataclasses import dataclass


_ALL_A_METRICS: frozenset[str] = frozenset({
    "A_attn_entropy",
    "A_effective_rank",
    "A_top_singular",
    "A_spectral_decay_rate",
    "A_anisotropy_index",
    "A_gini",
    "A_effective_rank_thr",
    "A_attn_distance",
})

_ALL_S_METRICS: frozenset[str] = frozenset({
    "S_effective_rank",
    "S_top_singular",
})

_ALL_SC_METRICS: frozenset[str] = frozenset({
    "Sc_effective_rank",
    "Sc_top_singular",
    "Sc_spectral_decay_rate",
    "Sc_anisotropy_index",
    "Sc_gini",
    "Sc_effective_rank_thr",
})

ALL_METRICS: frozenset[str] = _ALL_A_METRICS | _ALL_S_METRICS | _ALL_SC_METRICS


@dataclass
class MetricConfig:
    """
    Configuración de qué métricas calcular en ``compute_head_metrics``.

    Por defecto se calculan todas las métricas disponibles. Desactivar
    familias completas reduce el tiempo de cómputo, especialmente las
    métricas S_ que requieren una SVD adicional sobre log(A).

    La familia Sc_ no se controla desde aquí: se calcula automáticamente
    cuando el ``CapturedRun`` contiene scores (``capture_scores=True`` en
    ``CaptureConfig``). Si los scores no fueron capturados, los campos
    Sc_* contienen NaN independientemente de esta configuración.

    Attributes
    ----------
    compute_A_metrics:
        Calcular las 10 métricas de la familia A sobre la matriz de
        atención post-softmax. Incluye: ``A_attn_entropy``,
        ``A_effective_rank``, ``A_top_singular``,
        ``A_anisotropy_index``, ``A_gini``,
        ``A_attn_distance``.
    compute_S_metrics:
        Calcular las 3 métricas de la familia S sobre log(A).
        Incluye: ``S_effective_rank``,
        ``S_top_singular``. Requiere una SVD extra por capa.
    eps:
        Valor mínimo para evitar log(0) en log(A) y en las entropías.
        No suele necesitar ajuste (por defecto 1e-12).
    normalize_before_svd:
        Si True (por defecto), normaliza A por filas antes de calcular
        la SVD. Desactivar solo si A ya está normalizada externamente.
    er_threshold:
        Umbral relativo para ``A_effective_rank_thr``: se cuentan los
        valores singulares que superan ``er_threshold · σ₁``.
        Por defecto 0.01 (1% de σ₁). Valores típicos: 0.01, 0.05, 0.1.
    causal:
        Si True (por defecto), ``A_attn_distance`` solo considera la
        parte triangular inferior (j ≤ i), adecuado para modelos
        decoder-only como GPT. Si False, usa |i-j| sobre toda la
        matriz, adecuado para modelos encoder (BERT, RoBERTa).
    """

    compute_A_metrics:    bool  = True
    compute_S_metrics:    bool  = True
    eps:                  float = 1e-12
    normalize_before_svd: bool  = True
    er_threshold:         float = 0.01
    causal:               bool  = True

    def active_metrics(self) -> frozenset[str]:
        """
        Devuelve el conjunto de nombres de métricas que se calcularán.

        Nota: las métricas Sc_* no se incluyen aquí porque su disponibilidad
        depende de si los scores fueron capturados, no de esta configuración.

        Returns
        -------
        frozenset[str] con los nombres de los campos de ``HeadMetrics``
        que ``compute_head_metrics`` intentará calcular.
        """
        active: set[str] = set()
        if self.compute_A_metrics:
            active |= _ALL_A_METRICS
        if self.compute_S_metrics:
            active |= _ALL_S_METRICS
        return frozenset(active)

    def __post_init__(self) -> None:
        if not self.compute_A_metrics and not self.compute_S_metrics:
            raise ValueError(
                "MetricConfig: al menos uno de compute_A_metrics o "
                "compute_S_metrics debe ser True."
            )
        if self.eps <= 0:
            raise ValueError(
                f"MetricConfig.eps debe ser > 0, recibido: {self.eps}"
            )
        if not (0.0 < self.er_threshold < 1.0):
            raise ValueError(
                f"MetricConfig.er_threshold debe estar en (0, 1), "
                f"recibido: {self.er_threshold}"
            )