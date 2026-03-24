"""
Cómputo de métricas espectrales y de entropía sobre matrices de atención.

Flujo típico:
    CapturedRun  →  compute_head_metrics()  →  HeadMetrics
    [HeadMetrics, ...]  →  stack_metric() / top_sensitive_heads()

Exports
-------
Métricas principales
  compute_head_metrics      — calcula HeadMetrics completo a partir de CapturedRun

Espectrales originales
  singular_values           — SVD de matrices de atención
  effective_rank            — rango efectivo Roy & Vetterli (continuo)
  top_singular_value        — valor singular dominante σ₁

Nuevas métricas espectrales
  eigenvalue_decay          — δ = σ₂/σ₁
  spectral_decay_rate       — α del ajuste log(σᵢ) = -αi + β
  anisotropy_index          — (σmax-σmin)/σmean
  gini_coefficient          — coeficiente de Gini sobre sv

Entropía y distancia de atención
  normalize_attn            — normaliza filas de A para que sumen 1
  attention_entropy         — H de Shannon promediada sobre posiciones Q
  per_query_entropy         — H de Shannon por posición Q (sin promediar)
  attention_distance        — distancia media de atención ponderada (tokens)

Agregación sobre textos
  stack_metric              — (N,L,H) apilando HeadMetrics
  mean_over_texts           — (L,H) promedio sobre textos
  top_sensitive_heads       — cabezas con mayor Δ entre dos condiciones
"""

from attnspectra.analysis.metrics import compute_head_metrics
from attnspectra.analysis.spectra import (
    singular_values,
    effective_rank,
    top_singular_value,
    spectral_decay_rate,
    anisotropy_index,
    gini_coefficient,
)
from attnspectra.analysis.attention import (
    normalize_attn,
    attention_entropy,
    per_query_entropy,
    attention_distance,
)
from attnspectra.analysis.aggregation import (
    stack_metric,
    mean_over_texts,
    top_sensitive_heads,
)

__all__ = [
    "compute_head_metrics",
    "singular_values",
    "effective_rank",
    "top_singular_value",
    "spectral_decay_rate",
    "anisotropy_index",
    "gini_coefficient",
    "normalize_attn",
    "attention_entropy",
    "per_query_entropy",
    "attention_distance",
    "stack_metric",
    "mean_over_texts",
    "top_sensitive_heads",
]