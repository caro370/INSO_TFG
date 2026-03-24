"""
core
====
Tipos, contratos y utilidades transversales de attnspectra.

Exports
-------
CapturedRun     — resultado de un forward pass con atención capturada
HeadMetrics     — métricas espectrales y de entropía por capa × cabeza
ModelInfo       — metadatos del modelo origen
AttentionTensor — alias de tipo (B, H, Q, K)
set_seed        — fijar semillas de reproducibilidad
get_rng_state   — snapshot del estado RNG
set_rng_state   — restaurar estado RNG
"""

from attnspectra.core.types import (
    CapturedRun,
    HeadMetrics,
    ModelInfo,
    AttentionTensor,
    LayerCache,
    CacheList,
)
from attnspectra.core.seeds import set_seed, get_rng_state, set_rng_state
from attnspectra.core.validate import (
    check_attention_tensor,
    check_captured_run,
    check_head_metrics_array,
)

__all__ = [
    "CapturedRun",
    "HeadMetrics",
    "ModelInfo",
    "AttentionTensor",
    "LayerCache",
    "CacheList",
    "set_seed",
    "get_rng_state",
    "set_rng_state",
    "check_attention_tensor",
    "check_captured_run",
    "check_head_metrics_array",
]