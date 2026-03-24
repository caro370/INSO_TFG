"""
capture
=======
Normalización de tensores de atención, helpers de selección y
orquestador de alto nivel del pipeline.

Exports
-------
normalize_hf_attentions  — convierte tupla HF → List[Tensor(B,H,Q,K)]
normalize_cache_list     — convierte lista de caches GPT custom → List[Tensor]
get_layer_head           — extrae matriz (Q,K) de una capa y cabeza
pick_tokens              — selecciona token_strs por índice o slice
get_content_attention    — submatriz sin tokens de prefijo
get_metric_layer         — métrica de todas las cabezas de una capa
get_metric_head          — métrica de todas las capas de una cabeza
most_attended_tokens     — tokens más atendidos (promedio sobre queries)
capture_and_compute      — captura + métricas en una sola llamada
capture_batch            — procesa lista de textos/ids
run_degradation_exp      — pipeline completo EXP1 (degradación)
run_style_comparison     — pipeline completo EXP3 (comparación de estilos)
"""

from attnspectra.capture.normalize import (
    normalize_hf_attentions,
    normalize_cache_list,
)
from attnspectra.capture.selectors import (
    get_layer_head,
    pick_tokens,
    get_content_attention,
    get_metric_layer,
    get_metric_head,
    most_attended_tokens,
)
from attnspectra.capture.api import (
    capture_and_compute,
    capture_batch,
    run_degradation_exp,
    run_style_comparison,
)

__all__ = [
    # normalize
    "normalize_hf_attentions",
    "normalize_cache_list",
    # selectors
    "get_layer_head",
    "pick_tokens",
    "get_content_attention",
    "get_metric_layer",
    "get_metric_head",
    "most_attended_tokens",
    # api
    "capture_and_compute",
    "capture_batch",
    "run_degradation_exp",
    "run_style_comparison",
]