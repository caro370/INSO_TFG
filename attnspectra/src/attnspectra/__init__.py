"""
Extracción, análisis espectral y visualización de matrices de atención
de modelos transformer.
"""

__version__ = "0.1.0"

from attnspectra.core.types import CapturedRun, HeadMetrics, ModelInfo
from attnspectra.config.capture  import CaptureConfig, TargetSpec
from attnspectra.config.metrics  import MetricConfig, ALL_METRICS
from attnspectra.config.plotting import PlotConfig
from attnspectra.core.device  import get_device, to_device, device_info
from attnspectra.adapters.base       import BaseAdapter
from attnspectra.adapters.custom_gpt import CustomGPTAdapter

try:
    from attnspectra.adapters.hf import HFTransformerAdapter
except ImportError:
    pass

from attnspectra.capture.api import (
    capture_and_compute, capture_batch,
    run_degradation_exp, run_style_comparison,
)
from attnspectra.capture.selectors import (
    get_layer_head, pick_tokens, get_content_attention,
    get_metric_layer, get_metric_head, most_attended_tokens,
)
from attnspectra.analysis.metrics import compute_head_metrics
from attnspectra.analysis.aggregation import (
    stack_metric, mean_over_texts, top_sensitive_heads,
)
from attnspectra.analysis.attention import (
    normalize_attn, attention_entropy, attention_distance,
)
from attnspectra.analysis.spectra import (
    singular_values, effective_rank, top_singular_value, spectral_decay_rate, anisotropy_index,
    gini_coefficient, 
)
from attnspectra.transforms.token_perturbations import (
    shuffle_content, replace_fraction, random_iid,
    make_variants, trim_keep_prefix,
)
from attnspectra.viz.attention_maps import plot_attention_matrix, plot_attention_matrix_interactive
from attnspectra.viz.heatmaps       import heatmap_delta, heatmap_metric
from attnspectra.viz.lines          import plot_delta_lines_by_condition, plot_metric_by_layer
from attnspectra.storage.save import save_run, save_metrics
from attnspectra.storage.load import load_run, load_metrics
from attnspectra.core.seeds import set_seed

__all__ = [
    "__version__",
    "CapturedRun", "HeadMetrics", "ModelInfo",
    "CaptureConfig", "TargetSpec", "MetricConfig", "ALL_METRICS", "PlotConfig",
    "get_device", "to_device", "device_info",
    "BaseAdapter", "CustomGPTAdapter",
    "capture_and_compute", "capture_batch",
    "run_degradation_exp", "run_style_comparison",
    "get_layer_head", "pick_tokens", "get_content_attention",
    "get_metric_layer", "get_metric_head", "most_attended_tokens",
    "compute_head_metrics",
    "stack_metric", "mean_over_texts", "top_sensitive_heads",
    "normalize_attn", "attention_entropy", "attention_distance",
    "singular_values", "effective_rank", "top_singular_value", "spectral_decay_rate",
    "anisotropy_index", "gini_coefficient", "effective_rank_threshold",
    "shuffle_content", "replace_fraction", "random_iid",
    "make_variants", "trim_keep_prefix",
    "plot_attention_matrix", "plot_attention_matrix_interactive", "heatmap_delta", "heatmap_metric",
    "plot_delta_lines_by_condition", "plot_metric_by_layer",
    "save_run", "save_metrics", "load_run", "load_metrics",
    "set_seed",
    
]