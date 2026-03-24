"""
config
======
Dataclasses de configuración del paquete attnspectra.

Exports
-------
CaptureConfig   — qué capturar durante el forward pass
TargetSpec      — especificación de una capa/cabeza concreta
MetricConfig    — qué métricas calcular y con qué parámetros
PlotConfig      — opciones estéticas para todas las visualizaciones
"""

from attnspectra.config.capture import CaptureConfig, TargetSpec
from attnspectra.config.metrics import MetricConfig, ALL_METRICS
from attnspectra.config.plotting import PlotConfig

__all__ = [
    "CaptureConfig",
    "TargetSpec",
    "MetricConfig",
    "ALL_METRICS",
    "PlotConfig",
]