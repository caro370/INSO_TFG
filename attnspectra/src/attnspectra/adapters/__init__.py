"""
Adapters que envuelven modelos concretos y producen ``CapturedRun``.

El resto del paquete solo conoce ``BaseAdapter`` y ``CapturedRun``;
los detalles de cada modelo quedan encapsulados aquí.

Exports
-------
BaseAdapter          — clase abstracta que todo adapter debe implementar
CustomGPTAdapter     — adapter para el GPT decoder-only del TFG
HFTransformerAdapter — adapter para modelos HuggingFace (requiere [hf])
"""

from attnspectra.adapters.base import BaseAdapter
from attnspectra.adapters.custom_gpt import CustomGPTAdapter

try:
    from attnspectra.adapters.hf import HFTransformerAdapter
    _hf_available = True
except ImportError:
    _hf_available = False

__all__ = [
    "BaseAdapter",
    "CustomGPTAdapter",
]

if _hf_available:
    __all__.append("HFTransformerAdapter")