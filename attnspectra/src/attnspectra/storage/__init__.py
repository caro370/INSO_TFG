"""
istorageo
==
Serialización y deserialización de resultados a disco.

Formatos
--------
  CapturedRun  →  .npz   (arrays numpy comprimidos + metadata JSON embebido)
  HeadMetrics  →  .json  (arrays como listas + metadata del esquema)

El campo ``schema_version`` en cada archivo permite detectar
incompatibilidades cuando el formato cambia entre versiones del paquete.

Exports
-------
save_run        — guarda un CapturedRun en .npz
save_metrics    — guarda un HeadMetrics en .json
load_run        — carga un CapturedRun desde .npz
load_metrics    — carga un HeadMetrics desde .json
SCHEMA_VERSION  — versión actual del esquema de serialización
"""

from attnspectra.storage.save import save_run, save_metrics
from attnspectra.storage.load import load_run, load_metrics
from attnspectra.storage.formats import SCHEMA_VERSION

__all__ = [
    "save_run",
    "save_metrics",
    "load_run",
    "load_metrics",
    "SCHEMA_VERSION",
]