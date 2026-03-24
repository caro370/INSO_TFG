"""
Deserialización de CapturedRun y HeadMetrics desde disco.

Retrocompatible: ficheros guardados con versiones anteriores del paquete
se cargan sin errores, rellenando campos ausentes con NaN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from attnspectra.core.types import CapturedRun, HeadMetrics, ModelInfo


def load_run(path: str | Path) -> CapturedRun:
    """Carga un ``CapturedRun`` desde un archivo ``.npz``."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    meta     = json.loads(bytes(data["_meta"]).decode("utf-8"))
    n_layers = meta["n_layers"]

    attentions: list[torch.Tensor | None] = [None] * n_layers
    for key in data.files:
        if key.startswith("attentions_L"):
            li = int(key.removeprefix("attentions_L"))
            attentions[li] = torch.from_numpy(data[key])

    mi = meta["model_info"]
    model_info = ModelInfo(
        name=mi["name"],
        architecture=mi["architecture"],
        n_layers=mi["n_layers"],
        n_heads=mi["n_heads"],
        d_model=mi["d_model"],
    )

    return CapturedRun(
        input_ids=torch.from_numpy(data["input_ids"]),
        attentions=attentions,
        scores=None,
        token_strs=meta["token_strs"],
        model_info=model_info,
        style_idx=meta.get("style_idx"),
    )


def load_metrics(path: str | Path) -> HeadMetrics:
    """
    Carga un ``HeadMetrics`` desde un archivo ``.json``.

    Retrocompatible: campos ausentes se rellenan con NaN.
    Esto incluye los campos Sc_* (introducidos en v0.2.0) y cualquier
    campo añadido en versiones futuras.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    n_layers = data["n_layers"]
    n_heads  = data["n_heads"]
    shape    = (n_layers, n_heads)

    def _load(key: str) -> np.ndarray:
        if key in data:
            return np.array(data[key])
        return np.full(shape, np.nan)

    return HeadMetrics(
        # A
        A_attn_entropy=_load("A_attn_entropy"),
        A_effective_rank=_load("A_effective_rank"),
        A_top_singular=_load("A_top_singular"),
        A_spectral_decay_rate=_load("A_spectral_decay_rate"),
        A_anisotropy_index=_load("A_anisotropy_index"),
        A_gini=_load("A_gini"),
        A_attn_distance=_load("A_attn_distance"),
        # S
        S_effective_rank=_load("S_effective_rank"),
        S_top_singular=_load("S_top_singular"),
        # Sc
        Sc_effective_rank=_load("Sc_effective_rank"),
        Sc_top_singular=_load("Sc_top_singular"),
        Sc_spectral_decay_rate=_load("Sc_spectral_decay_rate"),
        Sc_anisotropy_index=_load("Sc_anisotropy_index"),
        Sc_gini=_load("Sc_gini"),
        # escalares
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=data["seq_len"],
    )