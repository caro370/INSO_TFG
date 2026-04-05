"""
Serialización de CapturedRun y HeadMetrics a disco.

Formatos
--------
  CapturedRun  →  archivo .npz  (arrays numpy comprimidos + metadata JSON embebido)
  HeadMetrics  →  archivo .json (21 campos de métricas + escalares)

Contenido de un .npz (CapturedRun):
  - ``input_ids``         : array (B, T)
  - ``attentions_L000``, ``attentions_L001``, ...: un array (B, H, Q, K) por capa no nula
  - ``_meta``             : JSON con model_info, token_strs, style_idx, schema_version

Nota: los scores (Q·K^T pre-softmax) no se serializan para reducir el tamaño.

Contenido de un .json (HeadMetrics):
  - Familia A (10 campos): A_attn_entropy, A_effective_rank, A_spectral_entropy,
    A_top_singular, A_eigenvalue_decay, A_spectral_decay_rate, A_anisotropy_index,
    A_gini, A_effective_rank_thr, A_attn_distance
  - Familia S (3 campos): S_effective_rank, S_spectral_entropy, S_top_singular
  - Familia Sc (8 campos): Sc_effective_rank, Sc_spectral_entropy, Sc_top_singular,
    Sc_eigenvalue_decay, Sc_spectral_decay_rate, Sc_anisotropy_index,
    Sc_gini, Sc_effective_rank_thr
  - Escalares: n_layers, n_heads, seq_len, schema_version
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from attnspectra.core.types import CapturedRun, HeadMetrics
from attnspectra.storage.formats import SCHEMA_VERSION


def save_run(run: CapturedRun, path: str | Path) -> None:
    """
    Guarda un ``CapturedRun`` en formato ``.npz``.

    El archivo contiene:
      - ``input_ids``          : array (B, T)
      - ``attentions_L{i}``    : array (B, H, Q, K) para cada capa no nula
      - ``_meta``              : JSON con model_info, token_strs, style_idx, versión
    """
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}

    arrays["input_ids"] = run.input_ids.numpy()

    for i, A in enumerate(run.attentions):
        if A is not None:
            key = f"attentions_L{i:03d}"
            arrays[key] = A.cpu().numpy() if isinstance(A, torch.Tensor) else A

    meta = {
        "schema_version": SCHEMA_VERSION,
        "n_layers": run.n_layers,
        "seq_len": run.seq_len,
        "batch_size": run.batch_size,
        "token_strs": run.token_strs,
        "style_idx": run.style_idx,
        "model_info": {
            "name": run.model_info.name,
            "architecture": run.model_info.architecture,
            "n_layers": run.model_info.n_layers,
            "n_heads": run.model_info.n_heads,
            "d_model": run.model_info.d_model,
        },
    }
    arrays["_meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    np.savez_compressed(path, **arrays)


def save_metrics(metrics: HeadMetrics, path: str | Path) -> None:
    """Guarda un ``HeadMetrics`` como JSON."""
    path = Path(path)
    data = {
        "schema_version": SCHEMA_VERSION,
        "n_layers": metrics.n_layers,
        "n_heads": metrics.n_heads,
        "seq_len": metrics.seq_len,
        **{k: v.tolist() for k, v in metrics.as_dict().items()},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")