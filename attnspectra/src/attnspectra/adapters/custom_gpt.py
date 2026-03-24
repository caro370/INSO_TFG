"""
Adapter para el GPT decoder-only implementado en el TFG.

Este modelo expone atención mediante el argumento ``capture`` en su forward(),
que devuelve una lista de dicts con claves ``attn_pre``, ``attn_post``, ``scores``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from attnspectra.adapters.base import BaseAdapter
from attnspectra.config.capture import CaptureConfig
from attnspectra.core.types import CapturedRun, ModelInfo


class CustomGPTAdapter(BaseAdapter):
    """
    Adapter para el modelo GPT custom del TFG.

    El modelo debe tener la siguiente signatura de forward:

        model(idx, style_idx, capture=None, return_style_logits=False)
            -> (logits, caches)      si return_style_logits=False
            -> (logits, caches, style_logits)  si return_style_logits=True

    donde ``caches`` es una lista de dicts con claves opcionales:
        - ``attn_pre``   : (B, H, T, T) — atención post-softmax
        - ``attn_post``  : (B, H, T, T) — atención post-dropout
        - ``scores``     : (B, H, T, T) — logits pre-softmax

    Parameters
    ----------
    model:
        Instancia del GPT custom, ya en el device correcto.
    tokenizer:
        Tokenizer del modelo (HuggingFace tokenizers o similar).
    n_layers:
        Número de capas del transformer.
    n_heads:
        Número de cabezas por capa.
    d_model:
        Dimensión del modelo.
    model_name:
        Nombre descriptivo del modelo (para logs y serialización).
    device:
        Device de PyTorch.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        n_layers: int,
        n_heads: int,
        d_model: int,
        model_name: str = "custom-gpt",
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._n_layers = n_layers
        self._n_heads = n_heads
        self._d_model = d_model
        self._model_name = model_name
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            architecture="decoder",
            n_layers=self._n_layers,
            n_heads=self._n_heads,
            d_model=self._d_model,
        )

    @torch.no_grad()
    def capture_from_ids(
        self,
        input_ids: torch.Tensor,
        config: CaptureConfig,
        style_idx: torch.Tensor | int | None = None,
        **kwargs: object,
    ) -> CapturedRun:
        """
        Parameters
        ----------
        input_ids:
            (B, T) tensor de ids.
        config:
            CaptureConfig con ``capture_scores``, ``capture_attn_pre``, etc.
        style_idx:
            Tensor (B,) o int con el índice de estilo. Si es int se expande al batch.
        """
        input_ids = input_ids.to(self.device)
        B = input_ids.shape[0]

        # Preparar style_idx
        if style_idx is None:
            sidx = torch.zeros(B, dtype=torch.long, device=self.device)
        elif isinstance(style_idx, int):
            sidx = torch.full((B,), style_idx, dtype=torch.long, device=self.device)
        else:
            sidx = style_idx.to(self.device)

        # Construir dict de capture según config
        capture_dict: dict[str, bool] = {
            "attn_pre": getattr(config, "capture_attn_pre", True),
            "attn_post": getattr(config, "capture_attn_post", False),
            "scores": getattr(config, "capture_scores", False),
        }

        _, caches = self.model(
            input_ids,
            style_idx=sidx,
            capture=capture_dict,
            return_style_logits=False,
        )

        attentions, scores = self._parse_caches(caches, capture_dict)
        token_strs = self._decode_tokens(input_ids[0].tolist())

        return CapturedRun(
            input_ids=input_ids.cpu(),
            attentions=attentions,
            scores=scores if capture_dict.get("scores") else None,
            token_strs=token_strs,
            model_info=self.model_info,
            style_idx=sidx[0].item() if sidx is not None else None,
        )

    def _parse_caches(
        self,
        caches: list[dict[str, torch.Tensor] | None],
        capture_dict: dict[str, bool],
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        """Extrae attentions y scores de la lista de caches del modelo."""
        attentions: list[torch.Tensor | None] = []
        scores_list: list[torch.Tensor | None] = []

        which_attn = "attn_pre" if capture_dict.get("attn_pre") else "attn_post"

        for cache in caches:
            if cache is None:
                attentions.append(None)
                scores_list.append(None)
                continue
            attentions.append(cache.get(which_attn))        # (B,H,T,T) o None
            scores_list.append(cache.get("scores"))         # (B,H,T,T) o None

        return attentions, scores_list

    def _decode_tokens(self, ids: list[int]) -> list[str]:
        tok = self.tokenizer
        if hasattr(tok, "decode"):
            # Decodificamos uno a uno para obtener el string de cada token
            return [tok.decode([i]) for i in ids]
        raise TypeError(f"No se sabe cómo decodificar con el tokenizer {type(tok)}.")