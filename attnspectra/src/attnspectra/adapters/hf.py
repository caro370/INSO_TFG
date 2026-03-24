"""
Adapter para modelos HuggingFace Transformers (BERT, GPT-2, T5, etc.).

Soporta dos estrategias de captura:
  1. ``output_attentions=True``  — la forma más limpia, cuando el modelo lo soporta.
  2. Forward hooks                — fallback para modelos que no exponen output_attentions.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from attnspectra.adapters.base import BaseAdapter
from attnspectra.capture.normalize import normalize_hf_attentions
from attnspectra.config.capture import CaptureConfig
from attnspectra.core.types import CapturedRun, ModelInfo


class HFTransformerAdapter(BaseAdapter):
    """
    Adapter para cualquier modelo HuggingFace con interfaz estándar
    (``AutoModel``, ``BertModel``, ``GPT2Model``, ``T5Model``, etc.).

    Parameters
    ----------
    model:
        Modelo HuggingFace ya cargado y en el device correcto.
    tokenizer:
        Tokenizer asociado al modelo.
    architecture:
        Uno de ``"encoder"``, ``"decoder"``, ``"encoder-decoder"``.
    device:
        Device de PyTorch. Si ``None``, se infiere del primer parámetro del modelo.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        architecture: str = "encoder",
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.architecture = architecture
        self.device = device or next(model.parameters()).device
        self.model.eval()

    @property
    def model_info(self) -> ModelInfo:
        cfg = getattr(self.model, "config", None)
        name = getattr(cfg, "_name_or_path", "unknown-hf-model") if cfg else "unknown-hf-model"

        n_layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(cfg, "num_layers", None)
            or -1
        )
        n_heads = (
            getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "num_heads", None)
            or -1
        )
        d_model = (
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or -1
        )
        return ModelInfo(
            name=name,
            architecture=self.architecture,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
        )

    @torch.no_grad()
    def capture_from_ids(
        self,
        input_ids: torch.Tensor,
        config: CaptureConfig,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> CapturedRun:
        """
        Ejecuta el forward pass y devuelve un ``CapturedRun``.

        Para modelos encoder-decoder (T5) puedes pasar ``decoder_input_ids``.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        fwd_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "output_attentions": True,
        }
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask
        if decoder_input_ids is not None:
            fwd_kwargs["decoder_input_ids"] = decoder_input_ids.to(self.device)

        outputs = self.model(**fwd_kwargs)

        # HF devuelve attentions como tupla de tensores (una por capa)
        # shape de cada elemento: (B, H, Q, K)
        raw_attentions: tuple[torch.Tensor, ...] | None = getattr(outputs, "attentions", None)
        raw_cross: tuple[torch.Tensor, ...] | None = getattr(outputs, "cross_attentions", None)

        # Seleccionamos encoder, decoder o cross según config
        chosen = self._pick_attentions(raw_attentions, raw_cross, config)
        attentions = normalize_hf_attentions(chosen, config)

        # Decodificamos tokens de la primera secuencia
        token_strs = self._decode_tokens(input_ids[0].tolist())

        return CapturedRun(
            input_ids=input_ids.cpu(),
            attentions=attentions,
            scores=None,          # HF no expone scores fácilmente; extensible vía hooks
            token_strs=token_strs,
            model_info=self.model_info,
        )

    def _pick_attentions(
        self,
        enc_attn: tuple[torch.Tensor, ...] | None,
        cross_attn: tuple[torch.Tensor, ...] | None,
        config: CaptureConfig,
    ) -> tuple[torch.Tensor, ...]:
        which = getattr(config, "which_attentions", "encoder")
        if which == "cross" and cross_attn is not None:
            return cross_attn
        if enc_attn is not None:
            return enc_attn
        raise ValueError(
            "El modelo no devolvió attentions. Comprueba que output_attentions=True "
            "es soportado por este modelo HuggingFace."
        )

    def _decode_tokens(self, ids: list[int]) -> list[str]:
        tok = self.tokenizer
        if hasattr(tok, "convert_ids_to_tokens"):
            tokens = tok.convert_ids_to_tokens(ids)
            return [str(t) for t in tokens]
        # Fallback: decodificar uno a uno
        return [tok.decode([i]) for i in ids]