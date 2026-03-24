"""
Clase base abstracta para todos los adapters de attnspectra.

Un adapter es el único componente que conoce los detalles del modelo concreto
(PyTorch custom, HuggingFace, etc.). Todo el resto del paquete consume
``CapturedRun``, que es lo que el adapter debe producir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from attnspectra.config.capture import CaptureConfig
from attnspectra.core.types import CapturedRun, ModelInfo


class BaseAdapter(ABC):
    """
    Interfaz que todo adapter debe implementar.

    Subclases concretas:
      - ``HFTransformerAdapter``  (adapters/hf.py)
      - ``CustomGPTAdapter``      (adapters/custom_gpt.py)
    """

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Devuelve los metadatos del modelo envuelto."""
        ...

    @abstractmethod
    def capture_from_ids(
        self,
        input_ids: torch.Tensor,
        config: CaptureConfig,
        **kwargs: object,
    ) -> CapturedRun:
        """
        Ejecuta un forward pass y devuelve un ``CapturedRun``.

        Parameters
        ----------
        input_ids:
            Tensor de forma (B, T) con los ids de entrada.
        config:
            Configuración de qué capturar (layers, heads, scores, etc.).
        **kwargs:
            Argumentos adicionales específicos del modelo (p.ej. style_idx).
        """
        ...

    def capture_from_texts(
        self,
        texts: Sequence[str],
        config: CaptureConfig,
        tokenizer: object | None = None,
        **kwargs: object,
    ) -> CapturedRun:
        """
        Versión de conveniencia que tokeniza y llama a ``capture_from_ids``.

        La implementación por defecto requiere que el adapter tenga un atributo
        ``self.tokenizer``. Subclases pueden sobreescribir esto.
        """
        tok = tokenizer or getattr(self, "tokenizer", None)
        if tok is None:
            raise ValueError(
                "No se proporcionó tokenizer. Pásalo en capture_from_texts() "
                "o asígnalo como self.tokenizer en el adapter."
            )

        # Soporte mínimo para tokenizers HF y custom
        if hasattr(tok, "batch_encode_plus"):
            enc = tok.batch_encode_plus(
                list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"]
        elif hasattr(tok, "encode_batch"):
            # tokenizers (HuggingFace tokenizers library)
            encodings = tok.encode_batch(list(texts))
            max_len = max(len(e.ids) for e in encodings)
            input_ids = torch.zeros(len(encodings), max_len, dtype=torch.long)
            for i, e in enumerate(encodings):
                input_ids[i, : len(e.ids)] = torch.tensor(e.ids)
        else:
            raise TypeError(
                f"No se sabe cómo usar el tokenizer de tipo {type(tok)}. "
                "Implementa capture_from_texts() en tu adapter."
            )

        return self.capture_from_ids(input_ids, config, **kwargs)