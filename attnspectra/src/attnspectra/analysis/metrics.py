from __future__ import annotations

import numpy as np
import torch

from attnspectra.analysis.attention import (
    normalize_attn,
    attention_entropy,
    attention_distance,
)
from attnspectra.analysis.spectra import (
    singular_values,
    effective_rank,
    top_singular_value,
    spectral_decay_rate,
    anisotropy_index,
    gini_coefficient,
)
from attnspectra.core.types import CapturedRun, HeadMetrics


def compute_head_metrics(
    run: CapturedRun,
    eps: float = 1e-12,
    causal: bool = True,
) -> HeadMetrics:
    """
    Calcula ``HeadMetrics`` a partir de un ``CapturedRun``.

    Métricas calculadas por capa y cabeza (shape ``(n_layers, n_heads)``):

    Familia A — sobre la matriz de atención post-softmax:
      A_attn_entropy        entropía de Shannon de la distribución de atención
      A_effective_rank      rango efectivo continuo (Roy & Vetterli 2007)
      A_top_singular        valor singular dominante σ₁
      A_spectral_decay_rate α del ajuste log(σᵢ) = -αi + β
      A_anisotropy_index    (σmax - σmin) / σmean
      A_gini                coeficiente de Gini sobre sv
      A_attn_distance       distancia media de atención ponderada (tokens)

    Familia S — sobre S = log(A), aproximación a scores pre-softmax:
      S_effective_rank      rango efectivo de log(A)
      S_top_singular        σ₁ de log(A)

    Familia Sc — sobre scores Q·K^T directamente (solo si capture_scores=True):
      Sc_effective_rank      rango efectivo de los scores reales
      Sc_top_singular        σ₁ de los scores
      Sc_spectral_decay_rate α del ajuste log-lineal de los scores
      Sc_anisotropy_index    (σmax - σmin) / σmean de los scores
      Sc_gini                coeficiente de Gini de los scores

    Si los scores no están disponibles, los campos Sc_* contienen NaN.
    Las capas no capturadas también se rellenan con NaN.

    Parameters
    ----------
    run:
        Resultado de un forward pass capturado.
    eps:
        Valor mínimo para evitar log(0) y divisiones por cero.
    causal:
        Si True (defecto), ``attention_distance`` solo considera la parte
        triangular inferior (modelos decoder-only como GPT).
        Si False, usa |i-j| sobre toda la matriz (modelos encoder como BERT).

    Returns
    -------
    ``HeadMetrics`` con arrays de shape ``(n_layers, n_heads)``.
    """
    n_layers = run.n_layers
    n_heads  = _infer_n_heads(run.attentions)
    shape    = (n_layers, n_heads)

    # Familia A
    A_ae   = np.full(shape, np.nan)
    A_er   = np.full(shape, np.nan)
    A_top  = np.full(shape, np.nan)
    A_sdr  = np.full(shape, np.nan)
    A_ai   = np.full(shape, np.nan)
    A_gini = np.full(shape, np.nan)
    A_dist = np.full(shape, np.nan)

    # Familia S
    S_er   = np.full(shape, np.nan)
    S_top  = np.full(shape, np.nan)

    # Familia Sc
    Sc_er  = np.full(shape, np.nan)
    Sc_top = np.full(shape, np.nan)
    Sc_sdr = np.full(shape, np.nan)
    Sc_ai  = np.full(shape, np.nan)
    Sc_gin = np.full(shape, np.nan)

    has_scores = run.scores is not None

    for li, A_raw in enumerate(run.attentions):
        if A_raw is None:
            continue

        # Familia A
        A  = normalize_attn(A_raw)       # (B, H, Q, K) — filas suman 1
        sv = singular_values(A)          # (B, H, min(Q,K))

        A_ae[li, :]  = attention_entropy(A)[0].detach().cpu().numpy()
        A_er[li, :]  = effective_rank(sv)[0].detach().cpu().numpy()
        A_top[li, :] = top_singular_value(sv)[0].detach().cpu().numpy()
        A_ai[li, :]  = anisotropy_index(sv, eps=eps)[0].detach().cpu().numpy()
        A_gini[li, :] = gini_coefficient(sv, eps=eps)[0].detach().cpu().numpy()
        A_dist[li, :] = attention_distance(A, causal=causal)[0].detach().cpu().numpy()

        try:
            A_sdr[li, :] = spectral_decay_rate(sv, eps=eps)[0].detach().cpu().numpy()
        except Exception:
            pass  # lstsq puede fallar con sv degenerados

        # Familia S
        S    = torch.log(A.clamp_min(eps))   # (B, H, Q, K) — valores ≤ 0
        sv_s = singular_values(S)
        S_er[li, :]  = effective_rank(sv_s)[0].detach().cpu().numpy()
        S_top[li, :] = top_singular_value(sv_s)[0].detach().cpu().numpy()

        # Familia Sc: solo si se capturaron los scores
        if has_scores and run.scores[li] is not None:
            Sc = run.scores[li].float()          # (B, H, Q, K) — logits reales
            sv_sc = singular_values(Sc)          # (B, H, min(Q,K))

            Sc_er[li, :]  = effective_rank(sv_sc)[0].detach().cpu().numpy()
            Sc_top[li, :] = top_singular_value(sv_sc)[0].detach().cpu().numpy()
            Sc_ai[li, :]  = anisotropy_index(sv_sc, eps=eps)[0].detach().cpu().numpy()
            Sc_gin[li, :] = gini_coefficient(sv_sc, eps=eps)[0].detach().cpu().numpy()
            
            try:
                Sc_sdr[li, :] = spectral_decay_rate(sv_sc, eps=eps)[0].detach().cpu().numpy()
            except Exception:
                pass

    return HeadMetrics(
        # A
        A_attn_entropy=A_ae,
        A_effective_rank=A_er,
        A_top_singular=A_top,
        A_spectral_decay_rate=A_sdr,
        A_anisotropy_index=A_ai,
        A_gini=A_gini,
        A_attn_distance=A_dist,
        # S
        S_effective_rank=S_er,
        S_top_singular=S_top,
        # Sc
        Sc_effective_rank=Sc_er,
        Sc_top_singular=Sc_top,
        Sc_spectral_decay_rate=Sc_sdr,
        Sc_anisotropy_index=Sc_ai,
        Sc_gini=Sc_gin,
        # escalares
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=run.seq_len,
    )


def _infer_n_heads(attentions: list[torch.Tensor | None]) -> int:
    for A in attentions:
        if A is not None:
            return A.shape[1]
    raise ValueError("Todas las capas son None; no se puede inferir n_heads.")