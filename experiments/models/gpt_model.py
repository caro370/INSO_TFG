"""
Arquitectura GPT decoder-only con RoPE, style embeddings y mecanismo
de captura de atención.

Clases:
GPT                  — modelo completo
GPTBlock             — bloque transformer (atención + FFN)
CausalSelfAttention  — atención causal con RoPE
FeedForward          — red feed-forward con GELU

Funciones auxiliares:
build_rope_cache     — precalcula cos/sin para RoPE
apply_rope           — aplica rotary position embedding a q/k
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# RoPE (Rotary Position Embedding)

def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    base: int = 10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precalcula las matrices cos y sin para RoPE.

    Returns
    -------
    cos, sin : tensores de shape (1, 1, seq_len, head_dim // 2)
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos()[None, None, :, :]  # (1,1,T,half)
    sin = freqs.sin()[None, None, :, :]  # (1,1,T,half)
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Aplica RoPE a un tensor de queries o keys.

    Parameters
    ----------
    x   : (B, H, T, D)
    cos : (1, 1, T, D//2)
    sin : (1, 1, T, D//2)
    """
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:half*2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    if D > 2*half:
        rest = x[..., 2*half:]
        return torch.cat([out1, out2, rest], dim=-1)
    return torch.cat([out1, out2], dim=-1)


# Módulos

class CausalSelfAttention(nn.Module):
    """
    Atención causal multi-cabeza con RoPE.

    Soporta captura de tensores intermedios mediante el argumento ``capture``:

        capture = {
            "attn_pre":  True,   # atención post-softmax, pre-dropout
            "attn_post": False,  # atención post-dropout
            "scores":    False,  # logits pre-softmax
            "qkv":       False,  # queries, keys y values
        }

    Si ``capture`` es ``None`` o un dict vacío, no se captura nada
    y el forward es equivalente al estándar.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        rope_base: int = 10000,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        cos, sin = build_rope_cache(max_seq_len, self.head_dim, device=torch.device("cpu"), base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        causal = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal, persistent=False)


    def _ensure_device(self, x):
        if self.rope_cos.device != x.device:
            self.rope_cos = self.rope_cos.to(x.device)
            self.rope_sin = self.rope_sin.to(x.device)
        if self.causal_mask.device != x.device:
            self.causal_mask = self.causal_mask.to(x.device)

    def forward(
        self,
        x: torch.Tensor,
        capture: dict[str, bool] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        B, T, C = x.shape
        assert T <= self.max_seq_len, "T supera max_seq_len del attention"

        self._ensure_device(x)

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[..., :T, :]
        sin = self.rope_sin[..., :T, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn_pre = torch.softmax(scores, dim=-1)
        attn_post = self.attn_dropout(attn_pre) if (self.training and self.dropout_p > 0) else attn_pre

        out = attn_post @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        if not capture:
            return out, None

        cache = {}
        if capture.get("scores", False):
            cache["scores"] = scores
        if capture.get("attn_pre", False):
            cache["attn_pre"] = attn_pre
        if capture.get("attn_post", False):
            cache["attn_post"] = attn_post
        if capture.get("qkv", False):
            cache["q"] = q
            cache["k"] = k
            cache["v"] = v
        return out, cache


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        style_emb=None,
        capture: dict[str, bool] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        a, cache = self.attn(self.ln1(x), capture=capture)
        x = x + a
        x = x + self.ff(self.ln2(x))
        if style_emb is not None:
            x = x + style_emb
        return x, cache


class GPT(nn.Module):
    """
    GPT decoder-only con style embeddings y mecanismo de captura de atención.

    Parameters
    ----------
    vocab_size : tamaño del vocabulario
    block_size : longitud máxima de secuencia
    n_layers   : número de bloques transformer
    d_model    : dimensión del modelo
    n_heads    : número de cabezas de atención
    d_ff       : dimensión interna de la FFN
    dropout    : tasa de dropout
    n_styles   : número de estilos (2 en el TFG: wiki y poem)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layers: int = 8,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        n_styles: int = 2,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.style_emb = nn.Embedding(n_styles, d_model)

        self.style_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_layers)
        ])
        
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, max_seq_len=block_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.style_head = nn.Linear(d_model, n_styles)

    def forward(
        self,
        idx: torch.Tensor,
        style_idx: torch.Tensor,
        capture: dict[str, bool] | None = None,
        return_style_logits: bool = False,
        pad_id: int = 0,
    ) -> tuple:
        """
        Parameters
        ----------
        idx              : (B, T) tensor de token ids
        style_idx        : (B,) tensor de índices de estilo
        capture          : dict de flags para capturar tensores intermedios
        return_style_logits : si True devuelve también logits de clasificación de estilo
        pad_id           : id del token de padding

        Returns
        -------
        (logits, caches)                        si return_style_logits=False
        (logits, caches, style_logits)          si return_style_logits=True
        """
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = idx.size(1)

        x = self.tok_emb(idx)

        s = self.style_emb(style_idx).unsqueeze(1)  # (B,1,C)
        x = self.drop(x)

        caches = [] if capture else None
        for i, blk in enumerate(self.blocks):
            s_i = self.style_projs[i](s)
            x, cache = blk(x, style_emb=s_i, capture=capture)
            if capture:
                caches.append(cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if not return_style_logits:
            return logits, caches

        last_nonpad = (idx != pad_id).long().sum(dim=1) - 1  # (B,)
        last_nonpad = last_nonpad.clamp(min=0)

        h_last = x[torch.arange(B, device=x.device), last_nonpad, :]  # (B,C)

        style_logits = self.style_head(h_last)
        return logits, caches, style_logits