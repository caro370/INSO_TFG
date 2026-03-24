"""
Tests unitarios para:
  - analysis/attention.py  (normalize_attn, attention_entropy, per_query_entropy)
  - analysis/spectra.py    (singular_values, effective_rank,
                            top_singular_value, eigenvalue_decay,
                            spectral_decay_rate, anisotropy_index,
                            gini_coefficient)
"""

import math

import numpy as np
import pytest
import torch

from attnspectra.analysis.attention import (
    attention_entropy,
    normalize_attn,
    per_query_entropy,
)
from attnspectra.analysis.spectra import (
    anisotropy_index,
    effective_rank,
    gini_coefficient,
    singular_values,
    spectral_decay_rate,
    top_singular_value,
)


# Fixtures compartidas

B, H, Q, K = 1, 4, 8, 8


def uniform_attn() -> torch.Tensor:
    """Matriz de atención uniforme (máxima entropía)."""
    return torch.full((B, H, Q, K), 1.0 / K)


def identity_attn() -> torch.Tensor:
    """Cada query atiende solo a su propia posición (mínima entropía)."""
    eye = torch.eye(Q, K).unsqueeze(0).unsqueeze(0)
    return eye.expand(B, H, Q, K)


def random_attn(seed: int = 42) -> torch.Tensor:
    """Atención aleatoria normalizada por filas."""
    torch.manual_seed(seed)
    raw = torch.rand(B, H, Q, K)
    return raw / raw.sum(dim=-1, keepdim=True)


def uniform_sv(k: int = K) -> torch.Tensor:
    """Valores singulares uniformes — distribución plana."""
    return torch.ones(B, H, k)


def rank1_sv(k: int = K) -> torch.Tensor:
    """Solo el primer valor singular no nulo — rango 1."""
    sv = torch.zeros(1, 1, k)
    sv[..., 0] = 1.0
    return sv


def decaying_sv(k: int = K, base: float = 2.0) -> torch.Tensor:
    """Valores singulares con caída exponencial: σᵢ = base^(-i)."""
    i = torch.arange(k, dtype=torch.float32)
    sv = base ** (-i)
    return sv.unsqueeze(0).unsqueeze(0).expand(B, H, k)



# TestNormalizeAttn

class TestNormalizeAttn:
    def test_rows_sum_to_one(self):
        A = random_attn()
        out = normalize_attn(A)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    def test_already_normalized_unchanged(self):
        A = uniform_attn()
        out = normalize_attn(A)
        assert torch.allclose(out, A, atol=1e-6)

    def test_nan_replaced(self):
        A = torch.full((1, 1, 4, 4), float("nan"))
        out = normalize_attn(A)
        assert not torch.isnan(out).any()

    def test_inf_replaced(self):
        A = torch.full((1, 1, 4, 4), float("inf"))
        out = normalize_attn(A)
        assert not torch.isinf(out).any()

    def test_shape_preserved(self):
        A = random_attn()
        assert normalize_attn(A).shape == A.shape

    def test_nonnegative(self):
        A = random_attn()
        assert (normalize_attn(A) >= 0).all()



# TestAttentionEntropy

class TestAttentionEntropy:
    def test_output_shape(self):
        A = uniform_attn()
        assert attention_entropy(A).shape == (B, H)

    def test_uniform_is_maximum(self):
        """La distribución uniforme tiene la mayor entropía posible: log(K)."""
        H_uni  = attention_entropy(uniform_attn())
        H_rand = attention_entropy(random_attn())
        assert (H_uni >= H_rand - 1e-5).all()

    def test_identity_is_minimum(self):
        """La distribución delta (identidad) tiene entropía ≈ 0."""
        H = attention_entropy(identity_attn())
        assert (H < 0.01).all()

    def test_uniform_value(self):
        """H(uniforme) ≈ log(K)."""
        H = attention_entropy(uniform_attn())
        assert torch.allclose(H, torch.full_like(H, math.log(K)), atol=1e-4)

    def test_nonnegative(self):
        assert (attention_entropy(random_attn()) >= 0).all()



# TestPerQueryEntropy

class TestPerQueryEntropy:
    def test_output_shape(self):
        assert per_query_entropy(random_attn()).shape == (B, H, Q)

    def test_mean_equals_attention_entropy(self):
        """Media sobre Q de per_query_entropy debe igualar attention_entropy."""
        A = random_attn()
        assert torch.allclose(
            per_query_entropy(A).mean(dim=-1),
            attention_entropy(A),
            atol=1e-5,
        )

    def test_nonnegative(self):
        assert (per_query_entropy(random_attn()) >= 0).all()



# TestSingularValues

class TestSingularValues:
    def test_output_shape_square(self):
        M = torch.rand(B, H, Q, K)
        assert singular_values(M).shape == (B, H, min(Q, K))

    def test_output_shape_rectangular(self):
        M = torch.rand(2, 3, 6, 10)
        assert singular_values(M).shape == (2, 3, 6)

    def test_nonnegative(self):
        assert (singular_values(torch.rand(B, H, Q, K)) >= 0).all()

    def test_sorted_descending(self):
        sv = singular_values(torch.rand(B, H, Q, K))
        diffs = sv[..., :-1] - sv[..., 1:]
        assert (diffs >= -1e-6).all()

    def test_rank1_matrix(self):
        """Matriz rango 1 tiene un solo valor singular no nulo."""
        u = torch.rand(Q, 1)
        v = torch.rand(1, K)
        M = (u @ v).unsqueeze(0).unsqueeze(0)
        sv = singular_values(M).squeeze()
        assert sv[0] > 1e-4
        assert (sv[1:] < 1e-4).all()



# TestEffectiveRank

class TestEffectiveRank:
    def test_uniform_sv_equals_k(self):
        """Si todos los sv son iguales, rango efectivo = K."""
        er = effective_rank(uniform_sv())
        assert torch.allclose(er, torch.full_like(er, float(K)), atol=1e-4)

    def test_rank1_equals_one(self):
        """Si solo hay un sv no nulo, rango efectivo ≈ 1."""
        er = effective_rank(rank1_sv())
        assert torch.allclose(er, torch.ones_like(er), atol=1e-4)

    def test_output_shape(self):
        assert effective_rank(torch.rand(B, H, K)).shape == (B, H)

    def test_bounded_above_by_k(self):
        assert (effective_rank(torch.rand(B, H, K)) <= K + 1e-5).all()

    def test_bounded_below_by_one(self):
        assert (effective_rank(torch.rand(B, H, K).clamp_min(1e-6)) >= 1.0 - 1e-5).all()

    def test_monotone_with_concentration(self):
        """Más concentración en un valor → menor rango efectivo."""
        sv_flat = torch.ones(1, 1, 8)
        sv_conc = torch.tensor([[[8., 1., 1., 1., 1., 1., 1., 1.]]])
        assert effective_rank(sv_flat) > effective_rank(sv_conc)


# TestTopSingularValue

class TestTopSingularValue:
    def test_output_shape(self):
        assert top_singular_value(torch.rand(B, H, K)).shape == (B, H)

    def test_equals_first_element(self):
        sv = torch.rand(B, H, K)
        assert torch.allclose(top_singular_value(sv), sv[..., 0])

    def test_is_largest(self):
        sv, _ = torch.rand(B, H, K).sort(dim=-1, descending=True)
        assert torch.allclose(top_singular_value(sv), sv[..., 0])

    def test_rank1_matrix_value(self):
        """Para matriz rango 1 = u·vᵀ, el sv = ||u|| · ||v||."""
        u = torch.tensor([[1.0, 2.0, 3.0]])
        v = torch.tensor([[4.0, 5.0]])
        M = (u.T @ v).unsqueeze(0).unsqueeze(0)
        sv  = singular_values(M)
        tsv = top_singular_value(sv).item()
        assert abs(tsv - (u.norm() * v.norm()).item()) < 1e-4




# TestSpectralDecayRate

class TestSpectralDecayRate:
    def test_output_shape(self):
        sv = torch.rand(B, H, K).clamp_min(1e-6)
        assert spectral_decay_rate(sv).shape == (B, H)

    def test_uniform_sv_near_zero(self):
        """
        Si todos los sv son iguales, log(σᵢ) es constante: la pendiente α ≈ 0.
        """
        alpha = spectral_decay_rate(uniform_sv())
        assert torch.allclose(alpha, torch.zeros_like(alpha), atol=1e-4)

    def test_faster_decay_gives_larger_alpha(self):
        """
        sv con caída más rápida → α mayor.
        decaying_sv(base=4) cae más rápido que decaying_sv(base=2).
        """
        alpha_slow = spectral_decay_rate(decaying_sv(base=2.0))
        alpha_fast = spectral_decay_rate(decaying_sv(base=4.0))
        assert (alpha_fast > alpha_slow).all()

    def test_nonnegative_for_descending_sv(self):
        """
        Para sv ordenados de mayor a menor, log(σᵢ) es decreciente → α ≥ 0.
        """
        sv, _ = torch.rand(B, H, K).clamp_min(1e-6).sort(dim=-1, descending=True)
        alpha = spectral_decay_rate(sv)
        assert (alpha >= -1e-4).all()

    def test_known_exponential_decay(self):
        """
        sv = exp(-α₀·i) → log(σᵢ) = -α₀·i.
        El ajuste debe recuperar α₀ exactamente.
        """
        alpha0 = 0.5
        K_test = 8
        i  = torch.arange(K_test, dtype=torch.float32)
        sv = torch.exp(-alpha0 * i).unsqueeze(0).unsqueeze(0)  # (1,1,K)
        alpha = spectral_decay_rate(sv)
        assert torch.allclose(alpha, torch.tensor([[alpha0]]), atol=1e-4)

    def test_batch_shape_preserved(self):
        sv = torch.rand(2, 6, K).clamp_min(1e-6)
        assert spectral_decay_rate(sv).shape == (2, 6)


# TestAnisotropyIndex

class TestAnisotropyIndex:
    def test_output_shape(self):
        assert anisotropy_index(torch.rand(B, H, K)).shape == (B, H)

    def test_uniform_sv_is_zero(self):
        """
        Si todos los sv son iguales, σmax = σmin → AI = 0.
        """
        ai = anisotropy_index(uniform_sv())
        assert torch.allclose(ai, torch.zeros_like(ai), atol=1e-5)

    def test_nonnegative(self):
        sv = torch.rand(B, H, K).clamp_min(1e-6)
        assert (anisotropy_index(sv) >= 0).all()

    def test_monotone_with_spread(self):
        """
        Mayor diferencia entre σmax y σmin → mayor AI.
        """
        sv_narrow = torch.tensor([[[2.0, 1.8, 1.6, 1.4]]])
        sv_wide   = torch.tensor([[[4.0, 1.0, 1.0, 0.1]]])
        assert anisotropy_index(sv_wide) > anisotropy_index(sv_narrow)

    def test_known_value(self):
        """
        sv = [3, 2, 1] → AI = (3-1) / mean([3,2,1]) = 2 / 2 = 1.0.
        """
        sv = torch.tensor([[[3.0, 2.0, 1.0]]])
        ai = anisotropy_index(sv)
        assert torch.allclose(ai, torch.tensor([[1.0]]), atol=1e-5)

    def test_single_element(self):
        """
        Un solo sv → σmax = σmin → AI = 0.
        """
        sv = torch.tensor([[[5.0]]])
        ai = anisotropy_index(sv)
        assert torch.allclose(ai, torch.zeros_like(ai), atol=1e-5)

    def test_batch_shape_preserved(self):
        assert anisotropy_index(torch.rand(2, 6, K)).shape == (2, 6)



# TestGiniCoefficient

class TestGiniCoefficient:
    def test_output_shape(self):
        assert gini_coefficient(torch.rand(B, H, K)).shape == (B, H)

    def test_uniform_sv_is_zero(self):
        """
        Distribución igualitaria → G = 0.
        Con sv = [c, c, ..., c]: G = (2·Σᵢ i·c)/(K·K·c) − (K+1)/K
        = (2·c·K(K+1)/2)/(K²c) − (K+1)/K = (K+1)/K − (K+1)/K = 0.
        """
        g = gini_coefficient(uniform_sv())
        assert torch.allclose(g, torch.zeros_like(g), atol=1e-5)

    def test_bounded_in_zero_one(self):
        """G ∈ [0, 1)."""
        g = gini_coefficient(torch.rand(B, H, K).clamp_min(1e-6))
        assert (g >= -1e-6).all()
        assert (g <  1.0 + 1e-6).all()

    def test_concentrated_is_higher_than_uniform(self):
        """Mayor concentración → mayor Gini."""
        g_uni  = gini_coefficient(uniform_sv())
        g_conc = gini_coefficient(rank1_sv())
        assert (g_conc >= g_uni).all()

    def test_invariant_to_input_order(self):
        """
        Gini no depende del orden: la implementación ordena internamente.
        """
        sv     = torch.tensor([[[1.0, 3.0, 2.0, 4.0]]])
        sv_asc = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        sv_desc= torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
        g_orig = gini_coefficient(sv)
        g_asc  = gini_coefficient(sv_asc)
        g_desc = gini_coefficient(sv_desc)
        assert torch.allclose(g_orig, g_asc,  atol=1e-5)
        assert torch.allclose(g_orig, g_desc, atol=1e-5)

    def test_known_two_values(self):
        """
        sv = [1, 3] (ascendente): K=2, Σᵢ i·σᵢ = 1·1 + 2·3 = 7,
        denominador = 2·4 = 8, G = 2·7/8 − 3/2 = 1.75 − 1.5 = 0.25.
        """
        sv = torch.tensor([[[1.0, 3.0]]])
        g  = gini_coefficient(sv)
        assert torch.allclose(g, torch.tensor([[0.25]]), atol=1e-5)

    def test_batch_shape_preserved(self):
        assert gini_coefficient(torch.rand(2, 6, K)).shape == (2, 6)
