"""
Tests focalizados en normalize_attn y casos edge de attention_entropy.

Complementa test_spectra.py con casos más exhaustivos de normalización,
valores extremos y comportamiento numérico.
"""

import pytest
import torch

from attnspectra.analysis.attention import attention_entropy, normalize_attn


class TestNormalizeAttnEdgeCases:
    def test_zero_matrix_does_not_crash(self):
        """Una matriz de ceros no debe lanzar excepción (divide-by-eps)."""
        A = torch.zeros(1, 1, 4, 4)
        out = normalize_attn(A)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_single_nonzero_per_row(self):
        """Una fila con un solo elemento no nulo → esa posición debe ser 1.0."""
        A = torch.zeros(1, 1, 3, 3)
        A[0, 0, 0, 2] = 5.0
        A[0, 0, 1, 0] = 3.0
        A[0, 0, 2, 1] = 1.0
        out = normalize_attn(A)
        assert torch.allclose(out[0, 0, 0], torch.tensor([0., 0., 1.]), atol=1e-6)
        assert torch.allclose(out[0, 0, 1], torch.tensor([1., 0., 0.]), atol=1e-6)

    def test_large_values_do_not_overflow(self):
        A = torch.full((1, 1, 4, 4), 1e30)
        out = normalize_attn(A)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_negative_values_handled(self):
        """Valores negativos (p.ej. scores sin softmax) no deben causar NaN."""
        A = torch.randn(1, 1, 4, 4)  # puede tener negativos
        out = normalize_attn(A)
        assert not torch.isnan(out).any()

    def test_mixed_nan_and_valid(self):
        A = torch.rand(1, 1, 4, 4)
        A[0, 0, 2, :] = float("nan")
        out = normalize_attn(A)
        assert not torch.isnan(out).any()

    def test_3d_tensor(self):
        """normalize_attn debe funcionar con shape (..., Q, K) genérico."""
        A = torch.rand(3, 5)
        out = normalize_attn(A)
        assert out.shape == (3, 5)
        assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)

    def test_batch_independence(self):
        """Normalizar en batch debe dar el mismo resultado que normalizar por separado."""
        torch.manual_seed(7)
        A0 = torch.rand(1, 2, 4, 4)
        A1 = torch.rand(1, 2, 4, 4)
        A_batch = torch.cat([A0, A1], dim=0)   # (2, 2, 4, 4)

        out_batch = normalize_attn(A_batch)
        out0 = normalize_attn(A0)
        out1 = normalize_attn(A1)

        assert torch.allclose(out_batch[0], out0[0], atol=1e-6)
        assert torch.allclose(out_batch[1], out1[0], atol=1e-6)


class TestAttentionEntropyEdgeCases:
    def test_entropy_increases_with_uniformity(self):
        """Más uniforme → mayor entropía. Verificamos con gradación."""
        K = 8
        entropies = []
        for conc in [8.0, 4.0, 2.0, 1.0]:
            sv = torch.ones(1, 1, 1, K)
            sv[0, 0, 0, 0] = conc
            A = sv / sv.sum(dim=-1, keepdim=True)
            entropies.append(attention_entropy(A).item())
        # Debe ser estrictamente creciente (menos concentrado → más entropía)
        for i in range(len(entropies) - 1):
            assert entropies[i] <= entropies[i + 1] + 1e-6

    def test_symmetric_attention_entropy(self):
        """Permutar columnas no cambia la entropía."""
        torch.manual_seed(3)
        A = torch.rand(1, 1, 4, 6)
        A = A / A.sum(dim=-1, keepdim=True)
        perm = torch.randperm(6)
        A_perm = A[..., perm]
        H = attention_entropy(A)
        H_perm = attention_entropy(A_perm)
        assert torch.allclose(H, H_perm, atol=1e-6)

    def test_causal_mask_does_not_break_entropy(self):
        """Simula atención causal: filas superiores tienen menos keys disponibles."""
        T = 6
        A = torch.zeros(1, 1, T, T)
        for i in range(T):
            # Solo puede atender a posiciones ≤ i
            n_valid = i + 1
            A[0, 0, i, :n_valid] = 1.0 / n_valid
        H = attention_entropy(A)
        assert not torch.isnan(H).any()
        assert (H >= 0).all()