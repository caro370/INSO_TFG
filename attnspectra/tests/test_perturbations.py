"""Tests para transforms/token_perturbations.py."""

import pytest
from attnspectra.transforms.token_perturbations import (
    shuffle_content,
    replace_fraction,
    random_iid,
    trim_keep_prefix,
    make_variants,
)

PREFIX_LEN = 3
VOCAB_SIZE = 100
BASE_IDS = list(range(20))  # [0,1,2, ...,19]  (prefix = [0,1,2])


class TestShuffleContent:
    def test_prefix_preserved(self):
        out = shuffle_content(BASE_IDS, prefix_len=PREFIX_LEN, seed=42)
        assert out[:PREFIX_LEN] == BASE_IDS[:PREFIX_LEN]

    def test_length_preserved(self):
        out = shuffle_content(BASE_IDS, prefix_len=PREFIX_LEN, seed=42)
        assert len(out) == len(BASE_IDS)

    def test_same_tokens_different_order(self):
        out = shuffle_content(BASE_IDS, prefix_len=PREFIX_LEN, seed=42)
        assert sorted(out) == sorted(BASE_IDS)

    def test_reproducible(self):
        a = shuffle_content(BASE_IDS, prefix_len=PREFIX_LEN, seed=7)
        b = shuffle_content(BASE_IDS, prefix_len=PREFIX_LEN, seed=7)
        assert a == b


class TestReplaceFraction:
    def test_prefix_preserved(self):
        out = replace_fraction(BASE_IDS, frac=0.5, vocab_size=VOCAB_SIZE,
                               prefix_len=PREFIX_LEN, seed=42)
        assert out[:PREFIX_LEN] == BASE_IDS[:PREFIX_LEN]

    def test_length_preserved(self):
        out = replace_fraction(BASE_IDS, frac=0.5, vocab_size=VOCAB_SIZE,
                               prefix_len=PREFIX_LEN, seed=42)
        assert len(out) == len(BASE_IDS)

    def test_zero_fraction_no_change(self):
        out = replace_fraction(BASE_IDS, frac=0.0, vocab_size=VOCAB_SIZE,
                               prefix_len=PREFIX_LEN, seed=42)
        assert out == BASE_IDS

    def test_invalid_frac_raises(self):
        with pytest.raises(ValueError):
            replace_fraction(BASE_IDS, frac=1.5, vocab_size=VOCAB_SIZE)

    def test_approximately_correct_fraction(self):
        # Con frac=0.5 y 17 tokens de contenido, esperamos ~8-9 reemplazos
        out = replace_fraction(BASE_IDS, frac=0.5, vocab_size=VOCAB_SIZE,
                               prefix_len=PREFIX_LEN, seed=42)
        content_changed = sum(
            a != b for a, b in zip(BASE_IDS[PREFIX_LEN:], out[PREFIX_LEN:])
        )
        assert content_changed > 0


class TestRandomIID:
    def test_prefix_preserved(self):
        out = random_iid(BASE_IDS, vocab_size=VOCAB_SIZE,
                         prefix_len=PREFIX_LEN, seed=42)
        assert out[:PREFIX_LEN] == BASE_IDS[:PREFIX_LEN]

    def test_length_preserved(self):
        out = random_iid(BASE_IDS, vocab_size=VOCAB_SIZE,
                         prefix_len=PREFIX_LEN, seed=42)
        assert len(out) == len(BASE_IDS)

    def test_content_differs_from_original(self):
        out = random_iid(BASE_IDS, vocab_size=VOCAB_SIZE,
                         prefix_len=PREFIX_LEN, seed=42)
        # Es estadísticamente casi imposible que sean todos iguales
        assert out[PREFIX_LEN:] != BASE_IDS[PREFIX_LEN:]

    def test_values_in_vocab_range(self):
        out = random_iid(BASE_IDS, vocab_size=VOCAB_SIZE,
                         prefix_len=PREFIX_LEN, seed=42)
        assert all(0 <= v < VOCAB_SIZE for v in out[PREFIX_LEN:])


class TestTrimKeepPrefix:
    def test_shorter_returns_none(self):
        assert trim_keep_prefix(BASE_IDS, target_len=100, prefix_len=PREFIX_LEN) is None

    def test_exact_length_returns_copy(self):
        out = trim_keep_prefix(BASE_IDS, target_len=len(BASE_IDS), prefix_len=PREFIX_LEN)
        assert out == BASE_IDS
        assert out is not BASE_IDS  # debe ser una copia

    def test_longer_trims_correctly(self):
        ids = list(range(30))
        out = trim_keep_prefix(ids, target_len=10, prefix_len=PREFIX_LEN)
        assert len(out) == 10
        assert out[:PREFIX_LEN] == ids[:PREFIX_LEN]


class TestMakeVariants:
    def test_all_keys_present(self):
        variants = make_variants(BASE_IDS, vocab_size=VOCAB_SIZE,
                                 prefix_len=PREFIX_LEN, seed=42)
        expected = {"clean", "shuffle", "replace_10%", "replace_30%", "replace_50%", "random_iid"}
        assert set(variants.keys()) == expected

    def test_clean_is_copy(self):
        variants = make_variants(BASE_IDS, vocab_size=VOCAB_SIZE,
                                 prefix_len=PREFIX_LEN, seed=42)
        assert variants["clean"] == BASE_IDS
        assert variants["clean"] is not BASE_IDS

    def test_all_same_length(self):
        variants = make_variants(BASE_IDS, vocab_size=VOCAB_SIZE,
                                 prefix_len=PREFIX_LEN, seed=42)
        lengths = {len(v) for v in variants.values()}
        assert lengths == {len(BASE_IDS)}