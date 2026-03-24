"""
Transformaciones de secuencias de token ids para experimentos de perturbación.
"""

from __future__ import annotations

import random

def shuffle_content(
    ids: list[int],
    prefix_len: int = 3,
    seed: int | None = None,
) -> list[int]:
    """
    Baraja aleatoriamente los tokens de contenido, preservando el prefijo.

    Parameters
    ----------
    ids:
        Lista de token ids completa (prefijo + contenido).
    prefix_len:
        Número de tokens de prefijo a preservar (p.ej. BOS + style + SEP = 3).
    seed:
        Semilla para reproducibilidad.

    Returns
    -------
    Nueva lista con el prefijo intacto y el contenido barajado.
    """
    rng = random.Random(seed)
    prefix = ids[:prefix_len]
    content = list(ids[prefix_len:])
    rng.shuffle(content)
    return prefix + content


def replace_fraction(
    ids: list[int],
    frac: float = 0.2,
    vocab_size: int = 32_000,
    prefix_len: int = 3,
    seed: int | None = None,
) -> list[int]:
    """
    Reemplaza una fracción aleatoria de tokens de contenido con ids aleatorios.

    Parameters
    ----------
    ids:
        Lista de token ids.
    frac:
        Fracción de tokens de contenido a reemplazar (entre 0 y 1).
    vocab_size:
        Tamaño del vocabulario para generar ids aleatorios.
    prefix_len:
        Tokens de prefijo a preservar.
    seed:
        Semilla para reproducibilidad.

    Returns
    -------
    Nueva lista con el prefijo intacto y ``frac`` tokens reemplazados.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac debe estar en [0,1], se recibió {frac}.")

    rng = random.Random(seed)
    out = list(ids)
    content_indices = list(range(prefix_len, len(ids)))
    k = int(len(content_indices) * frac)
    if k <= 0:
        return out
    chosen = rng.sample(content_indices, k=k)
    for i in chosen:
        out[i] = rng.randint(0, vocab_size - 1)
    return out


def random_iid(
    ids: list[int],
    vocab_size: int = 32_000,
    prefix_len: int = 3,
    seed: int | None = None,
) -> list[int]:
    """
    Sustituye todos los tokens de contenido por ids uniformemente aleatorios.
    Preserva el prefijo y la longitud total.

    Parameters
    ----------
    ids:
        Lista de token ids.
    vocab_size:
        Tamaño del vocabulario.
    prefix_len:
        Tokens de prefijo a preservar.
    seed:
        Semilla para reproducibilidad.

    Returns
    -------
    Nueva lista con el prefijo intacto y contenido completamente aleatorio.
    """
    rng = random.Random(seed)
    prefix = ids[:prefix_len]
    T = len(ids) - prefix_len
    content = [rng.randint(0, vocab_size - 1) for _ in range(T)]
    return prefix + content


def trim_keep_prefix(
    ids: list[int],
    target_len: int,
    prefix_len: int = 3,
) -> list[int] | None:
    """
    Ajusta ``ids`` a exactamente ``target_len`` tokens preservando el prefijo.

    Si ``len(ids) < target_len`` devuelve ``None`` (secuencia demasiado corta).
    Si ``len(ids) > target_len`` recorta por el final del contenido.
    Si ``len(ids) == target_len`` devuelve una copia.

    Returns
    -------
    Lista de ``target_len`` tokens, o ``None``.
    """
    if len(ids) < target_len:
        return None
    if len(ids) == target_len:
        return list(ids)
    keep_tail = target_len - prefix_len
    return list(ids[:prefix_len]) + list(ids[-keep_tail:])


def make_variants(
    ids: list[int],
    vocab_size: int = 32_000,
    prefix_len: int = 3,
    seed: int = 123,
) -> dict[str, list[int]]:
    """
    Genera el conjunto estándar de variantes perturbadas para los experimentos.

    Returns
    -------
    Dict con claves: ``"clean"``, ``"shuffle"``, ``"replace_10%"``,
    ``"replace_30%"``, ``"replace_50%"``, ``"random_iid"``.
    """
    return {
        "clean":        list(ids),
        "shuffle":      shuffle_content(ids, prefix_len=prefix_len, seed=seed),
        "replace_10%":  replace_fraction(ids, frac=0.1, vocab_size=vocab_size, prefix_len=prefix_len, seed=seed),
        "replace_30%":  replace_fraction(ids, frac=0.3, vocab_size=vocab_size, prefix_len=prefix_len, seed=seed),
        "replace_50%":  replace_fraction(ids, frac=0.5, vocab_size=vocab_size, prefix_len=prefix_len, seed=seed),
        "random_iid":   random_iid(ids, vocab_size=vocab_size, prefix_len=prefix_len, seed=seed),
    }