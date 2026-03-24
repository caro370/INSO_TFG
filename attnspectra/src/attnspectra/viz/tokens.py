"""
Utilidades para formatear etiquetas de tokens en visualizaciones.
"""

from __future__ import annotations


def format_token_labels(tokens: list[str], max_len: int = 12) -> list[str]:
    """
    Escapa caracteres especiales y trunca tokens largos para ejes de gráficos.
    """
    result = []
    for t in tokens:
        t = t.replace("\n", "\\n").replace("\t", "\\t")
        if len(t) > max_len:
            t = t[:max_len] + "…"
        result.append(t)
    return result