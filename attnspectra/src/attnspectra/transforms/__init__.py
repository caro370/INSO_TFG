"""
transforms
==========
Transformaciones de secuencias de token ids.

Todas las funciones son **puras**: reciben una ``list[int]``
y devuelven una ``list[int]`` nueva sin modificar la entrada.
Son la base de los experimentos de degradación.

Exports
-------
shuffle_content     — baraja tokens de contenido preservando el prefijo
replace_fraction    — reemplaza una fracción de tokens por ids aleatorios
random_iid          — sustituye todo el contenido por ids uniformes i.i.d.
trim_keep_prefix    — ajusta la longitud preservando el prefijo
make_variants       — genera el conjunto estándar de 6 variantes perturbadas
"""

from attnspectra.transforms.token_perturbations import (
    shuffle_content,
    replace_fraction,
    random_iid,
    trim_keep_prefix,
    make_variants,
)

__all__ = [
    "shuffle_content",
    "replace_fraction",
    "random_iid",
    "trim_keep_prefix",
    "make_variants",
]