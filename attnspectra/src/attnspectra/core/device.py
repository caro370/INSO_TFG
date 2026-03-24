"""
Helpers para gestión de device (CPU / CUDA / MPS).

Funciones pequeñas de conveniencia que evitan repetir la misma lógica
de detección de device en adapters, notebooks y scripts de experimentos.
"""

from __future__ import annotations

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Devuelve el mejor device disponible.

    Orden de preferencia: CUDA → MPS (Apple Silicon) → CPU.

    Parameters
    ----------
    prefer_gpu:
        Si False, devuelve siempre CPU aunque haya GPU disponible.
        Útil para reproducibilidad exacta o debugging.

    Returns
    -------
    ``torch.device`` listo para usar.
    """
    if not prefer_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(
    obj: torch.Tensor | list | dict,
    device: torch.device | str,
) -> torch.Tensor | list | dict:
    """
    Mueve tensores, listas de tensores o dicts de tensores al device indicado.

    Parameters
    ----------
    obj:
        Tensor, lista de tensores (o None) o dict con tensores como valores.
    device:
        Device destino.

    Returns
    -------
    El mismo tipo que la entrada, con todos los tensores en ``device``.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, list):
        return [
            item.to(device) if isinstance(item, torch.Tensor) else item
            for item in obj
        ]

    if isinstance(obj, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in obj.items()
        }

    raise TypeError(
        f"to_device: tipo no soportado '{type(obj).__name__}'. "
        "Se esperaba Tensor, list o dict."
    )


def device_info(device: torch.device | None = None) -> str:
    """
    Devuelve un string descriptivo del device y su memoria disponible.

    Parameters
    ----------
    device:
        Device a describir. Si es None, usa ``get_device()``.

    Returns
    -------
    String con el nombre del device y, si es CUDA, la memoria disponible.
    """
    if device is None:
        device = get_device()

    name = str(device)

    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / 1024 ** 3
        free  = (props.total_memory - torch.cuda.memory_allocated(idx)) / 1024 ** 3
        return f"{name}  |  VRAM total: {total:.1f} GB  |  libre: {free:.1f} GB"

    return name