"""
Utilidades de reproducibilidad: fijar semillas y hacer snapshots del estado RNG.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fija las semillas de Python, NumPy y PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict[str, Any]:
    """Captura el estado completo de los generadores aleatorios."""
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_all"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restaura el estado de los generadores aleatorios desde un snapshot."""
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and "cuda_random_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_random_all"])