"""Gestión de semillas y dispositivo para reproducibilidad."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de ``random``, ``numpy`` y ``torch`` (CPU y CUDA).

    La reproducibilidad es importante en seguridad adversarial: si el ataque
    usa un arranque aleatorio (PGD), repetir el experimento debe dar los
    mismos números, si no no hay forma de comparar defensas.

    Args:
        seed: entero arbitrario. Se usa el mismo valor en todas las librerías.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Devuelve el mejor dispositivo disponible (cuda > mps > cpu).

    Args:
        prefer: fuerza un dispositivo concreto (``"cpu"``, ``"cuda"``, ``"mps"``).
            Si es ``None``, se elige automáticamente.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
