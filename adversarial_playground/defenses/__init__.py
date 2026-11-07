"""Defensas contra ataques adversariales.

Tres familias, de la más simple a la más sólida:

* :mod:`transforms` — transformaciones de entrada (baratas, débiles solas).
* :mod:`training` — entrenamiento adversarial PGD-AT (estándar de oro empírico).
* :mod:`smoothing` — randomized smoothing (garantías certificables L2).
"""

from adversarial_playground.defenses.smoothing import RandomizedSmoothing
from adversarial_playground.defenses.training import (
    AdvTrainingConfig,
    adversarial_loss,
    generate_pgd_batch,
)
from adversarial_playground.defenses.transforms import (
    BitDepthReduction,
    GaussianNoise,
    JPEGCompression,
    TotalVarianceMinimization,
)

__all__ = [
    "AdvTrainingConfig",
    "adversarial_loss",
    "generate_pgd_batch",
    "BitDepthReduction",
    "JPEGCompression",
    "GaussianNoise",
    "TotalVarianceMinimization",
    "RandomizedSmoothing",
]
