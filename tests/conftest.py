"""Fixtures compartidas de los tests.

Estrategia: para mantener los tests **rápidos y deterministas** usamos el
MnistCNN *sin entrenar* (inicialización sembrada) sobre imágenes aleatorias
en [0, 1]. Matemáticamente los ataques funcionan igual sobre un modelo sin
entrenar (el gradiente existe); la etiqueta "verdadera" de cada muestra es
la propia predicción del modelo, así que al principio el modelo acierta el
100% y el éxito del ataque == cambio de predicción.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture()
def model():
    from adversarial_playground.models import MnistCNN

    torch.manual_seed(123)
    m = MnistCNN()
    m.eval()
    return m


@pytest.fixture()
def batch():
    return torch.rand(8, 1, 28, 28)


@pytest.fixture()
def labels(model, batch):
    """Etiqueta de cada imagen = su predicción limpia (modelo acierta 100%)."""
    with torch.no_grad():
        return model(batch).argmax(dim=1)
