"""Tests de la maquinaria de evaluación y utilidades."""

import torch

from adversarial_playground.attacks.factory import build_attack
from adversarial_playground.evaluation.metrics import (
    evaluate_attack,
    robust_accuracy_sweep,
)
from adversarial_playground.utils.data import denormalize


class _TinyLoader:
    """Loader sintético de 2 batches (evita descargar datasets en CI)."""

    def __init__(self, model, n_batches=2, size=8):
        self.model = model
        self.n_batches = n_batches
        self.size = size

    def __iter__(self):
        for _ in range(self.n_batches):
            x = torch.rand(self.size, 1, 28, 28)
            with torch.no_grad():
                y = self.model(x).argmax(1)
            yield x, y


def test_evaluate_attack_sanity(model):
    loader = _TinyLoader(model)
    attack = build_attack("fgsm", model, eps=0.4)
    ev = evaluate_attack(model, attack, loader, show_progress=False)

    assert ev.n_samples == 16
    assert ev.clean_acc == 1.0  # etiquetas = predicciones del propio modelo
    assert 0.0 <= ev.robust_acc <= 1.0
    assert ev.mean_l2 > 0


def test_evaluate_attack_max_samples(model):
    loader = _TinyLoader(model)
    attack = build_attack("fgsm", model, eps=0.4)
    ev = evaluate_attack(model, attack, loader, max_samples=10, show_progress=False)
    assert ev.n_samples == 10


def test_robust_sweep_shape(model):
    loader = _TinyLoader(model)
    factory = lambda eps: build_attack("fgsm", model, eps=eps)  # noqa: E731
    rows = robust_accuracy_sweep(model, factory, [0.1, 0.3], loader, max_samples=8)
    assert len(rows) == 2
    assert rows[0]["eps"] == 0.1 and "robust_acc" in rows[0]


def test_denormalize_cifar_stretches_contrast():
    x = torch.rand(1, 3, 8, 8) * 0.2 + 0.4  # rango estrecho
    out = denormalize(x, "cifar10")
    assert out.min() >= 0 and out.max() <= 1
    assert out.max() > x.max()  # estiró el contraste


def test_denormalize_cifar_accepts_single_image():
    """Las funciones de visualización pasan imágenes (C, H, W), no lotes."""
    x = torch.rand(3, 8, 8) * 0.2 + 0.4
    out = denormalize(x, "cifar10")
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1


def test_denormalize_mnist_is_identity_range():
    x = torch.rand(1, 1, 8, 8)
    out = denormalize(x, "mnist")
    assert out.min() >= 0 and out.max() <= 1
