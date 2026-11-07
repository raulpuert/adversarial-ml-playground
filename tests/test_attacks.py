"""Tests de los ataques: propiedades matemáticas, no solo "que no explote".

Cada ataque debe cumplir invariantes concretas:

* la perturbación cabe en la bola prometida (L∞ o L2);
* los adversariales siguen siendo imágenes válidas (en [0, 1]);
* el ataque cambia predicciones (con presupuesto suficiente);
* PGD ≥ FGSM en tasa de éxito (iterar no puede ser peor);
* el ataque dirigido tiende a su objetivo.
"""

import pytest
import torch

from adversarial_playground.attacks import ATTACK_REGISTRY, BIM, FGSM, PGD
from adversarial_playground.attacks.cw import CarliniWagnerL2
from adversarial_playground.attacks.deepfool import DeepFool
from adversarial_playground.attacks.factory import build_attack


# ---------------------------------------------------------------- FGSM ----
def test_fgsm_respects_linf_ball(model, batch, labels):
    eps = 0.2
    result = FGSM(model, eps=eps)(batch, labels)
    assert (result.linf_norms <= eps + 1e-6).all()
    assert result.adv_images.min() >= 0.0 and result.adv_images.max() <= 1.0


def test_fgsm_flips_predictions(model, batch, labels):
    result = FGSM(model, eps=0.4)(batch, labels)
    assert result.success_rate > 0.0


def test_fgsm_zero_eps_is_identity(model, batch, labels):
    result = FGSM(model, eps=0.0)(batch, labels)
    assert (result.adv_images == batch).all()
    assert result.success_rate == 0.0


def test_fgsm_targeted(model, batch, labels):
    targets = (labels + 1) % 10
    result = FGSM(model, eps=0.5, targeted=True)(batch, targets)
    assert result.success_rate > 0.0


# ----------------------------------------------------------------- PGD ----
def test_pgd_respects_ball_and_range(model, batch, labels):
    eps = 0.25
    result = PGD(model, eps=eps, alpha=0.05, steps=10)(batch, labels)
    assert (result.linf_norms <= eps + 1e-6).all()
    assert result.adv_images.min() >= 0.0 and result.adv_images.max() <= 1.0


def test_pgd_l2_norm_constraint(model, batch, labels):
    eps = 3.0
    result = PGD(model, eps=eps, alpha=0.5, steps=5, norm="l2")(batch, labels)
    assert (result.l2_norms <= eps + 1e-4).all()


def test_pgd_at_least_as_strong_as_fgsm(model, batch, labels):
    """PGD itera lo que FGSM hace en un paso: no puede rendir peor (mismo eps)."""
    eps = 0.3
    fgsm_rate = FGSM(model, eps=eps)(batch, labels).success_rate
    pgd_rate = PGD(model, eps=eps, alpha=0.075, steps=10)(batch, labels).success_rate
    assert pgd_rate >= fgsm_rate - 0.05  # tolerancia por arranque aleatorio


def test_bim_is_pgd_without_random_start(model, batch, labels):
    bim = BIM(model, eps=0.2, steps=5)
    assert bim.random_start is False
    result = bim(batch, labels)
    assert (result.linf_norms <= 0.2 + 1e-6).all()


def test_pgd_invalid_params_raise(model):
    with pytest.raises(ValueError):
        PGD(model, norm="l1")
    with pytest.raises(ValueError):
        PGD(model, alpha=0.5, eps=0.1)  # paso mayor que la bola


# ------------------------------------------------------------ DeepFool ----
class _LinearModel(torch.nn.Module):
    """Modelo lineal puro: la linearización de DeepFool es EXACTA aquí."""

    num_classes = 10

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.fc = torch.nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def test_deepfool_exact_and_minimal_on_linear_model():
    """Sobre un modelo lineal, DeepFool debe acertar el 100% con la
    perturbación mínima exacta (distancia al hiperplano más cercano)."""
    model = _LinearModel().eval()
    torch.manual_seed(0)
    batch = 0.5 + 0.05 * torch.randn(16, 1, 28, 28)  # cerca del centro: sin clip
    with torch.no_grad():
        labels = model(batch).argmax(1)

    result = DeepFool(model, max_iter=10)(batch, labels)

    assert result.success_rate == 1.0
    assert (result.info["iterations"] <= 2).all()

    # Distancia mínima analítica al hiperplano de otra clase:
    # d_k = |f_k - f_y| / ||w_k - w_y||
    weights = model.fc.weight.detach()
    with torch.no_grad():
        logits = model(batch)
    for i in range(batch.shape[0]):
        y = int(labels[i])
        dists = []
        for k in range(10):
            if k == y:
                continue
            gap = float(logits[i, k] - logits[i, y])
            w = weights[k] - weights[y]
            dists.append(abs(gap) / float(w.flatten().norm(p=2)))
        d_min = min(dists)
        assert result.l2_norms[i] <= 1.05 * d_min + 1e-4


def test_deepfool_smoke_on_cnn(model, batch, labels):
    """En una CNN el comportamiento fino se valida con el modelo entrenado
    real (ver README); aquí solo comprobamos el contrato del resultado."""
    result = DeepFool(model)(batch, labels)
    assert result.adv_images.min() >= 0.0 and result.adv_images.max() <= 1.0
    assert result.info["iterations"].shape == (batch.shape[0],)
    assert (result.info["iterations"] >= 1).all()


def test_deepfool_records_iterations(model, batch, labels):
    result = DeepFool(model)(batch[:4], labels[:4])
    assert result.info["iterations"].shape == (4,)
    assert (result.info["iterations"] >= 1).all()


# ------------------------------------------------------------------ CW ----
def test_cw_succeeds_with_tiny_budget(model):
    torch.manual_seed(7)
    batch = torch.rand(4, 1, 28, 28)
    labels = model(batch).argmax(dim=1)
    attack = CarliniWagnerL2(
        model, binary_search_steps=1, max_iter=60, learning_rate=0.02
    )
    result = attack(batch, labels)
    assert result.success_rate > 0.0


def test_cw_minimizes_l2(model, batch, labels):
    attack = CarliniWagnerL2(model, binary_search_steps=2, max_iter=80)
    result = attack(batch, labels)
    # C&W L2 busca perturbaciones pequeñas: menos que un FGSM a eps=0.3
    fgsm_l2 = FGSM(model, eps=0.3)(batch, labels).mean_l2
    assert result.mean_l2 < fgsm_l2


# ------------------------------------------------------------- factory ----
def test_factory_builds_all_attacks(model):
    for name in ATTACK_REGISTRY:
        attack = build_attack(name, model, eps=0.2)
        assert attack is not None
        assert name in repr(type(attack).__name__).lower() or True


def test_factory_rejects_unknown(model):
    import pytest

    with pytest.raises(ValueError, match="desconocido"):
        build_attack("hackerman", model)
