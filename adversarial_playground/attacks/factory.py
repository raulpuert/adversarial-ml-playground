"""Construcción de ataques por nombre con valores por defecto razonables.

Usada por la CLI y los scripts para que ``--attack pgd --eps 0.3`` haga lo
esperado sin configurar cada parámetro a mano.
"""

from __future__ import annotations

from torch import nn

from adversarial_playground.attacks.base import BaseAttack
from adversarial_playground.attacks.cw import CarliniWagnerL2
from adversarial_playground.attacks.deepfool import DeepFool
from adversarial_playground.attacks.fgsm import FGSM
from adversarial_playground.attacks.pgd import BIM, PGD


def build_attack(
    name: str,
    model: nn.Module,
    eps: float = 0.3,
    steps: int | None = None,
    targeted: bool = False,
) -> BaseAttack:
    """Instancia un ataque del registro por nombre.

    Args:
        name: ``"fgsm"``, ``"bim"``, ``"pgd"``, ``"deepfool"`` o ``"cw"``.
        model: modelo objetivo.
        eps: presupuesto L∞ (ignorado por DeepFool y C&W).
        steps: iteraciones para BIM/PGD (por defecto 10 y 40 respectivamente).
        targeted: ataque dirigido (no soportado por DeepFool/C&W).

    Raises:
        ValueError: si ``name`` no es un ataque conocido.
    """
    name = name.lower()
    if name == "fgsm":
        return FGSM(model, eps=eps, targeted=targeted)
    if name == "bim":
        return BIM(model, eps=eps, steps=steps or 10, targeted=targeted)
    if name == "pgd":
        return PGD(model, eps=eps, alpha=eps / 4, steps=steps or 40, targeted=targeted)
    if name == "deepfool":
        if targeted:
            raise ValueError("DeepFool no soporta ataque dirigido")
        return DeepFool(model)
    if name == "cw":
        if targeted:
            raise ValueError("Usa CarliniWagnerL2(..., targeted=True) directamente")
        return CarliniWagnerL2(model, targeted=False)
    raise ValueError(f"Ataque '{name}' desconocido. Disponibles: fgsm, bim, pgd, deepfool, cw")
