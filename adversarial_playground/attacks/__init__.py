"""Ataques de evasión (*evasion attacks*) implementados desde cero.

Todos comparten la interfaz de :class:`BaseAttack`::

    attack = PGD(model, eps=0.3, alpha=0.075, steps=40)
    result = attack(images, labels)          # -> AttackResult
    result.success_rate                       # fracción de muestras engañadas

Registro ``ATTACK_REGISTRY`` para construir ataques por nombre desde la CLI.
"""

from adversarial_playground.attacks.base import AttackResult, BaseAttack
from adversarial_playground.attacks.cw import CarliniWagnerL2
from adversarial_playground.attacks.deepfool import DeepFool
from adversarial_playground.attacks.fgsm import FGSM
from adversarial_playground.attacks.pgd import BIM, PGD, project_l2, project_linf

ATTACK_REGISTRY: dict[str, type[BaseAttack]] = {
    "fgsm": FGSM,
    "bim": BIM,
    "pgd": PGD,
    "deepfool": DeepFool,
    "cw": CarliniWagnerL2,
}

__all__ = [
    "AttackResult",
    "BaseAttack",
    "FGSM",
    "BIM",
    "PGD",
    "DeepFool",
    "CarliniWagnerL2",
    "project_linf",
    "project_l2",
    "ATTACK_REGISTRY",
]
