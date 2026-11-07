"""FGSM — *Fast Gradient Sign Method* (Goodfellow, Shlens & Szegedy, 2014).

El ataque adversarial clásico: **un solo paso** en la dirección del signo del
gradiente de la pérdida respecto a la entrada.

    x_adv = clip( x + eps * sign(∇_x L(f(x), y)) )

Es el ataque más rápido posible (una pasada forward + una backward) y el
punto de partida conceptual de todos los demás. Su debilidad: un solo paso
"se pasa" o "se queda corto" — por eso existen las versiones iterativas
(BIM, PGD).

Referencia: https://arxiv.org/abs/1412.6572
"""

import torch
from torch import Tensor, nn

from adversarial_playground.attacks.base import AttackResult, BaseAttack


class FGSM(BaseAttack):
    """Ataque de un paso sobre la bola L∞ de radio ``eps``.

    Args:
        model: modelo objetivo (logits, imágenes en [0, 1]).
        eps: radio de la perturbación L∞. Para MNIST en [0, 1], 0.1-0.3 son
            valores habituales (con 0.3 la distorsión es visible).
        targeted: si ``True``, *minimiza* la pérdida hacia ``labels`` (que en
            ese caso deben ser las clases objetivo, no las verdaderas).
        clip_min, clip_max: rango válido de píxeles.

    Example:
        >>> attack = FGSM(model, eps=0.15)
        >>> result = attack(images, labels)
        >>> result.success_rate  # doctest: +SKIP
        0.87
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.3,
        targeted: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, clip_min, clip_max)
        if eps < 0:
            raise ValueError(f"eps debe ser >= 0, llega {eps}")
        self.eps = eps
        self.targeted = targeted

    def forward(self, images: Tensor, labels: Tensor) -> AttackResult:
        self._validate_batch(images, labels)
        x = images.clone().detach().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(self.model(x), labels)
        grad = torch.autograd.grad(loss, x)[0]

        # sign() hace que cada píxel se empuje lo máximo posible (bola L∞)
        # con la dirección del gradiente. Dirigido: descendemos hacia el target.
        step = self.eps * grad.sign()
        adv = x.detach() - step if self.targeted else x.detach() + step
        adv = self.clamp(adv)
        return self.compute_result(images, adv, labels, targeted=self.targeted)
