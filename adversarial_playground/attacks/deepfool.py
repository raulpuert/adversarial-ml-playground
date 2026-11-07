"""DeepFool (Moosavi-Dezfooli, Fawzi & Frossard, 2016).

A diferencia de FGSM/PGD (que gastan *todo* el presupuesto ``eps``), DeepFool
busca la **perturbación mínima** que cruza la frontera de decisión:

1. Aproxima cada clase como un hiperplano (linearización de primer orden).
2. Calcula el vector que lleva al hiperplano de la clase "más cercana".
3. Repite hasta que cambia la predicción, con un pequeño *overshoot*.

El resultado es una perturbación minúscula (norma típicamente 10x menor que
PGD) — ideal para medir "cuánto margen le queda a una muestra" y para
visualizar lo frágil de la frontera de decisión.

Referencia: https://arxiv.org/abs/1511.04599
"""

import torch
from torch import Tensor, nn

from adversarial_playground.attacks.base import AttackResult, BaseAttack


class DeepFool(BaseAttack):
    """Ataque de perturbación mínima (L2) por linearización iterativa.

    Args:
        model: modelo objetivo (logits, imágenes en [0, 1]).
        overshoot: factor extra aplicado a la perturbación final para
            garantizar que se cruza la frontera (0.02 en el paper).
        max_iter: iteraciones máximas por muestra (50 basta en MNIST).
        clip_min, clip_max: rango válido de píxeles.

    Note:
        Es un ataque **por muestra** (no vectorizado por lote): la búsqueda
        del hiperplano más cercano es inherencialmente secuencial. Para
        evaluar sobre el test completo, usa un subconjunto (p. ej. 1000
        imágenes): las tasas de éxito son estables con eso.

    Example:
        >>> attack = DeepFool(model)
        >>> result = attack(images, labels)
        >>> result.mean_l2  # doctest: +SKIP
        0.04
    """

    def __init__(
        self,
        model: nn.Module,
        overshoot: float = 0.02,
        max_iter: int = 50,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, clip_min, clip_max)
        self.overshoot = overshoot
        self.max_iter = max_iter

    def _closest_hyperplane(
        self, logits: Tensor, class_grads: Tensor, true_label: int
    ) -> tuple[int, Tensor, Tensor]:
        """Devuelve ``(k, w, f)`` del hiperplano de la clase más cercana.

        Para cada clase ``k != y``, la linearización de primer orden de la
        diferencia de logits alrededor del punto actual es
        ``f_k(x+δ) ≈ f_k(x) + w_k · δ`` con ``w_k = ∇f_k − ∇f_y``.
        Elegimos la clase con menor distancia ``|f_k| / ||w_k||₂``.
        """
        # Todo en modo "números": aquí solo se leen valores.
        logits = logits.detach()
        class_grads = class_grads.detach()
        f_y = logits[true_label]
        grad_y = class_grads[true_label]
        best_k, best_score = -1, float("inf")
        best_w, best_f = None, None
        for k in range(logits.shape[0]):
            if k == true_label:
                continue
            w = class_grads[k] - grad_y
            f = logits[k] - f_y
            norm = w.flatten().norm(p=2).clamp_min(1e-12)
            score = float(f.abs()) / float(norm)
            if score < best_score:
                best_k, best_score, best_w, best_f = k, score, w, f
        return best_k, best_w, best_f

    def _attack_single(self, image: Tensor, label: Tensor) -> tuple[Tensor, int]:
        """DeepFool sobre una imagen ``(1, C, H, W)``; devuelve (adv, iters).

        Forma iterativa práctica del algoritmo (como en las implementaciones
        de referencia): en cada iteración se lineariza en el punto actual, se
        da **un paso completo** hacia el hiperplano de la clase más cercana
        y se recorta al dominio válido. Las iteraciones se acercan a la
        frontera real como un método de Newton (los pasos encojen al
        encojerse ``|f|``).
        """
        x = image.clone().detach()
        adv = x.clone()
        iters = 0

        for _ in range(self.max_iter):
            iters += 1
            x = x.detach().requires_grad_(True)
            logits = self.model(x)[0]

            if int(logits.argmax()) != int(label):
                adv = x.detach()
                break

            # Jacobiano completo en una pasada (vmap sobre las clases).
            # ATENCIÓN al orden de ejes de functional.jacobian: devuelve
            # (*output, *input) → aquí (K, 1, C, H, W): el gradiente de cada
            # clase respecto a la imagen.
            jac = torch.autograd.functional.jacobian(
                lambda z: self.model(z)[0], x, vectorize=True
            )
            class_grads = jac.reshape(logits.shape[0], *x.shape[1:])

            k, w, f = self._closest_hyperplane(logits, class_grads, int(label))
            # Paso completo al hiperplano de la clase k (paper, ecuación 6):
            # delta = (|f| / ||w||²) · w — con |f| en valor absoluto, para
            # moverse HACIA el hiperplano (f < 0: la clase k aún pierde).
            norm_sq = float(w.flatten().norm(p=2).clamp_min(1e-12)) ** 2
            delta = (float(f.abs()) / norm_sq) * w
            x = self.clamp(x + (1.0 + self.overshoot) * delta)
        else:
            adv = x.detach()

        return adv.detach(), iters

    def forward(self, images: Tensor, labels: Tensor) -> AttackResult:
        self._validate_batch(images, labels)
        adv_list: list[Tensor] = []
        iters_used: list[int] = []
        for i in range(images.shape[0]):
            adv_i, iters = self._attack_single(images[i : i + 1], labels[i])
            adv_list.append(adv_i)
            iters_used.append(iters)

        adv = torch.cat(adv_list, dim=0)
        result = self.compute_result(images, adv, labels, targeted=False)
        result.info["iterations"] = torch.tensor(iters_used, dtype=torch.float32)
        return result
