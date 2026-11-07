"""PGD — *Projected Gradient Descent* (Madry et al., 2017).

El "ataque universal de primer orden": FGSM repetido **con proyección** sobre
la bola de radio ``eps`` y, opcionalmente, **arranque aleatorio** dentro de
la bola.

    x₀ = x + U(-eps, eps)                (arranque aleatorio)
    x_{t+1} = Π( clip(x_t + alpha · sign(∇_x L)) , ||x - x_orig|| ≤ eps )

Es hoy el estándar *de facto*: la robustez de un modelo se mide contra PGD,
y el entrenamiento adversarial de Madry se entrena con él. Con
``random_start=False`` se recupera BIM/I-FGSM (Kurakin et al., 2016).

Referencias:
    * Madry et al.: https://arxiv.org/abs/1706.06083
    * BIM (Kurakin et al.): https://arxiv.org/abs/1607.02533
"""

import torch
from torch import Tensor, nn

from adversarial_playground.attacks.base import AttackResult, BaseAttack


def project_linf(delta: Tensor, eps: float | Tensor) -> Tensor:
    """Proyecta la perturbación sobre la bola L∞: recorta a ``[-eps, eps]``."""
    return delta.clamp(-eps, eps)


def project_l2(delta: Tensor, eps: float | Tensor) -> Tensor:
    """Proyecta la perturbación sobre la bola L2 de radio ``eps`` por muestra.

    Si ``||delta_i||_2 > eps``, se reescala a longitud exactamente ``eps``;
    en otro caso se deja igual (la bola es convexa y el punto ya está dentro).
    """
    flat = delta.flatten(1)
    norms = flat.norm(p=2, dim=1).clamp_min(1e-12)
    # escala = min(1, eps / ||delta||) por muestra
    scale = torch.minimum(torch.ones_like(norms), eps / norms)
    return delta * scale.view(-1, *([1] * (delta.ndim - 1)))


class PGD(BaseAttack):
    """Ataque iterativo sobre bolas L∞ o L2, con proyección tras cada paso.

    Args:
        model: modelo objetivo (logits, imágenes en [0, 1]).
        eps: radio de la bola de restricción.
        alpha: tamaño de paso por iteración (regla práctica: ``eps / 4`` en
            L∞; en L2, ``eps / 4`` sobre la bola L2 también funciona bien).
        steps: número de iteraciones (10 para experimentos rápidos, 40-100
            para la evaluación "seria").
        norm: ``"linf"`` (por defecto) o ``"l2"``.
        random_start: si ``True``, arranca desde un punto uniforme dentro de
            la bola. Importante: mejora la cobertura de la bola y evita que
            defensas con gradientes "enmascarados" en x=0 sobrevivan.
        targeted: ataque dirigido (minimiza CE hacia ``labels``).
        clip_min, clip_max: rango válido de píxeles.

    Example:
        >>> attack = PGD(model, eps=0.3, alpha=0.075, steps=40)
        >>> result = attack(images, labels)
        >>> result.success_rate  # doctest: +SKIP
        0.99
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.3,
        alpha: float | None = None,
        steps: int = 40,
        norm: str = "linf",
        random_start: bool = True,
        targeted: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(model, clip_min, clip_max)
        if norm not in ("linf", "l2"):
            raise ValueError(f"norm debe ser 'linf' o 'l2', llega {norm!r}")
        if steps < 1:
            raise ValueError(f"steps debe ser >= 1, llega {steps}")
        if alpha is None:
            alpha = eps / 4.0
        if alpha <= 0 or alpha > eps:
            raise ValueError(
                f"alpha ({alpha}) debe estar en (0, eps={eps}] para que el "
                f"ataque avance sin saltarse la bola entera"
            )
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.norm = norm
        self.random_start = random_start
        self.targeted = targeted
        self.project = project_linf if norm == "linf" else project_l2

    def _project(self, delta: Tensor) -> Tensor:
        return self.project(delta, self.eps)

    def _step(self, delta: Tensor, grad: Tensor) -> Tensor:
        """Un paso de ascenso (o descenso, si es dirigido) + proyección.

        Trabaja sobre la **perturbación** ``delta = x - x_orig`` (relativa al
        original, no sobre la imagen absoluta): así la proyección sobre la
        bola de radio ``eps`` es exacta.
        """
        if self.norm == "linf":
            step = self.alpha * grad.sign()
        else:
            # Paso unitario en la dirección del gradiente (L2) escalado por alpha.
            per_sample_norm = grad.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
            step = self.alpha * grad / per_sample_norm.view(-1, 1, 1, 1)
        step = -step if self.targeted else step
        return self._project(delta + step)

    def forward(self, images: Tensor, labels: Tensor) -> AttackResult:
        self._validate_batch(images, labels)
        x_orig = images.clone().detach()

        if self.random_start:
            if self.norm == "linf":
                delta = torch.empty_like(x_orig).uniform_(-self.eps, self.eps)
            else:
                # Punto uniforme en la bola L2: dirección aleatoria, radio
                # aleatorio ~ sqrt(U(0, 1)) para densidad uniforme.
                direction = torch.randn_like(x_orig)
                dir_norm = direction.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                direction = direction / dir_norm.clamp_min(1e-12)
                radius = self.eps * torch.rand(
                    x_orig.shape[0], 1, 1, 1, device=x_orig.device
                ).sqrt()
                delta = direction * radius
            x = self.clamp(x_orig + delta).detach()
        else:
            x = x_orig.clone()

        for _ in range(self.steps):
            x.requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(self.model(x), labels)
            grad = torch.autograd.grad(loss, x)[0]
            # Ascenso + proyección sobre la bola + clip al rango válido.
            delta = self._step(x.detach() - x_orig, grad)
            x = self.clamp(x_orig + delta).detach()

        adv = x.detach()
        return self.compute_result(images, adv, labels, targeted=self.targeted)


class BIM(PGD):
    """BIM / I-FGSM (Kurakin et al., 2016): PGD sin arranque aleatorio.

    Históricamente anterior a PGD; hoy se entiende como un caso particular.
    Se mantiene como clase propia porque la literatura compara las tres
    variantes (FGSM < BIM < PGD) y es cómodo nombrarlas.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 0.3,
        alpha: float | None = None,
        steps: int = 10,
        norm: str = "linf",
        targeted: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        if alpha is None:
            alpha = eps / 10.0  # BIM clásico usa pasos más finos
        super().__init__(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            norm=norm,
            random_start=False,
            targeted=targeted,
            clip_min=clip_min,
            clip_max=clip_max,
        )
