"""Clase base compartida por todos los ataques.

Todo ataque de evasión de este paquete:

* envuelve un modelo (espera imágenes en ``[0, 1]``, igual que el modelo);
* implementa ``forward(images, labels) -> AttackResult``;
* soporta ataque **no dirigido** (que falle la clase verdadera) y, donde
  tiene sentido, **dirigido** (que prediga una clase objetivo concreta);
* calcula las métricas básicas del resultado: éxito y tamaño de la
  perturbación en norma L2 y L∞.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class AttackResult:
    """Resultado de aplicar un ataque a un lote de imágenes.

    Attributes:
        adv_images: ejemplos adversariales ``(B, C, H, W)``, dentro de los
            límites del clip (``[0, 1]`` normalmente).
        adv_labels: predicción del modelo sobre ``adv_images``.
        success: máscara booleana ``(B,)`` — ``True`` si el ataque engañó al
            modelo en esa muestra (no dirigido: ``adv_labels != labels``).
        l2_norms: norma L2 de la perturbación por muestra.
        linf_norms: norma L∞ de la perturbación por muestra.
        info: métricas adicionales del ataque (iteraciones usadas, valor de
            ``c`` final en C&W, etc.).
    """

    adv_images: Tensor
    adv_labels: Tensor
    success: Tensor
    l2_norms: Tensor
    linf_norms: Tensor
    info: dict[str, Tensor] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fracción de muestras engañadas (batch completo)."""
        return float(self.success.float().mean().item())

    @property
    def mean_l2(self) -> float:
        return float(self.l2_norms.mean().item())

    @property
    def mean_linf(self) -> float:
        return float(self.linf_norms.mean().item())


class BaseAttack(ABC):
    """Base común para ataques de evasión (*evasion attacks*).

    Args:
        model: modelo objetivo; recibe imágenes en ``[0, 1]`` y devuelve logits.
        clip_min: valor mínimo permitido de píxel (0.0 en imágenes en [0,1]).
        clip_max: valor máximo permitido de píxel (1.0).
    """

    def __init__(self, model: nn.Module, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        self.model = model
        self.clip_min = clip_min
        self.clip_max = clip_max
        # Nota: NO congelamos los parámetros del modelo. Los ataques calculan
        # gradiente solo respecto a la *entrada* con torch.autograd.grad (que
        # no acumula en .grad), así que el mismo ataque sirve tanto para
        # evaluar modelos en eval() como para generar adversariales dentro
        # del entrenamiento adversarial (donde los parámetros deben poder
        # recibir gradiente después).

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def __call__(self, images: Tensor, labels: Tensor) -> AttackResult:
        """Alias de :meth:`forward` para que ``attack(x, y)`` sea ergonómico."""
        return self.forward(images, labels)

    @abstractmethod
    def forward(self, images: Tensor, labels: Tensor) -> AttackResult:
        """Genera ejemplos adversariales para el lote dado.

        Args:
            images: imágenes limpias ``(B, C, H, W)`` en ``[clip_min, clip_max]``.
            labels: etiquetas verdaderas ``(B,)`` (o objetivo, si es dirigido).
        """

    # ------------------------------------------------------------------ #
    # Helpers reutilizables
    # ------------------------------------------------------------------ #
    def predict(self, images: Tensor) -> Tensor:
        """Logits del modelo sobre un lote (sin gradiente)."""
        with torch.no_grad():
            return self.model(images)

    def clamp(self, images: Tensor) -> Tensor:
        """Recorta al rango válido de píxeles."""
        return images.clamp(self.clip_min, self.clip_max)

    def compute_result(
        self, clean_images: Tensor, adv_images: Tensor, labels: Tensor, targeted: bool = False
    ) -> AttackResult:
        """Construye el :class:`AttackResult` a partir del lote adversarial.

        Calcula predicciones del modelo sobre los adversariales, máscara de
        éxito y normas de la perturbación ``delta = adv - clean``.
        """
        delta = adv_images - clean_images
        adv_labels = self.predict(adv_images).argmax(dim=1)
        if targeted:
            success = adv_labels == labels
        else:
            success = adv_labels != labels
        return AttackResult(
            adv_images=adv_images.detach(),
            adv_labels=adv_labels,
            success=success,
            l2_norms=delta.flatten(1).norm(p=2, dim=1),
            linf_norms=delta.flatten(1).abs().amax(dim=1),
        )

    @staticmethod
    def _validate_batch(images: Tensor, labels: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"Se esperaban imágenes (B, C, H, W); llega {tuple(images.shape)}"
            )
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"images ({images.shape[0]}) y labels ({labels.shape[0]}) no coinciden"
            )

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if k != "model")
        name = type(self).__name__
        return f"{name}(model={type(self.model).__name__}{', ' + args if args else ''})"
